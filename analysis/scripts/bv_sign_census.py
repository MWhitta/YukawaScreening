"""Sign census of per-shell bond-valence fits.

Reproduces the numbers quoted in the Supplemental Material paragraph
"Sign census of the per-shell fits": counts of fitted single-valence
cation--oxygen shells with B < 0, their split across the z<n / z=n / z>n
regimes, the near-degeneracy statistics, and representative per-species
B < 0 fractions.

Population: all `status == "fitted"` records of both groups (oxides and
hydroxides) in the `oxygen_authoritative` payload of the consolidated
store, restricted to single-valence oxidation-state assignments
(`oxi_state_label != "mixed"`) with defined B, oxidation state, and
coordination number.

Usage: python analysis/scripts/bv_sign_census.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

STORE_PATH = Path("data/processed/bond_valence/consolidated_store.json")

REPRESENTATIVE_SPECIES = [
    ("Pd", 2), ("Au", 3), ("Cu", 1),
    ("Eu", 2), ("Rb", 1), ("K", 1), ("Sr", 2), ("Cs", 1),
]


def main() -> None:
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    payload = store["datasets"]["oxygen_authoritative"]["payload"]

    n_fitted_total = 0
    rows: list[tuple[str, int, int, float, float | None]] = []
    for element, groups in payload.items():
        for records in groups.values():
            for rec in records:
                if rec.get("status") != "fitted":
                    continue
                n_fitted_total += 1
                if rec.get("oxi_state_label") == "mixed":
                    continue
                b, cn, z = rec.get("B"), rec.get("cn"), rec.get("oxi_state")
                if b is None or cn is None or z is None:
                    continue
                diag = rec.get("fit_diagnostics") or {}
                rows.append((element, int(z), int(cn), float(b), diag.get("bond_length_range")))

    negative = [r for r in rows if r[3] < 0]
    print(f"fitted shells, all records: {n_fitted_total}")
    print(f"single-valence with defined (B, z, n): {len(rows)}")
    print(f"B < 0: {len(negative)} ({100 * len(negative) / len(rows):.1f}%)")

    for name, cond in (("z<n", lambda z, c: z < c),
                       ("z=n", lambda z, c: z == c),
                       ("z>n", lambda z, c: z > c)):
        sub = [r for r in rows if cond(r[1], r[2])]
        sub_neg = [r for r in sub if r[3] < 0]
        print(f"  {name}: {len(sub_neg)}/{len(sub)} = {100 * len(sub_neg) / len(sub):.1f}% B<0")

    spread_neg = [r[4] for r in negative if r[4] is not None]
    spread_pos = [r[4] for r in rows if r[3] > 0 and r[4] is not None]
    print(f"median bond-length range: B<0 {statistics.median(spread_neg):.4f} A, "
          f"B>0 {statistics.median(spread_pos):.4f} A")
    print(f"range < 0.05 A: B<0 {100 * sum(s < 0.05 for s in spread_neg) / len(spread_neg):.0f}%, "
          f"B>0 {100 * sum(s < 0.05 for s in spread_pos) / len(spread_pos):.0f}%")

    counts = Counter((r[0], r[1]) for r in rows)
    counts_neg = Counter((r[0], r[1]) for r in negative)
    for element, z in REPRESENTATIVE_SPECIES:
        n, nn = counts[(element, z)], counts_neg[(element, z)]
        print(f"  {element}{z}+: {nn}/{n} = {100 * nn / n:.1f}% B<0")


if __name__ == "__main__":
    main()
