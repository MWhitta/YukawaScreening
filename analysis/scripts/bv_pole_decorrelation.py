"""Decorrelation of fitted B from R0 at the z = n pole, across the corpus.

Tests the screening-collapse claim that at z = n the leading-order link
between the fitted parameters degenerates (R0 ceases to constrain B), so
per-structure (R0, B) points of a species stop organizing along a line.

Within every (element, z, n) cell with at least MIN_N clean fits, the
Pearson correlation |r(B, R0)| across structures is computed, then
summarized as a function of ln(z/n). Only unregularized linear fits
enter (fit_strategy == "linear_ls", no degenerate fallback applied) —
the pipeline's degenerate/regularized modes pull B toward a 0.37-A
prior, which would fake decorrelation at the pole.

Usage: python analysis/scripts/bv_pole_decorrelation.py
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STORE_PATH = Path("data/processed/bond_valence/consolidated_store.json")
MIN_N = 30


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main() -> None:
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    payload = store["datasets"]["oxygen_authoritative"]["payload"]

    cells: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)
    strategies: Counter[tuple[bool, str]] = Counter()
    for element, groups in payload.items():
        for records in groups.values():
            for rec in records:
                if rec.get("status") != "fitted":
                    continue
                if rec.get("oxi_state_label") == "mixed":
                    continue
                b, r0, cn, z = rec.get("B"), rec.get("R0"), rec.get("cn"), rec.get("oxi_state")
                if None in (b, r0, cn, z):
                    continue
                diag = rec.get("fit_diagnostics") or {}
                strategy = str(diag.get("fit_strategy") or rec.get("fit_strategy") or "")
                strategies[(int(z) == int(cn), strategy)] += 1
                if strategy != "linear_ls" or diag.get("degenerate_fit_applied"):
                    continue
                cells[(element, int(z), int(cn))].append((float(r0), float(b)))

    print("fit-strategy mix (is_pole, strategy -> count):")
    for (is_pole, strategy), count in sorted(strategies.items()):
        print(f"  {'z=n ' if is_pole else 'z!=n'}  {strategy or '<none>':28s} {count}")

    # |r(B, R0)| per cell, grouped by |ln(z/n)|
    profile: dict[str, list[float]] = defaultdict(list)
    pole_cells: list[tuple[str, int, int, int, float]] = []
    for (element, z, cn), pairs in sorted(cells.items()):
        if len(pairs) < MIN_N:
            continue
        r = abs(pearson([p[0] for p in pairs], [p[1] for p in pairs]))
        if z == cn:
            pole_cells.append((element, z, cn, len(pairs), r))
            profile["0 (pole)"].append(r)
            continue
        u = abs(math.log(z / cn))
        if u < 0.35:
            key = "(0, 0.35)"
        elif u < 0.75:
            key = "[0.35, 0.75)"
        else:
            key = ">= 0.75"
        profile[key].append(r)

    print(f"\n|r(B, R0)| within (element, z, n) cells, clean linear fits, N >= {MIN_N}:")
    for key in ["0 (pole)", "(0, 0.35)", "[0.35, 0.75)", ">= 0.75"]:
        rs = profile.get(key, [])
        if not rs:
            continue
        print(f"  |ln(z/n)| {key:12s}: {len(rs):3d} cells, median |r| = {statistics.median(rs):.3f}, "
              f"min = {min(rs):.3f}, max = {max(rs):.3f}")

    print("\nz = n cells individually:")
    for element, z, cn, n, r in sorted(pole_cells, key=lambda t: -t[3]):
        print(f"  {element}{z}+ at n={cn}: N = {n:4d}, |r(B, R0)| = {r:.3f}")


if __name__ == "__main__":
    main()
