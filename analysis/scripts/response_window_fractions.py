"""Corpus-wide statistics quoted in the round 2 response letter (Block 5).

Fraction of single-valence fitted shells whose per-structure (R0, B) lie
inside a physical window, and the bond-length-range contrast between
shells inside and outside it.  Reads the consolidated store only.

Run from the repo root:  python3 analysis/scripts/response_window_fractions.py
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "processed" / "bond_valence" / "consolidated_store.json"
WINDOW = {"R0": (0.0, 4.0), "B": (-1.0, 2.0)}          # plotted window
CONVENTIONAL = {"R0": (0.0, 4.0), "B": (0.0, 1.0)}     # conventional range


def inside(rec: dict, win: dict) -> bool:
    return win["R0"][0] < rec["R0"] < win["R0"][1] and win["B"][0] < rec["B"] < win["B"][1]


def main() -> None:
    payload = json.loads(STORE.read_text())["datasets"]["oxygen_authoritative"]["payload"]
    recs = [
        (el, r)
        for el, buckets in payload.items()
        for lst in buckets.values()
        for r in lst
        if r["status"] == "fitted"
        and r["oxi_state_label"] == "pure"
        and all(isinstance(r[k], (int, float)) and math.isfinite(r[k]) for k in ("R0", "B", "cn", "oxi_state"))
    ]
    n = len(recs)
    print(f"single-valence fitted shells: {n}")
    for name, win in (("plotted window 0<R0<4, -1<B<2 A", WINDOW), ("conventional 0<R0<4, 0<B<1 A", CONVENTIONAL)):
        k = sum(inside(r, win) for _, r in recs)
        print(f"{name}: {k}/{n} = {100 * k / n:.1f}%")
    rng_in = [r["fit_diagnostics"]["bond_length_range"] for _, r in recs if inside(r, WINDOW)]
    rng_out = [r["fit_diagnostics"]["bond_length_range"] for _, r in recs if not inside(r, WINDOW)]
    print(f"median bond-length range inside window {statistics.median(rng_in):.3f} A (n={len(rng_in)}), "
          f"outside {statistics.median(rng_out):.3f} A (n={len(rng_out)})")
    # Where R0 falls relative to the shell's bond-length interval [Rmin, Rmax]
    # (letter Block 8: R0 is generally outside the interval, below it for z<n).
    def rel(r: dict) -> str:
        z = int(round(r["oxi_state"]))
        return "z<n" if z < r["cn"] else ("z=n" if z == r["cn"] else "z>n")
    for c in ("z<n", "z=n", "z>n"):
        sub = [r for _, r in recs if rel(r) == c]
        dg = [r["fit_diagnostics"] for r in sub]
        below = sum(r["R0"] < d["bond_length_min"] for r, d in zip(sub, dg))
        within = sum(d["bond_length_min"] <= r["R0"] <= d["bond_length_max"] for r, d in zip(sub, dg))
        above = sum(r["R0"] > d["bond_length_max"] for r, d in zip(sub, dg))
        print(f"{c}: n={len(sub)}  R0 below Rmin {100 * below / len(sub):.0f}%  "
              f"within [Rmin,Rmax] {100 * within / len(sub):.0f}%  above Rmax {100 * above / len(sub):.0f}%")
    sub = [r for _, r in recs if rel(r) == "z<n" and r["B"] > 0]
    print(f"z<n, B>0: median (Rbar - R0) = {statistics.median(r['fit_diagnostics']['bond_length_mean'] - r['R0'] for r in sub):.3f} A, "
          f"median bond-length range = {statistics.median(r['fit_diagnostics']['bond_length_range'] for r in sub):.3f} A")
    for z in (1, 2, 3, 4):
        co = [r for el, r in recs if el == "Co" and int(round(r["oxi_state"])) == z]
        k = sum(inside(r, WINDOW) for r in co)
        print(f"Co{z}+: {k}/{len(co)} inside plotted window = {100 * k / len(co):.0f}%")


if __name__ == "__main__":
    main()
