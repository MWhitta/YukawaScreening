"""Histogram of the per-structure Yukawa screening length across the corpus.

lambda = (-R' + sqrt(R'^2 + 4 B R')) / 2 with R' = the shell's mean bond
length, for every single-valence fitted shell in the consolidated store.
Both branches are shown (B > 0 gives lambda > 0, B < 0 gives lambda < 0 when
B >= -R'/4; shells with B < -R'/4 have no real root and are counted only).

Run:  analysis/.venv/bin/python analysis/scripts/plot_lambda_distribution.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "processed" / "bond_valence" / "consolidated_store.json"
OUT = ROOT / "theory" / "prm_revision" / "response_figures" / "lambda_per_structure_hist.png"


def main() -> None:
    payload = json.loads(STORE.read_text())["datasets"]["oxygen_authoritative"]["payload"]
    lam, n_complex, rbar = [], 0, []
    for buckets in payload.values():
        for lst in buckets.values():
            for r in lst:
                if r["status"] != "fitted" or r["oxi_state_label"] != "pure":
                    continue
                if not all(isinstance(r[k], (int, float)) and math.isfinite(r[k]) for k in ("R0", "B", "cn", "oxi_state")):
                    continue
                Rp, B = r["fit_diagnostics"]["bond_length_mean"], r["B"]
                rbar.append(Rp)
                disc = Rp * Rp + 4 * B * Rp
                if disc < 0:
                    n_complex += 1
                    continue
                lam.append((-Rp + math.sqrt(disc)) / 2)
    lam = np.array(lam)
    pos, neg = lam[lam > 0], lam[lam < 0]
    Rm = float(np.median(rbar))
    lam_conv = (-Rm + math.sqrt(Rm * Rm + 4 * 0.37 * Rm)) / 2   # lambda for B = 0.37 A at the median R'
    print(f"median R' = {Rm:.3f} A; lambda for B = 0.37 A there = {lam_conv:.3f} A")
    print(f"real lambda: {len(lam)}  (positive {len(pos)}, negative {len(neg)}); no real root: {n_complex}")
    print(f"positive: median {np.median(pos):.3f} A, IQR {np.percentile(pos,25):.3f}-{np.percentile(pos,75):.3f} A, "
          f"p95 {np.percentile(pos,95):.3f} A, max {pos.max():.1f} A; negative: median {np.median(neg):.3f} A, min {neg.min():.2f} A")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300, facecolor="white")
    # (a) linear axis, -1 to 3 A, tails folded into the end bins
    lo, hi = -1.0, 3.0
    edges = np.linspace(lo, hi, 161)
    a.hist(np.clip(pos, lo, hi - 1e-9), bins=edges, color="tab:blue", alpha=0.75, label=f"B > 0  (n = {len(pos):,})")
    a.hist(np.clip(neg, lo + 1e-9, hi), bins=edges, color="tab:red", alpha=0.75, label=f"B < 0, real root  (n = {len(neg):,})")
    a.axvline(lam_conv, color="k", lw=0.8, ls="--", label=f"λ for B = 0.37 Å at median R′ ({lam_conv:.2f} Å)")
    a.set_xlim(lo, hi); a.set_xlabel("λ (Å)"); a.set_ylabel("shells")
    a.set_title("(a) per-structure λ, linear scale", fontsize=8, loc="left")
    a.text(0.98, 0.97, f"{(pos > hi).sum():,} shells with λ > {hi:g} Å folded into last bin\n"
           f"{(neg < lo).sum():,} with λ < {lo:g} Å folded into first bin\n"
           f"{n_complex:,} shells with B < −R′/4 have no real root",
           transform=a.transAxes, ha="right", va="top", fontsize=5.5)
    a.legend(fontsize=5.5, loc="center right")
    # (b) log axis of |lambda|, full range
    lb = np.logspace(-4, 2, 121)
    b.hist(pos, bins=lb, color="tab:blue", alpha=0.75, label="λ > 0")
    b.hist(-neg, bins=lb, color="tab:red", alpha=0.75, label="|λ|, λ < 0")
    b.set_xscale("log"); b.set_xlabel("|λ| (Å)"); b.set_ylabel("shells")
    b.set_title("(b) full range, log scale", fontsize=8, loc="left")
    b.axvline(lam_conv, color="k", lw=0.8, ls="--")
    b.legend(fontsize=5.5, loc="upper left")
    for ax in (a, b):
        ax.tick_params(labelsize=6); ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
