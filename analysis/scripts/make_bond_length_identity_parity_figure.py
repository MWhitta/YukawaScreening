#!/usr/bin/env python3
"""Render the SI bond-length-identity parity figure from the consolidated store.

The figure has two parity panels:
  Left:  fitted R0 vs (mean bond length) + B*ln(z/n).
  Right: fitted B  vs (R0 - mean bond length) / ln(z/n).

Inclusion rule (documented filter): every record in
``data/processed/bond_valence/consolidated_store.json::oxygen_authoritative``
that is

  * status == "fitted"
  * oxi_state_label != "mixed"
  * z != n  (excludes the formal pole at the screening/anti-screening crossover)
  * 0 < R0 < 4 Å, B > 0  (screening branch, physically sensible R0 bounds)
  * fit_diagnostics.bond_length_mean is finite

The script prints the panel-level Pearson r, MAE, and N so the SI text can be
kept in sync with the regenerated figure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = REPO_ROOT / "data" / "processed" / "bond_valence" / "consolidated_store.json"
DATASET_KEY = "oxygen_authoritative"
FIGURES_DIR = REPO_ROOT / "theory" / "figures"
OUTPUT_NAME = "r0_bond_length_identity_parity.png"


def _iter_fitted_records():
    payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))["datasets"][DATASET_KEY]["payload"]
    for cat, buckets in payload.items():
        for bucket in ("oxides", "hydroxides"):
            for record in buckets.get(bucket, []):
                yield record


def collect_parity_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs_r0, ys_r0 = [], []
    xs_b, ys_b = [], []
    for record in _iter_fitted_records():
        if record.get("status") != "fitted":
            continue
        if record.get("oxi_state_label") == "mixed":
            continue
        r0 = record.get("R0")
        b_value = record.get("B")
        cn = record.get("cn")
        z = record.get("oxi_state")
        diag = record.get("fit_diagnostics") or {}
        rbar = diag.get("bond_length_mean")
        if None in (r0, b_value, cn, z, rbar):
            continue
        if cn == 0 or z == 0 or z == cn:
            continue
        if not (0.0 < float(r0) < 4.0):
            continue
        if float(b_value) <= 0.0:
            continue
        ratio = float(z) / float(cn)
        if ratio <= 0.0:
            continue
        ln_zn = math.log(ratio)
        if not math.isfinite(ln_zn):
            continue
        xs_r0.append(float(rbar) + float(b_value) * ln_zn)
        ys_r0.append(float(r0))
        xs_b.append((float(r0) - float(rbar)) / ln_zn)
        ys_b.append(float(b_value))
    return (
        np.asarray(xs_r0),
        np.asarray(ys_r0),
        np.asarray(xs_b),
        np.asarray(ys_b),
    )


def _panel(ax, x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str, title: str) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    mae = float(np.mean(np.abs(x - y))) if len(x) else float("nan")

    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="0.55", lw=0.9, zorder=1)
    ax.scatter(x, y, s=2.5, c="#1f77b4", alpha=0.30, edgecolors="none", zorder=2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="0.92", linewidth=0.5)
    ax.text(
        0.04,
        0.96,
        f"$N = {len(x):,}$\n$r = {pearson:.3f}$\nMAE $= {mae:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.80", "linewidth": 0.5},
    )
    return pearson, mae


def main() -> None:
    xs_r0, ys_r0, xs_b, ys_b = collect_parity_points()
    if len(xs_r0) == 0:
        raise RuntimeError("No parity points collected — check the consolidated store.")

    with plt.rc_context({
        "font.size": 8.5,
        "axes.labelsize": 9.2,
        "axes.titlesize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
    }):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.1, 3.5))
        r_left, mae_left = _panel(
            ax_l,
            xs_r0,
            ys_r0,
            xlabel=r"$\bar R + B\,\ln(z/n)$ ($\AA$)",
            ylabel=r"fitted $R_0$ ($\AA$)",
            title=r"Bond-length identity",
        )
        r_right, mae_right = _panel(
            ax_r,
            xs_b,
            ys_b,
            xlabel=r"$(R_0 - \bar R)/\ln(z/n)$ ($\AA$)",
            ylabel=r"fitted $B$ ($\AA$)",
            title=r"Slope identity",
        )
        fig.tight_layout(pad=0.7, w_pad=1.4)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / OUTPUT_NAME
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    print(f"Wrote {out}")
    print(
        f"R0 panel: N={len(xs_r0):,}  Pearson r={r_left:.4f}  MAE={mae_left:.4f} A"
    )
    print(
        f"B  panel: N={len(xs_b):,}  Pearson r={r_right:.4f}  MAE={mae_right:.4f} A"
    )


if __name__ == "__main__":
    main()
