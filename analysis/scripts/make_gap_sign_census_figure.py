"""SI figure: band-gap-resolved sign census of the per-shell fits.

Stacked histogram of the Materials Project band gap for every clean
(linear_ls, non-degenerate, single-valence) fitted cation--oxygen
shell, stacked by the sign of the fitted softness B. Metallic hosts
(band gap exactly zero) are drawn as a separate stacked bar left of
the axis so they do not read as part of the gapped continuum.

Band gaps come from data/processed/theory/mp_band_gaps.json, a
mid -> band_gap map fetched from the Materials Project summary
endpoint. If the file is absent, it is rebuilt over the corpus mids
(requires MP_API_KEY). Because the API masks material_id in its
responses, the fetch batches mids so that no two materials in a batch
share a chemical formula, and maps each returned formula_pretty back
to its requested mid.

Output: theory/figures/si_band_gap_sign_census.png

Usage: python analysis/scripts/make_gap_sign_census_figure.py
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STORE_PATH = Path("data/processed/bond_valence/consolidated_store.json")
GAPS_PATH = Path("data/processed/theory/mp_band_gaps.json")
OUT_PATH = Path("theory/figures/si_band_gap_sign_census.png")

BLUE = "#2B5FAC"   # B > 0 (screening sign)
ORANGE = "#C96A00" # B < 0 (anti-screening sign)
INK, MUTED = "#1A1A1A", "#555555"


def clean_shells(store: dict) -> list[tuple[str, float]]:
    """(mid, B) for every clean single-valence linear_ls fit."""
    shells = []
    payload = store["datasets"]["oxygen_authoritative"]["payload"]
    for groups in payload.values():
        for records in groups.values():
            for rec in records:
                if rec.get("status") != "fitted" or rec.get("oxi_state_label") == "mixed":
                    continue
                diag = rec.get("fit_diagnostics") or {}
                strategy = str(diag.get("fit_strategy") or rec.get("fit_strategy") or "")
                if strategy != "linear_ls" or diag.get("degenerate_fit_applied"):
                    continue
                if rec.get("B") is None:
                    continue
                shells.append((str(rec["mid"]), float(rec["B"])))
    return shells


def fetch_band_gaps(store: dict) -> dict[str, float]:
    mid_formula = {}
    payload = store["datasets"]["oxygen_authoritative"]["payload"]
    for groups in payload.values():
        for records in groups.values():
            for rec in records:
                if rec.get("status") == "fitted":
                    mid_formula[str(rec["mid"])] = str(rec.get("formula") or "")
    groups_by_formula = defaultdict(list)
    for mid, formula in sorted(mid_formula.items()):
        groups_by_formula[formula].append(mid)
    slots = defaultdict(list)
    for members in groups_by_formula.values():
        for i, mid in enumerate(members):
            slots[i].append(mid)
    batches = []
    for members in slots.values():
        for j in range(0, len(members), 140):
            batches.append(members[j:j + 140])

    key = os.environ["MP_API_KEY"]
    mid_gap: dict[str, float] = {}
    for chunk in batches:
        url = ("https://api.materialsproject.org/materials/summary/"
               f"?material_ids={','.join(chunk)}&_fields=formula_pretty,band_gap&_limit=150")
        out = subprocess.run(
            ["curl", "-s", "--max-time", "90", "-H", f"X-API-KEY: {key}", url],
            capture_output=True, text=True, check=True,
        ).stdout
        formula_to_mid = {mid_formula[m]: m for m in chunk}
        docs = json.loads(out).get("data", [])
        counts = defaultdict(int)
        for doc in docs:
            counts[doc.get("formula_pretty")] += 1
        for doc in docs:
            formula = doc.get("formula_pretty")
            if counts[formula] > 1:
                continue
            mid = formula_to_mid.get(formula)
            if mid is not None and doc.get("band_gap") is not None:
                mid_gap[mid] = float(doc["band_gap"])
    return mid_gap


def main() -> None:
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    if GAPS_PATH.exists():
        mid_gap = json.loads(GAPS_PATH.read_text(encoding="utf-8"))
    else:
        mid_gap = fetch_band_gaps(store)
        GAPS_PATH.write_text(json.dumps(mid_gap), encoding="utf-8")

    shells = [(mid_gap[m], b) for m, b in clean_shells(store) if m in mid_gap]
    gaps = np.array([g for g, _ in shells])
    bsign = np.array([b < 0 for _, b in shells])
    n_total = len(shells)

    metal = gaps == 0.0
    n_metal_neg = int((metal & bsign).sum())
    n_metal_pos = int((metal & ~bsign).sum())

    bins = np.arange(0.0, 8.6 + 0.25, 0.25)
    pos, _ = np.histogram(gaps[~metal & ~bsign], bins=bins)
    neg, _ = np.histogram(gaps[~metal & bsign], bins=bins)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=300)
    ax.bar(bins[:-1], pos, width=0.25, align="edge", color=BLUE,
           edgecolor="white", linewidth=0.6, zorder=3, label="$B>0$")
    ax.bar(bins[:-1], neg, width=0.25, align="edge", bottom=pos, color=ORANGE,
           edgecolor="white", linewidth=0.6, zorder=3, label="$B<0$")
    ax.bar(-0.65, n_metal_pos, width=0.42, color=BLUE, edgecolor="white",
           linewidth=0.6, zorder=3)
    ax.bar(-0.65, n_metal_neg, width=0.42, bottom=n_metal_pos, color=ORANGE,
           edgecolor="white", linewidth=0.6, zorder=3)

    n_metal = n_metal_pos + n_metal_neg
    ax.annotate(
        f"metallic hosts ($E_g=0$)\n{n_metal:,} shells, "
        f"{100 * n_metal_neg / n_metal:.0f}% with $B<0$",
        xy=(-0.42, n_metal * 0.99), xytext=(0.55, n_metal * 0.965),
        fontsize=8.5, color=MUTED, ha="left", va="top", linespacing=1.45,
        arrowprops=dict(arrowstyle="-", color="0.6", linewidth=0.7))

    gapped = ~metal
    frac_gapped = 100 * (gapped & bsign).sum() / gapped.sum()
    ax.text(4.9, n_metal * 0.62,
            f"gapped hosts: {int(gapped.sum()):,} shells,\n"
            f"{frac_gapped:.0f}% with $B<0$",
            fontsize=8.5, color=MUTED, ha="left", va="top", linespacing=1.45)

    ax.set_xlabel("Materials Project band gap $E_g$ (eV)", fontsize=10, color=INK)
    ax.set_ylabel("fitted shells", fontsize=10, color=INK)
    ax.set_title(f"Sign of the fitted softness across the band-gap distribution "
                 f"({n_total:,} clean shells)", fontsize=10.5, color=INK, pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="0.75", fontsize=9)
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.45")
    ax.tick_params(colors="0.35", labelsize=9)
    ax.set_xlim(-1.0, 8.6)
    ax.set_ylim(0, n_metal * 1.14)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")
    print(f"shells: {n_total}, metallic: {n_metal} ({100*n_metal_neg/n_metal:.1f}% B<0), "
          f"gapped: {int(gapped.sum())} ({frac_gapped:.1f}% B<0)")


if __name__ == "__main__":
    main()
