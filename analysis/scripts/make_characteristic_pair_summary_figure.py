#!/usr/bin/env python3
"""Render the characteristic-pair summary panel used in the Supplemental Material."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_SUMMARY_PATH = REPO_ROOT / "data" / "processed" / "theory" / "master_oxygen_summary_theory.json"
FIGURES_DIR = REPO_ROOT / "theory" / "figures"
OUTPUT_NAME = "group1_group2_lanthanide_r0_star_vs_b_star.png"

PANEL_SPECS = [
    {
        "title": "Group 1 + H",
        "elements": ["H", "Li", "Na", "K", "Rb", "Cs"],
        "color": "#1f77b4",
    },
    {
        "title": "Group 2",
        "elements": ["Be", "Mg", "Ca", "Sr", "Ba"],
        "color": "#d95f02",
    },
    {
        "title": "Lanthanides",
        "elements": ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
        "color": "#2a6f6b",
    },
]


def _load_rows() -> list[dict[str, object]]:
    return json.loads(MASTER_SUMMARY_PATH.read_text(encoding="utf-8"))["master_rows"]


def main() -> None:
    rows = _load_rows()
    by_element = {str(row["element"]): row for row in rows}

    with plt.rc_context(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=True)
        for ax, spec in zip(axes, PANEL_SPECS):
            panel_rows = [by_element[element] for element in spec["elements"] if element in by_element]
            x_values = [float(row["R0_star"]) for row in panel_rows]
            y_values = [float(row["B_star"]) for row in panel_rows]

            ax.scatter(
                x_values,
                y_values,
                s=38,
                c=spec["color"],
                edgecolors="black",
                linewidths=0.45,
                zorder=3,
            )

            for row in panel_rows:
                ax.annotate(
                    str(row["element"]),
                    (float(row["R0_star"]), float(row["B_star"])),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7.1,
                )

            x_lo = min(x_values)
            x_hi = max(x_values)
            y_lo = min(y_values)
            y_hi = max(y_values)
            x_pad = max(0.03, 0.08 * (x_hi - x_lo or 1.0))
            y_pad = max(0.02, 0.10 * (y_hi - y_lo or 1.0))

            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
            ax.set_title(str(spec["title"]))
            ax.set_xlabel(r"$R_0^\ast$ ($\AA$)")
            ax.grid(True, color="0.90", linewidth=0.5)

        axes[0].set_ylabel(r"$B^\ast$ ($\AA$)")
        fig.tight_layout(pad=0.5, w_pad=0.8)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / OUTPUT_NAME
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
