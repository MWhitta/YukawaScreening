#!/usr/bin/env python3
"""Render the PRL Fig. 2 charge-density benchmark panels from checked-in JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from critmin.viz.manuscript import manuscript_subplots


BENCHMARK_PATH = REPO_ROOT / "data" / "processed" / "theory" / "charge_density_benchmark.json"
FIGURES_DIR = REPO_ROOT / "theory" / "figures"
PRL_FIGURE = "thomas_fermi_reff_prl.png"
SI_FIGURE = "thomas_fermi_reff_null_models_si.png"

GROUP1_COLOR = "#60A0C8"
GROUP2_COLOR = "#E09050"
PANEL_A_MARKER_EDGE = "#4A3A2A"
PANEL_B_STYLES = {
    "d": {"color": "#6A91C3", "marker": "o", "label": r"$d$-block"},
    "p": {"color": "#A7C48A", "marker": "s", "label": r"$p$-block"},
    "f": {"color": "#C3A4C2", "marker": "D", "label": r"$f$-block"},
}
PANEL_A_LABEL_OFFSETS = {
    "Be": (8, 10),
    "Li": (-8, 12),
    "Mg": (10, -10),
    "Ca": (-8, 12),
    "Sr": (-6, 12),
    "Na": (-10, -20),
    "Ba": (-8, 12),
    "K": (12, -18),
    "Cs": (8, -18),
    "Rb": (14, -18),
}
OUTLIER_LABEL_OFFSETS = {
    "B": (16, -2),
    "P": (10, 14),
    "As": (10, 10),
    "Au": (12, -16),
}
VALID_BLOCKS = {"s", "d", "p", "f"}
PANEL_B_BLOCKS = ("d", "p", "f")


def _load_benchmark() -> dict[str, object]:
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _styled_point(point: dict[str, object]) -> dict[str, object]:
    styled = dict(point)
    block = str(styled.get("block") or "")
    if block not in VALID_BLOCKS:
        raise ValueError(f"Benchmark point is missing a valid explicit block label: {point}")
    styled["block"] = block
    return styled


def _charge_label(point: dict[str, object]) -> str:
    return rf"{point['element']}$^{{{int(point['z'])}+}}$"


def _plot_block_points(
    ax: plt.Axes,
    points: list[dict[str, object]],
    *,
    open_symbols: bool,
    size: float,
) -> None:
    for block in PANEL_B_BLOCKS:
        block_points = [point for point in points if point["block"] == block]
        if not block_points:
            continue
        style = PANEL_B_STYLES[block]
        xs = [float(point["m_i"]) for point in block_points]
        ys = [float(point["r_eff"]) for point in block_points]
        marker = str(style["marker"])
        color = str(style["color"])
        if open_symbols:
            ax.scatter(
                xs,
                ys,
                s=size,
                marker=marker,
                facecolors="none",
                edgecolors="black",
                linewidths=0.55,
                alpha=0.92,
                zorder=5,
            )
            ax.scatter(
                xs,
                ys,
                s=size * 0.82,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                alpha=0.94,
                zorder=6,
            )
            continue
        ax.scatter(
            xs,
            ys,
            s=size,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.66,
            zorder=4,
        )


def render_prl_figure(payload: dict[str, object]) -> Path:
    points = [_styled_point(dict(point)) for point in payload["points"]]
    for point in points:
        panel = str(point["panel"])
        block = str(point["block"])
        if panel == "a" and block != "s":
            raise ValueError(f"Panel a point must be tagged as s-block: {point}")
        if panel == "b" and block not in PANEL_B_BLOCKS:
            raise ValueError(f"Panel b point must carry an explicit d/p/f block tag: {point}")
    group1 = [point for point in points if point["family"] == "Group 1"]
    group2 = [point for point in points if point["family"] == "Group 2"]
    other_fit = [
        point
        for point in points
        if point["panel"] == "b" and point["included_in_regression"]
    ]
    outliers = [
        point
        for point in points
        if point["panel"] == "b" and point["is_outlier"]
    ]
    panel_a_fit = payload["summary"]["panel_a_fit"]
    panel_b_fit = payload["summary"]["panel_b_fit"]

    fig, axes, style = manuscript_subplots(
        "full_width_2panel",
        panel_height_ratio=0.55,
        sharex=False,
        sharey=True,
        constrained_layout=False,
    )
    ax1, ax2 = axes

    with plt.rc_context(
        style.rcparams()
        | {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10.6,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.1,
        }
    ):
        lo, hi = 1.30, 3.10
        ylo, yhi = 1.20, 2.72
        fit_x = np.linspace(lo, hi, 200)

        for ax in (ax1, ax2):
            ax.plot([lo, hi], [lo, hi], "--", color="0.68", lw=1.05, alpha=0.95, zorder=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(ylo, yhi)
            ax.grid(True, color="0.80", linewidth=0.6, alpha=0.45)
            ax.tick_params(direction="out", length=4.0, width=0.8, pad=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.0)
            ax.spines["bottom"].set_linewidth(1.0)
            ax.spines["left"].set_color("0.45")
            ax.spines["bottom"].set_color("0.45")
            ax.set_xlabel(r"$m_i$ ($\AA$)")

        ax1.set_ylabel(r"$r_{\mathrm{eff}}$ ($\AA$)")
        ax2.tick_params(labelleft=False)

        ax1.text(
            0.02,
            0.98,
            "a)",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=10.8,
            fontweight="bold",
        )
        ax2.text(
            0.02,
            0.98,
            "b)",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=10.8,
            fontweight="bold",
        )

        ax1.plot(
            fit_x,
            float(panel_a_fit["slope"]) * fit_x + float(panel_a_fit["intercept"]),
            color="0.12",
            lw=1.55,
            zorder=2,
        )
        ax1.scatter(
            [float(point["m_i"]) for point in group1],
            [float(point["r_eff"]) for point in group1],
            s=88,
            marker="o",
            c=GROUP1_COLOR,
            edgecolors="#2F475C",
            linewidths=0.65,
            alpha=0.96,
            zorder=4,
        )
        ax1.scatter(
            [float(point["m_i"]) for point in group2],
            [float(point["r_eff"]) for point in group2],
            s=98,
            marker="s",
            c=GROUP2_COLOR,
            edgecolors=PANEL_A_MARKER_EDGE,
            linewidths=0.65,
            alpha=0.96,
            zorder=4,
        )
        for point in group1 + group2:
            dx, dy = PANEL_A_LABEL_OFFSETS.get(str(point["element"]), (8, 8))
            ax1.annotate(
                str(point["element"]),
                (float(point["m_i"]), float(point["r_eff"])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=7.7,
                color="0.10",
            )
        ax1.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=GROUP1_COLOR,
                    markeredgecolor="#2F475C",
                    markeredgewidth=0.65,
                    markersize=4.8,
                    label="Group 1",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markerfacecolor=GROUP2_COLOR,
                    markeredgecolor=PANEL_A_MARKER_EDGE,
                    markeredgewidth=0.65,
                    markersize=5.0,
                    label="Group 2",
                ),
            ],
            loc="lower right",
            frameon=True,
            framealpha=0.90,
            edgecolor="0.35",
            borderpad=0.55,
            labelspacing=0.55,
        )

        ax2.plot(
            fit_x,
            float(panel_b_fit["slope"]) * fit_x + float(panel_b_fit["intercept"]),
            color="0.12",
            lw=1.55,
            zorder=2,
        )
        _plot_block_points(ax2, other_fit, open_symbols=False, size=86.0)
        _plot_block_points(ax2, outliers, open_symbols=True, size=106.0)
        for point in outliers:
            dx, dy = OUTLIER_LABEL_OFFSETS.get(str(point["element"]), (10, 10))
            block = str(point["block"])
            ax2.annotate(
                _charge_label(point),
                (float(point["m_i"]), float(point["r_eff"])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.8,
                color=str(PANEL_B_STYLES[block]["color"]),
            )
        ax2.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=PANEL_B_STYLES["d"]["color"], markeredgecolor="#444444", markeredgewidth=0.65, markersize=4.9, label=str(PANEL_B_STYLES["d"]["label"])),
                Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=PANEL_B_STYLES["p"]["color"], markeredgecolor="#444444", markeredgewidth=0.65, markersize=5.0, label=str(PANEL_B_STYLES["p"]["label"])),
                Line2D([0], [0], marker="D", linestyle="None", markerfacecolor=PANEL_B_STYLES["f"]["color"], markeredgecolor="#444444", markeredgewidth=0.65, markersize=4.9, label=str(PANEL_B_STYLES["f"]["label"])),
            ],
            loc="lower right",
            frameon=True,
            framealpha=0.90,
            edgecolor="0.35",
            borderpad=0.55,
            labelspacing=0.55,
        )
        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.28, top=0.95, wspace=0.11)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / PRL_FIGURE
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def render_si_null_model(payload: dict[str, object]) -> Path:
    null_models = payload["summary"]["panel_a_null_models"]
    metric_labels = [r"$m_i$", r"$R_0$", "Shannon radius"]
    metric_values = [
        float(null_models["mae_m_i"]),
        float(null_models["mae_R0"]),
        float(null_models["mae_shannon_radius"]),
    ]
    metric_colors = ["#202020", "#4daf4a", "#984ea3"]

    with plt.rc_context(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        bars = ax.bar(
            metric_labels,
            metric_values,
            color=metric_colors,
            edgecolor="black",
            linewidth=0.45,
            width=0.62,
        )
        for bar, value in zip(bars, metric_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
            )
        ax.set_ylabel(r"LOOCV MAE ($\AA$)")
        ax.set_title("Group 1/2 null-model check")
        ax.set_ylim(0.0, max(metric_values) * 1.25)
        ax.grid(axis="y", color="0.90", linewidth=0.5)
        fig.tight_layout(pad=0.35)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / SI_FIGURE
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return out


def main() -> None:
    payload = _load_benchmark()
    prl_out = render_prl_figure(payload)
    si_out = render_si_null_model(payload)
    print(f"Wrote {prl_out}")
    print(f"Wrote {si_out}")


if __name__ == "__main__":
    main()
