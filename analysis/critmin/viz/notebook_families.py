"""Shared helpers for the repeated bond-valence family notebooks.

These utilities pull the notebook-local analysis and plotting helpers out of
the family notebooks so the notebooks can stay as thin orchestration layers.
The focus here is the repeated oxide-family workflow used by the d-block,
post-transition, nonmetal/halogen, actinide, and Group 1/2 notebooks.
"""

from __future__ import annotations

import glob
import os
import shutil
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from critmin.analysis.bond_valence_theory import (
    collect_oxygen_fit_lines,
    fit_weighted_alpha_line,
    group_convergence,
    iter_oxygen_cn_fit_points,
    minimum_spread_intersection,
)
from critmin.viz.manuscript import (
    apply_manuscript_axis_style,
    manuscript_subplots,
)
from critmin.viz.notebook_setup import (
    cn_linestyle_map,
    load_grouped_payload,
    merge_grouped_payloads,
    merge_oxygen_records,
    tint_color,
    xo_color,
    xo_shade_map,
)

OxidationStateColorFn = Callable[[str], str]
LegendLabelFormatter = Callable[[int, int], str]
LegendEntryFormatter = Callable[[int], str]
IntersectionFn = Callable[[Sequence[Mapping[str, Any]]], dict[str, Any] | None]

_SUPERSCRIPT_TO_DIGIT = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
}


def latest_payload(prefix: str) -> str | None:
    """Return the newest payload matching *prefix*, excluding aux variants."""
    matches = sorted(
        path
        for path in glob.glob(f"{prefix}*.json")
        if not path.endswith("_high_uncertainty.json")
        and not path.endswith(".phase_a.json")
    )
    return matches[-1] if matches else None


def latest_high_uncertainty_payload(prefix: str) -> str | None:
    """Return the newest ``*_high_uncertainty.json`` payload for *prefix*."""
    matches = sorted(glob.glob(f"{prefix}*_high_uncertainty.json"))
    return matches[-1] if matches else None


def latest_existing_path(pattern: str) -> str | None:
    """Return the lexicographically latest existing path matching *pattern*."""
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def existing_paths(paths: Sequence[str | os.PathLike[str] | None]) -> list[str]:
    """Filter *paths* to the entries that currently exist."""
    return [os.fspath(path) for path in paths if path and os.path.exists(path)]


def load_grouped_sources(
    accepted_paths: Sequence[str | os.PathLike[str] | None],
    high_uncertainty_paths: Sequence[str | os.PathLike[str] | None],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load and merge accepted and high-uncertainty grouped payload sources."""
    payloads: list[dict[str, dict[str, list[dict[str, Any]]]]] = []
    for source in existing_paths(accepted_paths):
        payloads.append(load_grouped_payload(source))
    for source in existing_paths(high_uncertainty_paths):
        payloads.append(load_grouped_payload(source, default_status="high_uncertainty"))
    return merge_grouped_payloads(*payloads) if payloads else {}


def pairwise_median_intersection(
    lines: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return a robust median pairwise intersection for heterogeneous CN mixes."""
    if len(lines) < 2:
        return None

    beta = np.array([float(line["beta"]) for line in lines], dtype=float)
    beta0 = np.array([float(line["beta0"]) for line in lines], dtype=float)
    r0_pairs: list[float] = []
    for (i, _), (j, _) in combinations(enumerate(lines), 2):
        delta_beta = beta[i] - beta[j]
        if abs(delta_beta) > 1.0e-10:
            r0_pairs.append(-(beta0[i] - beta0[j]) / delta_beta)
    if not r0_pairs:
        return None

    r0_star = float(np.median(np.asarray(r0_pairs, dtype=float)))
    b_values = beta * r0_star + beta0
    b_star = float(np.median(b_values))
    sigma_b = float(np.std(b_values))
    b_range = float(np.max(b_values) - np.min(b_values))
    return {
        "R0_star": r0_star,
        "B_star": b_star,
        "sigma_B_at_R0_star": sigma_b,
        "pct_sigma_B_at_R0_star": (
            float(100.0 * sigma_b / abs(b_star)) if abs(b_star) > 1.0e-12 else float("nan")
        ),
        "B_range_at_R0_star": b_range,
        "n_lines": len(lines),
    }


def sheaf_intersection(
    alpha_table: pd.DataFrame,
    *,
    group: str | None = None,
) -> dict[str, Any] | None:
    """Return the weighted least-squares intersection of alpha lines."""
    rows = alpha_table if group is None else alpha_table[alpha_table["group"] == group]
    if len(rows) < 2:
        return None

    alpha0 = rows["alpha0"].to_numpy(dtype=float)
    alpha1 = rows["alpha1"].to_numpy(dtype=float)
    weights = rows["total_inliers"].to_numpy(dtype=float)
    design = np.column_stack([-alpha0, np.ones(len(alpha0), dtype=float)])
    normal = design.T @ np.diag(weights) @ design
    rhs = design.T @ np.diag(weights) @ alpha1
    try:
        params = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return None

    residuals = alpha0 * params[0] + alpha1 - params[1]
    return {
        "beta_sheaf": float(params[0]),
        "beta0_sheaf": float(params[1]),
        "residuals": residuals,
        "cations": list(rows["cation"]),
        "rms_residual": float(np.sqrt((weights * residuals**2).sum() / weights.sum())),
    }


def plot_regression_comparison(
    data: Mapping[str, Any],
    cations: Sequence[str],
    cn_fits: Mapping[str, Any],
    target_cns: Sequence[int] | None = None,
    *,
    title: str,
    figure_path: str | os.PathLike[str],
    ncols: int = 5,
    legend_label_formatter: LegendLabelFormatter | None = None,
) -> Figure:
    """Plot OLS / Huber / RANSAC CN fits against the underlying B-vs-R0 points."""
    del target_cns

    if legend_label_formatter is None:
        legend_label_formatter = lambda cn, n: f"CN{cn} (n={n})"

    nrows = (len(cations) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 4.6 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, cation in zip(axes_flat, cations):
        oxygen_records = merge_oxygen_records(data, cation)
        cat_cns = sorted({record.get("cn") for record in oxygen_records if record.get("cn") is not None})
        if not cat_cns:
            ax.set_axis_off()
            continue

        cn_colors = xo_shade_map(cation, cat_cns, light_start=0.0, light_stop=0.42)
        for cn in cat_cns:
            cn_records = [record for record in oxygen_records if record.get("cn") == cn]
            r0 = np.array([record["R0"] for record in cn_records], dtype=float)
            b = np.array([record["B"] for record in cn_records], dtype=float)
            cn_block = cn_fits.get(cation, {}).get(cn, {})

            ransac_fit = cn_block.get("oxygen_ransac")
            inlier_mask = np.ones(len(cn_records), dtype=bool)
            if ransac_fit and ransac_fit.get("inlier_mask") is not None:
                mask = np.asarray(ransac_fit["inlier_mask"], dtype=bool)
                if len(mask) == len(cn_records):
                    inlier_mask = mask
            if (~inlier_mask).any():
                ax.scatter(
                    r0[~inlier_mask],
                    b[~inlier_mask],
                    color=cn_colors[cn],
                    marker="x",
                    s=42,
                    linewidths=1.0,
                    alpha=0.7,
                    zorder=4,
                )
            ax.scatter(
                r0[inlier_mask],
                b[inlier_mask],
                color=cn_colors[cn],
                marker="o",
                s=20,
                edgecolors="black",
                linewidths=0.2,
                alpha=0.6,
                zorder=3,
            )

            r0_lo = float(r0.min())
            r0_hi = float(r0.max())
            if np.isclose(r0_lo, r0_hi):
                r0_lo -= 0.05
                r0_hi += 0.05
            x_line = np.linspace(r0_lo, r0_hi, 120)

            for fit_key, linestyle, linewidth, alpha in (
                ("oxygen", ":", 1.2, 0.5),
                ("oxygen_huber", "--", 1.8, 0.8),
                ("oxygen_ransac", "-", 2.0, 0.95),
            ):
                fit = cn_block.get(fit_key)
                if fit is None:
                    continue
                y_line = fit["beta"] * x_line + fit["beta0"]
                label = None
                if fit_key == "oxygen_ransac":
                    n_inliers = int(fit.get("n_inliers", fit.get("n", len(cn_records))))
                    label = legend_label_formatter(int(cn), n_inliers)
                ax.plot(
                    x_line,
                    y_line,
                    color=cn_colors[cn],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=2,
                    label=label,
                )

        ax.set_title(f"{cation}–O", fontsize=14)
        ax.set_xlabel("R₀ (Å)", fontsize=11)
        ax.grid(alpha=0.2)
        if ax in axes_flat[::ncols]:
            ax.set_ylabel("B (Å)", fontsize=11)
        ax.legend(fontsize=6, loc="best")

    for ax in axes_flat[len(cations) :]:
        ax.set_visible(False)

    style_handles = [
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.2, label="OLS"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.8, label="Huber"),
        Line2D([0], [0], color="gray", linestyle="-", linewidth=2.0, label="RANSAC"),
    ]
    fig.legend(
        handles=style_handles,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def plot_oxygen_fit_line_panels(
    oxygen_cn_analysis: Mapping[str, Mapping[str, Any]],
    group_key: str,
    group_label: str,
    *,
    figure_path: str | os.PathLike[str],
    intersection_strategy_by_cation: Mapping[str, IntersectionFn] | None = None,
    legend_label_formatter: LegendLabelFormatter | None = None,
) -> Figure:
    """Plot CN-resolved oxygen fit lines for one cation group."""
    if legend_label_formatter is None:
        legend_label_formatter = lambda cn, n: f"CN{cn} (n={n})"

    group_analysis = oxygen_cn_analysis[group_key]
    cations = group_analysis["cations"]
    target_cns = group_analysis["target_cns"]
    cn_fits = group_analysis["cn_fits"]

    fig, axes = plt.subplots(1, len(cations), figsize=(4.2 * len(cations), 4.4))
    if len(cations) == 1:
        axes = [axes]

    present_cns = sorted(
        {
            cn
            for cation in cations
            for cn in target_cns
            if (
                cn_fits.get(cation, {}).get(cn, {}).get("oxygen_ransac")
                or cn_fits.get(cation, {}).get(cn, {}).get("oxygen")
            )
        }
    )
    cn_styles = cn_linestyle_map(present_cns)
    intersection_strategy_by_cation = dict(intersection_strategy_by_cation or {})

    for ax, cation in zip(axes, cations):
        lines = collect_oxygen_fit_lines(cn_fits, cation, target_cns)
        if not lines:
            ax.text(0.5, 0.5, "No all-CN O fits", ha="center", va="center")
            ax.set_axis_off()
            continue

        cn_colors = xo_shade_map(
            cation,
            [line["cn"] for line in lines],
            light_start=0.0,
            light_stop=0.42,
        )
        r0_plot_values: list[float] = []
        b_plot_values: list[float] = []
        for line in lines:
            r0_lo = float(line["R0_min"])
            r0_hi = float(line["R0_max"])
            if not np.isfinite(r0_lo) or not np.isfinite(r0_hi):
                continue
            if np.isclose(r0_lo, r0_hi):
                r0_lo -= 0.05
                r0_hi += 0.05
            x_line = np.linspace(r0_lo, r0_hi, 120)
            y_line = float(line["beta"]) * x_line + float(line["beta0"])
            ax.plot(
                x_line,
                y_line,
                color=cn_colors[int(line["cn"])],
                linestyle=cn_styles[int(line["cn"])],
                linewidth=2.0,
                alpha=0.95,
                label=legend_label_formatter(int(line["cn"]), int(line["n"])),
            )
            r0_plot_values.extend([r0_lo, r0_hi])
            b_plot_values.extend([float(y_line.min()), float(y_line.max())])

        intersection_fn = intersection_strategy_by_cation.get(cation, minimum_spread_intersection)
        intersection = intersection_fn(lines)
        if intersection is not None and np.isfinite(intersection["R0_star"]):
            ax.scatter(
                intersection["R0_star"],
                intersection["B_star"],
                marker="o",
                s=120,
                facecolors="white",
                edgecolors=xo_color(cation),
                linewidths=1.8,
                zorder=6,
                label="least-spread point",
            )
            r0_plot_values.append(float(intersection["R0_star"]))
            b_plot_values.append(float(intersection["B_star"]))

        if r0_plot_values and b_plot_values:
            r0_pad = max(0.03, 0.08 * (max(r0_plot_values) - min(r0_plot_values) or 1.0))
            b_pad = max(0.03, 0.08 * (max(b_plot_values) - min(b_plot_values) or 1.0))
            ax.set_xlim(min(r0_plot_values) - r0_pad, max(r0_plot_values) + r0_pad)
            ax.set_ylim(min(b_plot_values) - b_pad, max(b_plot_values) + b_pad)

        ax.set_title(f"{cation}–O", fontsize=16)
        ax.set_xlabel("R₀ (Å)", fontsize=16)
        ax.grid(alpha=0.2)
        if ax is axes[0]:
            ax.set_ylabel("B (Å)")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"{group_label}: X-O", fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def oxidation_state_element(label: str) -> str:
    """Return the element prefix from a superscript oxidation-state label."""
    return label.rstrip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def _species_cn_colors(
    species_label: str,
    ordered_cns: Sequence[int],
    *,
    color_fn: OxidationStateColorFn,
    base_color_by_element: Mapping[str, str] | None = None,
) -> dict[int, Any]:
    if not ordered_cns:
        return {}
    if len(ordered_cns) == 1:
        return {int(ordered_cns[0]): color_fn(species_label)}

    element = oxidation_state_element(species_label)
    base_color = None
    if base_color_by_element is not None:
        base_color = base_color_by_element.get(element)
    if base_color is None:
        base_color = color_fn(species_label)
    mixes = np.linspace(0.0, 0.42, len(ordered_cns))
    return {
        int(cn): tint_color(base_color, float(mix))
        for cn, mix in zip(ordered_cns, mixes)
    }


def plot_species_fit_line_panels(
    cn_fits: Mapping[str, Any],
    cations: Sequence[str],
    target_cns: Sequence[int],
    *,
    group_label: str,
    figure_path: str | os.PathLike[str],
    color_fn: OxidationStateColorFn,
    base_color_by_element: Mapping[str, str] | None = None,
    legend_label_formatter: LegendLabelFormatter | None = None,
) -> Figure:
    """Plot oxi-state/species CN-fit panels in an element-grouped grid."""
    if legend_label_formatter is None:
        legend_label_formatter = lambda cn, n: f"CN{cn} (n={n})"

    present_cns = sorted(
        {
            cn
            for cation in cations
            for cn in target_cns
            if (
                cn_fits.get(cation, {}).get(cn, {}).get("oxygen_ransac")
                or cn_fits.get(cation, {}).get(cn, {}).get("oxygen")
            )
        }
    )
    cn_styles = cn_linestyle_map(present_cns)

    elem_species: OrderedDict[str, list[str]] = OrderedDict()
    for label in cations:
        elem_species.setdefault(oxidation_state_element(label), []).append(label)

    max_cols = max(len(species) for species in elem_species.values())
    n_rows = len(elem_species)
    fig, axes = plt.subplots(
        n_rows,
        max_cols,
        figsize=(4.5 * max_cols, 4.0 * n_rows),
        squeeze=False,
    )

    for row_idx, (_element, species_list) in enumerate(elem_species.items()):
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(species_list):
                ax.set_axis_off()
                continue

            cation = species_list[col_idx]
            lines = collect_oxygen_fit_lines(cn_fits, cation, target_cns)
            if not lines:
                ax.text(0.5, 0.5, f"{cation}\nNo fits", ha="center", va="center", fontsize=10)
                ax.set_axis_off()
                continue

            ordered_cns = sorted({int(line["cn"]) for line in lines})
            cn_colors = _species_cn_colors(
                cation,
                ordered_cns,
                color_fn=color_fn,
                base_color_by_element=base_color_by_element,
            )
            r0_vals: list[float] = []
            b_vals: list[float] = []
            for line in lines:
                r0_lo = float(line["R0_min"])
                r0_hi = float(line["R0_max"])
                if not np.isfinite(r0_lo) or not np.isfinite(r0_hi):
                    continue
                if np.isclose(r0_lo, r0_hi):
                    r0_lo -= 0.05
                    r0_hi += 0.05
                x_line = np.linspace(r0_lo, r0_hi, 120)
                y_line = float(line["beta"]) * x_line + float(line["beta0"])
                ax.plot(
                    x_line,
                    y_line,
                    color=cn_colors.get(int(line["cn"]), color_fn(cation)),
                    linestyle=cn_styles[int(line["cn"])],
                    linewidth=2.0,
                    alpha=0.95,
                    label=legend_label_formatter(int(line["cn"]), int(line["n"])),
                )
                r0_vals.extend([r0_lo, r0_hi])
                b_vals.extend([float(y_line.min()), float(y_line.max())])

            intersection = minimum_spread_intersection(lines)
            if intersection is not None and np.isfinite(intersection["R0_star"]):
                ax.scatter(
                    intersection["R0_star"],
                    intersection["B_star"],
                    marker="o",
                    s=100,
                    facecolors="white",
                    edgecolors=color_fn(cation),
                    linewidths=1.8,
                    zorder=6,
                )
                r0_vals.append(float(intersection["R0_star"]))
                b_vals.append(float(intersection["B_star"]))

            if r0_vals and b_vals:
                r0_pad = max(0.03, 0.08 * (max(r0_vals) - min(r0_vals) or 1.0))
                b_pad = max(0.03, 0.08 * (max(b_vals) - min(b_vals) or 1.0))
                ax.set_xlim(min(r0_vals) - r0_pad, max(r0_vals) + r0_pad)
                ax.set_ylim(min(b_vals) - b_pad, max(b_vals) + b_pad)

            ax.set_title(f"{cation}–O", fontsize=13)
            ax.set_xlabel("R₀ (Å)", fontsize=10)
            ax.grid(alpha=0.2)
            if col_idx == 0:
                ax.set_ylabel("B (Å)", fontsize=10)
            ax.legend(fontsize=5.5, loc="best")

    fig.suptitle(f"{group_label}: Oxi-state-resolved n fit lines", fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


def build_species_intersection_table(
    cn_fits: Mapping[str, Any],
    cations: Sequence[str],
    target_cns: Sequence[int],
    *,
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a species-level least-spread intersection table."""
    rows: list[dict[str, Any]] = []
    for cation in cations:
        lines = collect_oxygen_fit_lines(cn_fits, cation, target_cns)
        result = minimum_spread_intersection(lines)
        if result is None:
            continue
        rows.append(
            {
                "element": oxidation_state_element(cation),
                "species": cation,
                **result,
                "lines_used": ", ".join(f"CN{line['cn']}" for line in lines),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) and order is not None:
        order_lookup = {cation: idx for idx, cation in enumerate(order)}
        frame = (
            frame.assign(_order=frame["species"].map(order_lookup))
            .sort_values("_order")
            .drop(columns=["_order"])
            .reset_index(drop=True)
        )
    return frame


def save_manuscript_figure(
    fig: Figure,
    filename: str,
    *,
    figures_dir: str | os.PathLike[str],
    theory_figures_dir: str | os.PathLike[str] | None = None,
    dpi: int = 300,
) -> Path:
    """Save a figure into the notebook figures dir and optionally mirror it."""
    out = Path(figures_dir) / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if theory_figures_dir is not None:
        theory_dir = Path(theory_figures_dir)
        if theory_dir.is_dir() and theory_dir.resolve() != Path(figures_dir).resolve():
            shutil.copy2(out, theory_dir / filename)
    return out


def export_species_fit_line_atlas(
    cn_fits: Mapping[str, Any],
    cations: Sequence[str],
    target_cns: Sequence[int],
    *,
    filename_prefix: str,
    figures_dir: str | os.PathLike[str],
    theory_figures_dir: str | os.PathLike[str] | None = None,
    color_fn: OxidationStateColorFn,
    base_color_by_element: Mapping[str, str] | None = None,
    legend_entry_formatter: LegendEntryFormatter | None = None,
    ncols: int = 3,
    chunk_size: int | None = None,
    panel_height_ratio: float = 0.82,
) -> list[str]:
    """Export a chunked manuscript-safe atlas of species CN fit-line panels."""
    if ncols != 3:
        raise ValueError("export_species_fit_line_atlas currently supports ncols=3 only")
    if legend_entry_formatter is None:
        legend_entry_formatter = lambda cn: fr"$n={cn}$"
    if chunk_size is None:
        chunk_size = ncols * 4

    present_cns = sorted(
        {
            cn
            for cation in cations
            for cn in target_cns
            if (
                cn_fits.get(cation, {}).get(cn, {}).get("oxygen_ransac")
                or cn_fits.get(cation, {}).get(cn, {}).get("oxygen")
            )
        }
    )
    cn_styles = cn_linestyle_map(present_cns)
    species_chunks = [list(cations[idx : idx + chunk_size]) for idx in range(0, len(cations), chunk_size)]
    output_files: list[str] = []

    for part_idx, species_chunk in enumerate(species_chunks, start=1):
        nrows = int(np.ceil(len(species_chunk) / ncols))
        fig, axes, style = manuscript_subplots(
            "full_width_3panel",
            nrows=nrows,
            panel_height_ratio=panel_height_ratio,
        )
        axes_flat = np.atleast_1d(axes).ravel()

        for idx, cation in enumerate(species_chunk):
            ax = axes_flat[idx]
            lines = collect_oxygen_fit_lines(cn_fits, cation, target_cns)
            if not lines:
                ax.text(0.5, 0.5, "No O fits", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            ordered_cns = sorted({int(line["cn"]) for line in lines})
            cn_colors = _species_cn_colors(
                cation,
                ordered_cns,
                color_fn=color_fn,
                base_color_by_element=base_color_by_element,
            )
            r0_values: list[float] = []
            b_values: list[float] = []
            for line in lines:
                r0_lo = float(line["R0_min"])
                r0_hi = float(line["R0_max"])
                if not np.isfinite(r0_lo) or not np.isfinite(r0_hi):
                    continue
                if np.isclose(r0_lo, r0_hi):
                    r0_lo -= 0.05
                    r0_hi += 0.05
                x_line = np.linspace(r0_lo, r0_hi, 120)
                y_line = float(line["beta"]) * x_line + float(line["beta0"])
                ax.plot(
                    x_line,
                    y_line,
                    color=cn_colors.get(int(line["cn"]), color_fn(cation)),
                    linestyle=cn_styles[int(line["cn"])],
                    linewidth=style.line_width_pt + 0.1,
                    alpha=0.95,
                    zorder=2,
                )
                r0_values.extend([float(x_line.min()), float(x_line.max())])
                b_values.extend([float(y_line.min()), float(y_line.max())])

            intersection = minimum_spread_intersection(lines)
            if intersection is not None and np.isfinite(intersection["R0_star"]):
                ax.scatter(
                    intersection["R0_star"],
                    intersection["B_star"],
                    marker="o",
                    s=36,
                    facecolors="white",
                    edgecolors=color_fn(cation),
                    linewidths=1.0,
                    zorder=4,
                )
                r0_values.append(float(intersection["R0_star"]))
                b_values.append(float(intersection["B_star"]))

            if r0_values and b_values:
                r0_pad = max(0.03, 0.08 * (max(r0_values) - min(r0_values) or 1.0))
                b_pad = max(0.03, 0.08 * (max(b_values) - min(b_values) or 1.0))
                ax.set_xlim(min(r0_values) - r0_pad, max(r0_values) + r0_pad)
                ax.set_ylim(min(b_values) - b_pad, max(b_values) + b_pad)

            apply_manuscript_axis_style(
                ax,
                style,
                title=f"{cation}–O",
                xlabel="R₀ (Å)" if idx >= (nrows - 1) * ncols else None,
                ylabel="B (Å)" if idx % ncols == 0 else None,
            )

        for ax in axes_flat[len(species_chunk) :]:
            ax.set_axis_off()

        chunk_cns = sorted(
            {
                int(line["cn"])
                for cation in species_chunk
                for line in collect_oxygen_fit_lines(cn_fits, cation, target_cns)
            }
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color="0.25",
                linestyle=cn_styles[cn],
                linewidth=style.line_width_pt + 0.1,
                label=legend_entry_formatter(int(cn)),
            )
            for cn in chunk_cns
        ]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="0.25",
                linestyle="",
                markerfacecolor="white",
                label="least-spread point",
            )
        )
        legend_ncol = min(6, len(legend_handles))
        legend_nrows = int(np.ceil(len(legend_handles) / legend_ncol))
        legend_band_in = 0.30 * legend_nrows + 0.55
        fig_w_in, fig_h_in = fig.get_size_inches()
        fig.set_size_inches(fig_w_in, fig_h_in + legend_band_in)
        try:
            fig.set_layout_engine(None)
        except (AttributeError, ValueError):
            try:
                fig.set_constrained_layout(False)
            except AttributeError:
                pass
        new_height_in = fig_h_in + legend_band_in
        legend_band_frac = legend_band_in / new_height_in
        fig.tight_layout(rect=(0.0, legend_band_frac, 1.0, 1.0))
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=legend_ncol,
            frameon=False,
            fontsize=style.legend_pt,
        )
        fig.align_labels()
        filename = f"{filename_prefix}_part{part_idx:02d}.png"
        save_manuscript_figure(
            fig,
            filename,
            figures_dir=figures_dir,
            theory_figures_dir=theory_figures_dir,
        )
        plt.close(fig)
        output_files.append(filename)

    return output_files


def parse_oxidation_state_label(label: str) -> tuple[str, int | None]:
    """Parse a superscript oxidation-state label such as ``Fe³⁺``."""
    element = ""
    digits = ""
    for char in label:
        if char in _SUPERSCRIPT_TO_DIGIT:
            digits += _SUPERSCRIPT_TO_DIGIT[char]
        elif char.isalpha():
            element += char
    digits = digits.replace("+", "").replace("-", "")
    return element, int(digits) if digits else None


def extract_beta_vs_charge(
    cn_fits: Mapping[str, Any],
    cations: Sequence[str],
    *,
    target_cn: int,
    min_n: int = 5,
    max_sigma_beta: float = 5.0,
    min_r2: float = 0.3,
    prefer_ransac: bool = True,
    include_crossover: bool = True,
) -> pd.DataFrame:
    """Extract a fixed-CN beta-vs-formal-charge table with quality filters."""
    rows: list[dict[str, Any]] = []
    for species in cations:
        cn_dict = cn_fits.get(species, {}).get(target_cn, {})
        fit = cn_dict.get("oxygen_ransac") if prefer_ransac else None
        if fit is None:
            fit = cn_dict.get("oxygen")
        if fit is None:
            continue

        n_inliers = int(fit.get("n_inliers", fit.get("n", 0)))
        if n_inliers < min_n:
            continue
        sigma_beta = float(fit.get("beta_stderr", 0.0))
        r2 = float(fit.get("r2", 0.0))
        if sigma_beta > max_sigma_beta or r2 < min_r2:
            continue

        element, charge = parse_oxidation_state_label(species)
        if charge is None:
            continue
        if not include_crossover and charge == target_cn:
            continue

        rows.append(
            {
                "element": element,
                "species": species,
                "charge": charge,
                "cn": int(target_cn),
                "beta": float(fit["beta"]),
                "beta0": float(fit["beta0"]),
                "sigma_beta": sigma_beta,
                "n": n_inliers,
                "r2": r2,
                "is_crossover": charge == target_cn,
            }
        )
    return pd.DataFrame(rows)


def _weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - mean) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _cn_log_pole_model(z: np.ndarray, amplitude: float, z_cross: float) -> np.ndarray:
    return amplitude / np.log(z / z_cross)


def fit_cn_log_pole(
    frame: pd.DataFrame,
    *,
    cn: int,
    fix_crossover: bool = True,
    min_points: int = 5,
) -> dict[str, Any] | None:
    """Fit ``beta = A / ln(z / z_cross)`` to the off-pole data at fixed CN."""
    off_pole = frame[frame["charge"] != cn].sort_values(["charge", "species"])
    if len(off_pole) < min_points or off_pole["charge"].nunique() < 3:
        return None

    z = off_pole["charge"].to_numpy(dtype=float)
    beta = off_pole["beta"].to_numpy(dtype=float)
    weights = np.clip(off_pole["n"].to_numpy(dtype=float), 1.0, None)
    sigma = 1.0 / np.sqrt(weights)

    if fix_crossover:
        def model(z_values: np.ndarray, amplitude: float) -> np.ndarray:
            return _cn_log_pole_model(z_values, amplitude, float(cn))

        popt, _ = curve_fit(
            model,
            z,
            beta,
            p0=[1.0],
            sigma=sigma,
            absolute_sigma=False,
            bounds=([0.0], [20.0]),
            maxfev=50_000,
        )
        pred = model(z, *popt)
        amplitude = float(popt[0])
        z_cross = float(cn)
    else:
        lower = max(z[z < cn].max() + 1.0e-3, 0.1)
        upper = z[z > cn].min() - 1.0e-3
        popt, _ = curve_fit(
            _cn_log_pole_model,
            z,
            beta,
            p0=[1.0, float(cn) - 0.01],
            sigma=sigma,
            absolute_sigma=False,
            bounds=([0.0, lower], [20.0, upper]),
            maxfev=50_000,
        )
        pred = _cn_log_pole_model(z, *popt)
        amplitude = float(popt[0])
        z_cross = float(popt[1])

    return {
        "cn": int(cn),
        "amplitude": amplitude,
        "z_cross": z_cross,
        "fixed_crossover": bool(fix_crossover),
        "r2_weighted": float(_weighted_r2(beta, pred, weights)),
        "rmse_weighted": float(np.sqrt(np.sum(weights * (beta - pred) ** 2) / weights.sum())),
        "n_points": int(len(off_pole)),
        "total_inliers": int(weights.sum()),
    }


def plot_fixed_cn_beta_panel(
    ax: Axes,
    frame: pd.DataFrame,
    cn: int,
    *,
    fit: Mapping[str, Any] | None = None,
    show_legend: bool = True,
    annotate_crossover: bool = False,
    title: str | bool | None = None,
    element_order: Sequence[str] | None = None,
    colors_by_element: Mapping[str, str] | None = None,
    series_line_alpha: float = 1.0,
    fit_zorder: float = 2.0,
    crossover_annotation_fontsize: float | None = None,
) -> None:
    """Plot beta versus oxidation state for one fixed CN branch."""
    shown_elements: set[str] = set()
    crossover_labels: list[tuple[str, float, float, str]] = []
    if element_order is None:
        element_order = sorted(frame["element"].unique())

    for element in element_order:
        element_data = frame[frame["element"] == element].sort_values("charge")
        if element_data.empty:
            continue

        color = (colors_by_element or {}).get(element, "#666666")
        left = element_data[element_data["charge"] < cn]
        cross = element_data[element_data["charge"] == cn]
        right = element_data[element_data["charge"] > cn]

        label = element if element not in shown_elements else None
        if not left.empty:
            ax.plot(
                left["charge"],
                left["beta"],
                "-o",
                color=color,
                linewidth=1.5,
                markersize=6.5,
                alpha=series_line_alpha,
                label=label,
                zorder=3,
            )
            ax.scatter(
                left["charge"],
                left["beta"],
                s=6.5**2,
                color=color,
                edgecolors=color,
                linewidths=0.8,
                zorder=4,
            )
            ax.errorbar(
                left["charge"],
                left["beta"],
                yerr=left["sigma_beta"],
                fmt="none",
                ecolor=color,
                capsize=2.5,
                alpha=0.4,
                zorder=2,
            )
            shown_elements.add(element)
            label = None

        if not right.empty:
            ax.plot(
                right["charge"],
                right["beta"],
                "-o",
                color=color,
                linewidth=1.5,
                markersize=6.5,
                alpha=series_line_alpha,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                label=label,
                zorder=3,
            )
            ax.scatter(
                right["charge"],
                right["beta"],
                s=6.5**2,
                facecolors="white",
                edgecolors=color,
                linewidths=1.4,
                zorder=4,
            )
            ax.errorbar(
                right["charge"],
                right["beta"],
                yerr=right["sigma_beta"],
                fmt="none",
                ecolor=color,
                capsize=2.5,
                alpha=0.4,
                zorder=2,
            )
            shown_elements.add(element)
            label = None

        if not cross.empty:
            ax.scatter(
                cross["charge"],
                cross["beta"],
                marker="D",
                s=72,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                label=label,
                zorder=4,
            )
            ax.errorbar(
                cross["charge"],
                cross["beta"],
                yerr=cross["sigma_beta"],
                fmt="none",
                ecolor=color,
                capsize=2.5,
                alpha=0.45,
                zorder=2,
            )
            shown_elements.add(element)
            if annotate_crossover:
                for _, row in cross.iterrows():
                    crossover_labels.append(
                        (str(row["species"]), float(row["charge"]), float(row["beta"]), color)
                    )

    if fit is not None:
        z_cross = float(fit["z_cross"])
        z_min = max(0.8, float(frame["charge"].min()) - 0.2)
        z_max = float(frame["charge"].max()) + 0.2
        left_x = np.linspace(z_min, z_cross - 0.05, 300)
        right_x = np.linspace(z_cross + 0.05, z_max, 300)
        ax.plot(
            left_x,
            _cn_log_pole_model(left_x, float(fit["amplitude"]), z_cross),
            "k--",
            linewidth=1.4,
            alpha=0.8,
            label=rf"$\beta = {fit['amplitude']:.2f} / \ln(z/{z_cross:.0f})$",
            zorder=fit_zorder,
        )
        ax.plot(
            right_x,
            _cn_log_pole_model(right_x, float(fit["amplitude"]), z_cross),
            "k--",
            linewidth=1.4,
            alpha=0.8,
            zorder=fit_zorder,
        )
        ax.axvline(z_cross, color="black", linewidth=0.9, linestyle=":", alpha=0.55)
    else:
        ax.axvline(cn, color="grey", linewidth=0.8, linestyle=":", alpha=0.45)

    for species, charge, beta, color in crossover_labels:
        ax.annotate(
            species,
            (charge, beta),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=7 if crossover_annotation_fontsize is None else crossover_annotation_fontsize,
            color=color,
        )

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(sorted(frame["charge"].unique()))
    ax.set_xlabel("Formal charge (z)")
    ax.set_ylabel("β")
    if title is None:
        ax.set_title(f"n = {cn}")
    elif title is not False:
        ax.set_title(str(title))
    ax.grid(alpha=0.2)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                label=r"$z = n$",
                markerfacecolor="lightgrey",
                markeredgecolor="black",
                markersize=7,
            )
        )
        labels.append(r"$z = n$")
        ax.legend(handles, labels, fontsize=7.5, loc="best", ncol=2)

    data_beta = frame["beta"].to_numpy(dtype=float)
    pad = max(0.4, 0.08 * (data_beta.max() - data_beta.min()))
    ax.set_ylim(data_beta.min() - pad, data_beta.max() + pad)


__all__ = [
    "build_species_intersection_table",
    "collect_oxygen_fit_lines",
    "existing_paths",
    "export_species_fit_line_atlas",
    "extract_beta_vs_charge",
    "fit_cn_log_pole",
    "fit_weighted_alpha_line",
    "group_convergence",
    "iter_oxygen_cn_fit_points",
    "latest_existing_path",
    "latest_high_uncertainty_payload",
    "latest_payload",
    "load_grouped_sources",
    "minimum_spread_intersection",
    "oxidation_state_element",
    "pairwise_median_intersection",
    "parse_oxidation_state_label",
    "plot_fixed_cn_beta_panel",
    "plot_oxygen_fit_line_panels",
    "plot_regression_comparison",
    "plot_species_fit_line_panels",
    "save_manuscript_figure",
    "sheaf_intersection",
]
