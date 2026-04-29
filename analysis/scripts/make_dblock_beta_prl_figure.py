"""Generate the PRL two-panel beta-versus-charge figure.

This script renders the PRL Fig. 1 pole-law panels for the maximum subset of
the authoritative 103-species oxygen dataset that furnishes usable off-pole
fits at coordination numbers 4 and 6.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from sklearn.exceptions import UndefinedMetricWarning

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from critmin.analysis.bond_valence_store import (  # noqa: E402
    DEFAULT_OXYGEN_AUTHORITATIVE_SOURCE_SPEC,
)
from critmin.analysis.bond_valence_theory import (  # noqa: E402
    build_unified_oxygen_cn_fits,
    load_grouped_payload_json,
    observed_coordination_numbers,
    oxidation_state_species_label,
)
from critmin.viz.manuscript import (  # noqa: E402
    apply_manuscript_axis_style,
    manuscript_subplots,
    style_manuscript_legend,
)
from critmin.viz.notebook_families import (  # noqa: E402
    plot_fixed_cn_beta_panel,
    save_manuscript_figure,
)
from critmin.viz.notebook_setup import init_notebook  # noqa: E402


FIGURES_DIR = ROOT / "theory" / "figures"
THEORY_FIGURES_DIR = ROOT / "theory" / "figures"
MASTER_SUMMARY_PATH = ROOT / "data" / "processed" / "theory" / "master_oxygen_summary_theory.json"
OUTPUT_NAME = "prl_oxygen_beta_vs_charge_cn4_cn6.png"

_SUPERSCRIPT_TO_DIGIT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def parse_charge_from_label(label: str) -> tuple[str, int | None]:
    plain = str(label).translate(_SUPERSCRIPT_TO_DIGIT)
    element = "".join(char for char in plain if char.isalpha())
    suffix = plain[len(element):]
    sign = -1 if "-" in suffix else 1
    digits = "".join(char for char in suffix if char.isdigit())
    charge = sign * int(digits) if digits else None
    return element, charge


def master_species_rows() -> list[tuple[str, str, int]]:
    payload = json.loads(MASTER_SUMMARY_PATH.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, int]] = []
    for row in payload["master_rows"]:
        label = str(row.get("species") or row.get("cation") or row.get("label") or "")
        element, charge = parse_charge_from_label(label)
        if charge is None:
            family_key = str(row.get("family_key") or "")
            if family_key == "group1":
                charge = 1
            elif family_key == "group2":
                charge = 2
            elif family_key == "lanthanides":
                charge = 3
            else:
                raise ValueError(f"Cannot infer oxidation state for master row: {row}")
            label = oxidation_state_species_label(element, charge)
        rows.append((label, element, charge))
    return rows


def build_authoritative_species_payload() -> tuple[dict[str, dict[str, list[dict]]], list[tuple[str, str, int]]]:
    species_rows = master_species_rows()
    grouped = load_grouped_payload_json(DEFAULT_OXYGEN_AUTHORITATIVE_SOURCE_SPEC)
    species_payload: dict[str, dict[str, list[dict]]] = {}
    missing: list[str] = []

    for label, element, charge in species_rows:
        by_oxi: dict[str, list[dict]] = {"oxides": [], "hydroxides": []}
        cat_data = grouped.get(element, {"oxides": [], "hydroxides": []})
        for bucket in ("oxides", "hydroxides"):
            for record in cat_data.get(bucket, []):
                if record.get("status", "fitted") != "fitted":
                    continue
                if record.get("oxi_state_label") == "mixed":
                    continue
                record_charge = record.get("oxi_state")
                if record_charge is None:
                    continue
                if int(record_charge) == charge:
                    by_oxi[bucket].append(record)
        if by_oxi["oxides"] or by_oxi["hydroxides"]:
            species_payload[label] = by_oxi
        else:
            missing.append(label)

    if missing:
        raise RuntimeError(
            "Master summary species missing from the authoritative grouped payload: "
            + ", ".join(missing)
        )
    return species_payload, species_rows


def extract_master_beta_charge_frames() -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, object]], dict[str, object]]:
    species_payload, species_rows = build_authoritative_species_payload()
    species_labels = [label for label, _element, _charge in species_rows]
    target_cns = observed_coordination_numbers(species_payload, species_labels)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        cn_fits, _outliers = build_unified_oxygen_cn_fits(
            species_payload,
            species_labels,
            target_cns=target_cns,
        )

    beta_charge_frames: dict[int, pd.DataFrame] = {}
    panel_stats: dict[int, dict[str, object]] = {}
    all_rows: list[dict[str, object]] = []
    present_species: set[str] = set()

    for cn in (4, 6):
        rows: list[dict[str, object]] = []
        crossover_species: list[str] = []
        for label, element, charge in species_rows:
            cn_block = cn_fits.get(label, {}).get(cn, {})
            fit = cn_block.get("oxygen_ransac") or cn_block.get("oxygen")
            if fit is None:
                continue

            n_inliers = int(fit.get("n_inliers", fit.get("n", 0)) or 0)
            sigma_beta = float(fit.get("beta_stderr") or 0.0)
            r2_raw = fit.get("r2")
            r2 = float(r2_raw) if isinstance(r2_raw, (int, float)) and r2_raw is not None else float("nan")
            if n_inliers < 4 or sigma_beta > 5.0 or (not np.isnan(r2) and r2 < 0.3):
                continue

            row = {
                "element": element,
                "species": label,
                "charge": charge,
                "cn": cn,
                "beta": float(fit["beta"]),
                "beta0": float(fit["beta0"]),
                "sigma_beta": sigma_beta,
                "n": n_inliers,
                "r2": r2,
                "is_crossover": charge == cn,
            }
            if charge == cn:
                crossover_species.append(label)
                continue
            rows.append(row)
            all_rows.append(row)
            present_species.add(label)

        frame = pd.DataFrame(rows).sort_values(["charge", "species"]).reset_index(drop=True)
        beta_charge_frames[cn] = frame
        panel_stats[cn] = {
            "species_count": int(frame["species"].nunique()) if not frame.empty else 0,
            "point_count": int(len(frame)),
            "weight_sum": int(frame["n"].sum()) if not frame.empty else 0,
            "crossover_species": crossover_species,
        }

    if not all_rows:
        raise RuntimeError("No off-pole CN 4/6 beta-charge points were available for export.")

    weights = np.asarray([max(int(row["n"]), 1) for row in all_rows], dtype=float)
    observed = np.asarray([float(row["beta"]) for row in all_rows], dtype=float)
    predicted = np.asarray(
        [1.0 / np.log(float(row["charge"]) / float(row["cn"])) for row in all_rows],
        dtype=float,
    )
    mean_beta = np.average(observed, weights=weights)
    ss_res = np.sum(weights * (observed - predicted) ** 2)
    ss_tot = np.sum(weights * (observed - mean_beta) ** 2)
    overall_stats = {
        "species_count": len(present_species),
        "point_count": len(all_rows),
        "weighted_r2": float(1.0 - ss_res / ss_tot),
        "weighted_mae": float(np.sum(weights * np.abs(observed - predicted)) / weights.sum()),
        "missing_species": sorted(set(species_labels) - present_species),
    }
    return beta_charge_frames, panel_stats, overall_stats


def build_element_color_map(elements: list[str]) -> dict[str, str]:
    palette: list[str] = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3"):
        cmap = plt.get_cmap(cmap_name)
        count = int(getattr(cmap, "N", 20))
        for index in range(count):
            palette.append(to_hex(cmap(index / max(count - 1, 1))))
    if len(elements) > len(palette):
        cmap = plt.get_cmap("hsv")
        needed = len(elements) - len(palette)
        for index in range(needed):
            palette.append(to_hex(cmap(index / max(needed, 1))))
    return {element: palette[index] for index, element in enumerate(elements)}


def render_figure(beta_charge_frames: dict[int, pd.DataFrame]) -> Path:
    panels = [
        (cn, beta_charge_frames[cn])
        for cn in (4, 6)
        if cn in beta_charge_frames and beta_charge_frames[cn]["element"].nunique() >= 2
    ]
    if not panels:
        raise RuntimeError("No CN 4/6 beta-charge panels were available for export.")

    fig, axes, style = manuscript_subplots(
        "full_width_2panel",
        ncols=len(panels),
        panel_height_ratio=0.78,
        sharey=False,
        constrained_layout=False,
    )
    if len(panels) == 1:
        axes = [axes]

    elements = sorted({element for _cn, frame in panels for element in frame["element"].unique()})
    colors_by_element = build_element_color_map(elements)

    with plt.rc_context(style.rcparams()):
        for ax, (cn, frame) in zip(axes, panels):
            fit = {"amplitude": 1.0, "z_cross": float(cn)}
            plot_fixed_cn_beta_panel(
                ax,
                frame,
                cn,
                fit=fit,
                show_legend=False,
                annotate_crossover=True,
                colors_by_element=colors_by_element,
                title=False,
                series_line_alpha=0.35,
                fit_zorder=5.0,
                crossover_annotation_fontsize=style.annotation_pt - 1.0,
            )
            apply_manuscript_axis_style(
                ax,
                style,
                title=None,
                xlabel=None,
                ylabel=r"$\beta$",
            )
            ax.set_xlabel("")
            ax.text(
                0.03,
                0.96,
                fr"$n = {cn}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=style.annotation_pt,
            )
            ax.xaxis.labelpad = 0
            ax.margins(x=0.04, y=0.06)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="0.3",
                linestyle="",
                markerfacecolor="0.3",
                label=r"$z < n$",
                markersize=5.2,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="0.3",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="0.3",
                label=r"$z > n$",
                markersize=5.2,
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=style.line_width_pt,
                label=r"$\beta = 1 / \ln(z/n)$",
            ),
        ]
        fig.subplots_adjust(bottom=0.27, top=0.95, wspace=0.24)
        fig.supxlabel(r"$z$", fontsize=style.axis_label_pt, y=0.09)
        legend_ax = fig.add_axes([0.10, 0.01, 0.80, 0.05])
        legend_ax.set_axis_off()
        style_manuscript_legend(
            legend_ax,
            style,
            handles=legend_handles,
            loc="center",
            fontsize=style.legend_pt,
            title_fontsize=style.legend_pt,
            ncol=3,
            columnspacing=1.0,
            handlelength=1.7,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        fig.align_labels()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return save_manuscript_figure(
        fig,
        OUTPUT_NAME,
        figures_dir=FIGURES_DIR,
        theory_figures_dir=THEORY_FIGURES_DIR,
        dpi=300,
    )


def main() -> None:
    init_notebook()
    beta_charge_frames, panel_stats, overall_stats = extract_master_beta_charge_frames()
    out = render_figure(beta_charge_frames)
    print(f"Wrote {out}")
    print(
        "CN4/CN6 coverage:",
        f"{panel_stats[4]['species_count']} species at n=4,",
        f"{panel_stats[6]['species_count']} species at n=6,",
        f"{overall_stats['species_count']} species total",
        f"({overall_stats['point_count']} points).",
    )
    print(
        "Weighted metrics:",
        f"R^2={overall_stats['weighted_r2']:.3f},",
        f"MAE={overall_stats['weighted_mae']:.3f}.",
    )
    if overall_stats["missing_species"]:
        print("Species not shown in CN4/CN6 panels:", ", ".join(overall_stats["missing_species"]))


if __name__ == "__main__":
    main()
