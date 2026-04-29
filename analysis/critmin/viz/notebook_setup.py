"""Shared setup and visualization helpers for bond-valence analysis notebooks.

Call :func:`init_notebook` at the top of each notebook to configure
matplotlib defaults, import common packages into the notebook namespace,
and register the standard cation color palette.  Then use the data-loading
and analysis helpers as needed.

Usage in a notebook cell::

    from critmin.viz.notebook_setup import *
    init_notebook()
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from critmin.viz.bond_valence import (
    fit_bv_regression,
    fit_bv_regression_huber,
    fit_bv_regression_ransac,
)
from critmin.viz.manuscript import (
    MANUSCRIPT_HALF_WIDTH_IN,
    MANUSCRIPT_TEXT_WIDTH_IN,
    ManuscriptFigureStyle,
    apply_manuscript_axis_style,
    get_manuscript_figure_style,
    manuscript_annotation_kwargs,
    manuscript_subplots,
    style_manuscript_legend,
    validate_manuscript_style,
)

# ── Cation group rosters ─────────────────────────────────────────────────────

G1_CATIONS: list[str] = ["Li", "Na", "K", "Rb", "Cs"]
G2_ALL_CATIONS: list[str] = ["Be", "Mg", "Ca", "Sr", "Ba"]
DBLOCK_3D_ELEMENTS: list[str] = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
LN_CATIONS: list[str] = [
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
]

# ── Color palettes ───────────────────────────────────────────────────────────

XO_PAIR_COLORS: dict[str, str] = {
    # Group 1/2
    "Li": "#332288", "Na": "#4477AA", "K": "#66CCEE", "Rb": "#44AA99",
    "Cs": "#117733", "Be": "#999933", "Mg": "#D9B44A", "Ca": "#CC6677",
    "Sr": "#AA4499", "Ba": "#882255",
    # Lanthanides
    "La": "#332288", "Ce": "#4477AA", "Pr": "#66CCEE", "Nd": "#44AA99",
    "Sm": "#117733", "Eu": "#999933", "Gd": "#D9B44A", "Tb": "#CC6677",
    "Dy": "#AA4499", "Ho": "#882255", "Er": "#661100", "Tm": "#6699CC",
    "Yb": "#888888", "Lu": "#333333",
}

CN_MARKERS: list[str] = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
CN_LINESTYLES: list = [
    "-", "--", "-.", ":",
    (0, (5, 1.25)), (0, (3, 1, 1, 1)), (0, (1, 1)),
    (0, (3, 1, 1, 1, 1, 1)), (0, (6, 2)), (0, (2, 1.2)),
]


# ── Matplotlib initialization ────────────────────────────────────────────────

def init_notebook() -> None:
    """Apply publication-style matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ── Color / marker helpers ──────────────────────────────────────────────────

def xo_color(cation: str, alpha: float | None = None):
    """Return the standard color for a cation–anion pair."""
    color = XO_PAIR_COLORS.get(cation, "#666666")
    return mcolors.to_rgba(color, alpha) if alpha is not None else color


def tint_color(color, mix_with_white: float = 0.0):
    """Lighten *color* by blending toward white."""
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    white = np.ones(3, dtype=float)
    return tuple((1.0 - mix_with_white) * rgb + mix_with_white * white)


def xo_shade_map(cation: str, values, light_start: float = 0.0, light_stop: float = 0.35):
    """Map *values* to progressively tinted shades of the cation color."""
    ordered = list(values)
    if not ordered:
        return {}
    mixes = ([light_start] if len(ordered) == 1
             else np.linspace(light_start, light_stop, len(ordered)))
    base = XO_PAIR_COLORS.get(cation, "#666666")
    return {value: tint_color(base, float(mix)) for value, mix in zip(ordered, mixes)}


def cn_marker_map(values):
    """Map *values* to cycling marker shapes."""
    return {v: CN_MARKERS[i % len(CN_MARKERS)] for i, v in enumerate(values)}


def cn_linestyle_map(values):
    """Map *values* to cycling line styles."""
    return {v: CN_LINESTYLES[i % len(CN_LINESTYLES)] for i, v in enumerate(values)}


# ── Data loading and filtering ───────────────────────────────────────────────

def load_grouped_payload(
    path: str, *, default_status: str | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load a grouped BV payload JSON, optionally stamping a default status."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if default_status is None:
        return data
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cat, cat_data in data.items():
        if not isinstance(cat_data, dict):
            continue
        out[cat] = {"oxides": [], "hydroxides": []}
        for bucket in ("oxides", "hydroxides"):
            for record in cat_data.get(bucket, []):
                copied = dict(record)
                copied.setdefault("status", default_status)
                out[cat][bucket].append(copied)
    return out


def merge_grouped_payloads(
    *payloads: Mapping[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Merge multiple grouped payloads, first-wins by ``mid``."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: dict[str, dict[str, set[str]]] = {}
    for payload in payloads:
        for cat, cat_data in payload.items():
            if not isinstance(cat_data, dict):
                continue
            out.setdefault(cat, {"oxides": [], "hydroxides": []})
            seen.setdefault(cat, {"oxides": set(), "hydroxides": set()})
            for bucket in ("oxides", "hydroxides"):
                for record in cat_data.get(bucket, []):
                    key = record.get("mid") or json.dumps(record, sort_keys=True)
                    if key in seen[cat][bucket]:
                        continue
                    seen[cat][bucket].add(key)
                    out[cat][bucket].append(record)
    return out


def filter_by_status(
    data: Mapping[str, Any], keep: tuple[str, ...] = ("fitted",),
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return a copy keeping only records whose status is in *keep*."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cat, cat_data in data.items():
        if not isinstance(cat_data, dict):
            continue
        out[cat] = {}
        for bucket in ("oxides", "hydroxides"):
            out[cat][bucket] = [
                r for r in cat_data.get(bucket, [])
                if r.get("status", "fitted") in keep
            ]
    return out


def inject_cn_from_trackers(
    data: dict[str, dict[str, list[dict[str, Any]]]],
    tracker_paths: Sequence[str],
) -> None:
    """Inject ``cn`` from tracker candidates into records that lack it."""
    cn_lookup: dict[str, int] = {}
    for tracker_path in tracker_paths:
        if not os.path.exists(tracker_path):
            continue
        with open(tracker_path) as f:
            tracker = json.load(f)
        for cat in tracker:
            for typ in ("oxides", "hydroxides"):
                candidates = tracker[cat].get(typ, {})
                cand_list = (candidates.get("candidates", [])
                             if isinstance(candidates, dict) else candidates)
                for c in cand_list:
                    if isinstance(c, dict) and c.get("cn_mode") is not None:
                        cn_lookup[c["mid"]] = c["cn_mode"]
    for cat in data:
        d = data[cat]
        if isinstance(d, dict):
            for typ in ("oxides", "hydroxides"):
                for r in d.get(typ, []):
                    if r.get("cn") is None and r.get("mid") in cn_lookup:
                        r["cn"] = cn_lookup[r["mid"]]


def report_coverage(
    label: str,
    all_data: Mapping[str, Any],
    filt_data: Mapping[str, Any],
    cations: Sequence[str],
) -> None:
    """Print a coverage summary for one cation group."""
    status_counts: Counter[str] = Counter()
    for c in cations:
        for typ in ("oxides", "hydroxides"):
            for r in all_data.get(c, {}).get(typ, []):
                status_counts[r.get("status", "fitted")] += 1
    total_all = sum(status_counts.values())
    total_filt = sum(
        len(filt_data.get(c, {}).get(typ, []))
        for c in cations for typ in ("oxides", "hydroxides")
    )
    with_cn = sum(
        1 for c in cations for typ in ("oxides", "hydroxides")
        for r in filt_data.get(c, {}).get(typ, []) if r.get("cn") is not None
    )
    print(f"  {label}: {total_all} total records")
    for s in ("fitted", "high_uncertainty", "boundary_capped"):
        if status_counts[s]:
            print(f"    {s}: {status_counts[s]}")
    print(f"  Using {total_filt} fitted records ({with_cn} with CN)")
    for c in cations:
        n = (len(filt_data.get(c, {}).get("oxides", []))
             + len(filt_data.get(c, {}).get("hydroxides", [])))
        print(f"    {c}: {n}")
    print()


# ── Analysis helpers ─────────────────────────────────────────────────────────

def observed_target_cns(
    data: Mapping[str, Any], cations: Sequence[str],
) -> list[int]:
    """Return sorted list of all CN values observed across *cations*."""
    observed: set[int] = set()
    for cat in cations:
        for bucket in ("oxides", "hydroxides"):
            for record in data.get(cat, {}).get(bucket, []):
                cn = record.get("cn")
                if cn is not None:
                    observed.add(int(cn))
    return sorted(observed)


def merge_oxygen_records(
    data: Mapping[str, Any], cation: str,
) -> list[dict[str, Any]]:
    """Pool oxide + hydroxide records for one cation, tagging source_bucket."""
    merged: list[dict[str, Any]] = []
    for bucket in ("oxides", "hydroxides"):
        for record in data.get(cation, {}).get(bucket, []):
            copied = dict(record)
            copied["source_bucket"] = bucket
            merged.append(copied)
    return merged


def augment_fit_with_ranges(
    fit: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    inlier_mask=None,
) -> dict[str, Any]:
    """Enrich a fit dict with R0/B min/max from *records*."""
    enriched = dict(fit)
    if not records:
        return enriched
    if inlier_mask is not None:
        selected = [r for r, keep in zip(records, inlier_mask) if keep]
    else:
        selected = list(records)
    if not selected:
        selected = list(records)
    r0_values = np.array([r["R0"] for r in selected], dtype=float)
    b_values = np.array([r["B"] for r in selected], dtype=float)
    enriched.update({
        "R0_min": float(np.min(r0_values)),
        "R0_max": float(np.max(r0_values)),
        "B_min": float(np.min(b_values)),
        "B_max": float(np.max(b_values)),
    })
    return enriched


def preferred_cn_fit(cn_block: dict[str, Any]) -> dict[str, Any] | None:
    """Return the Huber fit if available, then RANSAC, then OLS."""
    return (cn_block.get("oxygen_huber")
            or cn_block.get("oxygen_ransac")
            or cn_block.get("oxygen"))


def oxygen_cn_summary_table(
    data: Mapping[str, Any],
    cations: Sequence[str],
    *,
    target_cns: Sequence[int],
    ransac: bool = True,
    huber: bool = True,
) -> tuple[dict, list[dict[str, Any]]]:
    """Print a per-cation, per-CN summary table and return (cn_fits, outliers)."""
    header = (
        f"{'Cation':<6s}  {'CN':>3s}  {'Fit':<7s}  {'n':>4s}  "
        f"{'β':>8s}  {'β₀':>8s}  {'σβ':>8s}  {'σβ₀':>8s}  {'R²':>6s}"
    )
    if ransac:
        header += f"  {'outliers':>8s}"
    print(header)
    print("-" * len(header))

    cn_fits: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    all_outliers: list[dict[str, Any]] = []

    for cat in cations:
        cn_fits[cat] = {}
        oxygen_records = merge_oxygen_records(data, cat)
        for cn in target_cns:
            cn_fits[cat][cn] = {}
            cn_records = [r for r in oxygen_records if r.get("cn") == cn]
            fit_records = [{"R0": r["R0"], "B": r["B"]} for r in cn_records]

            f_ols = fit_bv_regression(fit_records) if len(fit_records) >= 5 else None
            if f_ols:
                f_ols = augment_fit_with_ranges(f_ols, cn_records)
                cn_fits[cat][cn]["oxygen"] = f_ols
                print(
                    f"{cat:<6s}  {cn:>3d}  {'OLS':<7s}  {f_ols['n']:>4d}  "
                    f"{f_ols['beta']:>8.4f}  {f_ols['beta0']:>8.4f}  "
                    f"{f_ols['beta_stderr']:>8.4f}  {f_ols['beta0_stderr']:>8.4f}  "
                    f"{f_ols['r2']:>6.3f}"
                    + ("" if not ransac else "          ")
                )
            else:
                print(
                    f"{cat:<6s}  {cn:>3d}  {'OLS':<7s}  {len(fit_records):>4d}  (too few)"
                )

            if huber:
                f_huber = (fit_bv_regression_huber(fit_records)
                           if len(fit_records) >= 5 else None)
                if f_huber:
                    outlier_mask = np.asarray(
                        f_huber.get("outlier_mask", []), dtype=bool,
                    )
                    huber_inlier = ~outlier_mask if len(outlier_mask) == len(cn_records) else None
                    f_huber = augment_fit_with_ranges(
                        f_huber, cn_records, inlier_mask=huber_inlier,
                    )
                    cn_fits[cat][cn]["oxygen_huber"] = f_huber
                    print(
                        f"{'':6s}  {'':>3s}  {'Huber':<7s}  {f_huber.get('n_inliers', f_huber['n']):>4d}  "
                        f"{f_huber['beta']:>8.4f}  {f_huber['beta0']:>8.4f}  "
                        f"{f_huber['beta_stderr']:>8.4f}  {f_huber['beta0_stderr']:>8.4f}  "
                        f"{f_huber['r2']:>6.3f}  {f_huber.get('n_outliers', 0):>4d}/{f_huber['n']:>4d}"
                    )

            if not ransac:
                continue
            f_ransac = (fit_bv_regression_ransac(fit_records)
                        if len(fit_records) >= 5 else None)
            if not f_ransac:
                continue
            inlier_mask = np.asarray(f_ransac["inlier_mask"], dtype=bool)
            f_ransac = augment_fit_with_ranges(
                f_ransac, cn_records, inlier_mask=inlier_mask,
            )
            cn_fits[cat][cn]["oxygen_ransac"] = f_ransac
            print(
                f"{'':6s}  {'':>3s}  {'RANSAC':<7s}  {f_ransac['n_inliers']:>4d}  "
                f"{f_ransac['beta']:>8.4f}  {f_ransac['beta0']:>8.4f}  "
                f"{f_ransac['beta_stderr']:>8.4f}  {f_ransac['beta0_stderr']:>8.4f}  "
                f"{f_ransac['r2']:>6.3f}  {f_ransac['n_outliers']:>4d}/{f_ransac['n']:>4d}"
            )
            for idx_out in np.where(~inlier_mask)[0]:
                record = cn_records[idx_out]
                residual = record["B"] - (
                    f_ransac["beta"] * record["R0"] + f_ransac["beta0"]
                )
                all_outliers.append({
                    "cation": cat, "cn": cn,
                    "mid": record.get("mid", ""),
                    "formula_pretty": record.get(
                        "formula_pretty", record.get("formula", ""),
                    ),
                    "R0": record["R0"], "B": record["B"],
                    "residual": round(float(residual), 4),
                })

    return cn_fits, all_outliers


def build_cn_fits(
    data: Mapping[str, Any],
    cations: Sequence[str],
    target_cns: Sequence[int],
    *,
    ransac: bool = True,
    huber: bool = True,
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    """Build per-cation, per-CN OLS + Huber + RANSAC fits.

    Returns ``{cation: {cn: {"oxygen": fit, "oxygen_huber": fit, "oxygen_ransac": fit}}}``
    mirroring the structure used throughout the notebook analysis cells.
    """
    cn_fits: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for cat in cations:
        cn_fits[cat] = {}
        oxygen_records = merge_oxygen_records(data, cat)
        for cn in target_cns:
            cn_fits[cat][cn] = {}
            cn_records = [r for r in oxygen_records if r.get("cn") == cn]
            fit_records = [{"R0": r["R0"], "B": r["B"]} for r in cn_records]

            f_ols = fit_bv_regression(fit_records) if len(fit_records) >= 5 else None
            if f_ols:
                f_ols = augment_fit_with_ranges(f_ols, cn_records)
                cn_fits[cat][cn]["oxygen"] = f_ols

            if huber:
                f_huber = (fit_bv_regression_huber(fit_records)
                           if len(fit_records) >= 5 else None)
                if f_huber:
                    outlier_mask = np.asarray(
                        f_huber.get("outlier_mask", []), dtype=bool,
                    )
                    inlier_mask = ~outlier_mask if len(outlier_mask) == len(cn_records) else None
                    f_huber = augment_fit_with_ranges(
                        f_huber, cn_records, inlier_mask=inlier_mask,
                    )
                    cn_fits[cat][cn]["oxygen_huber"] = f_huber

            if ransac:
                f_ransac = (fit_bv_regression_ransac(fit_records)
                            if len(fit_records) >= 5 else None)
                if f_ransac:
                    inlier_mask = np.asarray(f_ransac["inlier_mask"], dtype=bool)
                    f_ransac = augment_fit_with_ranges(
                        f_ransac, cn_records, inlier_mask=inlier_mask,
                    )
                    cn_fits[cat][cn]["oxygen_ransac"] = f_ransac
    return cn_fits
