"""Theory-facing bond-valence summary helpers for manuscript payloads."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from critmin.analysis.bond_valence_tracking import BUCKETS, empty_bucket_payload, is_cation_payload
from critmin.analysis.bond_valence_store import (
    DEFAULT_OXYGEN_AUTHORITATIVE_SOURCE_SPEC,
    load_grouped_payload_source,
)
from critmin.analysis.config import (
    ACTINIDE_CATIONS,
    DBLOCK_3D_CATIONS,
    DBLOCK_4D_CATIONS,
    DBLOCK_5D_CATIONS,
    NONMETAL_HALOGEN_OXYGEN_CATIONS,
    POST_TRANSITION_OXYGEN_CATIONS,
)
from critmin.analysis.default_sources import (
    resolve_all_remaining_oxygen_accepted_sources,
    resolve_all_remaining_oxygen_high_uncertainty_sources,
    resolve_dblock_4d_post_oxygen_accepted_sources,
    resolve_dblock_4d_post_oxygen_high_uncertainty_sources,
    resolve_dblock_3d_oxygen_accepted_sources,
    resolve_dblock_3d_oxygen_high_uncertainty_sources,
)
from critmin.viz.bond_valence import fit_bv_regression, fit_bv_regression_ransac
from critmin.analysis.bond_valence import resolve_tracker_coordination

GROUP1_CATIONS: tuple[str, ...] = ("Li", "Na", "K", "Rb", "Cs")
GROUP2_CATIONS: tuple[str, ...] = ("Be", "Mg", "Ca", "Sr", "Ba")
LANTHANIDE_CATIONS: tuple[str, ...] = (
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
)
DEFAULT_GROUP12_ACCEPTED_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/group12_two_phase_20260410T230957Z.json",
)
DEFAULT_GROUP12_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/group12_two_phase_20260410T230957Z_high_uncertainty.json",
)
DEFAULT_GROUP12_TRACKERS: tuple[str, ...] = (
    "data/raw/ima/group12_candidates.json",
    "data/raw/ima/group1_candidates.json",
    "data/raw/ima/group2_candidates.json",
)
DEFAULT_DBLOCK_ACCEPTED_SOURCES: tuple[str, ...] = (
    resolve_dblock_3d_oxygen_accepted_sources()
)
DEFAULT_DBLOCK_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    resolve_dblock_3d_oxygen_high_uncertainty_sources()
)
DEFAULT_DBLOCK_TRACKERS: tuple[str, ...] = (
    "data/raw/ima/dblock_3d_candidates.json",
)
DEFAULT_LANTHANIDE_ACCEPTED_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/fblock_4f_two_phase.json",
)
DEFAULT_LANTHANIDE_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/fblock_4f_two_phase_high_uncertainty.json",
)
DEFAULT_LANTHANIDE_TRACKERS: tuple[str, ...] = (
    "data/raw/ima/fblock_4f_candidates.json",
)
DEFAULT_MASTER_TRANSITION_ACCEPTED_SOURCES: tuple[str, ...] = (
    *DEFAULT_DBLOCK_ACCEPTED_SOURCES,
    *resolve_dblock_4d_post_oxygen_accepted_sources(),
    *resolve_all_remaining_oxygen_accepted_sources(),
)
DEFAULT_MASTER_TRANSITION_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    *DEFAULT_DBLOCK_HIGH_UNCERTAINTY_SOURCES,
    *resolve_dblock_4d_post_oxygen_high_uncertainty_sources(),
    *resolve_all_remaining_oxygen_high_uncertainty_sources(),
)
DEFAULT_MASTER_TRANSITION_TRACKERS: tuple[str, ...] = (
    "data/raw/ima/dblock_3d_candidates.json",
    "data/raw/ima/dblock_4d_post_o_candidates_20260412T155839Z.json",
    "data/raw/ima/all_remaining_o_candidates_20260413T030533Z.json",
)
DEFAULT_MASTER_POST_ACCEPTED_SOURCES: tuple[str, ...] = (
    *resolve_dblock_4d_post_oxygen_accepted_sources(),
    "data/processed/remote_jobs/post_transition_remaining_o_two_phase_20260414T045256Z.json",
)
DEFAULT_MASTER_POST_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    *resolve_dblock_4d_post_oxygen_high_uncertainty_sources(),
    "data/processed/remote_jobs/post_transition_remaining_o_two_phase_20260414T045256Z_high_uncertainty.json",
)
DEFAULT_MASTER_POST_TRACKERS: tuple[str, ...] = (
    "data/raw/ima/dblock_4d_post_o_candidates_20260412T155839Z.json",
    "data/raw/ima/post_transition_remaining_o_candidates_20260414T045256Z.json",
)
DEFAULT_THEORY_OUTPUT_DIR = Path("data/processed/theory")
DEFAULT_DBLOCK_OXI_OUTLIERS: tuple[str, ...] = ()
DEFAULT_MASTER_MIN_LINES = 2
DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES: tuple[str, ...] = (
    DEFAULT_OXYGEN_AUTHORITATIVE_SOURCE_SPEC,
)
DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = ()
MASTER_FAMILY_ORDER: tuple[str, ...] = (
    "Group 1",
    "Group 2",
    "Lanthanides",
    "Actinides",
    "3d transition",
    "4d transition",
    "5d transition",
    "Post-transition / p-block",
    "Nonmetal / halogen",
)
_SUPERSCRIPTS = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SUPERSCRIPT_PLAIN = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def _record_identity(record: Mapping[str, Any]) -> str:
    mid = record.get("mid")
    if mid:
        return str(mid)
    return json.dumps(dict(record), sort_keys=True)


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_float(value)
    return value


def load_grouped_payload_json(
    path: str | Path,
    *,
    default_status: str | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load one grouped payload source.

    ``path`` may be either a grouped-payload JSON file or a consolidated-store
    reference of the form ``store.json::dataset_key``. Missing plain files are
    ignored so default export commands can tolerate partially refreshed
    worktrees.
    """
    return load_grouped_payload_source(path, default_status=default_status)


def merge_grouped_payloads_first_wins(
    *payloads: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Merge grouped payloads while preserving the first record seen for each MID."""
    merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    seen: dict[str, dict[str, set[str]]] = {}

    for payload in payloads:
        for cation, cation_payload in payload.items():
            if not is_cation_payload(cation_payload):
                continue
            cat = str(cation)
            merged.setdefault(cat, empty_bucket_payload())
            seen.setdefault(cat, {bucket: set() for bucket in BUCKETS})
            for bucket in BUCKETS:
                for record in cation_payload.get(bucket, []):
                    identity = _record_identity(record)
                    if identity in seen[cat][bucket]:
                        continue
                    seen[cat][bucket].add(identity)
                    merged[cat][bucket].append(dict(record))

    return merged


def filter_grouped_payload_by_status(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    keep: Sequence[str] = ("fitted",),
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return a grouped payload copy containing only matching statuses."""
    keep_set = {str(status) for status in keep}
    filtered: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cation, cation_payload in data.items():
        if not is_cation_payload(cation_payload):
            continue
        filtered[str(cation)] = empty_bucket_payload()
        for bucket in BUCKETS:
            filtered[str(cation)][bucket] = [
                dict(record)
                for record in cation_payload.get(bucket, [])
                if str(record.get("status", "fitted")) in keep_set
            ]
    return filtered


def _iter_tracker_cations(tracker: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    def _is_tracker_cation_block(value: Any) -> bool:
        return (
            isinstance(value, Mapping)
            and all(bucket in value for bucket in BUCKETS)
            and all(isinstance(value.get(bucket), (Mapping, list)) for bucket in BUCKETS)
        )

    if _is_tracker_cation_block(tracker):
        cation = tracker.get("cation")
        if cation:
            return [(str(cation), tracker)]
        return []

    items: list[tuple[str, Mapping[str, Any]]] = []
    for cation, payload in tracker.items():
        if _is_tracker_cation_block(payload) or is_cation_payload(payload):
            items.append((str(cation), payload))
    return items


def build_tracker_cn_lookup(*tracker_paths: str | Path) -> dict[str, int]:
    """Return ``mid -> cn_mode`` across one or more tracker JSON files."""
    lookup: dict[str, int] = {}
    for tracker_path in tracker_paths:
        path = Path(tracker_path)
        if not path.exists():
            continue
        tracker = json.loads(path.read_text(encoding="utf-8"))
        for _cation, payload in _iter_tracker_cations(tracker):
            for bucket in BUCKETS:
                candidates = payload.get(bucket, {})
                candidate_list = (
                    candidates.get("candidates", [])
                    if isinstance(candidates, dict)
                    else candidates
                )
                for candidate in candidate_list:
                    if not isinstance(candidate, dict):
                        continue
                    mid = candidate.get("mid")
                    cn, _cn_all = resolve_tracker_coordination(
                        cn_mode=candidate.get("cn_mode"),
                        cn_all=candidate.get("cn_all"),
                    )
                    if mid is None or cn is None:
                        continue
                    lookup[str(mid)] = int(cn)
    return lookup


def inject_tracker_coordination(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cn_lookup: Mapping[str, int],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Fill missing ``cn`` values from tracker metadata."""
    enriched: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cation, cation_payload in data.items():
        if not is_cation_payload(cation_payload):
            continue
        enriched[str(cation)] = empty_bucket_payload()
        for bucket in BUCKETS:
            for record in cation_payload.get(bucket, []):
                copied = dict(record)
                if copied.get("cn") is None:
                    mid = copied.get("mid")
                    if mid is not None and str(mid) in cn_lookup:
                        copied["cn"] = int(cn_lookup[str(mid)])
                enriched[str(cation)][bucket].append(copied)
    return enriched


def merge_oxygen_records(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cation: str,
) -> list[dict[str, Any]]:
    """Pool oxide and hydroxide records into one O-coordinated slice."""
    merged: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        for record in data.get(cation, {}).get(bucket, []):
            copied = dict(record)
            copied["source_bucket"] = bucket
            merged.append(copied)
    return merged


def observed_coordination_numbers(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cations: Sequence[str],
) -> list[int]:
    """Return the sorted coordination numbers observed across *cations*."""
    observed: set[int] = set()
    for cation in cations:
        for bucket in BUCKETS:
            for record in data.get(cation, {}).get(bucket, []):
                cn = record.get("cn")
                if cn is not None:
                    observed.add(int(cn))
    return sorted(observed)


def _valid_fit_record(record: Mapping[str, Any]) -> bool:
    return _finite_float(record.get("R0")) is not None and _finite_float(record.get("B")) is not None


def _augment_fit_with_ranges(
    fit: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    inlier_mask: Sequence[bool] | None = None,
) -> dict[str, Any]:
    enriched = dict(fit)
    if not records:
        return enriched

    if inlier_mask is not None:
        selected = [record for record, keep in zip(records, inlier_mask) if keep]
    else:
        selected = list(records)
    if not selected:
        selected = list(records)

    r0_values = np.array([float(record["R0"]) for record in selected], dtype=float)
    b_values = np.array([float(record["B"]) for record in selected], dtype=float)
    enriched.update(
        {
            "R0_min": float(np.min(r0_values)),
            "R0_max": float(np.max(r0_values)),
            "B_min": float(np.min(b_values)),
            "B_max": float(np.max(b_values)),
        }
    )
    return enriched


def _serialize_fit_summary(fit: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "beta",
        "beta0",
        "r2",
        "p",
        "n",
        "n_inliers",
        "n_outliers",
        "beta_stderr",
        "beta0_stderr",
        "beta_ci95",
        "beta0_ci95",
        "uncertainty_method",
        "uncertainty_n_bootstrap",
        "R0_min",
        "R0_max",
        "B_min",
        "B_max",
    )
    serialized = {key: _json_ready(fit.get(key)) for key in keys if key in fit}
    return serialized


def build_unified_oxygen_cn_fits(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cations: Sequence[str],
    *,
    target_cns: Sequence[int] | None = None,
    ransac: bool = True,
    min_points: int = 5,
) -> tuple[dict[str, dict[int, dict[str, dict[str, Any]]]], list[dict[str, Any]]]:
    """Fit CN-resolved ``B = beta * R0 + beta0`` lines on unified O records."""
    cn_values = list(target_cns) if target_cns is not None else observed_coordination_numbers(data, cations)
    cn_fits: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    outliers: list[dict[str, Any]] = []

    for cation in cations:
        cn_fits[cation] = {}
        oxygen_records = merge_oxygen_records(data, cation)
        for cn in cn_values:
            cn_fits[cation][int(cn)] = {}
            cn_records = [record for record in oxygen_records if record.get("cn") == cn]
            fit_records = [record for record in cn_records if _valid_fit_record(record)]
            fit_input = [
                {"R0": float(record["R0"]), "B": float(record["B"])}
                for record in fit_records
            ]
            if len(fit_input) < min_points:
                continue

            fit_ols = fit_bv_regression(fit_input)
            if fit_ols is not None:
                fit_ols = _augment_fit_with_ranges(fit_ols, fit_records)
                cn_fits[cation][int(cn)]["oxygen"] = _serialize_fit_summary(fit_ols)

            if not ransac:
                continue

            fit_ransac = fit_bv_regression_ransac(fit_input)
            if fit_ransac is None:
                continue

            inlier_mask = np.asarray(fit_ransac["inlier_mask"], dtype=bool)
            fit_ransac = _augment_fit_with_ranges(
                fit_ransac,
                fit_records,
                inlier_mask=inlier_mask,
            )
            cn_fits[cation][int(cn)]["oxygen_ransac"] = _serialize_fit_summary(fit_ransac)

            for index in np.where(~inlier_mask)[0]:
                record = fit_records[int(index)]
                residual = float(record["B"]) - (
                    float(fit_ransac["beta"]) * float(record["R0"]) + float(fit_ransac["beta0"])
                )
                outliers.append(
                    {
                        "cation": cation,
                        "cn": int(cn),
                        "mid": record.get("mid"),
                        "formula_pretty": record.get("formula_pretty", record.get("formula")),
                        "source_bucket": record.get("source_bucket"),
                        "R0": float(record["R0"]),
                        "B": float(record["B"]),
                        "residual": round(residual, 4),
                    }
                )

    return cn_fits, outliers


def iter_oxygen_cn_fit_points(
    cn_fits: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    cations: Sequence[str],
    target_cns: Sequence[int],
) -> list[dict[str, Any]]:
    """Return the per-CN unified-oxygen points used for alpha-line fitting."""
    points: list[dict[str, Any]] = []
    for cation in cations:
        for cn in target_cns:
            cn_block = cn_fits.get(cation, {}).get(int(cn), {})
            fit = cn_block.get("oxygen_ransac") or cn_block.get("oxygen")
            if fit is None:
                continue
            points.append(
                {
                    "cation": cation,
                    "cn": int(cn),
                    "beta": float(fit["beta"]),
                    "beta0": float(fit["beta0"]),
                    "n": int(fit.get("n_inliers", fit.get("n", 0)) or 0),
                }
            )
    return points


def fit_weighted_alpha_line(
    points: Sequence[Mapping[str, Any]],
    cation: str,
) -> dict[str, Any] | None:
    """Fit ``beta0 = alpha0 * beta + alpha1`` for one cation or species."""
    subset = [point for point in points if point["cation"] == cation]
    if len(subset) < 2:
        return None

    x = np.array([float(point["beta"]) for point in subset], dtype=float)
    y = np.array([float(point["beta0"]) for point in subset], dtype=float)
    w = np.array([max(int(point.get("n", 0)), 1) for point in subset], dtype=float)
    if np.allclose(x, x[0]):
        return None

    sw = float(w.sum())
    sx = float((w * x).sum() / sw)
    sy = float((w * y).sum() / sw)
    sxx = float((w * (x - sx) ** 2).sum())
    sxy = float((w * (x - sx) * (y - sy)).sum())
    if abs(sxx) < 1.0e-15:
        return None

    alpha0 = sxy / sxx
    alpha1 = sy - alpha0 * sx
    ss_res = float((w * (y - (alpha0 * x + alpha1)) ** 2).sum())
    ss_tot = float((w * (y - sy) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "cation": cation,
        "n_cn": len(subset),
        "total_inliers": int(w.sum()),
        "alpha0": float(alpha0),
        "alpha1": float(alpha1),
        "r2": float(r2),
        "beta_min": float(x.min()),
        "beta_max": float(x.max()),
    }


def build_alpha_line_table(
    cn_fits: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    cations: Sequence[str],
    target_cns: Sequence[int],
    *,
    group_label: str,
    include_element: bool = False,
) -> list[dict[str, Any]]:
    """Return ordered alpha-line rows for one manuscript panel family."""
    points = iter_oxygen_cn_fit_points(cn_fits, cations, target_cns)
    rows: list[dict[str, Any]] = []
    for cation in cations:
        fit = fit_weighted_alpha_line(points, cation)
        if fit is None:
            continue
        row = {
            "group": group_label,
            "cation": cation,
            "n_cn": int(fit["n_cn"]),
            "total_inliers": int(fit["total_inliers"]),
            "alpha0": float(fit["alpha0"]),
            "alpha1": float(fit["alpha1"]),
            "r2": _finite_float(fit["r2"]),
        }
        if include_element:
            row["element"] = cation.rstrip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
            row["species"] = cation
        rows.append(row)
    return rows


def group_convergence(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return the notebook-style least-spread beta* summary for alpha-lines."""
    if len(lines) < 2:
        return None

    a0 = np.array([float(line["alpha0"]) for line in lines], dtype=float)
    a1 = np.array([float(line["alpha1"]) for line in lines], dtype=float)
    d0 = a0 - a0.mean()
    d1 = a1 - a1.mean()
    denom = float((d0 ** 2).sum())
    if abs(denom) < 1.0e-15:
        return {
            "beta_star": None,
            "sigma_at_star": None,
            "mean_beta0_at_star": None,
            "sigma_at_0": float(a1.std()),
            "mean_beta0_at_0": float(a1.mean()),
            "n_lines": len(lines),
        }

    beta_star = -float((d0 * d1).sum() / denom)
    vals_at_star = a0 * beta_star + a1
    vals_at_zero = a1
    return {
        "beta_star": beta_star,
        "sigma_at_star": float(vals_at_star.std()),
        "mean_beta0_at_star": float(vals_at_star.mean()),
        "sigma_at_0": float(vals_at_zero.std()),
        "mean_beta0_at_0": float(vals_at_zero.mean()),
        "n_lines": len(lines),
    }


def collect_oxygen_fit_lines(
    cn_fits: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    cation: str,
    target_cns: Sequence[int],
) -> list[dict[str, Any]]:
    """Collect one cation's CN-resolved unified-oxygen lines."""
    lines: list[dict[str, Any]] = []
    for cn in target_cns:
        cn_block = cn_fits.get(cation, {}).get(int(cn), {})
        fit = cn_block.get("oxygen_ransac") or cn_block.get("oxygen")
        if fit is None:
            continue
        lines.append(
            {
                "cation": cation,
                "cn": int(cn),
                "beta": float(fit["beta"]),
                "beta0": float(fit["beta0"]),
                "R0_min": _finite_float(fit.get("R0_min")),
                "R0_max": _finite_float(fit.get("R0_max")),
                "n": int(fit.get("n_inliers", fit.get("n", 0)) or 0),
                "r2": float(fit.get("r2") or 0.0),
            }
        )
    return lines


def minimum_spread_intersection(
    lines: Sequence[Mapping[str, Any]],
    *,
    min_r2: float | None = None,
) -> dict[str, Any] | None:
    """Return the weighted least-spread intersection.

    Parameters
    ----------
    lines : sequence of line dicts
        Each dict must contain ``beta``, ``beta0``, and ``n``.
        If ``min_r2`` is set, each dict should also contain ``r2``.
    min_r2 : float or None
        If given, only lines with ``r2 >= min_r2`` are used in the
        intersection calculation.  Lines below the threshold are
        excluded before computing the weighted least-spread point.
    """
    if min_r2 is not None:
        lines = [l for l in lines if float(l.get("r2", 0.0)) >= min_r2]

    if len(lines) < 2:
        return None

    beta = np.array([float(line["beta"]) for line in lines], dtype=float)
    beta0 = np.array([float(line["beta0"]) for line in lines], dtype=float)
    weights = np.array([max(int(line.get("n", 0)), 1) for line in lines], dtype=float)

    sw = float(weights.sum())
    beta_mean = float((weights * beta).sum() / sw)
    beta0_mean = float((weights * beta0).sum() / sw)
    d_beta = beta - beta_mean
    d_beta0 = beta0 - beta0_mean
    denom = float((weights * d_beta ** 2).sum())

    if abs(denom) < 1.0e-15:
        r0_star = float("nan")
        b_values = beta0.copy()
    else:
        r0_star = -float((weights * d_beta * d_beta0).sum() / denom)
        b_values = beta * r0_star + beta0

    b_star = float((weights * b_values).sum() / sw)
    sigma_b = float(np.sqrt((weights * (b_values - b_star) ** 2).sum() / sw))
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


def build_intersection_table(
    cn_fits: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    cations: Sequence[str],
    target_cns: Sequence[int],
    *,
    group_label: str,
    include_element: bool = False,
    min_r2: float | None = None,
) -> list[dict[str, Any]]:
    """Return ordered least-spread intersection rows.

    Parameters
    ----------
    min_r2 : float or None
        If given, only CN lines with ``r2 >= min_r2`` are used in the
        intersection calculation.  Passed through to
        :func:`minimum_spread_intersection`.
    """
    rows: list[dict[str, Any]] = []
    for cation in cations:
        lines = collect_oxygen_fit_lines(cn_fits, cation, target_cns)
        result = minimum_spread_intersection(lines, min_r2=min_r2)
        if result is None:
            continue
        row = {
            "group": group_label,
            "cation": cation,
            "n_lines": int(result["n_lines"]),
            "R0_star": _finite_float(result["R0_star"]),
            "B_star": _finite_float(result["B_star"]),
            "sigma_B_at_R0_star": _finite_float(result["sigma_B_at_R0_star"]),
            "pct_sigma_B_at_R0_star": _finite_float(result["pct_sigma_B_at_R0_star"]),
            "B_range_at_R0_star": _finite_float(result["B_range_at_R0_star"]),
            "lines_used": [f"CN{line['cn']}" for line in lines],
        }
        if include_element:
            row["element"] = cation.rstrip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
            row["species"] = cation
            oxi_state = _parse_oxidation_state_from_label(cation)
            if oxi_state is not None:
                row["oxi_state"] = oxi_state
        rows.append(row)
    return rows


def oxidation_state_species_label(element: str, oxidation_state: int) -> str:
    """Return manuscript-style species labels such as ``Fe³⁺``."""
    sign = "+" if int(oxidation_state) >= 0 else "-"
    magnitude = abs(int(oxidation_state))
    return f"{element}{magnitude}{sign}".translate(_SUPERSCRIPTS)


def partition_dblock_by_oxidation_state(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    elements: Sequence[str] = DBLOCK_3D_CATIONS,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    list[str],
    dict[str, int],
]:
    """Partition d-block grouped payloads into pure oxidation-state species."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ordered_species: list[str] = []
    mixed_counts: dict[str, int] = {}

    for element in elements:
        cation_payload = data.get(element, empty_bucket_payload())
        by_oxidation_state: dict[int, dict[str, list[dict[str, Any]]]] = {}
        n_mixed = 0
        for bucket in BUCKETS:
            for record in cation_payload.get(bucket, []):
                if str(record.get("status", "fitted")) != "fitted":
                    continue
                if record.get("oxi_state_label") == "mixed":
                    n_mixed += 1
                    continue
                oxidation_state = record.get("oxi_state")
                if oxidation_state is None:
                    continue
                oxi = int(oxidation_state)
                by_oxidation_state.setdefault(oxi, empty_bucket_payload())
                by_oxidation_state[oxi][bucket].append(dict(record))
        mixed_counts[element] = n_mixed

        for oxidation_state in sorted(by_oxidation_state):
            species = oxidation_state_species_label(element, oxidation_state)
            grouped[species] = by_oxidation_state[oxidation_state]
            ordered_species.append(species)

    return grouped, ordered_species, mixed_counts


def build_dblock_cluster_summary(
    intersections: Sequence[Mapping[str, Any]],
    *,
    excluded_species: Sequence[str] = DEFAULT_DBLOCK_OXI_OUTLIERS,
) -> dict[str, Any]:
    """Summarize the oxidation-state resolved d-block ``(R0*, B*)`` cluster."""
    rows = [
        row
        for row in intersections
        if _finite_float(row.get("R0_star")) is not None and _finite_float(row.get("B_star")) is not None
    ]
    filtered = [row for row in rows if str(row.get("species", row.get("cation"))) not in excluded_species]

    def _sample_std(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    r0_values = np.array([float(row["R0_star"]) for row in filtered], dtype=float)
    b_values = np.array([float(row["B_star"]) for row in filtered], dtype=float)
    return {
        "excluded_species": list(excluded_species),
        "n_species_total": len(rows),
        "n_species_cluster": len(filtered),
        "cluster_species": [str(row.get("species", row.get("cation"))) for row in filtered],
        "R0_star_mean": float(r0_values.mean()) if len(r0_values) else None,
        "R0_star_std": _sample_std(r0_values) if len(r0_values) else None,
        "B_star_mean": float(b_values.mean()) if len(b_values) else None,
        "B_star_std": _sample_std(b_values) if len(b_values) else None,
        "R0_star_range": (
            [float(r0_values.min()), float(r0_values.max())] if len(r0_values) else []
        ),
        "B_star_range": (
            [float(b_values.min()), float(b_values.max())] if len(b_values) else []
        ),
    }


def _prepare_fitted_grouped_payload(
    *,
    accepted_sources: Sequence[str | Path],
    high_uncertainty_sources: Sequence[str | Path],
    tracker_paths: Sequence[str | Path],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load grouped payloads, keep fitted rows, and backfill missing CN values."""
    merged = _load_and_merge_sources(
        accepted_sources=accepted_sources,
        high_uncertainty_sources=high_uncertainty_sources,
    )
    fitted = filter_grouped_payload_by_status(merged, keep=("fitted",))
    cn_lookup = build_tracker_cn_lookup(*tracker_paths)
    return inject_tracker_coordination(fitted, cn_lookup)


def _build_intersection_family_payload(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cations: Sequence[str],
    *,
    group_label: str,
    include_element: bool = False,
    ransac: bool = True,
) -> dict[str, Any]:
    """Build one family's least-spread intersection rows."""
    target_cns = observed_coordination_numbers(data, cations)
    cn_fits, outliers = build_unified_oxygen_cn_fits(
        data,
        cations,
        target_cns=target_cns,
        ransac=ransac,
    )
    intersections = build_intersection_table(
        cn_fits,
        cations,
        target_cns,
        group_label=group_label,
        include_element=include_element,
    )
    return {
        "group_label": group_label,
        "cations": list(cations),
        "target_cns": target_cns,
        "n_outliers": len(outliers),
        "intersections": intersections,
    }


def _build_oxidation_state_family_payload(
    data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    elements: Sequence[str],
    *,
    group_label: str,
    ransac: bool = True,
) -> dict[str, Any]:
    """Build one oxidation-state-resolved family summary."""
    oxi_grouped, oxi_species, mixed_counts = partition_dblock_by_oxidation_state(
        data,
        elements=elements,
    )
    family = _build_intersection_family_payload(
        oxi_grouped,
        oxi_species,
        group_label=group_label,
        include_element=True,
        ransac=ransac,
    )
    total_pure_fitted = sum(
        len(oxi_grouped[species][bucket])
        for species in oxi_species
        for bucket in BUCKETS
    )
    family.update(
        {
            "elements": list(elements),
            "species": list(oxi_species),
            "mixed_counts": mixed_counts,
            "total_pure_fitted_records": total_pure_fitted,
        }
    )
    return family


def _parse_oxidation_state_from_label(label: str) -> int | None:
    plain = str(label).translate(_SUPERSCRIPT_PLAIN)
    suffix = plain.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    digits = "".join(ch for ch in suffix if ch.isdigit())
    if not digits:
        return None
    sign = -1 if "-" in suffix else 1
    return sign * int(digits)


def _master_sort_key(row: Mapping[str, Any]) -> tuple[int, str, int, str]:
    group = str(row.get("group") or "")
    label = str(row.get("label") or row.get("species") or row.get("cation") or "")
    element = str(row.get("element") or row.get("cation") or "")
    oxi_state = _parse_oxidation_state_from_label(label)
    try:
        order = MASTER_FAMILY_ORDER.index(group)
    except ValueError:
        order = len(MASTER_FAMILY_ORDER)
    return (order, element, oxi_state if oxi_state is not None else -1, label)


def _annotate_master_row(row: Mapping[str, Any], *, family_key: str) -> dict[str, Any]:
    copied = dict(row)
    label = str(copied.get("species") or copied.get("cation") or "")
    element = str(copied.get("element") or copied.get("cation") or "")
    copied["family_key"] = family_key
    copied["label"] = label
    copied["element"] = element
    explicit_oxi_state = copied.get("oxi_state")
    if explicit_oxi_state is not None:
        copied["oxi_state"] = int(explicit_oxi_state)
    else:
        copied["oxi_state"] = _parse_oxidation_state_from_label(label)
    return copied


def _sample_std_or_none(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return None
    return float(np.std(array, ddof=1))


def _summary_stats_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    r0_values = np.asarray([float(row["R0_star"]) for row in rows], dtype=float)
    b_values = np.asarray([float(row["B_star"]) for row in rows], dtype=float)
    return {
        "R0_star_mean": float(r0_values.mean()) if len(r0_values) else None,
        "R0_star_std": _sample_std_or_none(r0_values),
        "B_star_mean": float(b_values.mean()) if len(b_values) else None,
        "B_star_std": _sample_std_or_none(b_values),
    }


def build_master_oxygen_summary_payload_from_rows(
    family_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    min_lines: int = DEFAULT_MASTER_MIN_LINES,
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the pooled oxygen-only manuscript summary from family rows."""
    master_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    families: dict[str, Any] = {}

    for family_key, rows in family_rows.items():
        annotated = [_annotate_master_row(row, family_key=family_key) for row in rows]
        annotated.sort(key=_master_sort_key)
        included = [row for row in annotated
                    if int(row.get("n_lines", 0)) >= min_lines
                    and row.get("R0_star") is not None
                    and -0.5 <= float(row.get("B_star", 0) or 0) <= 1.5]
        excluded = [row for row in annotated if row not in included]
        group_label = str(annotated[0].get("group") or family_key) if annotated else family_key
        families[family_key] = {
            "group_label": group_label,
            "n_rows_total": len(annotated),
            "n_rows_included": len(included),
            "n_rows_excluded_sparse": len(excluded),
            "rows": included,
            "excluded_rows": excluded,
        }
        master_rows.extend(included)
        excluded_rows.extend(excluded)

    master_rows.sort(key=_master_sort_key)
    excluded_rows.sort(key=_master_sort_key)

    family_summary: list[dict[str, Any]] = []
    for group_label in MASTER_FAMILY_ORDER:
        subset = [row for row in master_rows if str(row.get("group")) == group_label]
        if not subset:
            continue
        summary = _summary_stats_from_rows(subset)
        family_summary.append(
            {
                "group": group_label,
                "n_points": len(subset),
                **summary,
            }
        )

    global_stats = (
        _summary_stats_from_rows(master_rows)
        if master_rows
        else {
            "R0_star_mean": None,
            "R0_star_std": None,
            "B_star_mean": None,
            "B_star_std": None,
        }
    )
    global_summary = {
        "n_rows": len(master_rows),
        "n_excluded_sparse_rows": len(excluded_rows),
        "min_lines_threshold": int(min_lines),
        **global_stats,
    }

    return {
        "workflow": "master_oxygen_summary",
        "sources": _json_ready(dict(sources or {})),
        "families": families,
        "master_rows": master_rows,
        "excluded_rows": excluded_rows,
        "family_summary": family_summary,
        "global_summary": global_summary,
    }


def _load_and_merge_sources(
    *,
    accepted_sources: Sequence[str | Path],
    high_uncertainty_sources: Sequence[str | Path],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    payloads: list[dict[str, dict[str, list[dict[str, Any]]]]] = []
    for source in accepted_sources:
        payloads.append(load_grouped_payload_json(source))
    for source in high_uncertainty_sources:
        payloads.append(load_grouped_payload_json(source, default_status="high_uncertainty"))
    return merge_grouped_payloads_first_wins(*payloads)


def _concat_unique_paths(*collections: Sequence[str | Path]) -> tuple[str | Path, ...]:
    merged: list[str | Path] = []
    seen: set[str] = set()
    for collection in collections:
        for item in collection:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return tuple(merged)


def build_group12_unified_oxygen_theory_payload(
    *,
    accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    tracker_paths: Sequence[str | Path] = DEFAULT_GROUP12_TRACKERS,
    ransac: bool = True,
) -> dict[str, Any]:
    """Build the manuscript's unified-oxygen summaries for Group 1 and Group 2."""
    all_group12 = _load_and_merge_sources(
        accepted_sources=accepted_sources,
        high_uncertainty_sources=high_uncertainty_sources,
    )
    fitted_group12 = filter_grouped_payload_by_status(all_group12, keep=("fitted",))
    cn_lookup = build_tracker_cn_lookup(*tracker_paths)
    fitted_group12 = inject_tracker_coordination(fitted_group12, cn_lookup)

    groups: dict[str, Any] = {}
    for group_key, group_label, cations in (
        ("group1", "Group 1", GROUP1_CATIONS),
        ("group2", "Group 2", GROUP2_CATIONS),
    ):
        group_data = {
            cation: fitted_group12.get(cation, empty_bucket_payload())
            for cation in cations
        }
        target_cns = observed_coordination_numbers(group_data, cations)
        cn_fits, outliers = build_unified_oxygen_cn_fits(
            group_data,
            cations,
            target_cns=target_cns,
            ransac=ransac,
        )
        alpha_lines = build_alpha_line_table(
            cn_fits,
            cations,
            target_cns,
            group_label=group_label,
        )
        intersections = build_intersection_table(
            cn_fits,
            cations,
            target_cns,
            group_label=group_label,
        )
        groups[group_key] = {
            "group_label": group_label,
            "cations": list(cations),
            "target_cns": target_cns,
            "cn_fits": _json_ready(cn_fits),
            "outliers": _json_ready(outliers),
            "alpha_lines": _json_ready(alpha_lines),
            "intersections": _json_ready(intersections),
            "group_convergence": _json_ready(group_convergence(alpha_lines)),
        }

    return {
        "sources": {
            "accepted_sources": [str(source) for source in accepted_sources],
            "high_uncertainty_sources": [str(source) for source in high_uncertainty_sources],
            "tracker_paths": [str(path) for path in tracker_paths],
        },
        "workflow": "unified_oxygen",
        "groups": groups,
    }


def build_master_oxygen_summary_payload(
    *,
    group12_payload: Mapping[str, Any] | None = None,
    group12_accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    group12_high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    group12_tracker_paths: Sequence[str | Path] = DEFAULT_GROUP12_TRACKERS,
    lanthanide_accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    lanthanide_high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    lanthanide_tracker_paths: Sequence[str | Path] = DEFAULT_LANTHANIDE_TRACKERS,
    transition_accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    transition_high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    transition_tracker_paths: Sequence[str | Path] = DEFAULT_MASTER_TRANSITION_TRACKERS,
    post_transition_accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    post_transition_high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    post_transition_tracker_paths: Sequence[str | Path] = DEFAULT_MASTER_POST_TRACKERS,
    ransac: bool = True,
    min_lines: int = DEFAULT_MASTER_MIN_LINES,
) -> dict[str, Any]:
    """Build the pooled oxygen-only manuscript summary payload."""
    if group12_payload is None:
        group12_payload = build_group12_unified_oxygen_theory_payload(
            accepted_sources=group12_accepted_sources,
            high_uncertainty_sources=group12_high_uncertainty_sources,
            tracker_paths=group12_tracker_paths,
            ransac=ransac,
        )

    lanthanide_data = _prepare_fitted_grouped_payload(
        accepted_sources=lanthanide_accepted_sources,
        high_uncertainty_sources=lanthanide_high_uncertainty_sources,
        tracker_paths=lanthanide_tracker_paths,
    )
    lanthanide_family = _build_intersection_family_payload(
        lanthanide_data,
        LANTHANIDE_CATIONS,
        group_label="Lanthanides",
        ransac=ransac,
    )

    transition_data = _prepare_fitted_grouped_payload(
        accepted_sources=transition_accepted_sources,
        high_uncertainty_sources=transition_high_uncertainty_sources,
        tracker_paths=transition_tracker_paths,
    )
    transition_3d_family = _build_oxidation_state_family_payload(
        transition_data,
        DBLOCK_3D_CATIONS,
        group_label="3d transition",
        ransac=ransac,
    )
    transition_4d_family = _build_oxidation_state_family_payload(
        transition_data,
        DBLOCK_4D_CATIONS,
        group_label="4d transition",
        ransac=ransac,
    )
    transition_5d_family = _build_oxidation_state_family_payload(
        transition_data,
        DBLOCK_5D_CATIONS,
        group_label="5d transition",
        ransac=ransac,
    )

    # Actinides are in the same all_remaining payload as the transition metals.
    actinide_family = _build_oxidation_state_family_payload(
        transition_data,
        ACTINIDE_CATIONS,
        group_label="Actinides",
        ransac=ransac,
    )

    # Nonmetal / halogen oxygen cations also live in the all_remaining payload.
    nonmetal_halogen_family = _build_oxidation_state_family_payload(
        transition_data,
        NONMETAL_HALOGEN_OXYGEN_CATIONS,
        group_label="Nonmetal / halogen",
        ransac=ransac,
    )

    post_transition_data = _prepare_fitted_grouped_payload(
        accepted_sources=post_transition_accepted_sources,
        high_uncertainty_sources=post_transition_high_uncertainty_sources,
        tracker_paths=post_transition_tracker_paths,
    )
    post_transition_family = _build_oxidation_state_family_payload(
        post_transition_data,
        POST_TRANSITION_OXYGEN_CATIONS,
        group_label="Post-transition / p-block",
        ransac=ransac,
    )

    payload = build_master_oxygen_summary_payload_from_rows(
        {
            "group1": group12_payload.get("groups", {}).get("group1", {}).get("intersections", []),
            "group2": group12_payload.get("groups", {}).get("group2", {}).get("intersections", []),
            "lanthanides": lanthanide_family["intersections"],
            "actinides": actinide_family["intersections"],
            "transition_3d": transition_3d_family["intersections"],
            "transition_4d": transition_4d_family["intersections"],
            "transition_5d": transition_5d_family["intersections"],
            "post_transition_pblock": post_transition_family["intersections"],
            "nonmetal_halogen": nonmetal_halogen_family["intersections"],
        },
        min_lines=min_lines,
        sources={
            "group12": {
                "accepted_sources": [str(source) for source in group12_accepted_sources],
                "high_uncertainty_sources": [str(source) for source in group12_high_uncertainty_sources],
                "tracker_paths": [str(path) for path in group12_tracker_paths],
            },
            "lanthanides": {
                "accepted_sources": [str(source) for source in lanthanide_accepted_sources],
                "high_uncertainty_sources": [str(source) for source in lanthanide_high_uncertainty_sources],
                "tracker_paths": [str(path) for path in lanthanide_tracker_paths],
            },
            "transition": {
                "accepted_sources": [str(source) for source in transition_accepted_sources],
                "high_uncertainty_sources": [str(source) for source in transition_high_uncertainty_sources],
                "tracker_paths": [str(path) for path in transition_tracker_paths],
            },
            "post_transition": {
                "accepted_sources": [str(source) for source in post_transition_accepted_sources],
                "high_uncertainty_sources": [str(source) for source in post_transition_high_uncertainty_sources],
                "tracker_paths": [str(path) for path in post_transition_tracker_paths],
            },
        },
    )
    payload["families"]["group1"].update(
        {
            "cations": list(group12_payload.get("groups", {}).get("group1", {}).get("cations", [])),
            "target_cns": list(group12_payload.get("groups", {}).get("group1", {}).get("target_cns", [])),
        }
    )
    payload["families"]["group2"].update(
        {
            "cations": list(group12_payload.get("groups", {}).get("group2", {}).get("cations", [])),
            "target_cns": list(group12_payload.get("groups", {}).get("group2", {}).get("target_cns", [])),
        }
    )
    payload["families"]["lanthanides"].update(
        {
            "cations": lanthanide_family["cations"],
            "target_cns": lanthanide_family["target_cns"],
        }
    )
    payload["families"]["transition_3d"].update(
        {
            "elements": transition_3d_family["elements"],
            "species": transition_3d_family["species"],
            "target_cns": transition_3d_family["target_cns"],
            "mixed_counts": transition_3d_family["mixed_counts"],
            "total_pure_fitted_records": transition_3d_family["total_pure_fitted_records"],
        }
    )
    payload["families"]["transition_4d"].update(
        {
            "elements": transition_4d_family["elements"],
            "species": transition_4d_family["species"],
            "target_cns": transition_4d_family["target_cns"],
            "mixed_counts": transition_4d_family["mixed_counts"],
            "total_pure_fitted_records": transition_4d_family["total_pure_fitted_records"],
        }
    )
    payload["families"]["transition_5d"].update(
        {
            "elements": transition_5d_family["elements"],
            "species": transition_5d_family["species"],
            "target_cns": transition_5d_family["target_cns"],
            "mixed_counts": transition_5d_family["mixed_counts"],
            "total_pure_fitted_records": transition_5d_family["total_pure_fitted_records"],
        }
    )
    payload["families"]["post_transition_pblock"].update(
        {
            "elements": post_transition_family["elements"],
            "species": post_transition_family["species"],
            "target_cns": post_transition_family["target_cns"],
            "mixed_counts": post_transition_family["mixed_counts"],
            "total_pure_fitted_records": post_transition_family["total_pure_fitted_records"],
        }
    )
    payload["families"]["actinides"].update(
        {
            "elements": actinide_family["elements"],
            "species": actinide_family["species"],
            "target_cns": actinide_family["target_cns"],
            "mixed_counts": actinide_family["mixed_counts"],
            "total_pure_fitted_records": actinide_family["total_pure_fitted_records"],
        }
    )
    payload["families"]["nonmetal_halogen"].update(
        {
            "elements": nonmetal_halogen_family["elements"],
            "species": nonmetal_halogen_family["species"],
            "target_cns": nonmetal_halogen_family["target_cns"],
            "mixed_counts": nonmetal_halogen_family["mixed_counts"],
            "total_pure_fitted_records": nonmetal_halogen_family["total_pure_fitted_records"],
        }
    )
    return payload


def build_dblock_unified_oxygen_theory_payload(
    *,
    accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    tracker_paths: Sequence[str | Path] = DEFAULT_DBLOCK_TRACKERS,
    excluded_species: Sequence[str] = DEFAULT_DBLOCK_OXI_OUTLIERS,
    ransac: bool = True,
) -> dict[str, Any]:
    """Build the manuscript's oxidation-state resolved d-block summaries."""
    dblock_all = _load_and_merge_sources(
        accepted_sources=accepted_sources,
        high_uncertainty_sources=high_uncertainty_sources,
    )
    cn_lookup = build_tracker_cn_lookup(*tracker_paths)
    dblock_all = inject_tracker_coordination(dblock_all, cn_lookup)
    dblock_oxi_data, ordered_species, mixed_counts = partition_dblock_by_oxidation_state(
        dblock_all,
        elements=DBLOCK_3D_CATIONS,
    )
    target_cns = observed_coordination_numbers(dblock_oxi_data, ordered_species)
    cn_fits, outliers = build_unified_oxygen_cn_fits(
        dblock_oxi_data,
        ordered_species,
        target_cns=target_cns,
        ransac=ransac,
    )
    alpha_lines = build_alpha_line_table(
        cn_fits,
        ordered_species,
        target_cns,
        group_label="D-block 3d",
        include_element=True,
    )
    intersections = build_intersection_table(
        cn_fits,
        ordered_species,
        target_cns,
        group_label="D-block 3d",
        include_element=True,
    )
    cluster_summary = build_dblock_cluster_summary(
        intersections,
        excluded_species=excluded_species,
    )

    total_pure_fitted = sum(
        len(dblock_oxi_data[species][bucket])
        for species in ordered_species
        for bucket in BUCKETS
    )

    return {
        "sources": {
            "accepted_sources": [str(source) for source in accepted_sources],
            "high_uncertainty_sources": [str(source) for source in high_uncertainty_sources],
            "tracker_paths": [str(path) for path in tracker_paths],
        },
        "workflow": "unified_oxygen_oxi_resolved",
        "group_label": "D-block 3d",
        "elements": list(DBLOCK_3D_CATIONS),
        "species": ordered_species,
        "mixed_counts": mixed_counts,
        "total_pure_fitted_records": total_pure_fitted,
        "target_cns": target_cns,
        "cn_fits": _json_ready(cn_fits),
        "outliers": _json_ready(outliers),
        "alpha_lines": _json_ready(alpha_lines),
        "intersections": _json_ready(intersections),
        "cluster_summary": _json_ready(cluster_summary),
    }


def _write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def export_manuscript_theory_payloads(
    *,
    output_dir: str | Path = DEFAULT_THEORY_OUTPUT_DIR,
    group12_accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    group12_high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    group12_tracker_paths: Sequence[str | Path] = DEFAULT_GROUP12_TRACKERS,
    dblock_accepted_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_ACCEPTED_SOURCES,
    dblock_high_uncertainty_sources: Sequence[str | Path] = DEFAULT_AUTHORITATIVE_OXYGEN_HIGH_UNCERTAINTY_SOURCES,
    dblock_tracker_paths: Sequence[str | Path] = DEFAULT_DBLOCK_TRACKERS,
    dblock_excluded_species: Sequence[str] = DEFAULT_DBLOCK_OXI_OUTLIERS,
    master_min_lines: int = DEFAULT_MASTER_MIN_LINES,
    ransac: bool = True,
) -> dict[str, Any]:
    """Write the manuscript theory payloads and return a manifest."""
    output_root = Path(output_dir)
    group12_payload = build_group12_unified_oxygen_theory_payload(
        accepted_sources=group12_accepted_sources,
        high_uncertainty_sources=group12_high_uncertainty_sources,
        tracker_paths=group12_tracker_paths,
        ransac=ransac,
    )
    dblock_payload = build_dblock_unified_oxygen_theory_payload(
        accepted_sources=dblock_accepted_sources,
        high_uncertainty_sources=dblock_high_uncertainty_sources,
        tracker_paths=dblock_tracker_paths,
        excluded_species=dblock_excluded_species,
        ransac=ransac,
    )
    master_payload = build_master_oxygen_summary_payload(
        group12_payload=group12_payload,
        group12_accepted_sources=group12_accepted_sources,
        group12_high_uncertainty_sources=group12_high_uncertainty_sources,
        group12_tracker_paths=group12_tracker_paths,
        transition_accepted_sources=dblock_accepted_sources,
        transition_high_uncertainty_sources=dblock_high_uncertainty_sources,
        transition_tracker_paths=_concat_unique_paths(
            dblock_tracker_paths,
            DEFAULT_MASTER_TRANSITION_TRACKERS,
        ),
        ransac=ransac,
        min_lines=master_min_lines,
    )

    group12_path = _write_json(output_root / "group12_unified_oxygen_theory.json", group12_payload)
    dblock_path = _write_json(output_root / "dblock_oxi_unified_oxygen_theory.json", dblock_payload)
    master_path = _write_json(output_root / "master_oxygen_summary_theory.json", master_payload)
    manifest = {
        "output_dir": str(output_root),
        "files": {
            "group12_unified_oxygen_theory": str(group12_path),
            "dblock_oxi_unified_oxygen_theory": str(dblock_path),
            "master_oxygen_summary_theory": str(master_path),
        },
        "sources": {
            "group12": group12_payload["sources"],
            "dblock": dblock_payload["sources"],
            "master_oxygen_summary": master_payload["sources"],
        },
        "summary": {
            "group1_alpha_lines": len(group12_payload["groups"]["group1"]["alpha_lines"]),
            "group2_alpha_lines": len(group12_payload["groups"]["group2"]["alpha_lines"]),
            "dblock_species": len(dblock_payload["species"]),
            "dblock_cluster_species": dblock_payload["cluster_summary"]["n_species_cluster"],
            "master_oxygen_rows": int(master_payload["global_summary"]["n_rows"]),
            "master_oxygen_excluded_sparse_rows": int(
                master_payload["global_summary"]["n_excluded_sparse_rows"]
            ),
            "master_oxygen_R0_star_mean": master_payload["global_summary"]["R0_star_mean"],
            "master_oxygen_B_star_mean": master_payload["global_summary"]["B_star_mean"],
        },
    }
    manifest_path = _write_json(output_root / "manuscript_theory_manifest.json", manifest)
    manifest["files"]["manifest"] = str(manifest_path)
    return manifest
