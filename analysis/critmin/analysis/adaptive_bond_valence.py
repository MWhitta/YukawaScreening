"""Adaptive fitting helpers for near-degenerate bond-valence systems.

This module provides the fallback solvers and prior infrastructure used when
the linear least-squares fast path in :mod:`critmin.analysis.bond_valence`
cannot determine ``(R0, B)`` uniquely — typically because the target-bond
design matrix is rank-1 (a single distinct ``s_ij``) or otherwise
ill-conditioned.

Pipeline flow
-------------

The full per-material fitting cascade (implemented across this module and
:mod:`critmin.analysis.bond_valence`) is::

    1. Linear LS fast path          — closed-form 2-param OLS on the
       (bond_valence.py)              cation valence-sum residual.
                                      Requires rank-2 design matrix.
            ↓ degenerate
    2. Anion-centered fallback      — fits (R₀, B) from O²⁻ valence sums.
       (bond_valence.py →              Three-level cascade:
        anion_valence.py)              a) de novo joint (all species)
                                       b) partial prior (sparse fixed)
                                       c) target-only (all others fixed)
            ↓ fails or no structure graph
    3. Legacy SBV optimizer         — brute-force ensemble (shgo,
       (bond_valence.py)              differential_evolution, etc.)

When ``degenerate_fit_mode`` is set (the adaptive path), degenerate
materials are routed through :func:`solve_degenerate_fit` instead of
the legacy optimizer::

    1. Linear LS fast path (non-degenerate materials only)
            ↓ degenerate
    2. solve_degenerate_fit with the selected mode:
       - ``"anion-centered"``      — O²⁻ valence sums (preferred for z=CN)
       - ``"pole-line"``           — analytical λ(R₀) from the screened-
                                     valence pole model (avoids the β-line
                                     singularity at z=CN)
       - ``"beta-line"``           — exact constraint to a CN-specific
                                     β-line prior
       - ``"cn-line-regularized"`` — soft pull toward a CN-specific β-line
       - ``"regularized-b"``       — Tikhonov pull on B toward a prior
       - ``"hierarchical-b"``      — fit R₀ only, B from hierarchy prior
       - ``"fixed-b"``             — fit R₀ only, B from global default

Public surface
--------------

* :func:`analyze_material_fit_design` — diagnose degeneracy from the
  target-bond design matrix (rank, condition number, log-sij spread).

* :func:`fit_many_materials_adaptive` — batch driver that routes materials
  through the selected degenerate-fit mode.

* :func:`solve_degenerate_fit` — per-material adaptive dispatcher.

* :func:`build_hierarchical_b_priors` /
  :func:`build_cn_line_priors` /
  :func:`build_cn_line_priors_from_payload` — construct prior tables.

When ``force_adaptive_solver=True`` is passed through the driver, every
material with a valid theoretical solution is routed through this module's
solvers regardless of whether ``analyze_material_fit_design`` flagged it as
degenerate.  This is the path used by Phase B of the two-phase
orchestrator.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import median, pstdev
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from critmin.analysis.bond_valence_diagnostics import summarize_target_bond_lengths
from critmin.analysis.config import (
    DEFAULT_DEGENERATE_B_PRIOR,
    DEFAULT_DEGENERATE_CN_LINE_REGULARIZATION,
    DEFAULT_DEGENERATE_CONDITION_NUMBER,
    DEFAULT_DEGENERATE_LOG_SIJ_RANGE,
    DEFAULT_DEGENERATE_REGULARIZATION,
    DEFAULT_MAX_B_STD,
    DEFAULT_MAX_R0_STD,
)

VALID_DEGENERATE_FIT_MODES = (
    "anion-centered",
    "beta-line",
    "cn-line-regularized",
    "fixed-b",
    "hierarchical-b",
    "pole-line",
    "regularized-b",
)

# Global oxygen-only mean from the master summary (Eq. 5 of manuscript).
_DEFAULT_POLE_R0_STAR: float = 1.94
_DEFAULT_POLE_B_STAR: float = 0.37
_BOND_PART_RE = re.compile(r"\d+")

# Lazy cache for the anion-centered known-species lookup.
_ANION_KNOWN_SPECIES_CACHE: dict[str, tuple[float, float]] | None = None


def _get_anion_known_species() -> dict[str, tuple[float, float]]:
    """Load and cache the master table as an anion-centered known-species dict."""
    global _ANION_KNOWN_SPECIES_CACHE
    if _ANION_KNOWN_SPECIES_CACHE is not None:
        return _ANION_KNOWN_SPECIES_CACHE

    from critmin.analysis.anion_valence import species_key
    from critmin.analysis.shell_thermodynamics import (
        DEFAULT_MASTER_SUMMARY_PATH,
    )

    try:
        import json
        data = json.loads(DEFAULT_MASTER_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        _ANION_KNOWN_SPECIES_CACHE = {}
        return _ANION_KNOWN_SPECIES_CACHE

    table: dict[str, tuple[float, float]] = {}
    for r in data.get("master_rows", []):
        elem = str(r["element"])
        oxi = r.get("oxi_state")
        if oxi is not None:
            table[species_key(elem, int(oxi))] = (float(r["R0_star"]), float(r["B_star"]))
        else:
            default_oxi = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
                           "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2}.get(elem)
            if default_oxi is not None:
                table[species_key(elem, default_oxi)] = (float(r["R0_star"]), float(r["B_star"]))

    _ANION_KNOWN_SPECIES_CACHE = table
    return _ANION_KNOWN_SPECIES_CACHE


def _load_tqdm():
    """Return ``tqdm`` when installed, otherwise a no-op iterator wrapper."""
    try:
        from tqdm import tqdm as tqdm_impl
    except ModuleNotFoundError:
        def tqdm_impl(iterable, *args, **kwargs):
            return iterable

    return tqdm_impl


tqdm = _load_tqdm()


@dataclass(slots=True)
class AdaptiveMaterialFitResult:
    """Summary-carrying fit result compatible with the SBV payload builder."""

    material: Any
    theoretical: Any | None
    summary: Any | None = None
    failure_reasons: Sequence[str] = ()

    @property
    def has_fit(self) -> bool:
        return self.summary is not None

    def aggregate(self, reducer=median):
        return self.summary


@dataclass(slots=True)
class HierarchicalBPrior:
    """Resolved shared-B prior for a degenerate material."""

    value: float
    level: str
    count: int
    std: float


@dataclass(slots=True)
class CnLinePrior:
    """Resolved CN-specific ``B = beta * R0 + beta0`` prior."""

    beta: float
    beta0: float
    level: str
    count: int
    r2: float | None


ResolvedDegeneratePrior = HierarchicalBPrior | CnLinePrior


def _matching_target_bond_rows(material: Any, theoretical: Any) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    cation = getattr(material, "cation", None)
    anion = getattr(material, "anion", None)
    if not cation or not anion or theoretical is None:
        return rows

    bond_valences = dict(getattr(theoretical, "bond_valences", {}) or {})
    bond_lengths = dict(getattr(theoretical, "bond_lengths", {}) or {})
    for bond_type in getattr(theoretical, "bond_types", ()):
        parts = _BOND_PART_RE.split(str(bond_type))
        if len(parts) < 2 or parts[0] != cation or parts[1] != anion:
            continue

        sij = bond_valences.get(bond_type)
        distance = bond_lengths.get(bond_type)
        if sij is None or distance is None:
            continue
        sij = float(sij)
        if sij <= 0:
            continue
        rows.append(
            {
                "bond_type": str(bond_type),
                "sij": sij,
                "log_sij": float(np.log(sij)),
                "bond_length": float(distance),
            }
        )
    return rows


def _collapse_rows(
    rows: Sequence[Mapping[str, float | str]],
    *,
    round_digits: int = 8,
) -> list[dict[str, float]]:
    collapsed: dict[tuple[float, float], dict[str, float]] = {}
    for row in rows:
        log_sij = float(row["log_sij"])
        bond_length = float(row["bond_length"])
        key = (round(log_sij, round_digits), round(bond_length, round_digits))
        target = collapsed.setdefault(
            key,
            {
                "log_sij": log_sij,
                "bond_length": bond_length,
                "weight": 0.0,
            },
        )
        target["weight"] += 1.0
    return list(collapsed.values())


def _json_safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    cast = float(value)
    if not math.isfinite(cast):
        return None
    return cast


def _design_arrays(
    collapsed_rows: Sequence[Mapping[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(
        [[1.0, -float(row["log_sij"])] for row in collapsed_rows],
        dtype=float,
    )
    y = np.asarray([float(row["bond_length"]) for row in collapsed_rows], dtype=float)
    w = np.asarray([float(row.get("weight", 1.0)) for row in collapsed_rows], dtype=float)
    return x, y, w


def analyze_material_fit_design(
    material: Any,
    theoretical: Any,
    *,
    condition_threshold: float = DEFAULT_DEGENERATE_CONDITION_NUMBER,
    log_sij_range_threshold: float = DEFAULT_DEGENERATE_LOG_SIJ_RANGE,
) -> dict[str, Any]:
    """Return linear-design diagnostics for one material's target bond set."""
    rows = _matching_target_bond_rows(material, theoretical)
    result_like = SimpleNamespace(material=material, theoretical=theoretical)
    diagnostics = dict(summarize_target_bond_lengths(result_like))
    diagnostics.update(
        {
            "n_target_bonds": int(len(rows)),
            "n_distinct_equations": 0,
            "log_sij_range": None,
            "design_rank": 0,
            "design_condition_number": None,
            "degenerate_detected": True,
            "degenerate_reason": "no_target_bonds",
        }
    )
    if not rows:
        return diagnostics

    collapsed_rows = _collapse_rows(rows)
    log_values = [float(row["log_sij"]) for row in rows]
    diagnostics["n_distinct_equations"] = int(len(collapsed_rows))
    diagnostics["log_sij_range"] = _json_safe_float(max(log_values) - min(log_values))

    x, _y, w = _design_arrays(collapsed_rows)
    diagnostics["design_rank"] = int(np.linalg.matrix_rank(x))
    xtwx = x.T @ (w[:, None] * x)
    if diagnostics["design_rank"] >= 2:
        diagnostics["design_condition_number"] = _json_safe_float(np.linalg.cond(xtwx))

    if diagnostics["n_distinct_equations"] < 2 or diagnostics["design_rank"] < 2:
        diagnostics["degenerate_reason"] = "rank_deficient"
        return diagnostics
    if (diagnostics["log_sij_range"] or 0.0) <= log_sij_range_threshold:
        diagnostics["degenerate_reason"] = "low_log_sij_spread"
        return diagnostics
    if (
        diagnostics["design_condition_number"] is not None
        and diagnostics["design_condition_number"] >= condition_threshold
    ):
        diagnostics["degenerate_reason"] = "ill_conditioned"
        return diagnostics

    diagnostics["degenerate_detected"] = False
    diagnostics["degenerate_reason"] = None
    return diagnostics


def _fit_with_legacy_algorithms(
    *,
    sbv: Any,
    material: Any,
    theoretical: Any,
    algorithms: Sequence[str],
    r0_bounds: tuple[float, float],
) -> Any:
    fits: list[Any] = []
    failure_reasons: list[str] = []

    for algorithm in algorithms:
        solver = sbv.BVParamSolver(algo=algorithm)
        solution = solver.solve_R0Bs(
            cation=material.cation,
            anion=material.anion,
            bond_type_list=theoretical.bond_types,
            networkValence_dict=theoretical.bond_valences,
            bondLen_dict=theoretical.bond_lengths,
            materID=material.material_id,
            chem_formula=material.formula_pretty,
            R0_bounds=r0_bounds,
        )
        if solution is None:
            if solver.last_failure_reason:
                failure_reasons.append(f"{algorithm}:{solver.last_failure_reason}")
            continue
        fits.append(
            sbv.AlgorithmFitResult(
                algorithm=str(algorithm),
                r0=float(solution[0]),
                b=float(solution[1]),
            )
        )

    return sbv.MaterialFitResult(
        material=material,
        theoretical=theoretical,
        fits=fits,
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
    )


def _hierarchy_keys(material: Any) -> list[tuple[str, tuple[Any, ...]]]:
    metadata = dict(getattr(material, "metadata", {}) or {})
    cation = str(getattr(material, "cation", ""))
    bucket = metadata.get("bucket")
    oxi_state = metadata.get("oxi_state")
    cn = metadata.get("cn")

    keys: list[tuple[str, tuple[Any, ...]]] = []
    if bucket is not None and oxi_state is not None and cn is not None:
        keys.append(("cation_bucket_oxi_cn", (cation, bucket, int(oxi_state), int(cn))))
    if bucket is not None and cn is not None:
        keys.append(("cation_bucket_cn", (cation, bucket, int(cn))))
    if bucket is not None and oxi_state is not None:
        keys.append(("cation_bucket_oxi", (cation, bucket, int(oxi_state))))
    if bucket is not None:
        keys.append(("cation_bucket", (cation, bucket)))
    if oxi_state is not None:
        keys.append(("cation_oxi", (cation, int(oxi_state))))
    keys.append(("cation", (cation,)))
    return keys


def _cn_line_keys(material: Any) -> list[tuple[str, tuple[Any, ...]]]:
    metadata = dict(getattr(material, "metadata", {}) or {})
    cation = str(getattr(material, "cation", ""))
    bucket = metadata.get("bucket")
    oxi_state = metadata.get("oxi_state")
    cn = metadata.get("cn")
    if cn is None:
        return []

    keys: list[tuple[str, tuple[Any, ...]]] = []
    if bucket is not None and oxi_state is not None:
        keys.append(("cation_bucket_oxi_cn", (cation, bucket, int(oxi_state), int(cn))))
    if bucket is not None:
        keys.append(("cation_bucket_cn", (cation, bucket, int(cn))))
    if oxi_state is not None:
        keys.append(("cation_oxi_cn", (cation, int(oxi_state), int(cn))))
    keys.append(("cation_cn", (cation, int(cn))))
    return keys


def _fit_cn_line(points: Sequence[tuple[float, float]]) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None

    r0_values = np.asarray([float(point[0]) for point in points], dtype=float)
    b_values = np.asarray([float(point[1]) for point in points], dtype=float)
    if np.allclose(r0_values, r0_values[0]):
        return None

    x = np.column_stack((r0_values, np.ones(len(r0_values), dtype=float)))
    beta, beta0 = np.linalg.lstsq(x, b_values, rcond=None)[0]
    residuals = b_values - (beta * r0_values + beta0)
    ss_tot = float(np.sum((b_values - b_values.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(beta), float(beta0), float(r2)


def build_hierarchical_b_priors(
    successful_results: Sequence[Any],
    *,
    default_b: float = DEFAULT_DEGENERATE_B_PRIOR,
) -> dict[tuple[str, tuple[Any, ...]], HierarchicalBPrior]:
    """Build empirical shared-B priors from informative fits."""
    values_by_key: dict[tuple[str, tuple[Any, ...]], list[float]] = defaultdict(list)

    for result in successful_results:
        summary = result.aggregate()
        if summary is None:
            continue
        if summary.r0_std >= DEFAULT_MAX_R0_STD or summary.b_std >= DEFAULT_MAX_B_STD:
            continue
        if not math.isfinite(float(summary.b)) or abs(float(summary.b)) > 2.0:
            continue
        for key in _hierarchy_keys(result.material):
            values_by_key[key].append(float(summary.b))

    priors: dict[tuple[str, tuple[Any, ...]], HierarchicalBPrior] = {}
    for key, values in values_by_key.items():
        priors[key] = HierarchicalBPrior(
            value=float(median(values)),
            level=key[0],
            count=len(values),
            std=float(pstdev(values)) if len(values) > 1 else 0.0,
        )

    priors[("global_default", ())] = HierarchicalBPrior(
        value=float(default_b),
        level="global_default",
        count=0,
        std=0.0,
    )
    return priors


def build_cn_line_priors(
    successful_results: Sequence[Any],
) -> dict[tuple[str, tuple[Any, ...]], CnLinePrior]:
    """Build empirical CN-line priors from informative fits."""
    values_by_key: dict[tuple[str, tuple[Any, ...]], list[tuple[float, float]]] = defaultdict(list)

    for result in successful_results:
        summary = result.aggregate()
        if summary is None:
            continue
        if summary.r0_std >= DEFAULT_MAX_R0_STD or summary.b_std >= DEFAULT_MAX_B_STD:
            continue
        if not math.isfinite(float(summary.r0)) or not math.isfinite(float(summary.b)):
            continue
        if abs(float(summary.b)) > 2.0:
            continue
        for key in _cn_line_keys(result.material):
            values_by_key[key].append((float(summary.r0), float(summary.b)))

    priors: dict[tuple[str, tuple[Any, ...]], CnLinePrior] = {}
    for key, points in values_by_key.items():
        fit = _fit_cn_line(points)
        if fit is None:
            continue
        beta, beta0, r2 = fit
        priors[key] = CnLinePrior(
            beta=beta,
            beta0=beta0,
            level=key[0],
            count=len(points),
            r2=_json_safe_float(r2),
        )

    return priors


def resolve_hierarchical_b_prior(
    material: Any,
    priors: Mapping[tuple[str, tuple[Any, ...]], HierarchicalBPrior],
    *,
    default_b: float = DEFAULT_DEGENERATE_B_PRIOR,
) -> HierarchicalBPrior:
    for key in _hierarchy_keys(material):
        prior = priors.get(key)
        if prior is not None:
            return prior
    return priors.get(
        ("global_default", ()),
        HierarchicalBPrior(
            value=float(default_b),
            level="global_default",
            count=0,
            std=0.0,
        ),
    )


def build_cn_line_priors_from_payload(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    max_r0_std: float = DEFAULT_MAX_R0_STD,
    max_b_std: float = DEFAULT_MAX_B_STD,
    max_abs_b: float = 2.0,
    min_r2: float = 0.3,
) -> dict[tuple[str, tuple[Any, ...]], CnLinePrior]:
    """Build CN-line priors from an existing grouped payload (JSON dict).

    This allows priors to be seeded from a payload that was generated in a
    previous run, rather than requiring in-memory ``MaterialFitResult`` objects.
    """
    values_by_key: dict[tuple[str, tuple[Any, ...]], list[tuple[float, float]]] = defaultdict(list)

    for cation, cation_data in payload.items():
        if not isinstance(cation_data, dict):
            continue
        for bucket in ("oxides", "hydroxides"):
            for record in cation_data.get(bucket, []):
                r0 = record.get("R0")
                b = record.get("B")
                r0_std = record.get("R0_std")
                b_std = record.get("B_std")
                cn = record.get("cn")
                if r0 is None or b is None or cn is None:
                    continue
                if r0_std is not None and float(r0_std) >= max_r0_std:
                    continue
                if b_std is not None and float(b_std) >= max_b_std:
                    continue
                if not math.isfinite(float(r0)) or not math.isfinite(float(b)):
                    continue
                if abs(float(b)) > max_abs_b:
                    continue

                oxi_state = record.get("oxi_state")
                point = (float(r0), float(b))

                if oxi_state is not None:
                    values_by_key[
                        ("cation_bucket_oxi_cn", (cation, bucket, int(oxi_state), int(cn)))
                    ].append(point)
                values_by_key[
                    ("cation_bucket_cn", (cation, bucket, int(cn)))
                ].append(point)
                if oxi_state is not None:
                    values_by_key[
                        ("cation_oxi_cn", (cation, int(oxi_state), int(cn)))
                    ].append(point)
                values_by_key[("cation_cn", (cation, int(cn)))].append(point)

    priors: dict[tuple[str, tuple[Any, ...]], CnLinePrior] = {}
    for key, points in values_by_key.items():
        fit = _fit_cn_line(points)
        if fit is None:
            continue
        beta, beta0, r2 = fit
        if r2 < min_r2:
            continue
        priors[key] = CnLinePrior(
            beta=beta,
            beta0=beta0,
            level=key[0],
            count=len(points),
            r2=r2,
        )

    return priors


def resolve_cn_line_prior(
    material: Any,
    priors: Mapping[tuple[str, tuple[Any, ...]], CnLinePrior],
) -> CnLinePrior | None:
    for key in _cn_line_keys(material):
        prior = priors.get(key)
        if prior is not None:
            return prior
    return None


def _solve_fixed_b(
    collapsed_rows: Sequence[Mapping[str, float]],
    *,
    b_value: float,
) -> tuple[float, float, float, float, float]:
    _x, y, w = _design_arrays(collapsed_rows)
    log_sij = np.asarray([float(row["log_sij"]) for row in collapsed_rows], dtype=float)
    adjusted = y + float(b_value) * log_sij
    r0 = float(np.average(adjusted, weights=w))
    residuals = y - (r0 - float(b_value) * log_sij)
    obs_total = max(int(np.sum(w)), 1)
    rss = float(np.sum(w * residuals**2))
    sigma2 = rss / max(obs_total - 1, 1) if obs_total > 1 else 0.0
    r0_std = math.sqrt(max(sigma2 / float(np.sum(w)), 0.0)) if np.sum(w) > 0 else 0.0
    return r0, float(b_value), float(r0_std), 0.0, rss


def _solve_regularized_b(
    collapsed_rows: Sequence[Mapping[str, float]],
    *,
    b_prior: float,
    condition_number: float | None,
    base_lambda: float = DEFAULT_DEGENERATE_REGULARIZATION,
) -> tuple[float, float, float, float, float, float]:
    x, y, w = _design_arrays(collapsed_rows)
    if condition_number is None:
        lambda_b = base_lambda * 10.0
    else:
        lambda_b = max(base_lambda, min(1.0e6, condition_number / DEFAULT_DEGENERATE_CONDITION_NUMBER))

    xtwx = x.T @ (w[:, None] * x)
    xtwy = x.T @ (w * y)
    penalty = np.asarray([[0.0, 0.0], [0.0, float(lambda_b)]], dtype=float)
    rhs = xtwy + np.asarray([0.0, float(lambda_b) * float(b_prior)], dtype=float)
    system = xtwx + penalty
    beta = np.linalg.pinv(system) @ rhs
    residuals = y - x @ beta
    obs_total = max(int(np.sum(w)), 1)
    rss = float(np.sum(w * residuals**2))
    sigma2 = rss / max(obs_total - 2, 1) if obs_total > 2 else 0.0
    cov = sigma2 * np.linalg.pinv(system)
    r0_std = math.sqrt(max(float(cov[0, 0]), 0.0))
    b_std = math.sqrt(max(float(cov[1, 1]), 0.0))
    return (
        float(beta[0]),
        float(beta[1]),
        float(r0_std),
        float(b_std),
        float(rss),
        float(lambda_b),
    )


def _solve_cn_line_regularized(
    collapsed_rows: Sequence[Mapping[str, float]],
    *,
    line_prior: CnLinePrior,
    condition_number: float | None,
    base_lambda: float = DEFAULT_DEGENERATE_CN_LINE_REGULARIZATION,
) -> tuple[float, float, float, float, float, float, float]:
    x, y, w = _design_arrays(collapsed_rows)
    if condition_number is None:
        lambda_line = base_lambda * 10.0
    else:
        lambda_line = max(
            base_lambda,
            min(1.0e6, condition_number / DEFAULT_DEGENERATE_CONDITION_NUMBER),
        )

    constraint = np.asarray([-float(line_prior.beta), 1.0], dtype=float)
    xtwx = x.T @ (w[:, None] * x)
    xtwy = x.T @ (w * y)
    system = xtwx + float(lambda_line) * np.outer(constraint, constraint)
    rhs = xtwy + float(lambda_line) * float(line_prior.beta0) * constraint
    solution = np.linalg.pinv(system) @ rhs
    residuals = y - x @ solution
    obs_total = max(int(np.sum(w)), 1)
    rss = float(np.sum(w * residuals**2))
    sigma2 = rss / max(obs_total - 2, 1) if obs_total > 2 else 0.0
    cov = sigma2 * np.linalg.pinv(system)
    r0_std = math.sqrt(max(float(cov[0, 0]), 0.0))
    b_std = math.sqrt(max(float(cov[1, 1]), 0.0))
    line_residual = float(solution[1] - float(line_prior.beta) * solution[0] - float(line_prior.beta0))
    return (
        float(solution[0]),
        float(solution[1]),
        float(r0_std),
        float(b_std),
        float(rss),
        float(lambda_line),
        line_residual,
    )


def _solve_beta_line(
    collapsed_rows: Sequence[Mapping[str, float]],
    *,
    line_prior: CnLinePrior,
) -> tuple[float, float, float, float, float]:
    """Solve for R₀ with B constrained exactly to the β-line B = β·R₀ + β₀.

    Substituting into R_ij = R₀ − B·ln(s_ij) gives a single-parameter
    weighted regression::

        R_ij + β₀·ln(s_ij) = R₀·(1 − β·ln(s_ij))

    so the weighted least-squares solution is::

        R₀ = Σ w_i x_i y_i / Σ w_i x_i²

    where x_i = 1 − β·ln(s_ij) and y_i = R_ij + β₀·ln(s_ij).
    """
    beta = float(line_prior.beta)
    beta0 = float(line_prior.beta0)

    log_sij = np.asarray([float(row["log_sij"]) for row in collapsed_rows], dtype=float)
    bond_length = np.asarray([float(row["bond_length"]) for row in collapsed_rows], dtype=float)
    w = np.asarray([float(row.get("weight", 1.0)) for row in collapsed_rows], dtype=float)

    x = 1.0 - beta * log_sij
    y = bond_length + beta0 * log_sij
    xtwx = float(np.sum(w * x * x))
    if xtwx <= 1e-12:
        raise ValueError("All bonds fall on the β-line singularity (1 − β·ln s = 0)")

    r0 = float(np.sum(w * x * y) / xtwx)
    b_value = beta * r0 + beta0

    residuals = y - r0 * x
    obs_total = max(int(np.sum(w)), 1)
    rss = float(np.sum(w * residuals**2))
    sigma2 = rss / max(obs_total - 1, 1) if obs_total > 1 else 0.0
    r0_std = math.sqrt(max(sigma2 / xtwx, 0.0))
    b_std = abs(beta) * r0_std

    return float(r0), float(b_value), float(r0_std), float(b_std), float(rss)


def _analytical_lambda(r0: float, z: int, cn: int, r0_star: float, b_star: float) -> float | None:
    """Analytical λ(R₀, z, CN) from the pole model β = 1/ln(z/CN).

    Returns the positive-branch screening parameter, or None if outside
    the validity window (B ≤ 0 or discriminant < 0).
    """
    if z <= 0 or cn <= 0 or z == cn or r0 <= 0:
        return None
    beta_pole = 1.0 / math.log(z / cn)
    b = beta_pole * (r0 - r0_star) + b_star
    if b <= 0:
        return None
    disc = r0 * r0 + 4.0 * b * r0
    if disc < 0:
        return None
    lam = (math.sqrt(disc) - r0) / 2.0
    return lam if lam > 0 else None


def _solve_lambda_parameterized(
    collapsed_rows: Sequence[Mapping[str, float]],
    *,
    z: int,
    cn: int,
    r0_star: float,
    b_star: float,
) -> tuple[float, float, float, float, float]:
    """Solve for R₀ using the analytical λ(R₀, CN), bypassing the β-line singularity.

    For each trial R₀, B is derived from λ via B = (λ² + λR₀)/R₀, and the
    residual R_ij − R₀ + B·ln(s_ij) is minimized.  The λ(R₀) curve is smooth
    everywhere within the validity window, including at the pole β·ln(q/CN) = 1
    where the β-line linear solve is singular.
    """
    log_sij = np.asarray([float(row["log_sij"]) for row in collapsed_rows], dtype=float)
    bond_length = np.asarray([float(row["bond_length"]) for row in collapsed_rows], dtype=float)
    w = np.asarray([float(row.get("weight", 1.0)) for row in collapsed_rows], dtype=float)

    # Check for the exact-pole case: beta*ln(q/CN) = 1.
    # At the pole, R(R0) is constant across the entire validity window,
    # so R0 and B are not independently determinable.  The fixed-point
    # assignment R0 = mean(R_ij), B = B* is the natural resolution.
    beta_pole = 1.0 / math.log(z / cn) if z != cn else 0.0
    mean_log_sij = float(np.mean(log_sij))
    pole_gain = abs(beta_pole * mean_log_sij)
    if pole_gain > 0.95:
        # Near or at the pole: use the fixed-point assignment.
        r0 = float(np.average(bond_length, weights=w)) + b_star * mean_log_sij
        b_value = b_star
        residuals = bond_length - r0 + b_value * log_sij
        rss = float(np.sum(w * residuals ** 2))
        obs_total = max(int(np.sum(w)), 1)
        sigma2 = rss / max(obs_total - 1, 1) if obs_total > 1 else 0.0
        n_eq = len(set(zip(bond_length.tolist(), log_sij.tolist())))
        r0_std = math.sqrt(sigma2 / max(obs_total, 1)) if n_eq > 1 else 0.0
        b_std = 0.0  # B is fixed at B*
        return float(r0), float(b_value), float(r0_std), float(b_std), float(rss)

    def _rss_for_r0(r0: float) -> float:
        lam = _analytical_lambda(r0, z, cn, r0_star, b_star)
        if lam is None or lam <= 0 or r0 <= 0:
            return math.inf
        b = (lam * lam + lam * r0) / r0
        residuals = bond_length - r0 + b * log_sij
        return float(np.sum(w * residuals ** 2))
    r0_max = r0_star + abs(b_star / beta_pole) - 1e-6 if abs(beta_pole) > 1e-8 else r0_star + 2.0
    r0_grid = np.linspace(max(1e-3, r0_star - 2.0), max(r0_max, r0_star + 0.1), 2000)
    best_r0 = float("nan")
    best_rss = math.inf
    for r0_trial in r0_grid:
        rss = _rss_for_r0(float(r0_trial))
        if rss < best_rss:
            best_rss = rss
            best_r0 = float(r0_trial)

    if not math.isfinite(best_r0):
        raise ValueError("No valid R₀ found within the λ validity window")

    # Refine with golden-section search
    from scipy.optimize import minimize_scalar

    lo = max(1e-3, best_r0 - 0.1)
    hi = min(r0_max, best_r0 + 0.1)
    result = minimize_scalar(_rss_for_r0, bounds=(lo, hi), method="bounded")
    r0 = float(result.x)
    rss = float(result.fun)

    lam = _analytical_lambda(r0, z, cn, r0_star, b_star)
    if lam is None or lam <= 0 or r0 <= 0:
        raise ValueError("Refined R₀ fell outside validity window")
    b_value = (lam * lam + lam * r0) / r0

    # Uncertainty from curvature at the minimum
    obs_total = max(int(np.sum(w)), 1)
    sigma2 = rss / max(obs_total - 1, 1) if obs_total > 1 else 0.0
    # Numerical second derivative for R₀ uncertainty
    dr = 1e-5
    rss_lo = _rss_for_r0(r0 - dr)
    rss_hi = _rss_for_r0(r0 + dr)
    d2rss = (rss_lo - 2 * rss + rss_hi) / (dr * dr) if math.isfinite(rss_lo) and math.isfinite(rss_hi) else 1e12
    r0_std = math.sqrt(max(2.0 * sigma2 / max(d2rss, 1e-12), 0.0))
    b_std = abs(b_value / r0) * r0_std if r0 > 0 else 0.0  # propagated

    return float(r0), float(b_value), float(r0_std), float(b_std), float(rss)


def _build_pole_line_prior(
    material: Any,
    *,
    r0_star: float = _DEFAULT_POLE_R0_STAR,
    b_star: float = _DEFAULT_POLE_B_STAR,
) -> CnLinePrior | None:
    """Build a CN-line prior from the pole model β = 1/ln(z/CN).

    Requires ``oxi_state`` and ``cn`` in ``material.metadata``.
    Returns ``None`` when the pole is undefined (z ≤ 0, CN ≤ 0, or z = CN).
    """
    metadata = dict(getattr(material, "metadata", {}) or {})
    oxi_state = metadata.get("oxi_state")
    cn = metadata.get("cn")
    if oxi_state is None or cn is None:
        return None
    z = int(oxi_state)
    n = int(cn)
    if z <= 0 or n <= 0 or z == n:
        return None
    beta_pole = 1.0 / math.log(z / n)
    beta0_pole = b_star - beta_pole * r0_star
    return CnLinePrior(
        beta=beta_pole,
        beta0=beta0_pole,
        level="pole_model",
        count=0,
        r2=None,
    )


def solve_degenerate_fit(
    material: Any,
    theoretical: Any,
    diagnostics: Mapping[str, Any],
    *,
    mode: str,
    priors: Mapping[tuple[str, tuple[Any, ...]], HierarchicalBPrior] | None = None,
    cn_line_priors: Mapping[tuple[str, tuple[Any, ...]], CnLinePrior] | None = None,
    default_b: float = DEFAULT_DEGENERATE_B_PRIOR,
    pole_r0_star: float = _DEFAULT_POLE_R0_STAR,
    pole_b_star: float = _DEFAULT_POLE_B_STAR,
) -> tuple[dict[str, Any], ResolvedDegeneratePrior]:
    """Solve a detected degenerate fit with the requested adaptive mode."""
    collapsed_rows = _collapse_rows(_matching_target_bond_rows(material, theoretical))
    if not collapsed_rows:
        raise ValueError("Degenerate fit requested without any target bond equations")

    if mode == "anion-centered":
        from critmin.analysis.anion_valence import try_anion_centered_for_material

        # Load master table for known species (lazy, cached per process)
        known = _get_anion_known_species()
        anion_result = try_anion_centered_for_material(
            material, known_species=known,
        )
        if anion_result is not None:
            prior = HierarchicalBPrior(
                value=float(anion_result["B"]),
                level="anion_centered",
                count=int(anion_result.get("anion_n_sites", 0)),
                std=0.0,
            )
            return anion_result, prior

        # Fall back to fixed-b if anion approach fails
        prior = resolve_hierarchical_b_prior(material, priors or {}, default_b=default_b)
        r0, b_value, r0_std, b_std, rss = _solve_fixed_b(collapsed_rows, b_value=prior.value)
        payload = {
            "R0": r0,
            "B": b_value,
            "R0_std": r0_std,
            "B_std": b_std,
            "rss": rss,
            "anion_centered_failed": True,
            "fit_strategy": "adaptive_fixed_b",
        }
        return payload, prior

    if mode == "fixed-b":
        prior = HierarchicalBPrior(
            value=float(default_b),
            level="fixed_global_default",
            count=0,
            std=0.0,
        )
        r0, b_value, r0_std, b_std, rss = _solve_fixed_b(collapsed_rows, b_value=prior.value)
        payload = {
            "R0": r0,
            "B": b_value,
            "R0_std": r0_std,
            "B_std": b_std,
            "rss": rss,
            "fit_strategy": "adaptive_fixed_b",
        }
        return payload, prior

    if mode == "hierarchical-b":
        prior = resolve_hierarchical_b_prior(material, priors or {}, default_b=default_b)
        r0, b_value, r0_std, b_std, rss = _solve_fixed_b(collapsed_rows, b_value=prior.value)
        payload = {
            "R0": r0,
            "B": b_value,
            "R0_std": r0_std,
            "B_std": b_std,
            "rss": rss,
            "fit_strategy": "adaptive_hierarchical_b",
        }
        return payload, prior

    if mode == "regularized-b":
        prior = resolve_hierarchical_b_prior(material, priors or {}, default_b=default_b)
        r0, b_value, r0_std, b_std, rss, lambda_b = _solve_regularized_b(
            collapsed_rows,
            b_prior=prior.value,
            condition_number=_json_safe_float(diagnostics.get("design_condition_number")),
        )
        payload = {
            "R0": r0,
            "B": b_value,
            "R0_std": r0_std,
            "B_std": b_std,
            "rss": rss,
            "regularization_lambda": lambda_b,
            "fit_strategy": "adaptive_regularized_b",
        }
        return payload, prior

    if mode == "beta-line":
        line_prior = resolve_cn_line_prior(material, cn_line_priors or {})
        if line_prior is None:
            prior = resolve_hierarchical_b_prior(material, priors or {}, default_b=default_b)
            r0, b_value, r0_std, b_std, rss = _solve_fixed_b(collapsed_rows, b_value=prior.value)
            payload = {
                "R0": r0,
                "B": b_value,
                "R0_std": r0_std,
                "B_std": b_std,
                "rss": rss,
                "cn_line_prior_missing": True,
                "fit_strategy": "adaptive_fixed_b",
            }
            return payload, prior

        r0, b_value, r0_std, b_std, rss = _solve_beta_line(
            collapsed_rows, line_prior=line_prior,
        )
        payload = {
            "R0": r0,
            "B": b_value,
            "R0_std": r0_std,
            "B_std": b_std,
            "rss": rss,
            "fit_strategy": "adaptive_beta_line",
        }
        return payload, line_prior

    if mode == "pole-line":
        metadata = dict(getattr(material, "metadata", {}) or {})
        oxi_state = metadata.get("oxi_state")
        cn_val = metadata.get("cn")
        if oxi_state is None or cn_val is None or int(oxi_state) <= 0 or int(cn_val) <= 0 or int(oxi_state) == int(cn_val):
            prior = resolve_hierarchical_b_prior(material, priors or {}, default_b=default_b)
            r0, b_value, r0_std, b_std, rss = _solve_fixed_b(collapsed_rows, b_value=prior.value)
            payload = {
                "R0": r0,
                "B": b_value,
                "R0_std": r0_std,
                "B_std": b_std,
                "rss": rss,
                "pole_prior_missing": True,
                "fit_strategy": "adaptive_fixed_b",
            }
            return payload, prior

        try:
            r0, b_value, r0_std, b_std, rss = _solve_lambda_parameterized(
                collapsed_rows,
                z=int(oxi_state),
                cn=int(cn_val),
                r0_star=pole_r0_star,
                b_star=pole_b_star,
            )
        except (ValueError, ZeroDivisionError):
            prior = resolve_hierarchical_b_prior(material, priors or {}, default_b=default_b)
            r0, b_value, r0_std, b_std, rss = _solve_fixed_b(collapsed_rows, b_value=prior.value)
            payload = {
                "R0": r0,
                "B": b_value,
                "R0_std": r0_std,
                "B_std": b_std,
                "rss": rss,
                "pole_line_failed": True,
                "fit_strategy": "adaptive_fixed_b",
            }
            return payload, prior

        pole_prior = _build_pole_line_prior(
            material, r0_star=pole_r0_star, b_star=pole_b_star,
        )
        lam = _analytical_lambda(r0, int(oxi_state), int(cn_val), pole_r0_star, pole_b_star)
        payload = {
            "R0": r0,
            "B": b_value,
            "R0_std": r0_std,
            "B_std": b_std,
            "rss": rss,
            "screening_lambda": lam,
            "fit_strategy": "adaptive_pole_line",
        }
        return payload, pole_prior or HierarchicalBPrior(
            value=b_value, level="pole_derived", count=0, std=b_std,
        )

    if mode == "cn-line-regularized":
        line_prior = resolve_cn_line_prior(material, cn_line_priors or {})
        if line_prior is None:
            prior = resolve_hierarchical_b_prior(material, priors or {}, default_b=default_b)
            r0, b_value, r0_std, b_std, rss, lambda_b = _solve_regularized_b(
                collapsed_rows,
                b_prior=prior.value,
                condition_number=_json_safe_float(diagnostics.get("design_condition_number")),
            )
            payload = {
                "R0": r0,
                "B": b_value,
                "R0_std": r0_std,
                "B_std": b_std,
                "rss": rss,
                "regularization_lambda": lambda_b,
                "cn_line_prior_missing": True,
                "fit_strategy": "adaptive_regularized_b",
            }
            return payload, prior

        r0, b_value, r0_std, b_std, rss, lambda_line, line_residual = _solve_cn_line_regularized(
            collapsed_rows,
            line_prior=line_prior,
            condition_number=_json_safe_float(diagnostics.get("design_condition_number")),
        )
        payload = {
            "R0": r0,
            "B": b_value,
            "R0_std": r0_std,
            "B_std": b_std,
            "rss": rss,
            "cn_line_regularization_lambda": lambda_line,
            "cn_line_residual": line_residual,
            "fit_strategy": "adaptive_cn_line_regularized",
        }
        return payload, line_prior

    raise ValueError(f"Unknown degenerate fit mode: {mode}")


def _updated_fit_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    requested_mode: str,
    fit_strategy: str,
    degenerate_fit_applied: bool,
    prior: ResolvedDegeneratePrior | None = None,
    rss: float | None = None,
    regularization_lambda: float | None = None,
    cn_line_regularization_lambda: float | None = None,
    cn_line_residual: float | None = None,
    cn_line_prior_missing: bool | None = None,
) -> dict[str, Any]:
    updated = dict(diagnostics)
    updated["degenerate_fit_mode_requested"] = str(requested_mode)
    updated["degenerate_fit_applied"] = bool(degenerate_fit_applied)
    updated["fit_strategy"] = str(fit_strategy)
    if isinstance(prior, HierarchicalBPrior):
        updated["b_prior"] = _json_safe_float(prior.value)
        updated["prior_b_std"] = _json_safe_float(prior.std)
        updated["prior_level"] = str(prior.level)
        updated["prior_count"] = int(prior.count)
    if isinstance(prior, CnLinePrior):
        updated["cn_line_beta"] = _json_safe_float(prior.beta)
        updated["cn_line_beta0"] = _json_safe_float(prior.beta0)
        updated["cn_line_prior_level"] = str(prior.level)
        updated["cn_line_prior_count"] = int(prior.count)
        updated["cn_line_prior_r2"] = _json_safe_float(prior.r2)
    if rss is not None:
        updated["residual_rss"] = _json_safe_float(rss)
    if regularization_lambda is not None:
        updated["regularization_lambda"] = _json_safe_float(regularization_lambda)
    if cn_line_regularization_lambda is not None:
        updated["cn_line_regularization_lambda"] = _json_safe_float(
            cn_line_regularization_lambda
        )
    if cn_line_residual is not None:
        updated["cn_line_residual"] = _json_safe_float(cn_line_residual)
    if cn_line_prior_missing is not None:
        updated["cn_line_prior_missing"] = bool(cn_line_prior_missing)
    return updated


def fit_many_materials_adaptive(
    *,
    sbv: Any,
    materials: Sequence[Any],
    algorithms: Sequence[str],
    r0_bounds: tuple[float, float],
    degenerate_fit_mode: str,
    progress: bool = False,
    desc: str = "Fitting bond valence parameters",
    seed_informative_results: Sequence[Any] | None = None,
    seed_cn_line_priors: Mapping[tuple[str, tuple[Any, ...]], CnLinePrior] | None = None,
    force_adaptive_solver: bool = False,
) -> list[Any]:
    """Fit materials while routing degenerate designs through adaptive solvers.

    *seed_informative_results*, when provided, supplies additional well-determined
    fit results (from a prior run or an existing payload) that are included when
    building hierarchical-B and CN-line priors.  This is essential when the current
    batch consists entirely of degenerate materials and would otherwise produce no
    informative results to derive priors from.

    *seed_cn_line_priors*, when provided, supplies pre-built CN-line priors
    (e.g. from :func:`build_cn_line_priors_from_payload`) that are merged with
    the priors derived from the current batch.  Batch-derived priors take
    precedence when the same key exists in both.

    *force_adaptive_solver*, when True, routes **every** material with a valid
    theoretical solution through :func:`solve_degenerate_fit`, bypassing the
    ``degenerate_detected`` classification.  This is used by the two-phase
    runner (see :func:`critmin.analysis.bond_valence.fit_materials_project_group_two_phase`)
    to re-fit Phase-A high-uncertainty materials with global priors, regardless
    of whether the individual material's design matrix was flagged degenerate.
    When this flag is set, ``seed_cn_line_priors`` should already contain the
    globally-frozen priors; local informative results will be empty (since
    nothing is routed through the legacy optimizer) and therefore will not
    override the seed.
    """
    if degenerate_fit_mode not in VALID_DEGENERATE_FIT_MODES:
        raise ValueError(
            f"degenerate_fit_mode must be one of {VALID_DEGENERATE_FIT_MODES}, "
            f"got {degenerate_fit_mode!r}"
        )

    material_list = list(materials)
    service = sbv.ScreenedBondValenceService(
        algorithms=tuple(algorithms),
        r0_bounds=tuple(r0_bounds),
    )

    prepared: list[tuple[Any, Any, dict[str, Any]]] = []
    iterator = tqdm(material_list, desc=desc) if progress else material_list
    for material in iterator:
        theoretical = service.compute_theoretical(material)
        diagnostics = analyze_material_fit_design(material, theoretical)
        material.metadata["fit_diagnostics"] = dict(diagnostics)
        prepared.append((material, theoretical, diagnostics))

    results: list[Any | None] = [None] * len(prepared)
    informative_results: list[Any] = []
    degenerate_items: list[tuple[int, Any, Any, dict[str, Any]]] = []

    for index, (material, theoretical, diagnostics) in enumerate(prepared):
        if theoretical is None or not theoretical.has_solution:
            material.metadata["fit_strategy"] = "no_network_solution"
            material.metadata["fit_diagnostics"] = _updated_fit_diagnostics(
                diagnostics,
                requested_mode=degenerate_fit_mode,
                fit_strategy="no_network_solution",
                degenerate_fit_applied=False,
            )
            results[index] = AdaptiveMaterialFitResult(
                material=material,
                theoretical=theoretical,
                summary=None,
                failure_reasons=("no_network_solution",),
            )
            continue

        if int(diagnostics.get("n_target_bonds") or 0) == 0:
            material.metadata["fit_strategy"] = "no_target_bonds"
            material.metadata["fit_diagnostics"] = _updated_fit_diagnostics(
                diagnostics,
                requested_mode=degenerate_fit_mode,
                fit_strategy="no_target_bonds",
                degenerate_fit_applied=False,
            )
            results[index] = AdaptiveMaterialFitResult(
                material=material,
                theoretical=theoretical,
                summary=None,
                failure_reasons=("no_target_bonds",),
            )
            continue

        if not force_adaptive_solver and not diagnostics.get("degenerate_detected", True):
            material.metadata["fit_strategy"] = "legacy_optimizer"
            material.metadata["fit_diagnostics"] = _updated_fit_diagnostics(
                diagnostics,
                requested_mode=degenerate_fit_mode,
                fit_strategy="legacy_optimizer",
                degenerate_fit_applied=False,
            )
            result = _fit_with_legacy_algorithms(
                sbv=sbv,
                material=material,
                theoretical=theoretical,
                algorithms=algorithms,
                r0_bounds=r0_bounds,
            )
            informative_results.append(result)
            results[index] = result
            continue

        degenerate_items.append((index, material, theoretical, diagnostics))

    all_informative = list(seed_informative_results or []) + informative_results
    priors = build_hierarchical_b_priors(all_informative)
    cn_line_priors = dict(seed_cn_line_priors or {})
    cn_line_priors.update(build_cn_line_priors(all_informative))

    for index, material, theoretical, diagnostics in degenerate_items:
        solved, prior = solve_degenerate_fit(
            material,
            theoretical,
            diagnostics,
            mode=degenerate_fit_mode,
            priors=priors,
            cn_line_priors=cn_line_priors,
        )
        material.metadata["fit_strategy"] = str(solved["fit_strategy"])
        material.metadata["fit_diagnostics"] = _updated_fit_diagnostics(
            diagnostics,
            requested_mode=degenerate_fit_mode,
            fit_strategy=str(solved["fit_strategy"]),
            degenerate_fit_applied=True,
            prior=prior,
            rss=float(solved["rss"]),
            regularization_lambda=_json_safe_float(solved.get("regularization_lambda")),
            cn_line_regularization_lambda=_json_safe_float(
                solved.get("cn_line_regularization_lambda")
            ),
            cn_line_residual=_json_safe_float(solved.get("cn_line_residual")),
            cn_line_prior_missing=(
                bool(solved["cn_line_prior_missing"])
                if "cn_line_prior_missing" in solved
                else None
            ),
        )
        summary = sbv.MaterialFitSummary(
            material_id=str(material.material_id),
            cation=str(material.cation),
            anion=str(material.anion),
            formula_pretty=getattr(material, "formula_pretty", None),
            r0=float(solved["R0"]),
            b=float(solved["B"]),
            r0_std=float(solved["R0_std"]),
            b_std=float(solved["B_std"]),
            n_algos=1,
            algorithms=[str(solved["fit_strategy"])],
            metadata=material.metadata,
        )
        results[index] = AdaptiveMaterialFitResult(
            material=material,
            theoretical=theoretical,
            summary=summary,
            failure_reasons=(),
        )

    return [result for result in results if result is not None]
