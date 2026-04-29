"""Helpers for generating CriticalMineralProject bond-valence payloads.

This module wraps the ScreenedBondValence (SBV) fitter and adds three layers
of orchestration on top of it:

1. **Linear least-squares fast path** (:func:`fit_material_linear`,
   :func:`_solve_linear_bv`, :class:`_LinearFitResult`,
   :func:`_fit_one_material_with_fast_path`).
   The bond valence equation ``r_ij = R0 - B * ln(s_ij)`` is linear in
   ``(R0, B)`` once the network valence solver fixes ``s_ij`` from the
   structure topology.  For the typical non-degenerate case (rank-2 design
   matrix) this is a closed-form 2-parameter OLS, ~3000× faster per fit than
   the SBV optimizer ensemble (shgo / differential_evolution / dual_annealing
   / direct), and more accurate.  Materials whose target-bond design is
   rank-deficient (e.g. only one distinct ``s_ij``) automatically fall back
   to the SBV ensemble or, when ``degenerate_fit_mode`` is set, to the
   adaptive solver in :mod:`critmin.analysis.adaptive_bond_valence`.

2. **Two-phase orchestration** (:func:`fit_materials_project_group_two_phase`).
   Order-robust two-pass fitting:

   * **Phase A** runs the legacy fitter (linear LS fast path + SBV fallback)
     on every material with no priors.  High-certainty results
     (``R0_std < max_r0_std`` and ``B_std < max_b_std``) are persisted to a
     checkpoint JSON **before** any priors are computed.
   * **Prior build**: a single global table of CN-resolved β-line priors
     ``(β, β₀)`` is built from the frozen on-disk checkpoint via
     :func:`adaptive_bond_valence.build_cn_line_priors_from_payload`.
   * **Phase B** re-fits every Phase-A high-uncertainty material with the
     adaptive degenerate solver under ``degenerate_fit_mode``, seeded with
     the global priors and ``force_adaptive_solver=True`` so that even
     non-degenerate quarantined materials get refit against the prior.

   The result is invariant under shuffling the cation input list, because
   Phase A is priorless and the global prior table is frozen on disk before
   any Phase-B worker runs.

3. **Parallel batch executor with bounded memory growth**
   (:func:`fit_materials_project_group_parallel`).  Distributes per-material
   fits across a ``ProcessPoolExecutor`` and recycles each worker after
   ``max_tasks_per_child`` batches (default 50) to bound the per-worker
   memory growth that pymatgen / scipy accumulate over thousands of fits.
   Streams incremental progress via ``on_cation_complete`` and
   ``on_progress`` callbacks so partial results are visible (and persisted
   to disk) throughout long runs.

The ScreenedBondValence dependency is optional and is loaded dynamically.
Callers can either install the package into the active environment or
point at a local repo checkout via the ``screened_bond_valence_repo=``
argument.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from critmin.analysis.adaptive_bond_valence import (
    VALID_DEGENERATE_FIT_MODES,
    analyze_material_fit_design,
    build_cn_line_priors_from_payload,
    fit_many_materials_adaptive,
)
from critmin.analysis.config import DEFAULT_MAX_B_STD, DEFAULT_MAX_R0_STD, DEFAULT_R0_BOUNDS

_FORMULA_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9.]+)?")
_EXPLICIT_HYDROXYL_RE = re.compile(r"\(OH\)|OH(?![A-Za-z])")
_SPECIES_RE = re.compile(r"^([A-Z][a-z]?)(\d+(?:\.\d+)?)?([+-])$")
_BOND_TYPE_DIGIT_RE = re.compile(r"\d+")
_BUCKETS = ("oxides", "hydroxides")


@dataclass(slots=True)
class BondValenceSystemRun:
    """Artifacts for one fitted cation-anion system."""

    cation: str
    anion: str
    materials: list[Any]
    results: list[Any]
    payload: dict[str, list[dict[str, Any]]]
    failure_mids: dict[str, list[str]] = field(default_factory=dict)


def resolve_screened_bond_valence_repo(
    repo: str | Path | None = None,
) -> Path | None:
    """Return the ScreenedBondValence repo path if one is configured."""
    if repo is None:
        env_value = os.environ.get("SCREENED_BOND_VALENCE_REPO")
        if not env_value:
            return None
        repo = env_value

    repo_path = Path(repo).expanduser().resolve()
    package_dir = repo_path / "screened_bond_valence"
    if not package_dir.exists():
        raise FileNotFoundError(
            f"{repo_path} does not contain a screened_bond_valence package"
        )
    return repo_path


def load_screened_bond_valence(
    repo: str | Path | None = None,
) -> ModuleType:
    """Import ``screened_bond_valence`` with an optional local repo path."""
    repo_path = resolve_screened_bond_valence_repo(repo)
    if repo_path is not None:
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    try:
        return import_module("screened_bond_valence")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "screened_bond_valence is not importable. Install the dependency "
            "repo or set SCREENED_BOND_VALENCE_REPO / --sbv-repo."
        ) from exc


def _fetch_materials_project_inputs_for_material_ids(
    api_key: str,
    *,
    sbv: ModuleType,
    cation: str,
    anion: str,
    material_ids: Sequence[str],
    require_possible_species: bool = True,
) -> list[Any]:
    """Fetch exact Materials Project mids for targeted reruns."""
    try:
        from mp_api.client import MPRester
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mp_api is required to fetch targeted Materials Project materials."
        ) from exc

    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for material_id in material_ids:
        mid = str(material_id)
        if not mid or mid in seen_ids:
            continue
        seen_ids.add(mid)
        ordered_ids.append(mid)

    if not ordered_ids:
        return []

    with MPRester(api_key=api_key) as mpr:
        summary_docs = list(
            mpr.materials.summary.search(
                material_ids=ordered_ids,
                fields=[
                    "material_id",
                    "possible_species",
                    "formula_pretty",
                    "energy_above_hull",
                ],
            )
        )

    species_by_material: dict[str, tuple[str, ...]] = {}
    formula_by_material: dict[str, str | None] = {}
    energy_by_material: dict[str, float | None] = {}
    for doc in summary_docs:
        mid = str(doc.material_id)
        possible_species = tuple(doc.possible_species or ())
        if require_possible_species and not possible_species:
            continue
        species_by_material[mid] = possible_species
        formula_by_material[mid] = getattr(doc, "formula_pretty", None)
        energy_above_hull = getattr(doc, "energy_above_hull", None)
        energy_by_material[mid] = (
            None if energy_above_hull is None else float(energy_above_hull)
        )

    fetch_ids = (
        [mid for mid in ordered_ids if mid in species_by_material]
        if require_possible_species
        else ordered_ids
    )
    if not fetch_ids:
        return []

    with MPRester(api_key=api_key) as mpr:
        bond_docs = list(
            mpr.materials.bonds.search(
                material_ids=fetch_ids,
                fields=["material_id", "structure_graph", "formula_pretty"],
            )
        )

    bonds_by_material = {str(doc.material_id): doc for doc in bond_docs}
    materials: list[Any] = []
    for mid in fetch_ids:
        doc = bonds_by_material.get(mid)
        if doc is None:
            continue
        materials.append(
            sbv.BondValenceMaterial(
                material_id=mid,
                cation=cation,
                anion=anion,
                structure=doc.structure_graph.structure,
                structure_graph=doc.structure_graph,
                possible_species=species_by_material.get(mid, ()),
                formula_pretty=(
                    getattr(doc, "formula_pretty", None) or formula_by_material.get(mid)
                ),
                metadata={
                    "source": "materials_project",
                    "energy_above_hull": energy_by_material.get(mid),
                },
            )
        )

    return materials


def parse_formula_elements(formula: str | None) -> tuple[str, ...]:
    """Return element tokens from a pretty or reduced formula string."""
    if not formula:
        return ()
    return tuple(_FORMULA_ELEMENT_RE.findall(formula))


def parse_formula_tokens(formula: str | None) -> tuple[tuple[str, float], ...]:
    """Return ordered ``(element, count)`` tokens from a formula string."""
    if not formula:
        return ()

    tokens: list[tuple[str, float]] = []
    for elem, count_text in _FORMULA_TOKEN_RE.findall(formula):
        count = float(count_text) if count_text else 1.0
        tokens.append((elem, count))
    return tuple(tokens)


def classify_formula_hydrogen_environment(formula: str | None) -> str:
    """Classify the hydrogen-bearing environment of a formula.

    Returns one of:

    - ``"oxide"`` for H-free formulas
    - ``"hydroxide"`` for formulas that appear to contain hydroxyl groups
    - ``"other_hydrogen"`` for hydrates, ammonium salts, acidic oxyanions,
      hydronium-bearing phases, and other H-bearing formulas that should not be
      merged into the hydroxide bucket
    """
    tokens = parse_formula_tokens(formula)
    if not tokens:
        return "oxide"

    elements = {elem for elem, _count in tokens}
    if "H" not in elements:
        return "oxide"

    if isinstance(formula, str) and _EXPLICIT_HYDROXYL_RE.search(formula):
        return "hydroxide"

    for index, (elem, count) in enumerate(tokens[:-1]):
        next_elem, next_count = tokens[index + 1]
        if elem != "H" or next_elem != "O":
            continue

        # Reduced simple hydroxides such as CaH2O2 / MgHO are usually the
        # only case where multi-H H→O tokens should stay in the hydroxide bin.
        if len(tokens) == 3 and abs(count - next_count) < 1e-8:
            return "hydroxide"

        # Mixed oxyhydroxides often retain a single terminal OH in reduced
        # formulas, e.g. ...HO15 or ...(HO5)2.
        if abs(count - 1.0) < 1e-8:
            return "hydroxide"

    return "other_hydrogen"


def classify_formula_bucket(formula: str | None) -> str:
    """Classify a formula as ``oxides`` or ``hydroxides`` for O-based systems."""
    environment = classify_formula_hydrogen_environment(formula)
    return "hydroxides" if environment == "hydroxide" else "oxides"


def annotate_material_bucket(material: Any) -> None:
    """Store the derived environment bucket metadata on a material object."""
    environment = classify_formula_hydrogen_environment(
        getattr(material, "formula_pretty", None)
    )
    material.metadata.setdefault("hydrogen_environment", environment)
    material.metadata.setdefault(
        "bucket",
        "hydroxides" if environment == "hydroxide" else "oxides",
    )


def resolve_tracker_coordination(
    *,
    cn_mode: Any,
    cn_all: Any,
) -> tuple[int | None, list[int]]:
    """Return ``(cn, cn_all)`` while leaving mixed-CN entries unbinned."""
    normalized_cn_all: list[int] = []
    if isinstance(cn_all, Sequence) and not isinstance(cn_all, (str, bytes)):
        normalized_cn_all = sorted(
            {
                int(value)
                for value in cn_all
                if value is not None
            }
        )

    if len(normalized_cn_all) > 1:
        return None, normalized_cn_all

    if cn_mode is not None:
        resolved_cn = int(cn_mode)
    elif normalized_cn_all:
        resolved_cn = normalized_cn_all[0]
    else:
        resolved_cn = None

    return resolved_cn, normalized_cn_all


def candidate_tracker_metadata(
    tracker_path: str | Path,
    *,
    cation: str,
    statuses: Sequence[str] | None = ("completed",),
) -> dict[str, dict[str, Any]]:
    """Load per-material bucket/CN metadata from an existing candidate tracker."""
    tracker = json.loads(Path(tracker_path).read_text(encoding="utf-8"))
    allowed_statuses = None if statuses is None else set(statuses)

    if "cation" in tracker:
        tracker_cation = tracker.get("cation")
        if tracker_cation and tracker_cation != cation:
            raise ValueError(
                f"Candidate tracker cation {tracker_cation!r} does not match {cation!r}"
            )
        cation_block = tracker
    else:
        try:
            cation_block = tracker[cation]
        except KeyError as exc:
            raise KeyError(f"Cation {cation!r} not present in {tracker_path}") from exc

    metadata: dict[str, dict[str, Any]] = {}
    for bucket in _BUCKETS:
        block = cation_block.get(bucket, {})
        for candidate in block.get("candidates", []):
            status = candidate.get("status")
            if allowed_statuses is not None and status not in allowed_statuses:
                continue

            entry = {
                "bucket": bucket,
                "status": status,
            }
            cn, cn_all = resolve_tracker_coordination(
                cn_mode=candidate.get("cn_mode"),
                cn_all=candidate.get("cn_all"),
            )
            if cn is not None:
                entry["cn"] = cn
            if cn_all:
                entry["cn_all"] = cn_all
            oxi_state = candidate.get("oxi_state")
            if oxi_state is not None:
                entry["oxi_state"] = int(oxi_state)
            oxi_label = candidate.get("oxi_state_label")
            if oxi_label is not None:
                entry["oxi_state_label"] = str(oxi_label)

            metadata[candidate["mid"]] = entry

    return metadata


def _site_symbol(site: Any) -> str | None:
    specie = getattr(site, "specie", None)
    if specie is None:
        return None
    return getattr(specie, "symbol", str(specie))


def coordination_numbers_for_material(material: Any) -> list[int]:
    """Return per-site coordination numbers for the target cation in one material."""
    structure = getattr(material.structure_graph, "structure", material.structure)
    counts: list[int] = []

    for index, site in enumerate(structure):
        if _site_symbol(site) != material.cation:
            continue

        neighbors = material.structure_graph.get_connected_sites(index)
        anion_neighbors = [
            neighbor
            for neighbor in neighbors
            if _site_symbol(getattr(neighbor, "site", None)) == material.anion
        ]
        if anion_neighbors:
            counts.append(len(anion_neighbors))

    return counts


def coordination_mode(coordination_numbers: Sequence[int]) -> int | None:
    """Return the unique coordination number for an unambiguous list."""
    if not coordination_numbers:
        return None

    unique_counts = sorted({int(value) for value in coordination_numbers})
    if len(unique_counts) != 1:
        return None
    return unique_counts[0]


def annotate_material_coordination(material: Any) -> None:
    """Store CN metadata on a ScreenedBondValence material object."""
    counts = coordination_numbers_for_material(material)
    if not counts:
        return

    unique_counts = sorted({int(count) for count in counts})
    cn = coordination_mode(unique_counts)
    if cn is not None:
        material.metadata["cn"] = cn
    else:
        material.metadata.pop("cn", None)
    material.metadata["cn_all"] = unique_counts


def extract_cation_oxidation_states(
    possible_species: Sequence[Any],
    cation: str,
) -> tuple[int | None, str | None]:
    """Classify the oxidation state of *cation* from *possible_species* strings.

    Returns ``(oxi_state, label)`` where:

    - ``(N, "pure")`` if all cation species share oxidation state *N*
    - ``(max, "mixed")`` if multiple oxidation states are found
    - ``(None, None)`` if no matching cation species are present
    """
    oxi_states: set[int] = set()
    for species in possible_species:
        match = _SPECIES_RE.match(str(species).strip())
        if match is None:
            continue
        element, charge_str, sign = match.groups()
        if element != cation:
            continue
        charge = int(float(charge_str)) if charge_str else 1
        if sign == "-":
            charge = -charge
        oxi_states.add(charge)

    if not oxi_states:
        return None, None
    if len(oxi_states) == 1:
        return next(iter(oxi_states)), "pure"
    return max(oxi_states), "mixed"


def annotate_material_oxidation_state(material: Any) -> None:
    """Store oxidation-state metadata on a ScreenedBondValence material object."""
    possible_species = getattr(material, "possible_species", ())
    oxi_state, label = extract_cation_oxidation_states(
        possible_species, material.cation,
    )
    if oxi_state is not None:
        material.metadata["oxi_state"] = oxi_state
    if label is not None:
        material.metadata["oxi_state_label"] = label


def classify_summary_bucket(summary: Any) -> str:
    """Return the downstream bucket for one aggregated material summary."""
    metadata = dict(getattr(summary, "metadata", {}) or {})
    bucket = metadata.get("bucket")
    if bucket in _BUCKETS:
        return str(bucket)
    return classify_formula_bucket(getattr(summary, "formula_pretty", None))


def summary_to_critmin_dict(
    summary: Any,
    *,
    include_coordination: bool = True,
    r0_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Serialize a ScreenedBondValence summary to the compact CritMin schema."""
    payload = summary.to_compact_dict()

    metadata = dict(getattr(summary, "metadata", {}) or {})

    if include_coordination:
        cn, cn_all = resolve_tracker_coordination(
            cn_mode=metadata.get("cn"),
            cn_all=metadata.get("cn_all"),
        )
        if cn is not None:
            payload["cn"] = cn
        else:
            payload.pop("cn", None)
        if cn_all:
            payload["cn_all"] = cn_all
        else:
            payload.pop("cn_all", None)

    oxi_state = metadata.get("oxi_state")
    if oxi_state is not None:
        payload["oxi_state"] = int(oxi_state)
    oxi_label = metadata.get("oxi_state_label")
    if oxi_label is not None:
        payload["oxi_state_label"] = str(oxi_label)
    fit_strategy = metadata.get("fit_strategy")
    if fit_strategy is not None:
        payload["fit_strategy"] = str(fit_strategy)
    fit_diagnostics = metadata.get("fit_diagnostics")
    serialized_fit_diagnostics: dict[str, Any] | None = None
    if isinstance(fit_diagnostics, Mapping):
        serialized_fit_diagnostics = json.loads(json.dumps(dict(fit_diagnostics)))
    energy_above_hull = _as_float(metadata.get("energy_above_hull"))
    if energy_above_hull is not None:
        payload["energy_above_hull"] = energy_above_hull
    if r0_bounds is not None and tuple(r0_bounds) != DEFAULT_R0_BOUNDS:
        if serialized_fit_diagnostics is None:
            serialized_fit_diagnostics = {}
        serialized_fit_diagnostics["r0_bounds"] = [
            float(r0_bounds[0]),
            float(r0_bounds[1]),
        ]
    if serialized_fit_diagnostics:
        payload["fit_diagnostics"] = serialized_fit_diagnostics

    return payload


def empty_bucket_payload() -> dict[str, list[dict[str, Any]]]:
    """Return an empty two-bucket payload."""
    return {bucket: [] for bucket in _BUCKETS}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_high_uncertainty_record(
    record: Mapping[str, Any],
    *,
    max_r0_std: float = DEFAULT_MAX_R0_STD,
    max_b_std: float = DEFAULT_MAX_B_STD,
) -> bool:
    """Return whether a compact fit record exceeds the accepted uncertainty."""
    r0_std = _as_float(record.get("R0_std"))
    b_std = _as_float(record.get("B_std"))
    return (
        (r0_std is not None and r0_std >= max_r0_std)
        or (b_std is not None and b_std >= max_b_std)
    )


def separate_cation_payload_by_uncertainty(
    cation_payload: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    *,
    max_r0_std: float = DEFAULT_MAX_R0_STD,
    max_b_std: float = DEFAULT_MAX_B_STD,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Split one cation payload into accepted and high-uncertainty records."""
    accepted = empty_bucket_payload()
    high_uncertainty = empty_bucket_payload()
    if not cation_payload:
        return accepted, high_uncertainty

    for bucket in _BUCKETS:
        for record in cation_payload.get(bucket, []):
            copied = dict(record)
            current_status = copied.get("status")
            target_is_uncertain = is_high_uncertainty_record(
                record,
                max_r0_std=max_r0_std,
                max_b_std=max_b_std,
            )
            if target_is_uncertain:
                if current_status not in (None, "high_uncertainty"):
                    copied.setdefault("original_status", str(current_status))
                copied["status"] = "high_uncertainty"
                high_uncertainty[bucket].append(copied)
                continue

            if current_status == "high_uncertainty":
                copied.setdefault("original_status", "high_uncertainty")
            if current_status != "boundary_capped":
                copied["status"] = "fitted"
            accepted[bucket].append(copied)

    return _sorted_payload(accepted), _sorted_payload(high_uncertainty)


def separate_grouped_payload_by_uncertainty(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    max_r0_std: float = DEFAULT_MAX_R0_STD,
    max_b_std: float = DEFAULT_MAX_B_STD,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, list[dict[str, Any]]]],
]:
    """Split a grouped payload into accepted and high-uncertainty records."""
    accepted_grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    high_uncertainty_grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for cation, cation_payload in payload.items():
        accepted, high_uncertainty = separate_cation_payload_by_uncertainty(
            cation_payload,
            max_r0_std=max_r0_std,
            max_b_std=max_b_std,
        )
        accepted_grouped[str(cation)] = accepted
        if any(high_uncertainty[bucket] for bucket in _BUCKETS):
            high_uncertainty_grouped[str(cation)] = high_uncertainty

    return accepted_grouped, high_uncertainty_grouped


def _sorted_payload(
    payload: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    ordered: dict[str, list[dict[str, Any]]] = {}
    for bucket in _BUCKETS:
        records = [dict(record) for record in payload.get(bucket, [])]
        ordered[bucket] = sorted(records, key=lambda record: str(record.get("mid", "")))
    return ordered


def _failure_counts(results: Sequence[Any]) -> Counter[str]:
    failures: Counter[str] = Counter()
    for result in results:
        for reason in getattr(result, "failure_reasons", ()):
            failures[str(reason)] += 1
    return failures


def _append_failure_mid(
    failure_mids: dict[str, list[str]],
    reason: str,
    material_id: Any,
) -> None:
    reason_label = str(reason).strip()
    material_label = str(material_id).strip() if material_id is not None else ""
    if not reason_label or not material_label:
        return
    failure_mids.setdefault(reason_label, []).append(material_label)


def _sorted_failure_mids(
    failure_mids: Mapping[str, Sequence[Any]] | None,
) -> dict[str, list[str]]:
    if not failure_mids:
        return {}
    ordered: dict[str, list[str]] = {}
    for reason in sorted(str(key) for key in failure_mids.keys()):
        mids = sorted(
            {
                str(mid).strip()
                for mid in failure_mids.get(reason, ())
                if str(mid).strip()
            }
        )
        if mids:
            ordered[reason] = mids
    return ordered


def _merge_failure_mids(
    *sources: Mapping[str, Sequence[Any]] | None,
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for source in sources:
        if not source:
            continue
        for reason, mids in source.items():
            for mid in mids:
                _append_failure_mid(merged, str(reason), mid)
    return _sorted_failure_mids(merged)


def _result_material_id(result: Any) -> str:
    material = getattr(result, "material", None)
    material_id = getattr(material, "material_id", None)
    if material_id is not None:
        return str(material_id)

    summary = getattr(result, "summary", None)
    material_id = getattr(summary, "material_id", None)
    if material_id is not None:
        return str(material_id)

    material_id = getattr(result, "material_id", None)
    return str(material_id) if material_id is not None else ""


def _result_has_summary(result: Any) -> bool:
    if hasattr(result, "summary"):
        return getattr(result, "summary") is not None
    if hasattr(result, "aggregate"):
        try:
            return result.aggregate() is not None
        except Exception:
            return False
    return False


def _failure_mids_from_results(results: Sequence[Any]) -> dict[str, list[str]]:
    failure_mids: dict[str, list[str]] = {}
    for result in results:
        material_id = _result_material_id(result)
        reasons = tuple(str(reason) for reason in getattr(result, "failure_reasons", ()))
        if reasons:
            for reason in reasons:
                _append_failure_mid(failure_mids, reason, material_id)
            continue
        if not _result_has_summary(result):
            _append_failure_mid(failure_mids, "no_summary", material_id)
    return _sorted_failure_mids(failure_mids)


def _failure_details(
    results: Sequence[Any],
    extra_failure_mids: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[dict[str, int], dict[str, list[str]]]:
    failure_mids = _merge_failure_mids(_failure_mids_from_results(results), extra_failure_mids)
    failure_counts = Counter(_failure_counts(results))
    if extra_failure_mids:
        for reason, mids in extra_failure_mids.items():
            failure_counts[str(reason)] += sum(
                1 for mid in mids if str(mid).strip()
            )
    for reason, mids in failure_mids.items():
        failure_counts[str(reason)] = max(failure_counts.get(str(reason), 0), len(mids))
    return dict(failure_counts), failure_mids


def _parallel_failure_label(exc: Exception) -> str:
    return f"exception:{type(exc).__name__}"


# ─────────────────────────── linear least squares fast path ──────────────────
#
# The bond valence equation r_ij = R_0 - B * ln(s_ij) is *linear* in (R_0, B)
# once s_ij has been computed by the network valence solver.  ScreenedBondValence
# treats it as a generic nonlinear minimization problem and runs a global
# optimizer ensemble (shgo / differential_evolution / dual_annealing / direct /
# brute) over a sympy-evaluated objective.  For the non-degenerate case
# (rank-2 design matrix) this is enormously expensive: typical per-fit cost is
# ~2 minutes vs ~50 microseconds for the closed-form OLS solution to the same
# problem.  This helper provides the fast path; the legacy ensemble is kept as
# a fallback for the rank-1 case (where it is wrapped by the adaptive solver
# anyway) and as a safety net.


def _bond_type_elements(bond_type: str) -> tuple[str, str] | None:
    """Split a bond type string like ``"Li1O2"`` into ``("Li", "O")``.

    Mirrors the SBV ``BVParamSolver.get_eqs_for_R0B`` filter logic.
    """
    parts = _BOND_TYPE_DIGIT_RE.split(str(bond_type))
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _select_target_bonds(
    bond_types: Sequence[str],
    *,
    cation: str,
    anion: str,
) -> list[str]:
    """Filter ``bond_types`` to those matching the cation/anion pair."""
    targets: list[str] = []
    for bond_type in bond_types:
        elements = _bond_type_elements(bond_type)
        if elements is None:
            continue
        if elements[0] == str(cation) and elements[1] == str(anion):
            targets.append(str(bond_type))
    return targets


def _linear_bv_design(
    bond_valences: Mapping[str, float],
    bond_lengths: Mapping[str, float],
    target_bonds: Sequence[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the (x, y) arrays for the linear regression ``y = R0 + B * x``.

    ``x_i = -ln(s_ij)`` and ``y_i = r_ij`` for each target bond.  Returns
    ``None`` if any target bond has a non-positive ``s_ij`` (matching the SBV
    "negative_Sij" failure mode) or if no target bonds are present.
    """
    xs: list[float] = []
    ys: list[float] = []
    for bond in target_bonds:
        sij = bond_valences.get(bond)
        length = bond_lengths.get(bond)
        if sij is None or length is None:
            continue
        sij_f = float(sij)
        length_f = float(length)
        if sij_f <= 0.0:
            return None  # matches SBV "negative_Sij" failure
        if not (math.isfinite(sij_f) and math.isfinite(length_f)):
            continue
        xs.append(-math.log(sij_f))
        ys.append(length_f)
    if not xs:
        return None
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _solve_linear_bv(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    r0_bounds: tuple[float, float],
) -> tuple[float, float, float, float, float] | None:
    """Closed-form 2-parameter OLS for ``y = R0 + B * x`` with R0 bound projection.

    Returns ``(R0, B, R0_std, B_std, residual_rms)`` or ``None`` if
    rank-deficient (all x identical).

    **Important note on stderr semantics**: ``R0_std`` and ``B_std`` are
    reported as **0.0**, matching the SBV convention where the ``_std`` fields
    represent disagreement between independent optimizer algorithms (typically
    ~1e-6 when the ensemble converges).  The unique linear LS solution is the
    common minimum that all four SBV optimizers would converge to, so the
    "algorithm disagreement" stderr is exactly zero.

    The statistical regression standard error (which is what ``_std`` would
    *technically* mean for an OLS fit) is much larger and would clobber the
    user's ``--max-r0-std 0.1`` threshold if reported here.  Users who want
    that statistical uncertainty should look at ``residual_rms``, the
    root-mean-square residual of the linear fit, which is plumbed through to
    ``fit_diagnostics["linear_residual_rms"]`` by ``fit_material_linear``.
    """
    n = int(xs.size)
    if n < 1:
        return None
    # Robust rank-deficiency check using ptp (peak-to-peak) of xs against the
    # x-scale.  Pure Sxx<=0 checks miss the floating-point case where all xs
    # are nominally identical but the computed variance is a tiny positive
    # number.
    x_scale = max(float(np.abs(xs).max()), 1.0)
    if float(np.ptp(xs)) <= 1e-12 * x_scale:
        return None  # all x identical → rank-deficient
    xbar = float(xs.mean())
    ybar = float(ys.mean())
    centered_x = xs - xbar
    Sxx = float((centered_x * centered_x).sum())
    if Sxx <= 0.0:
        return None
    Sxy = float((centered_x * (ys - ybar)).sum())
    B = Sxy / Sxx
    R0 = ybar - B * xbar

    r0_lo, r0_hi = float(r0_bounds[0]), float(r0_bounds[1])
    if R0 < r0_lo or R0 > r0_hi:
        # Project R0 to the nearest bound and re-fit B with R0 fixed.
        # The 1D least-squares problem  min sum((y_i - R0 - B*x_i)^2)
        # has B = sum(x_i * (y_i - R0)) / sum(x_i^2).
        R0 = max(r0_lo, min(r0_hi, R0))
        denom = float((xs * xs).sum())
        if denom <= 0.0:
            return None
        numer = float((xs * (ys - R0)).sum())
        B = numer / denom

    # Residual RMS reflects how well the linear model fits the data; this is
    # surfaced via fit_diagnostics for downstream inspection but is NOT used
    # for the high-uncertainty classifier (see docstring).
    residuals = ys - (R0 + B * xs)
    rss = float((residuals * residuals).sum())
    residual_rms = math.sqrt(rss / max(n, 1))

    # SBV-compatible stderr: 0.0 because the unique LS solution has no
    # algorithm disagreement.  See docstring for the rationale.
    return float(R0), float(B), 0.0, 0.0, float(residual_rms)


def fit_material_linear(
    sbv: ModuleType,
    material: Any,
    theoretical: Any,
    *,
    r0_bounds: tuple[float, float],
) -> Any | None:
    """Closed-form linear fit for non-degenerate materials.

    Returns a ``sbv.MaterialFitSummary`` on success, or ``None`` if the
    material is rank-deficient (single distinct ``s_ij``) or has no usable
    target bonds.  Callers should fall back to the legacy SBV ensemble or the
    adaptive solver when this returns ``None``.

    The ``material`` is modified in place to set ``metadata["fit_strategy"] =
    "linear_ls"``, mirroring the convention used by the adaptive solver.
    """
    if theoretical is None or not getattr(theoretical, "has_solution", False):
        return None

    cation = str(getattr(material, "cation", ""))
    anion = str(getattr(material, "anion", ""))
    if not cation or not anion:
        return None

    target_bonds = _select_target_bonds(
        getattr(theoretical, "bond_types", ()) or (),
        cation=cation,
        anion=anion,
    )
    if not target_bonds:
        return None

    design = _linear_bv_design(
        getattr(theoretical, "bond_valences", {}) or {},
        getattr(theoretical, "bond_lengths", {}) or {},
        target_bonds,
    )
    if design is None:
        return None
    xs, ys = design

    solved = _solve_linear_bv(xs, ys, r0_bounds=tuple(r0_bounds))
    if solved is None:
        return None
    R0, B, R0_std, B_std, residual_rms = solved

    if not (math.isfinite(R0) and math.isfinite(B)):
        return None

    material.metadata["fit_strategy"] = "linear_ls"
    # Plumb the residual RMS through fit_diagnostics so downstream callers
    # (e.g. high_uncertainty.py) can inspect linear-fit quality even though
    # R0_std/B_std are intentionally zeroed for SBV-compatibility.
    fit_diagnostics = dict(material.metadata.get("fit_diagnostics") or {})
    fit_diagnostics["linear_residual_rms"] = float(residual_rms)
    fit_diagnostics["linear_n_bonds"] = int(xs.size)
    material.metadata["fit_diagnostics"] = fit_diagnostics

    return sbv.MaterialFitSummary(
        material_id=str(material.material_id),
        cation=cation,
        anion=anion,
        formula_pretty=getattr(material, "formula_pretty", None),
        r0=float(R0),
        b=float(B),
        r0_std=float(R0_std),
        b_std=float(B_std),
        n_algos=int(xs.size),
        algorithms=["linear_ls"],
        metadata=dict(material.metadata),
    )


@dataclass(slots=True)
class _LinearFitResult:
    """Adapter that mimics sbv.MaterialFitResult for the linear LS fast path.

    Cannot subclass or patch ``sbv.MaterialFitResult`` because it is declared
    with ``slots=True`` and refuses instance attribute assignment.  This
    adapter exposes the same public interface that ``build_summary_payload``
    relies on: ``failure_reasons`` (read by ``_failure_counts``) and an
    ``aggregate(reducer)`` method that returns a precomputed
    ``MaterialFitSummary``.
    """

    material: Any
    theoretical: Any
    summary: Any
    failure_reasons: tuple = ()

    def aggregate(self, reducer: Any = None) -> Any:
        return self.summary


def _try_anion_fallback(
    sbv: ModuleType,
    material: Any,
    *,
    r0_bounds: tuple[float, float],
) -> Any | None:
    """Attempt anion-centered fitting for one degenerate material.

    Returns a ``MaterialFitSummary`` on success, or ``None`` if the anion
    approach doesn't apply or fails quality checks.
    """
    try:
        from critmin.analysis.anion_valence import try_anion_centered_for_material
        from critmin.analysis.adaptive_bond_valence import _get_anion_known_species
    except ImportError:
        return None

    known = _get_anion_known_species()
    if not known:
        return None

    result = try_anion_centered_for_material(
        material,
        known_species=known,
        r0_bounds=r0_bounds,
    )
    if result is None:
        return None

    material.metadata["fit_strategy"] = "anion_centered"
    material.metadata.setdefault("fit_diagnostics", {})
    material.metadata["fit_diagnostics"]["anion_n_sites"] = result.get("anion_n_sites", 0)
    material.metadata["fit_diagnostics"]["anion_rms"] = result.get("anion_rms", 0.0)
    material.metadata["fit_diagnostics"]["anion_jac_rank"] = result.get("anion_jac_rank", 0)

    return sbv.MaterialFitSummary(
        material_id=str(material.material_id),
        cation=str(material.cation),
        anion=str(material.anion),
        formula_pretty=getattr(material, "formula_pretty", None),
        r0=float(result["R0"]),
        b=float(result["B"]),
        r0_std=0.0,
        b_std=0.0,
        n_algos=1,
        algorithms=["anion_centered"],
        metadata=material.metadata,
    )


def _fit_one_material_with_fast_path(
    sbv: ModuleType,
    service: Any,
    material: Any,
    *,
    r0_bounds: tuple[float, float],
) -> tuple[Any, str]:
    """Fit one material through the three-level cascade.

    Returns ``(result, strategy)`` where ``strategy`` is one of
    ``"linear_ls"``, ``"anion_centered"``, or ``"legacy_optimizer"``.

    Cascade::

        1. Linear LS       — closed-form OLS on rank-2 design matrix
              ↓ degenerate or linear LS failed
        2. Anion-centered  — O²⁻ valence sums (de novo → partial → target-only)
              ↓ fails or no structure graph
        3. Legacy SBV      — brute-force optimizer ensemble
    """
    try:
        theoretical = service.compute_theoretical(material)
    except Exception:
        # Theoretical solver failed entirely; let the legacy path produce a
        # proper failure record so its reason gets surfaced.
        results = service.fit_many([material], progress=False)
        return (results[0] if results else None), "legacy_optimizer"

    if theoretical is not None and getattr(theoretical, "has_solution", False):
        try:
            diagnostics = analyze_material_fit_design(material, theoretical)
        except Exception:
            diagnostics = {"degenerate_detected": True}
        material.metadata["fit_diagnostics"] = dict(diagnostics)

        if not diagnostics.get("degenerate_detected", True):
            summary = fit_material_linear(
                sbv,
                material,
                theoretical,
                r0_bounds=r0_bounds,
            )
            if summary is not None:
                return _LinearFitResult(
                    material=material,
                    theoretical=theoretical,
                    summary=summary,
                ), "linear_ls"

        # Degenerate or linear LS failed — try anion-centered before legacy.
        anion_summary = _try_anion_fallback(sbv, material, r0_bounds=r0_bounds)
        if anion_summary is not None:
            return _LinearFitResult(
                material=material,
                theoretical=theoretical,
                summary=anion_summary,
            ), "anion_centered"

    results = service.fit_many([material], progress=False)
    return (results[0] if results else None), "legacy_optimizer"


def _fit_materials_project_run(
    *,
    cation: str,
    anion: str,
    materials: Sequence[Any],
    algorithms: Sequence[str] | None,
    r0_bounds: tuple[float, float],
    screened_bond_valence_repo: str | Path | None,
    include_coordination: bool,
    degenerate_fit_mode: str | None,
    progress: bool,
    seed_cn_line_priors: Mapping | None = None,
    force_adaptive_solver: bool = False,
) -> BondValenceSystemRun:
    """Fit an already-prepared material list and build its compact payload."""
    sbv = load_screened_bond_valence(screened_bond_valence_repo)
    try:
        resolved_algorithms = tuple(algorithms or getattr(sbv, "DEFAULT_ALGORITHMS"))
        service_cls = sbv.ScreenedBondValenceService
        build_summary_payload = sbv.build_summary_payload
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ScreenedBondValence is available but its runtime dependencies are "
            "not installed. Run `pip install -e /path/to/ScreenedBondValence` "
            "in this environment first."
        ) from exc

    if degenerate_fit_mode is not None and degenerate_fit_mode not in VALID_DEGENERATE_FIT_MODES:
        raise ValueError(
            f"degenerate_fit_mode must be one of {VALID_DEGENERATE_FIT_MODES}, "
            f"got {degenerate_fit_mode!r}"
        )

    material_list = list(materials)
    explicit_failure_mids: dict[str, list[str]] = {}
    if degenerate_fit_mode is None:
        if force_adaptive_solver:
            raise ValueError(
                "force_adaptive_solver=True requires degenerate_fit_mode to be set"
            )
        service = service_cls(algorithms=resolved_algorithms, r0_bounds=tuple(r0_bounds))
        # Per-material loop: try the linear LS fast path; fall back to the
        # SBV ensemble for rank-deficient or otherwise ineligible materials.
        results = []
        iterator = material_list
        if progress:
            try:
                from tqdm import tqdm as _tqdm
                iterator = _tqdm(material_list, desc=f"Fitting {cation}-{anion} materials")
            except ModuleNotFoundError:
                pass
        for material in iterator:
            try:
                fit_result, _strategy = _fit_one_material_with_fast_path(
                    sbv,
                    service,
                    material,
                    r0_bounds=tuple(r0_bounds),
                )
            except Exception as exc:
                _append_failure_mid(
                    explicit_failure_mids,
                    _parallel_failure_label(exc),
                    getattr(material, "material_id", None),
                )
                continue
            if fit_result is None:
                _append_failure_mid(
                    explicit_failure_mids,
                    "no_summary",
                    getattr(material, "material_id", None),
                )
                continue
            results.append(fit_result)
    else:
        results = fit_many_materials_adaptive(
            sbv=sbv,
            materials=material_list,
            algorithms=resolved_algorithms,
            r0_bounds=tuple(r0_bounds),
            degenerate_fit_mode=degenerate_fit_mode,
            progress=progress,
            desc=f"Fitting {cation}-{anion} materials",
            seed_cn_line_priors=seed_cn_line_priors,
            force_adaptive_solver=force_adaptive_solver,
        )

    payload = build_summary_payload(
        results,
        classifier=classify_summary_bucket,
        serializer=lambda summary: summary_to_critmin_dict(
            summary,
            include_coordination=include_coordination,
            r0_bounds=tuple(r0_bounds),
        ),
    )
    failure_mids = _merge_failure_mids(
        _failure_mids_from_results(results),
        explicit_failure_mids,
    )

    return BondValenceSystemRun(
        cation=cation,
        anion=anion,
        materials=material_list,
        results=results,
        payload=_sorted_payload(payload),
        failure_mids=failure_mids,
    )


def fit_materials_project_system(
    *,
    api_key: str,
    cation: str,
    anion: str = "O",
    algorithms: Sequence[str] | None = None,
    energy_above_hull: tuple[float, float] = (0.0, 0.05),
    r0_bounds: tuple[float, float] = DEFAULT_R0_BOUNDS,
    require_possible_species: bool = True,
    screened_bond_valence_repo: str | Path | None = None,
    include_coordination: bool = True,
    candidate_tracker: str | Path | None = None,
    tracker_statuses: Sequence[str] | None = ("completed",),
    material_ids: Sequence[str] | None = None,
    degenerate_fit_mode: str | None = None,
    progress: bool = True,
    seed_cn_line_priors: Mapping | None = None,
    force_adaptive_solver: bool = False,
) -> BondValenceSystemRun:
    """Fetch MP inputs, fit one cation-anion system, and build a compact payload."""
    materials = prepare_cation_materials(
        api_key=api_key,
        cation=cation,
        anion=anion,
        energy_above_hull=energy_above_hull,
        material_ids=material_ids,
        require_possible_species=require_possible_species,
        screened_bond_valence_repo=screened_bond_valence_repo,
        include_coordination=include_coordination,
        candidate_tracker=candidate_tracker,
        tracker_statuses=tracker_statuses,
    )
    return _fit_materials_project_run(
        cation=cation,
        anion=anion,
        materials=materials,
        algorithms=algorithms,
        r0_bounds=tuple(r0_bounds),
        screened_bond_valence_repo=screened_bond_valence_repo,
        include_coordination=include_coordination,
        degenerate_fit_mode=degenerate_fit_mode,
        progress=progress,
        seed_cn_line_priors=seed_cn_line_priors,
        force_adaptive_solver=force_adaptive_solver,
    )


def prepare_cation_materials(
    *,
    api_key: str,
    cation: str,
    anion: str = "O",
    energy_above_hull: tuple[float, float] = (0.0, 0.05),
    material_ids: Sequence[str] | None = None,
    require_possible_species: bool = True,
    screened_bond_valence_repo: str | Path | None = None,
    include_coordination: bool = True,
    candidate_tracker: str | Path | None = None,
    tracker_statuses: Sequence[str] | None = ("completed",),
) -> list[Any]:
    """Fetch, filter, and CN-annotate materials for one cation without fitting.

    This is the first half of :func:`fit_materials_project_system`, useful when
    you want to distribute individual materials across a worker pool rather than
    fitting all materials for a cation in one shot.
    """
    sbv = load_screened_bond_valence(screened_bond_valence_repo)
    tracker_metadata: dict[str, dict[str, Any]] = {}
    requested_mids = {
        str(material_id)
        for material_id in (material_ids or ())
        if str(material_id)
    } or None
    if candidate_tracker is not None:
        tracker_metadata = candidate_tracker_metadata(
            candidate_tracker,
            cation=cation,
            statuses=tracker_statuses,
        )
        tracker_mids = set(tracker_metadata)
        requested_mids = tracker_mids if requested_mids is None else requested_mids & tracker_mids

    if requested_mids is not None:
        if not requested_mids:
            return []
        materials = _fetch_materials_project_inputs_for_material_ids(
            api_key,
            sbv=sbv,
            cation=cation,
            anion=anion,
            material_ids=sorted(requested_mids),
            require_possible_species=require_possible_species,
        )
    else:
        materials = sbv.fetch_materials_project_inputs(
            api_key,
            cation=cation,
            anion=anion,
            energy_above_hull=energy_above_hull,
            require_possible_species=require_possible_species,
        )

    if tracker_metadata:
        materials = [m for m in materials if m.material_id in tracker_metadata]

    for material in materials:
        if material.material_id in tracker_metadata:
            material.metadata.update(tracker_metadata[material.material_id])
        if "bucket" not in material.metadata:
            annotate_material_bucket(material)
        if include_coordination and "cn" not in material.metadata:
            annotate_material_coordination(material)
        if "oxi_state" not in material.metadata:
            annotate_material_oxidation_state(material)

    return materials


def fit_materials_project_group(
    *,
    api_key: str,
    cations: Sequence[str],
    anion: str = "O",
    algorithms: Sequence[str] | None = None,
    energy_above_hull: tuple[float, float] = (0.0, 0.05),
    r0_bounds: tuple[float, float] = DEFAULT_R0_BOUNDS,
    require_possible_species: bool = True,
    screened_bond_valence_repo: str | Path | None = None,
    include_coordination: bool = True,
    candidate_tracker: str | Path | None = None,
    tracker_statuses: Sequence[str] | None = ("completed",),
    degenerate_fit_mode: str | None = None,
    progress: bool = True,
    on_cation_complete: Callable[
        [str, dict[str, list[dict[str, Any]]], dict[str, Any]], None
    ] | None = None,
    seed_cn_line_priors: Mapping | None = None,
    force_adaptive_solver: bool = False,
    exclude_mids: set[str] | None = None,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[BondValenceSystemRun]]:
    """Fit multiple cations and return a nested payload plus run metadata.

    Parameters
    ----------
    on_cation_complete : callable, optional
        Called after each cation finishes with ``(cation, cation_payload, stats)``.
    """
    payload: dict[str, dict[str, list[dict[str, Any]]]] = {}
    runs: list[BondValenceSystemRun] = []

    for cation in cations:
        if exclude_mids:
            materials = prepare_cation_materials(
                api_key=api_key,
                cation=cation,
                anion=anion,
                energy_above_hull=energy_above_hull,
                require_possible_species=require_possible_species,
                screened_bond_valence_repo=screened_bond_valence_repo,
                include_coordination=include_coordination,
                candidate_tracker=candidate_tracker,
                tracker_statuses=tracker_statuses,
            )
            materials = [m for m in materials if m.material_id not in exclude_mids]
            if not materials:
                run = BondValenceSystemRun(
                    cation=cation,
                    anion=anion,
                    materials=[],
                    results=[],
                    payload=empty_bucket_payload(),
                )
            else:
                run = _fit_materials_project_run(
                    cation=cation,
                    anion=anion,
                    materials=materials,
                    algorithms=algorithms,
                    r0_bounds=tuple(r0_bounds),
                    screened_bond_valence_repo=screened_bond_valence_repo,
                    include_coordination=include_coordination,
                    degenerate_fit_mode=degenerate_fit_mode,
                    progress=progress,
                    seed_cn_line_priors=seed_cn_line_priors,
                    force_adaptive_solver=force_adaptive_solver,
                )
        else:
            run = fit_materials_project_system(
                api_key=api_key,
                cation=cation,
                anion=anion,
                algorithms=algorithms,
                energy_above_hull=energy_above_hull,
                r0_bounds=r0_bounds,
                require_possible_species=require_possible_species,
                screened_bond_valence_repo=screened_bond_valence_repo,
                include_coordination=include_coordination,
                candidate_tracker=candidate_tracker,
                tracker_statuses=tracker_statuses,
                degenerate_fit_mode=degenerate_fit_mode,
                progress=progress,
                seed_cn_line_priors=seed_cn_line_priors,
                force_adaptive_solver=force_adaptive_solver,
            )
        payload[cation] = run.payload
        runs.append(run)
        if on_cation_complete is not None:
            on_cation_complete(cation, run.payload, run_stats(run))

    return payload, runs


def _parallel_system_worker(config: dict[str, Any]) -> tuple[str, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Fit one cation system in a worker process and return serializable outputs."""
    run = fit_materials_project_system(
        api_key=config["api_key"],
        cation=config["cation"],
        anion=config["anion"],
        algorithms=config.get("algorithms"),
        energy_above_hull=tuple(config["energy_above_hull"]),
        r0_bounds=tuple(config.get("r0_bounds", DEFAULT_R0_BOUNDS)),
        require_possible_species=bool(config["require_possible_species"]),
        screened_bond_valence_repo=config.get("screened_bond_valence_repo"),
        include_coordination=bool(config["include_coordination"]),
        candidate_tracker=config.get("candidate_tracker"),
        tracker_statuses=config.get("tracker_statuses"),
        degenerate_fit_mode=config.get("degenerate_fit_mode"),
        progress=False,
    )
    return run.cation, run.payload, run_stats(run)


def _parallel_adaptive_system_worker(
    config: dict[str, Any],
) -> tuple[str, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Fit one prepared cation pool with the adaptive degenerate solver."""
    run = _fit_materials_project_run(
        cation=config["cation"],
        anion=config["anion"],
        materials=config["materials"],
        algorithms=config.get("algorithms"),
        r0_bounds=tuple(config.get("r0_bounds", DEFAULT_R0_BOUNDS)),
        screened_bond_valence_repo=config.get("screened_bond_valence_repo"),
        include_coordination=bool(config["include_coordination"]),
        degenerate_fit_mode=config.get("degenerate_fit_mode"),
        progress=False,
        seed_cn_line_priors=config.get("seed_cn_line_priors"),
        force_adaptive_solver=bool(config.get("force_adaptive_solver", False)),
    )
    return run.cation, run.payload, run_stats(run)


def _fit_material_batch_worker(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit a batch of (cation, material) pairs in a single worker process.

    Each item is fitted independently using a shared
    ``ScreenedBondValenceService``.  Returns a list of
    ``(cation, bucket, record_dict)`` tuples plus per-cation failure counts.
    """
    sbv = load_screened_bond_valence(config.get("screened_bond_valence_repo"))
    algorithms = tuple(config.get("algorithms") or getattr(sbv, "DEFAULT_ALGORITHMS"))
    service = sbv.ScreenedBondValenceService(
        algorithms=algorithms,
        r0_bounds=tuple(config.get("r0_bounds", DEFAULT_R0_BOUNDS)),
    )
    build_summary = sbv.build_summary_payload
    include_coordination = config["include_coordination"]

    outputs: list[tuple[str, str, dict[str, Any]]] = []
    failure_counts_by_cation: dict[str, Counter[str]] = {}
    failure_mids_by_cation: dict[str, dict[str, list[str]]] = {}
    r0_bounds_config = tuple(config.get("r0_bounds", DEFAULT_R0_BOUNDS))
    for cation, material in config["items"]:
        cation_failures = failure_counts_by_cation.setdefault(cation, Counter())
        cation_failure_mids = failure_mids_by_cation.setdefault(cation, {})
        try:
            fit_result, _strategy = _fit_one_material_with_fast_path(
                sbv,
                service,
                material,
                r0_bounds=r0_bounds_config,
            )
            results = [fit_result] if fit_result is not None else []
            item_failure_counts, item_failure_mids = _failure_details(results)
            material_id = getattr(material, "material_id", None)
            for reason, count in item_failure_counts.items():
                missing = max(0, int(count) - len(item_failure_mids.get(str(reason), ())))
                for _ in range(missing):
                    _append_failure_mid(item_failure_mids, str(reason), material_id)
            item_failure_mids = _sorted_failure_mids(item_failure_mids)
            cation_failures.update(item_failure_counts)
            cation_failure_mids = _merge_failure_mids(
                cation_failure_mids,
                item_failure_mids,
            )
            payload = build_summary(
                results,
                classifier=classify_summary_bucket,
                serializer=lambda s: summary_to_critmin_dict(
                    s,
                    include_coordination=include_coordination,
                    r0_bounds=r0_bounds_config,
                ),
            )
            records_added = 0
            for bucket, records in payload.items():
                for record in records:
                    outputs.append((cation, bucket, record))
                    records_added += 1
            if records_added == 0 and not item_failure_counts:
                cation_failures["no_summary"] += 1
                _append_failure_mid(
                    cation_failure_mids,
                    "no_summary",
                    getattr(material, "material_id", None),
                )
            failure_mids_by_cation[cation] = _sorted_failure_mids(cation_failure_mids)
        except Exception as exc:
            reason = _parallel_failure_label(exc)
            cation_failures[reason] += 1
            _append_failure_mid(
                cation_failure_mids,
                reason,
                getattr(material, "material_id", None),
            )
            failure_mids_by_cation[cation] = _sorted_failure_mids(cation_failure_mids)

    return {
        "records": outputs,
        "failure_counts_by_cation": {
            cation: dict(counts)
            for cation, counts in failure_counts_by_cation.items()
            if counts
        },
        "failure_mids_by_cation": {
            cation: mids
            for cation, mids in failure_mids_by_cation.items()
            if mids
        },
    }


def fit_materials_project_group_parallel(
    *,
    api_key: str,
    cations: Sequence[str],
    anion: str = "O",
    algorithms: Sequence[str] | None = None,
    energy_above_hull: tuple[float, float] = (0.0, 0.05),
    r0_bounds: tuple[float, float] = DEFAULT_R0_BOUNDS,
    require_possible_species: bool = True,
    screened_bond_valence_repo: str | Path | None = None,
    include_coordination: bool = True,
    candidate_tracker: str | Path | None = None,
    tracker_statuses: Sequence[str] | None = ("completed",),
    degenerate_fit_mode: str | None = None,
    max_workers: int = 1,
    on_cation_complete: Callable[
        [str, dict[str, list[dict[str, Any]]], dict[str, Any]], None
    ] | None = None,
    on_progress: Callable[
        [dict[str, dict[str, list[dict[str, Any]]]], int, int], None
    ] | None = None,
    exclude_mids: set[str] | None = None,
    batch_size: int = 10,
    seed_cn_line_priors: Mapping | None = None,
    force_adaptive_solver: bool = False,
    max_tasks_per_child: int | None = 50,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Fit multiple cations concurrently and return grouped payload plus stats.

    Work is distributed at the *material* level, not per-cation, so all
    workers stay busy even when cation pool sizes differ greatly.

    Parameters
    ----------
    on_cation_complete : callable, optional
        Called after every material for a cation has been processed, with
        ``(cation, cation_payload, stats)``.  Calls arrive as each cation
        finishes (not necessarily in input order).
    on_progress : callable, optional
        Called after every completed batch with
        ``(payload_so_far, materials_done, materials_total)``.
        Use this to write incremental results to disk regardless of cation
        boundaries.
    exclude_mids : set[str], optional
        Material IDs to skip (already fitted).  Extracted from a previously
        seeded output file to avoid re-fitting existing results.
    batch_size : int
        Number of materials per worker task.  Larger batches reduce IPC
        overhead; smaller batches improve load balancing. Only applies to the
        non-adaptive material-batch path; adaptive degenerate fits are still
        dispatched per cation so each worker sees the full cation pool.
    """
    if max_workers <= 1:
        payload, runs = fit_materials_project_group(
            api_key=api_key,
            cations=cations,
            anion=anion,
            algorithms=algorithms,
            energy_above_hull=energy_above_hull,
            r0_bounds=r0_bounds,
            require_possible_species=require_possible_species,
            screened_bond_valence_repo=screened_bond_valence_repo,
            include_coordination=include_coordination,
            candidate_tracker=candidate_tracker,
            tracker_statuses=tracker_statuses,
            degenerate_fit_mode=degenerate_fit_mode,
            progress=False,
            on_cation_complete=on_cation_complete,
            seed_cn_line_priors=seed_cn_line_priors,
            force_adaptive_solver=force_adaptive_solver,
            exclude_mids=exclude_mids,
        )
        return payload, [run_stats(run) for run in runs]

    if degenerate_fit_mode is not None:
        cation_material_counts: dict[str, int] = {}
        cation_materials: dict[str, list[Any]] = {}
        payload: dict[str, dict[str, list[dict[str, Any]]]] = {
            str(cation): empty_bucket_payload() for cation in cations
        }
        total_materials = 0
        for cation in cations:
            cation_name = str(cation)
            materials = prepare_cation_materials(
                api_key=api_key,
                cation=cation_name,
                anion=anion,
                energy_above_hull=energy_above_hull,
                require_possible_species=require_possible_species,
                screened_bond_valence_repo=screened_bond_valence_repo,
                include_coordination=include_coordination,
                candidate_tracker=candidate_tracker,
                tracker_statuses=tracker_statuses,
            )
            if exclude_mids:
                before = len(materials)
                materials = [m for m in materials if m.material_id not in exclude_mids]
                skipped = before - len(materials)
                if skipped:
                    sys.stderr.write(
                        f"[skip] {cation_name}: {skipped} materials already fitted, "
                        f"{len(materials)} remaining\n"
                    )
                    sys.stderr.flush()
            cation_material_counts[cation_name] = len(materials)
            cation_materials[cation_name] = materials
            total_materials += len(materials)

        if total_materials == 0:
            return payload, [
                {
                    "cation": str(cation),
                    "anion": anion,
                    "fetched_materials": 0,
                    "summarized_materials": 0,
                    "oxides": 0,
                    "hydroxides": 0,
                    "failure_counts": {},
                    "failure_mids": {},
                }
                for cation in cations
            ]

        summaries: dict[str, dict[str, Any]] = {}
        materials_done = 0
        active_cations = [
            str(cation)
            for cation in cations
            if cation_material_counts.get(str(cation), 0) > 0
        ]
        worker_count = min(max_workers, len(active_cations))
        adaptive_executor_kwargs: dict[str, Any] = {"max_workers": worker_count}
        # The adaptive path dispatches one task per cation, so a worker only
        # processes one task before being recycled if max_tasks_per_child=1.
        # That defeats the purpose; cap at the lower of (cations_per_worker, 1)
        # but allow override.
        if max_tasks_per_child is not None and max_tasks_per_child > 0:
            adaptive_executor_kwargs["max_tasks_per_child"] = int(max_tasks_per_child)
        with ProcessPoolExecutor(**adaptive_executor_kwargs) as executor:
            futures = {
                executor.submit(
                    _parallel_adaptive_system_worker,
                    {
                        "cation": cation,
                        "anion": anion,
                        "materials": cation_materials[cation],
                        "algorithms": tuple(algorithms) if algorithms else None,
                        "r0_bounds": tuple(r0_bounds),
                        "screened_bond_valence_repo": (
                            str(screened_bond_valence_repo)
                            if screened_bond_valence_repo
                            else None
                        ),
                        "include_coordination": bool(include_coordination),
                        "degenerate_fit_mode": degenerate_fit_mode,
                        "seed_cn_line_priors": dict(seed_cn_line_priors) if seed_cn_line_priors else None,
                        "force_adaptive_solver": bool(force_adaptive_solver),
                    },
                ): str(cation)
                for cation in active_cations
            }

            for future in as_completed(futures):
                cation, cation_payload, stats = future.result()
                payload[str(cation)] = cation_payload
                summaries[str(cation)] = stats
                materials_done += cation_material_counts.get(str(cation), 0)
                if on_cation_complete is not None:
                    on_cation_complete(str(cation), cation_payload, stats)
                if on_progress is not None:
                    on_progress(payload, materials_done, total_materials)

        ordered_payload = {str(cation): payload.get(str(cation), empty_bucket_payload()) for cation in cations}
        ordered_stats = [
            summaries.get(
                str(cation),
                {
                    "cation": str(cation),
                    "anion": anion,
                    "fetched_materials": cation_material_counts.get(str(cation), 0),
                    "summarized_materials": 0,
                    "oxides": 0,
                    "hydroxides": 0,
                    "failure_counts": {},
                    "failure_mids": {},
                },
            )
            for cation in cations
        ]
        return ordered_payload, ordered_stats

    # ── Phase 1: fetch all materials (sequential, I/O-bound) ─────────────
    cation_material_counts: dict[str, int] = {}
    cation_failure_counts: dict[str, Counter[str]] = {
        str(cation): Counter() for cation in cations
    }
    cation_failure_mids: dict[str, dict[str, list[str]]] = {
        str(cation): {} for cation in cations
    }
    all_items: list[tuple[str, Any]] = []

    for cation in cations:
        materials = prepare_cation_materials(
            api_key=api_key,
            cation=cation,
            anion=anion,
            energy_above_hull=energy_above_hull,
            require_possible_species=require_possible_species,
            screened_bond_valence_repo=screened_bond_valence_repo,
            include_coordination=include_coordination,
            candidate_tracker=candidate_tracker,
            tracker_statuses=tracker_statuses,
        )
        if exclude_mids:
            before = len(materials)
            materials = [m for m in materials if m.material_id not in exclude_mids]
            skipped = before - len(materials)
            if skipped:
                sys.stderr.write(
                    f"[skip] {cation}: {skipped} materials already fitted, "
                    f"{len(materials)} remaining\n"
                )
                sys.stderr.flush()
        cation_material_counts[cation] = len(materials)
        for m in materials:
            all_items.append((cation, m))

    if not all_items:
        empty: dict[str, dict[str, list[dict[str, Any]]]] = {
            c: {"oxides": [], "hydroxides": []} for c in cations
        }
        return empty, [
            {"cation": c, "anion": anion, "fetched_materials": 0,
             "summarized_materials": 0, "oxides": 0, "hydroxides": 0,
             "failure_counts": {}, "failure_mids": {}}
            for c in cations
        ]

    # ── Phase 2: batch and submit ────────────────────────────────────────
    actual_batch = max(1, min(batch_size, len(all_items)))
    batches = [
        all_items[i : i + actual_batch]
        for i in range(0, len(all_items), actual_batch)
    ]

    base_config: dict[str, Any] = {
        "screened_bond_valence_repo": (
            str(screened_bond_valence_repo) if screened_bond_valence_repo else None
        ),
        "algorithms": tuple(algorithms) if algorithms else None,
        "include_coordination": include_coordination,
        "r0_bounds": tuple(r0_bounds),
    }

    worker_count = min(max_workers, len(batches))

    # ── Phase 3: collect results as batches complete ─────────────────────
    payload: dict[str, dict[str, list[dict[str, Any]]]] = {
        cation: {"oxides": [], "hydroxides": []} for cation in cations
    }
    total_materials = len(all_items)
    materials_done = 0
    cation_materials_done: Counter[str] = Counter()
    cation_completed: set[str] = set()

    executor_kwargs: dict[str, Any] = {"max_workers": worker_count}
    if max_tasks_per_child is not None and max_tasks_per_child > 0:
        executor_kwargs["max_tasks_per_child"] = int(max_tasks_per_child)
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        # Submit all batches; tag each future with the cation counts it
        # contains so we can track per-cation completion.
        future_cation_counts: dict[Any, Counter[str]] = {}
        for batch in batches:
            batch_counts: Counter[str] = Counter()
            for cation, _ in batch:
                batch_counts[cation] += 1
            future = executor.submit(
                _fit_material_batch_worker,
                {**base_config, "items": batch},
            )
            future_cation_counts[future] = batch_counts

        for future in as_completed(future_cation_counts):
            batch_results = future.result()
            if isinstance(batch_results, list):
                batch_records = batch_results
                batch_failure_counts: dict[str, dict[str, int]] = {}
                batch_failure_mids: dict[str, dict[str, list[str]]] = {}
            else:
                batch_records = batch_results.get("records", [])
                batch_failure_counts = batch_results.get("failure_counts_by_cation", {})
                batch_failure_mids = batch_results.get("failure_mids_by_cation", {})

            for cation, counts in batch_failure_counts.items():
                cation_failure_counts.setdefault(str(cation), Counter()).update(
                    {str(reason): int(value) for reason, value in counts.items()}
                )
            for cation, mids in batch_failure_mids.items():
                cation_failure_mids[str(cation)] = _merge_failure_mids(
                    cation_failure_mids.get(str(cation), {}),
                    mids,
                )

            for cation, bucket, record in batch_records:
                payload[cation][bucket].append(record)

            # Track completion
            batch_material_count = sum(future_cation_counts[future].values())
            materials_done += batch_material_count

            for cation, count in future_cation_counts[future].items():
                cation_materials_done[cation] += count
                if (
                    cation not in cation_completed
                    and cation_materials_done[cation] >= cation_material_counts[cation]
                ):
                    cation_completed.add(cation)
                    payload[cation] = _sorted_payload(payload[cation])
                    if on_cation_complete is not None:
                        stats = {
                            "cation": cation,
                            "anion": anion,
                            "fetched_materials": cation_material_counts[cation],
                            "summarized_materials": sum(
                                len(r) for r in payload[cation].values()
                            ),
                            "oxides": len(payload[cation].get("oxides", [])),
                            "hydroxides": len(payload[cation].get("hydroxides", [])),
                            "failure_counts": dict(cation_failure_counts.get(cation, Counter())),
                            "failure_mids": cation_failure_mids.get(cation, {}),
                        }
                        on_cation_complete(cation, payload[cation], stats)

            # Progress callback after every batch
            if on_progress is not None:
                on_progress(payload, materials_done, total_materials)

    # ── Phase 4: order output by input cation order ──────────────────────
    for cation in cations:
        if cation not in cation_completed:
            payload[cation] = _sorted_payload(payload[cation])

    ordered_payload = {c: payload[c] for c in cations}
    ordered_stats = [
        {
            "cation": c,
            "anion": anion,
            "fetched_materials": cation_material_counts.get(c, 0),
            "summarized_materials": sum(len(r) for r in payload[c].values()),
            "oxides": len(payload[c].get("oxides", [])),
            "hydroxides": len(payload[c].get("hydroxides", [])),
            "failure_counts": dict(cation_failure_counts.get(c, Counter())),
            "failure_mids": cation_failure_mids.get(c, {}),
        }
        for c in cations
    ]
    return ordered_payload, ordered_stats


def _collect_mids(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> set[str]:
    """Return every material-id that appears anywhere in a grouped payload."""
    mids: set[str] = set()
    for cation_payload in payload.values():
        if not isinstance(cation_payload, Mapping):
            continue
        for bucket in _BUCKETS:
            for record in cation_payload.get(bucket, []):
                mid = record.get("mid")
                if mid:
                    mids.add(str(mid))
    return mids


def _merge_grouped_payload(
    base: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    incoming: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Merge two grouped payloads. Incoming records override base by ``mid``."""
    cations = list(dict.fromkeys([*base.keys(), *incoming.keys()]))
    merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cation in cations:
        base_cation = base.get(cation) or {}
        incoming_cation = incoming.get(cation) or {}
        cation_out = empty_bucket_payload()
        for bucket in _BUCKETS:
            by_mid: dict[str, dict[str, Any]] = {}
            for record in base_cation.get(bucket, []) or []:
                mid = str(record.get("mid", ""))
                if mid:
                    by_mid[mid] = dict(record)
            for record in incoming_cation.get(bucket, []) or []:
                mid = str(record.get("mid", ""))
                if mid:
                    by_mid[mid] = dict(record)
            cation_out[bucket] = sorted(by_mid.values(), key=lambda r: str(r.get("mid", "")))
        merged[cation] = cation_out
    return merged


def _tag_phase_b_records(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return a copy of the payload with ``phase_b_refit=True`` on every record."""
    tagged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cation, cation_payload in payload.items():
        cation_out = empty_bucket_payload()
        if not isinstance(cation_payload, Mapping):
            tagged[cation] = cation_out
            continue
        for bucket in _BUCKETS:
            for record in cation_payload.get(bucket, []) or []:
                copied = dict(record)
                copied["phase_b_refit"] = True
                cation_out[bucket].append(copied)
        tagged[cation] = cation_out
    return tagged


def fit_materials_project_group_two_phase(
    *,
    api_key: str,
    cations: Sequence[str],
    degenerate_fit_mode: str,
    phase_a_checkpoint: str | Path,
    anion: str = "O",
    algorithms: Sequence[str] | None = None,
    energy_above_hull: tuple[float, float] = (0.0, 0.05),
    r0_bounds: tuple[float, float] = DEFAULT_R0_BOUNDS,
    require_possible_species: bool = True,
    screened_bond_valence_repo: str | Path | None = None,
    include_coordination: bool = True,
    candidate_tracker: str | Path | None = None,
    tracker_statuses: Sequence[str] | None = ("completed",),
    max_workers: int = 1,
    batch_size: int = 10,
    max_r0_std: float = DEFAULT_MAX_R0_STD,
    max_b_std: float = DEFAULT_MAX_B_STD,
    prior_min_r2: float = 0.3,
    prior_max_abs_b: float = 2.0,
    max_tasks_per_child: int | None = 50,
    on_cation_complete: Callable[
        [str, dict[str, list[dict[str, Any]]], dict[str, Any]], None
    ] | None = None,
    on_progress: Callable[
        [dict[str, dict[str, list[dict[str, Any]]]], int, int], None
    ] | None = None,
) -> dict[str, Any]:
    """Two-phase order-robust fit with global β-line priors.

    **Phase A** runs the legacy fitter on every material with no adaptive
    solving and no priors.  All results are classified by uncertainty via
    :func:`separate_grouped_payload_by_uncertainty`; the high-certainty subset
    is **persisted to ``phase_a_checkpoint`` before any priors are built**.

    **Prior build** constructs a single global β-line prior table
    (:func:`critmin.analysis.adaptive_bond_valence.build_cn_line_priors_from_payload`)
    from the frozen Phase-A checkpoint.  These priors are not updated again.

    **Phase B** re-fits every Phase-A high-uncertainty material with the
    adaptive degenerate solver under ``degenerate_fit_mode``, seeded with the
    global priors.  ``force_adaptive_solver=True`` ensures that *all*
    quarantined materials are routed through the prior-based solver regardless
    of whether their individual design matrix was flagged degenerate.

    **Order robustness.**  Phase A fits depend only on per-material inputs
    (no priors).  The Phase-A checkpoint is serialized to disk before priors
    are built, so prior content is independent of cation processing order.
    Every Phase-B worker receives the same frozen ``global_priors`` object, so
    Phase-B results depend only on (material, global_priors), not on cation
    processing order.  Shuffling ``cations`` produces identical final payloads.

    Returns
    -------
    dict
        ``{
            "accepted_payload":     merged high-certainty payload (Phase A + promoted Phase B),
            "high_uncertainty_payload": records still above the uncertainty threshold after Phase B,
            "phase_a_payload":      raw Phase-A grouped payload,
            "phase_a_stats":        per-cation stats from Phase A,
            "phase_b_payload":      raw Phase-B grouped payload (None if Phase B was skipped),
            "phase_b_stats":        per-cation stats from Phase B,
            "phase_a_checkpoint":   Path to the persisted Phase-A checkpoint,
            "global_priors":        the frozen CnLinePrior table used in Phase B,
            "prior_coverage":       counts of priors by hierarchy level,
            "phase_b_promotions":   count of records promoted from high-uncertainty → accepted,
        }``
    """
    if degenerate_fit_mode not in VALID_DEGENERATE_FIT_MODES:
        raise ValueError(
            f"degenerate_fit_mode must be one of {VALID_DEGENERATE_FIT_MODES}, "
            f"got {degenerate_fit_mode!r}"
        )

    checkpoint_path = Path(phase_a_checkpoint)

    def _phase_a_cation_cb(
        cation: str,
        cation_payload: dict[str, list[dict[str, Any]]],
        stats: dict[str, Any],
    ) -> None:
        if on_cation_complete is None:
            return
        tagged_stats = dict(stats)
        tagged_stats["phase"] = "A"
        on_cation_complete(cation, cation_payload, tagged_stats)

    def _phase_a_progress_cb(
        payload_so_far: dict[str, dict[str, list[dict[str, Any]]]],
        done: int,
        total: int,
    ) -> None:
        if on_progress is None:
            return
        on_progress(payload_so_far, done, total)

    # ── Phase A: legacy fitting, no priors, no adaptive solving ──────────
    phase_a_payload, phase_a_stats = fit_materials_project_group_parallel(
        api_key=api_key,
        cations=cations,
        anion=anion,
        algorithms=algorithms,
        energy_above_hull=energy_above_hull,
        r0_bounds=r0_bounds,
        require_possible_species=require_possible_species,
        screened_bond_valence_repo=screened_bond_valence_repo,
        include_coordination=include_coordination,
        candidate_tracker=candidate_tracker,
        tracker_statuses=tracker_statuses,
        degenerate_fit_mode=None,
        max_workers=max_workers,
        on_cation_complete=_phase_a_cation_cb if on_cation_complete else None,
        on_progress=_phase_a_progress_cb if on_progress else None,
        batch_size=batch_size,
        max_tasks_per_child=max_tasks_per_child,
    )

    accepted_payload, quarantined_payload = separate_grouped_payload_by_uncertainty(
        phase_a_payload,
        max_r0_std=max_r0_std,
        max_b_std=max_b_std,
    )

    # ── Checkpoint: persist high-certainty records BEFORE building priors ──
    write_payload_json(accepted_payload, checkpoint_path)

    # ── Build global priors from the frozen checkpoint on disk ───────────
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        checkpoint_payload = json.load(fh)
    global_priors = build_cn_line_priors_from_payload(
        checkpoint_payload,
        max_r0_std=max_r0_std,
        max_b_std=max_b_std,
        max_abs_b=prior_max_abs_b,
        min_r2=prior_min_r2,
    )
    prior_coverage: Counter[str] = Counter()
    for (level, _key) in global_priors.keys():
        prior_coverage[str(level)] += 1

    # Cations that have any quarantined record → targets of Phase B
    phase_b_cations = [
        str(cation) for cation in cations if str(cation) in quarantined_payload
    ]

    phase_b_payload: dict[str, dict[str, list[dict[str, Any]]]] | None = None
    phase_b_stats: list[dict[str, Any]] = []
    phase_b_promotions = 0
    final_accepted = accepted_payload
    final_high_uncertainty = quarantined_payload

    if phase_b_cations:
        accepted_mids = _collect_mids(accepted_payload)

        def _phase_b_cation_cb(
            cation: str,
            cation_payload: dict[str, list[dict[str, Any]]],
            stats: dict[str, Any],
        ) -> None:
            if on_cation_complete is None:
                return
            tagged = _tag_phase_b_records({cation: cation_payload}).get(cation, cation_payload)
            tagged_stats = dict(stats)
            tagged_stats["phase"] = "B"
            on_cation_complete(cation, tagged, tagged_stats)

        def _phase_b_progress_cb(
            payload_so_far: dict[str, dict[str, list[dict[str, Any]]]],
            done: int,
            total: int,
        ) -> None:
            if on_progress is None:
                return
            on_progress(_tag_phase_b_records(payload_so_far), done, total)

        # ── Phase B: re-fit quarantined materials with global priors ─────
        phase_b_payload, phase_b_stats = fit_materials_project_group_parallel(
            api_key=api_key,
            cations=phase_b_cations,
            anion=anion,
            algorithms=algorithms,
            energy_above_hull=energy_above_hull,
            r0_bounds=r0_bounds,
            require_possible_species=require_possible_species,
            screened_bond_valence_repo=screened_bond_valence_repo,
            include_coordination=include_coordination,
            candidate_tracker=candidate_tracker,
            tracker_statuses=tracker_statuses,
            degenerate_fit_mode=degenerate_fit_mode,
            max_workers=max_workers,
            exclude_mids=accepted_mids,
            seed_cn_line_priors=global_priors,
            force_adaptive_solver=True,
            batch_size=batch_size,
            max_tasks_per_child=max_tasks_per_child,
            on_cation_complete=_phase_b_cation_cb if on_cation_complete else None,
            on_progress=_phase_b_progress_cb if on_progress else None,
        )

        phase_b_payload = _tag_phase_b_records(phase_b_payload)

        # Separate Phase B results by their own uncertainty.  Records that
        # now meet the threshold are promoted into the accepted payload.
        phase_b_accepted, phase_b_still_uncertain = separate_grouped_payload_by_uncertainty(
            phase_b_payload,
            max_r0_std=max_r0_std,
            max_b_std=max_b_std,
        )

        final_accepted = _merge_grouped_payload(accepted_payload, phase_b_accepted)

        # High-uncertainty pool: keep Phase A quarantined records that were
        # NOT re-fit or were re-fit but still high-uncertainty, overridden by
        # any Phase B result for the same mid.
        final_high_uncertainty = _merge_grouped_payload(
            quarantined_payload, phase_b_still_uncertain
        )
        # Remove from the high-uncertainty pool any record that got promoted.
        promoted_mids = _collect_mids(phase_b_accepted)
        if promoted_mids:
            pruned: dict[str, dict[str, list[dict[str, Any]]]] = {}
            for cation, cation_payload in final_high_uncertainty.items():
                cation_out = empty_bucket_payload()
                for bucket in _BUCKETS:
                    cation_out[bucket] = [
                        record
                        for record in cation_payload.get(bucket, [])
                        if str(record.get("mid", "")) not in promoted_mids
                    ]
                if any(cation_out[bucket] for bucket in _BUCKETS):
                    pruned[cation] = cation_out
            final_high_uncertainty = pruned
        phase_b_promotions = len(promoted_mids)

    return {
        "accepted_payload": final_accepted,
        "high_uncertainty_payload": final_high_uncertainty,
        "phase_a_payload": phase_a_payload,
        "phase_a_stats": phase_a_stats,
        "phase_b_payload": phase_b_payload,
        "phase_b_stats": phase_b_stats,
        "phase_a_checkpoint": checkpoint_path,
        "global_priors": global_priors,
        "prior_coverage": dict(prior_coverage),
        "phase_b_promotions": phase_b_promotions,
    }


def write_payload_json(
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a compact payload JSON file atomically."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def run_stats(run: BondValenceSystemRun) -> dict[str, Any]:
    """Return simple stats for CLI/reporting use."""
    summarized = sum(len(records) for records in run.payload.values())
    failure_counts, failure_mids = _failure_details(
        getattr(run, "results", ()),
        getattr(run, "failure_mids", {}),
    )
    return {
        "cation": run.cation,
        "anion": run.anion,
        "fetched_materials": len(run.materials),
        "summarized_materials": summarized,
        "oxides": len(run.payload.get("oxides", [])),
        "hydroxides": len(run.payload.get("hydroxides", [])),
        "failure_counts": failure_counts,
        "failure_mids": failure_mids,
    }
