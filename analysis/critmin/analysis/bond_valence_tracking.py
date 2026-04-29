"""Candidate tracking and coverage reporting for bond-valence payloads."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from critmin.analysis.bond_valence import (
    classify_formula_bucket,
    prepare_cation_materials,
    resolve_tracker_coordination,
)

BUCKETS = ("oxides", "hydroxides")
STATUS_ORDER = {
    "completed": 0,
    "high_uncertainty": 1,
    "ready": 2,
    "no_species": 3,
}


def empty_bucket_payload() -> dict[str, list[dict[str, Any]]]:
    return {"oxides": [], "hydroxides": []}


def is_cation_payload(value: object) -> bool:
    return isinstance(value, dict) and (
        isinstance(value.get("oxides"), list)
        or isinstance(value.get("hydroxides"), list)
    )


def record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    mid = record.get("mid")
    if mid is not None:
        return ("mid", str(mid))
    return ("record", json.dumps(dict(record), sort_keys=True))


def merge_bucket_records(
    existing_records: Sequence[Mapping[str, Any]],
    incoming_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for record in existing_records:
        merged[record_key(record)] = dict(record)
    for record in incoming_records:
        merged[record_key(record)] = dict(record)
    return list(merged.values())


def merge_cation_payload(
    existing: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    incoming: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    existing = existing or {}
    incoming = incoming or {}
    return {
        bucket: merge_bucket_records(
            existing.get(bucket, []),
            incoming.get(bucket, []),
        )
        for bucket in BUCKETS
    }


def load_payload_source(
    source: str | Path | tuple[str | Path, str],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], str]:
    cation_hint: str | None = None
    if isinstance(source, tuple):
        path, cation_hint = source
    else:
        path = source

    payload_path = Path(path)
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    if is_cation_payload(data):
        if cation_hint is None:
            raise ValueError(f"Single-cation payload {payload_path} requires a cation hint")
        return {cation_hint: merge_cation_payload(empty_bucket_payload(), data)}, str(payload_path)

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cation, cation_payload in data.items():
        if is_cation_payload(cation_payload):
            grouped[str(cation)] = merge_cation_payload(empty_bucket_payload(), cation_payload)
    return grouped, str(payload_path)


def merge_payload_sources(
    *sources: str | Path | tuple[str | Path, str],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[str]]]:
    merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_by_cation: dict[str, list[str]] = {}

    for source in sources:
        grouped, label = load_payload_source(source)
        for cation, payload in grouped.items():
            before = cation_record_count(merged.get(cation, {}))
            merged[cation] = merge_cation_payload(merged.get(cation, {}), payload)
            after = cation_record_count(merged[cation])
            if after > before or cation not in source_by_cation:
                source_by_cation.setdefault(cation, [])
                if label not in source_by_cation[cation]:
                    source_by_cation[cation].append(label)

    return merged, source_by_cation


def cation_record_count(cation_payload: Mapping[str, Sequence[Mapping[str, Any]]] | None) -> int:
    if not cation_payload:
        return 0
    return sum(len(cation_payload.get(bucket, [])) for bucket in BUCKETS)


def grouped_record_count(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> int:
    return sum(cation_record_count(cation_payload) for cation_payload in payload.values())


def normalize_seeded_payload(
    *,
    output_path: str | Path,
    cations: Sequence[str],
    grouped_output: bool,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    path = Path(output_path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if grouped_output and isinstance(data, dict):
        return {
            str(cation): merge_cation_payload(empty_bucket_payload(), payload)
            for cation, payload in data.items()
            if is_cation_payload(payload)
        }

    if is_cation_payload(data):
        return {
            str(cations[0]): merge_cation_payload(empty_bucket_payload(), data),
        }

    if (
        isinstance(data, dict)
        and cations
        and cations[0] in data
        and is_cation_payload(data[cations[0]])
    ):
        return {
            str(cations[0]): merge_cation_payload(
                empty_bucket_payload(),
                data[cations[0]],
            ),
        }

    return {}


def seeded_material_ids(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> set[str]:
    mids: set[str] = set()
    for cation_payload in payload.values():
        for bucket in BUCKETS:
            for record in cation_payload.get(bucket, []):
                mid = record.get("mid")
                if mid:
                    mids.add(str(mid))
    return mids


def merge_grouped_payloads(
    existing: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    incoming: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged = {
        str(cation): merge_cation_payload(empty_bucket_payload(), payload)
        for cation, payload in existing.items()
    }
    for cation, payload in incoming.items():
        merged[str(cation)] = merge_cation_payload(merged.get(str(cation), {}), payload)
    return merged


def merged_output_payload(
    *,
    cations: Sequence[str],
    grouped_output: bool,
    incremental_payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    if grouped_output:
        return {
            str(cation): merge_cation_payload(empty_bucket_payload(), payload)
            for cation, payload in incremental_payload.items()
        }
    return merge_cation_payload(
        empty_bucket_payload(),
        incremental_payload.get(str(cations[0]), empty_bucket_payload()),
    )


def _candidate_status(
    *,
    mid: str,
    has_species: bool,
    completed_mids: set[str],
    high_uncertainty_mids: set[str],
) -> str:
    if mid in completed_mids:
        return "completed"
    if mid in high_uncertainty_mids:
        return "high_uncertainty"
    if has_species:
        return "ready"
    return "no_species"


def _candidate_record(
    material: Any,
    *,
    completed_mids: set[str],
    high_uncertainty_mids: set[str],
) -> tuple[str, dict[str, Any]]:
    has_species = bool(getattr(material, "possible_species", ()))
    metadata = dict(getattr(material, "metadata", {}) or {})
    bucket = metadata.get("bucket")
    if bucket not in BUCKETS:
        bucket = classify_formula_bucket(getattr(material, "formula_pretty", None))
    cn_all = metadata.get("cn_all")
    cn_mode, cn_all = resolve_tracker_coordination(
        cn_mode=metadata.get("cn"),
        cn_all=cn_all,
    )
    if not cn_all:
        cn_all = None
    oxi_state = metadata.get("oxi_state")
    oxi_state_label = metadata.get("oxi_state_label")
    record: dict[str, Any] = {
        "mid": str(material.material_id),
        "formula": getattr(material, "formula_pretty", None),
        "has_species": has_species,
        "status": _candidate_status(
            mid=str(material.material_id),
            has_species=has_species,
            completed_mids=completed_mids,
            high_uncertainty_mids=high_uncertainty_mids,
        ),
        "cn_mode": cn_mode,
        "cn_all": cn_all,
    }
    if oxi_state is not None:
        record["oxi_state"] = int(oxi_state)
    if oxi_state_label is not None:
        record["oxi_state_label"] = str(oxi_state_label)
    return bucket, record


def _bucket_tracker_block(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(candidate.get("status")) for candidate in candidates)
    cn_counts = Counter(
        str(int(candidate["cn_mode"]))
        for candidate in candidates
        if candidate.get("cn_mode") is not None
    )
    ordered_candidates = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            STATUS_ORDER.get(str(candidate.get("status")), 99),
            str(candidate.get("mid", "")),
        ),
    )
    return {
        "total": len(ordered_candidates),
        "completed": status_counts.get("completed", 0),
        "high_uncertainty": status_counts.get("high_uncertainty", 0),
        "ready": status_counts.get("ready", 0),
        "no_species": status_counts.get("no_species", 0),
        "candidates": ordered_candidates,
        "cn_distribution": dict(sorted(cn_counts.items(), key=lambda item: int(item[0]))),
    }


def generate_mid_level_candidate_tracker(
    *,
    api_key: str,
    cations: Sequence[str],
    anion: str = "O",
    energy_above_hull: tuple[float, float] = (0.0, 0.05),
    screened_bond_valence_repo: str | Path | None = None,
    seed_payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    high_uncertainty_payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    include_coordination: bool = True,
    include_no_species: bool = True,
) -> dict[str, dict[str, Any]]:
    completed_mids = seeded_material_ids(seed_payload or {})
    high_uncertainty_mids = seeded_material_ids(high_uncertainty_payload or {})
    tracker: dict[str, dict[str, Any]] = {}

    for cation in cations:
        materials = prepare_cation_materials(
            api_key=api_key,
            cation=cation,
            anion=anion,
            energy_above_hull=energy_above_hull,
            require_possible_species=not include_no_species,
            screened_bond_valence_repo=screened_bond_valence_repo,
            include_coordination=include_coordination,
        )
        bucket_candidates = {bucket: [] for bucket in BUCKETS}
        for material in materials:
            bucket, record = _candidate_record(
                material,
                completed_mids=completed_mids,
                high_uncertainty_mids=high_uncertainty_mids,
            )
            bucket_candidates[bucket].append(record)

        tracker[str(cation)] = {
            bucket: _bucket_tracker_block(bucket_candidates[bucket])
            for bucket in BUCKETS
        }

    return tracker


def build_reprocessing_tracker(
    *,
    tracker: Mapping[str, Any],
    statuses: Sequence[str] = ("high_uncertainty",),
    cations: Sequence[str] | None = None,
    staged_status: str = "ready",
    selected_mids: Mapping[tuple[str, str], set[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a tracker-shaped subset staged for reprocessing.

    Selected candidates keep their bucket/CN metadata, gain ``original_status``,
    and have ``status`` rewritten to ``staged_status`` so the output can be
    passed directly back into ``--candidate-tracker`` with
    ``--tracker-statuses ready``.
    """
    target_statuses = {str(status) for status in statuses}
    cation_names = (
        [str(cation) for cation in cations]
        if cations is not None
        else [str(cation) for cation in tracker.keys()]
    )

    staged_tracker: dict[str, dict[str, Any]] = {}
    for cation in cation_names:
        cation_data = tracker.get(cation, {})
        bucket_blocks: dict[str, dict[str, Any]] = {}
        for bucket in BUCKETS:
            candidates = cation_data.get(bucket, {}).get("candidates", [])
            staged_candidates = []
            bucket_selected_mids = selected_mids.get((cation, bucket), set()) if selected_mids else set()
            for candidate in candidates:
                candidate_status = str(candidate.get("status"))
                candidate_mid = str(candidate.get("mid")) if candidate.get("mid") else None
                selected_by_status = candidate_status in target_statuses
                selected_by_mid = candidate_mid in bucket_selected_mids if candidate_mid else False
                if not (selected_by_status or selected_by_mid):
                    continue
                staged_candidate = dict(candidate)
                staged_candidate["original_status"] = candidate_status
                staged_candidate["status"] = staged_status
                staged_candidates.append(staged_candidate)
            bucket_blocks[bucket] = _bucket_tracker_block(staged_candidates)
        staged_tracker[cation] = bucket_blocks

    return staged_tracker


def write_tracker_json(tracker: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracker, indent=2), encoding="utf-8")
    return path


def payload_mid_sets(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[tuple[str, str], set[str]]:
    mids: dict[tuple[str, str], set[str]] = {}
    for cation, cation_payload in payload.items():
        for bucket in BUCKETS:
            mids[(str(cation), bucket)] = {
                str(record["mid"])
                for record in cation_payload.get(bucket, [])
                if record.get("mid")
            }
    return mids


def tracker_mid_sets(
    tracker: Mapping[str, Any],
) -> dict[tuple[str, str, str], set[str]]:
    mids: dict[tuple[str, str, str], set[str]] = {}
    for cation, cation_data in tracker.items():
        for bucket in BUCKETS:
            candidates = cation_data.get(bucket, {}).get("candidates", [])
            for status in ("completed", "high_uncertainty", "ready", "no_species"):
                mids[(str(cation), bucket, status)] = {
                    str(candidate["mid"])
                    for candidate in candidates
                    if candidate.get("mid") and candidate.get("status") == status
                }
    return mids


def build_progress_report(
    *,
    tracker: Mapping[str, Any],
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    high_uncertainty_payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
    cations: Sequence[str] | None = None,
) -> dict[str, Any]:
    if cations is None:
        cation_order = list(tracker.keys())
    else:
        cation_order = [str(cation) for cation in cations]

    payload_sets = payload_mid_sets(payload)
    uncertain_payload_sets = payload_mid_sets(high_uncertainty_payload or {})
    tracked_sets = tracker_mid_sets(tracker)
    report_cations: dict[str, Any] = {}
    all_missing_fit_mids: set[str] = set()

    for cation in cation_order:
        bucket_report: dict[str, Any] = {}
        for bucket in BUCKETS:
            completed = tracked_sets.get((cation, bucket, "completed"), set())
            high_uncertainty = tracked_sets.get((cation, bucket, "high_uncertainty"), set())
            ready = tracked_sets.get((cation, bucket, "ready"), set())
            no_species = tracked_sets.get((cation, bucket, "no_species"), set())
            payload_mids = payload_sets.get((cation, bucket), set())
            uncertain_payload_mids = uncertain_payload_sets.get((cation, bucket), set())
            missing_fit_mids = sorted(
                (completed - payload_mids)
                | (high_uncertainty - uncertain_payload_mids)
                | (ready - (payload_mids | uncertain_payload_mids))
            )
            all_missing_fit_mids.update(missing_fit_mids)
            bucket_report[bucket] = {
                "identified_total": len(completed | high_uncertainty | ready | no_species),
                "completed_total": len(completed),
                "high_uncertainty_total": len(high_uncertainty),
                "ready_total": len(ready),
                "no_species_total": len(no_species),
                "payload_total": len(payload_mids),
                "high_uncertainty_payload_total": len(uncertain_payload_mids),
                "covered_completed": len(completed & payload_mids),
                "covered_high_uncertainty": len(high_uncertainty & uncertain_payload_mids),
                "covered_ready": len(ready & payload_mids),
                "excluded_no_species": len(no_species),
                "missing_fit_total": len(missing_fit_mids),
                "missing_fit_mids": missing_fit_mids,
            }
        report_cations[cation] = bucket_report

    return {
        "cations": report_cations,
        "all_fit_eligible_mids_processed": not all_missing_fit_mids,
        "missing_fit_mid_total": len(all_missing_fit_mids),
        "missing_fit_mids": sorted(all_missing_fit_mids),
    }


def render_progress_markdown(
    *,
    tracker_path: str | Path,
    payload_sources: Sequence[str],
    high_uncertainty_payload_sources: Sequence[str] = (),
    report: Mapping[str, Any],
    title: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Tracker: `{tracker_path}`",
        f"- Payload sources: {', '.join(f'`{source}`' for source in payload_sources)}",
        (
            "- High-uncertainty payload sources: "
            + ", ".join(f'`{source}`' for source in high_uncertainty_payload_sources)
            if high_uncertainty_payload_sources
            else "- High-uncertainty payload sources: none"
        ),
        f"- All fit-eligible mids processed: `{'yes' if report['all_fit_eligible_mids_processed'] else 'no'}`",
        f"- Missing fit-eligible mids: `{report['missing_fit_mid_total']}`",
        "",
        "| Cation | Bucket | Identified | Completed | High uncertainty | Ready | No species | Payload | High-uncertainty payload | Missing fit mids |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for cation, cation_report in report["cations"].items():
        for bucket in BUCKETS:
            bucket_report = cation_report[bucket]
            lines.append(
                "| "
                + " | ".join(
                    [
                        cation,
                        bucket,
                        str(bucket_report["identified_total"]),
                        str(bucket_report["completed_total"]),
                        str(bucket_report["high_uncertainty_total"]),
                        str(bucket_report["ready_total"]),
                        str(bucket_report["no_species_total"]),
                        str(bucket_report["payload_total"]),
                        str(bucket_report["high_uncertainty_payload_total"]),
                        str(bucket_report["missing_fit_total"]),
                    ]
                )
                + " |"
            )

    if report["missing_fit_mid_total"]:
        lines.extend(
            [
                "",
                "## Missing Fit MIDs",
                "",
            ]
        )
        for cation, cation_report in report["cations"].items():
            for bucket in BUCKETS:
                missing = cation_report[bucket]["missing_fit_mids"]
                if not missing:
                    continue
                lines.append(
                    f"- `{cation}` `{bucket}`: {len(missing)} missing "
                    f"(sample: {', '.join(f'`{mid}`' for mid in missing[:10])})"
                )

    return "\n".join(lines) + "\n"
