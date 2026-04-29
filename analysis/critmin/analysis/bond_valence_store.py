"""Canonical storage helpers for harmonized bond-valence payload data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from critmin.analysis.bond_valence import resolve_tracker_coordination
from critmin.analysis.default_sources import (
    resolve_all_remaining_oxygen_accepted_sources,
    resolve_all_remaining_oxygen_high_uncertainty_sources,
    resolve_dblock_3d_oxygen_accepted_sources,
    resolve_dblock_3d_oxygen_high_uncertainty_sources,
    resolve_dblock_4d_post_oxygen_accepted_sources,
    resolve_dblock_4d_post_oxygen_high_uncertainty_sources,
)
from critmin.analysis.bond_valence_tracking import BUCKETS, empty_bucket_payload, is_cation_payload
from critmin.analysis.payload_schema import validate_grouped_payload

DEFAULT_GROUP12_ACCEPTED_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/group12_two_phase_20260410T230957Z.json",
)
DEFAULT_GROUP12_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/group12_two_phase_20260410T230957Z_high_uncertainty.json",
)
DEFAULT_DBLOCK_ACCEPTED_SOURCES: tuple[str, ...] = (
    resolve_dblock_3d_oxygen_accepted_sources()
)
DEFAULT_DBLOCK_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    resolve_dblock_3d_oxygen_high_uncertainty_sources()
)
DEFAULT_GROUP12_F_ACCEPTED_SOURCES: tuple[str, ...] = (
    "data/raw/ima/group12_f_bv_params.json",
)
DEFAULT_GROUP12_F_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    "data/raw/ima/group12_f_bv_params_high_uncertainty.json",
)
DEFAULT_LANTHANIDE_ACCEPTED_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/fblock_4f_two_phase.json",
)
DEFAULT_LANTHANIDE_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/fblock_4f_two_phase_high_uncertainty.json",
)
DEFAULT_POST_TRANSITION_ACCEPTED_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/post_transition_remaining_o_two_phase_20260414T045256Z.json",
)
DEFAULT_POST_TRANSITION_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    "data/processed/remote_jobs/post_transition_remaining_o_two_phase_20260414T045256Z_high_uncertainty.json",
)
DEFAULT_CONSOLIDATED_STORE_PATH = Path("data/processed/bond_valence/consolidated_store.json")
CONSOLIDATED_STORE_SCHEMA_VERSION = "critmin-bv-consolidated/v1"
DEFAULT_OXYGEN_AUTHORITATIVE_DATASET_KEY = "oxygen_authoritative"
DEFAULT_OXYGEN_AUTHORITATIVE_ACCEPTED_SOURCES: tuple[str, ...] = (
    *DEFAULT_GROUP12_ACCEPTED_SOURCES,
    *DEFAULT_DBLOCK_ACCEPTED_SOURCES,
    *resolve_dblock_4d_post_oxygen_accepted_sources(),
    *DEFAULT_POST_TRANSITION_ACCEPTED_SOURCES,
    *DEFAULT_LANTHANIDE_ACCEPTED_SOURCES,
    *resolve_all_remaining_oxygen_accepted_sources(),
)
DEFAULT_OXYGEN_AUTHORITATIVE_HIGH_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    *DEFAULT_GROUP12_HIGH_UNCERTAINTY_SOURCES,
    *DEFAULT_DBLOCK_HIGH_UNCERTAINTY_SOURCES,
    *resolve_dblock_4d_post_oxygen_high_uncertainty_sources(),
    *DEFAULT_POST_TRANSITION_HIGH_UNCERTAINTY_SOURCES,
    *DEFAULT_LANTHANIDE_HIGH_UNCERTAINTY_SOURCES,
    *resolve_all_remaining_oxygen_high_uncertainty_sources(),
)
DEFAULT_OXYGEN_AUTHORITATIVE_SOURCE_SPEC = (
    f"{DEFAULT_CONSOLIDATED_STORE_PATH}::{DEFAULT_OXYGEN_AUTHORITATIVE_DATASET_KEY}"
)
CANONICAL_RECORD_FIELDS: tuple[str, ...] = (
    "mid",
    "formula",
    "energy_above_hull",
    "R0",
    "B",
    "R0_std",
    "B_std",
    "n_algos",
    "cn",
    "cn_all",
    "oxi_state",
    "oxi_state_label",
    "fit_strategy",
    "fit_diagnostics",
    "status",
    "original_status",
    "schema_origin",
)
VALID_SCHEMA_ORIGINS = frozenset({"legacy", "adaptive"})
DEFAULT_CONSOLIDATED_DATASET_SPECS: dict[str, dict[str, Any]] = {
    DEFAULT_OXYGEN_AUTHORITATIVE_DATASET_KEY: {
        "title": "Authoritative oxygen union",
        "role": "authoritative",
        "description": (
            "Union of the best checked-in oxygen payloads, with first-wins "
            "precedence favoring the more targeted reruns where they overlap."
        ),
        "accepted_sources": DEFAULT_OXYGEN_AUTHORITATIVE_ACCEPTED_SOURCES,
        "high_uncertainty_sources": DEFAULT_OXYGEN_AUTHORITATIVE_HIGH_UNCERTAINTY_SOURCES,
    },
    "group12_oxygen": {
        "title": "Group 1+2 oxygen",
        "role": "component",
        "accepted_sources": DEFAULT_GROUP12_ACCEPTED_SOURCES,
        "high_uncertainty_sources": DEFAULT_GROUP12_HIGH_UNCERTAINTY_SOURCES,
    },
    "group12_fluoride": {
        "title": "Group 1+2 fluoride",
        "role": "component",
        "accepted_sources": DEFAULT_GROUP12_F_ACCEPTED_SOURCES,
        "high_uncertainty_sources": DEFAULT_GROUP12_F_HIGH_UNCERTAINTY_SOURCES,
    },
    "dblock_oxygen": {
        "title": "D-block 3d oxygen",
        "role": "component",
        "accepted_sources": DEFAULT_DBLOCK_ACCEPTED_SOURCES,
        "high_uncertainty_sources": DEFAULT_DBLOCK_HIGH_UNCERTAINTY_SOURCES,
    },
}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _record_identity(record: Mapping[str, Any]) -> str:
    mid = record.get("mid")
    if mid:
        return str(mid)
    return json.dumps(dict(record), sort_keys=True)


def _sorted_bucket_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted((dict(record) for record in records), key=lambda record: str(record.get("mid", "")))


def harmonize_bv_record(
    record: Mapping[str, Any],
    *,
    default_status: str | None = None,
) -> dict[str, Any]:
    """Return one record normalized to the canonical mixed legacy/adaptive schema."""
    copied = _json_copy(dict(record))
    fit_strategy = copied.get("fit_strategy")
    if fit_strategy is not None:
        fit_strategy = str(fit_strategy)
    fit_diagnostics = copied.get("fit_diagnostics")
    if not isinstance(fit_diagnostics, Mapping):
        fit_diagnostics = None
    else:
        fit_diagnostics = _json_copy(dict(fit_diagnostics))

    schema_origin = copied.get("schema_origin")
    if schema_origin is None:
        schema_origin = (
            "adaptive"
            if fit_strategy is not None or fit_diagnostics is not None
            else "legacy"
        )
    else:
        schema_origin = str(schema_origin)

    cn, cn_all = resolve_tracker_coordination(
        cn_mode=copied.get("cn"),
        cn_all=copied.get("cn_all"),
    )

    harmonized: dict[str, Any] = {
        "mid": copied.get("mid"),
        "formula": copied.get("formula"),
        "energy_above_hull": copied.get("energy_above_hull"),
        "R0": copied.get("R0"),
        "B": copied.get("B"),
        "R0_std": copied.get("R0_std"),
        "B_std": copied.get("B_std"),
        "n_algos": copied.get("n_algos"),
        "cn": cn,
        "cn_all": cn_all,
        "oxi_state": copied.get("oxi_state"),
        "oxi_state_label": copied.get("oxi_state_label"),
        "fit_strategy": fit_strategy,
        "fit_diagnostics": fit_diagnostics,
        "status": copied.get("status", default_status),
        "original_status": copied.get("original_status"),
        "schema_origin": schema_origin,
    }
    for key, value in copied.items():
        if key not in harmonized:
            harmonized[key] = value
    return harmonized


def harmonize_grouped_payload(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    default_status: str | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return a grouped payload with canonical record fields and bucket layout."""
    harmonized: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cation, cation_payload in payload.items():
        if not is_cation_payload(cation_payload):
            continue
        cat = str(cation)
        harmonized[cat] = empty_bucket_payload()
        for bucket in BUCKETS:
            records = cation_payload.get(bucket, [])
            harmonized[cat][bucket] = _sorted_bucket_records(
                harmonize_bv_record(record, default_status=default_status)
                for record in records
            )
    return harmonized


def load_harmonized_grouped_payload_json(
    path: str | Path,
    *,
    default_status: str | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load one grouped payload file and normalize it into the canonical schema."""
    payload_path = Path(path)
    if not payload_path.exists():
        return {}

    data = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return harmonize_grouped_payload(data, default_status=default_status)


def load_grouped_payload_source(
    source: str | Path,
    *,
    default_status: str | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load one grouped payload source.

    ``source`` may be either a plain payload JSON path or a consolidated-store
    reference of the form ``store.json::dataset_key``.
    """
    source_text = str(source)
    if "::" in source_text:
        store_path_text, dataset_key = source_text.split("::", 1)
        return load_consolidated_dataset_payload(dataset_key, path=store_path_text)
    return load_harmonized_grouped_payload_json(source, default_status=default_status)


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

    for cation, cation_payload in merged.items():
        for bucket in BUCKETS:
            cation_payload[bucket] = _sorted_bucket_records(cation_payload[bucket])
    return merged


def _format_validation_errors(errors: Mapping[str, Mapping[str, Sequence[str]]]) -> str:
    parts: list[str] = []
    for cation, cation_errors in errors.items():
        for location, messages in cation_errors.items():
            joined = "; ".join(str(message) for message in messages)
            parts.append(f"{cation}/{location}: {joined}")
            if len(parts) >= 8:
                return " | ".join(parts)
    return " | ".join(parts)


def summarize_grouped_payload(
    payload: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    """Return dataset-level counts and field coverage for one grouped payload."""
    total_records = 0
    status_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    schema_origin_counts: Counter[str] = Counter()
    present_counts: Counter[str] = Counter()
    coverage_fields = (
        "energy_above_hull",
        "cn",
        "oxi_state",
        "oxi_state_label",
        "fit_strategy",
        "fit_diagnostics",
        "original_status",
    )

    for _cation, cation_payload in payload.items():
        for bucket in BUCKETS:
            for record in cation_payload.get(bucket, []):
                total_records += 1
                bucket_counts[bucket] += 1
                status_counts[str(record.get("status"))] += 1
                schema_origin_counts[str(record.get("schema_origin"))] += 1
                for field in coverage_fields:
                    value = record.get(field)
                    if field == "fit_diagnostics":
                        if isinstance(value, Mapping):
                            present_counts[field] += 1
                        continue
                    if value is not None:
                        present_counts[field] += 1

    field_coverage = {
        field: {
            "present": int(present_counts.get(field, 0)),
            "coverage": (
                float(present_counts.get(field, 0)) / float(total_records)
                if total_records
                else 0.0
            ),
        }
        for field in coverage_fields
    }
    return {
        "cation_count": len(payload),
        "cation_names": sorted(str(cation) for cation in payload),
        "records_total": total_records,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "schema_origin_counts": dict(sorted(schema_origin_counts.items())),
        "field_coverage": field_coverage,
    }


def build_consolidated_dataset(
    *,
    dataset_key: str,
    title: str,
    role: str | None,
    description: str | None,
    accepted_sources: Sequence[str | Path],
    high_uncertainty_sources: Sequence[str | Path],
) -> dict[str, Any]:
    """Build one canonical dataset entry for the consolidated store."""
    payloads: list[dict[str, dict[str, list[dict[str, Any]]]]] = []
    for source in accepted_sources:
        payloads.append(load_harmonized_grouped_payload_json(source, default_status="fitted"))
    for source in high_uncertainty_sources:
        payloads.append(
            load_harmonized_grouped_payload_json(source, default_status="high_uncertainty")
        )
    merged = merge_grouped_payloads_first_wins(*payloads)
    errors = validate_grouped_payload(merged)
    if errors:
        raise ValueError(
            f"Consolidated dataset {dataset_key!r} failed validation: "
            f"{_format_validation_errors(errors)}"
        )

    return {
        "title": str(title),
        "role": str(role) if role is not None else "component",
        "description": str(description) if description is not None else None,
        "sources": {
            "accepted": [str(source) for source in accepted_sources],
            "high_uncertainty": [str(source) for source in high_uncertainty_sources],
        },
        "summary": summarize_grouped_payload(merged),
        "payload": merged,
    }


def build_consolidated_store(
    *,
    dataset_keys: Sequence[str] | None = None,
    dataset_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the canonical single-source BV data store."""
    specs = dict(dataset_specs or DEFAULT_CONSOLIDATED_DATASET_SPECS)
    selected_keys = list(dataset_keys) if dataset_keys is not None else list(specs)
    datasets: dict[str, Any] = {}

    for dataset_key in selected_keys:
        if dataset_key not in specs:
            raise KeyError(f"Unknown consolidated dataset: {dataset_key}")
        spec = specs[dataset_key]
        datasets[dataset_key] = build_consolidated_dataset(
            dataset_key=dataset_key,
            title=str(spec.get("title", dataset_key)),
            role=str(spec.get("role")) if spec.get("role") is not None else None,
            description=(
                str(spec.get("description")) if spec.get("description") is not None else None
            ),
            accepted_sources=spec.get("accepted_sources", ()),
            high_uncertainty_sources=spec.get("high_uncertainty_sources", ()),
        )

    return {
        "schema_version": CONSOLIDATED_STORE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "record_fields": list(CANONICAL_RECORD_FIELDS),
        "datasets": datasets,
    }


def load_consolidated_store(path: str | Path = DEFAULT_CONSOLIDATED_STORE_PATH) -> dict[str, Any]:
    """Load the canonical consolidated store JSON file."""
    store_path = Path(path)
    return json.loads(store_path.read_text(encoding="utf-8"))


def load_consolidated_dataset_payload(
    dataset_key: str,
    *,
    path: str | Path = DEFAULT_CONSOLIDATED_STORE_PATH,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return the grouped payload for one dataset from the consolidated store."""
    store = load_consolidated_store(path)
    datasets = store.get("datasets", {})
    if dataset_key not in datasets:
        raise KeyError(f"Dataset {dataset_key!r} not found in consolidated store")
    payload = datasets[dataset_key].get("payload", {})
    return harmonize_grouped_payload(payload)


def export_consolidated_payload_store(
    *,
    output_path: str | Path = DEFAULT_CONSOLIDATED_STORE_PATH,
    dataset_keys: Sequence[str] | None = None,
    dataset_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the canonical consolidated store and return a compact manifest."""
    store = build_consolidated_store(dataset_keys=dataset_keys, dataset_specs=dataset_specs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")

    datasets = {
        dataset_key: {
            "title": dataset["title"],
            "role": dataset.get("role"),
            "records_total": dataset["summary"]["records_total"],
            "status_counts": dataset["summary"]["status_counts"],
            "schema_origin_counts": dataset["summary"]["schema_origin_counts"],
        }
        for dataset_key, dataset in store["datasets"].items()
    }
    return {
        "output": str(output),
        "schema_version": store["schema_version"],
        "generated_at": store["generated_at"],
        "datasets": datasets,
    }
