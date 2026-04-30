#!/usr/bin/env python3
"""Build the machine-readable FAIR artifact catalog for the repository."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "metadata" / "artifact_catalog.json"
REPOSITORY_URL = "https://github.com/MWhitta/YukawaScreening"
MP_TERMS = "Materials Project Terms of Use (redistributed derived data); see metadata/DATA_LICENSES.md"


ARTIFACT_SPECS: list[dict[str, Any]] = [
    {
        "id": "ima_group1_candidates",
        "path": "data/raw/ima/group1_candidates.json",
        "title": "Group 1 IMA candidate tracker",
        "category": "raw-tracker",
        "description": "IMA-derived candidate tracker for Group 1 oxygen-containing materials queried from the Materials Project.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "ima_group2_candidates",
        "path": "data/raw/ima/group2_candidates.json",
        "title": "Group 2 IMA candidate tracker",
        "category": "raw-tracker",
        "description": "IMA-derived candidate tracker for Group 2 oxygen-containing materials queried from the Materials Project.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "ima_group12_candidates",
        "path": "data/raw/ima/group12_candidates.json",
        "title": "Group 1+2 combined IMA candidate tracker",
        "category": "raw-tracker",
        "description": "Merged Group 1 / Group 2 candidate tracker for the oxygen-authoritative screening workflow.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "ima_dblock_3d_candidates",
        "path": "data/raw/ima/dblock_3d_candidates.json",
        "title": "3d transition-metal candidate tracker",
        "category": "raw-tracker",
        "description": "IMA-derived candidate tracker for 3d transition-metal oxygen materials queried from the Materials Project.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "ima_dblock_4d_post_candidates",
        "path": "data/raw/ima/dblock_4d_post_o_candidates_20260412T155839Z.json",
        "title": "4d / post-transition oxygen candidate tracker",
        "category": "raw-tracker",
        "description": "Candidate tracker for the 4d transition-metal and post-transition oxygen expansion phase of the manuscript workflow.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "ima_all_remaining_oxygen_candidates",
        "path": "data/raw/ima/all_remaining_o_candidates_20260413T030533Z.json",
        "title": "All-remaining oxygen candidate tracker",
        "category": "raw-tracker",
        "description": "Candidate tracker for the remaining oxygen-bearing cation families included in the master manuscript summary.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "ima_post_transition_remaining_candidates",
        "path": "data/raw/ima/post_transition_remaining_o_candidates_20260414T045256Z.json",
        "title": "Post-transition remaining oxygen candidate tracker",
        "category": "raw-tracker",
        "description": "Candidate tracker for the late-stage post-transition oxygen family expansion in the manuscript workflow.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "ima_fblock_4f_candidates",
        "path": "data/raw/ima/fblock_4f_candidates.json",
        "title": "Lanthanide candidate tracker",
        "category": "raw-tracker",
        "description": "Candidate tracker for 4f-block oxygen-containing materials used in the lanthanide manuscript summary.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker",
    },
    {
        "id": "oxygen_authoritative_store",
        "path": "data/processed/bond_valence/consolidated_store.json",
        "title": "Oxygen-authoritative consolidated bond-valence store",
        "category": "processed-dataset",
        "description": "Canonical grouped payload containing per-structure (R0, B) fits and diagnostics for the oxygen-authoritative manuscript workflow.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "analysis/critmin/analysis/payload_schema.py::validate_grouped_payload",
        "generated_by": [
            "analysis/critmin/analysis/bond_valence_store.py"
        ],
    },
    {
        "id": "group12_unified_oxygen_theory",
        "path": "data/processed/theory/group12_unified_oxygen_theory.json",
        "title": "Group 1 / Group 2 unified oxygen theory summary",
        "category": "processed-dataset",
        "description": "Coordination-number-resolved fit summary for the Group 1 and Group 2 manuscript families.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#group12_unified_oxygen_theoryjson",
        "generated_by": [
            "analysis/critmin/analysis/bond_valence_theory.py"
        ],
    },
    {
        "id": "dblock_oxi_unified_oxygen_theory",
        "path": "data/processed/theory/dblock_oxi_unified_oxygen_theory.json",
        "title": "d-block oxidation-state-resolved oxygen theory summary",
        "category": "processed-dataset",
        "description": "Coordination-number-resolved fit summary for the oxidation-state-resolved d-block manuscript families.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#dblock_oxi_unified_oxygen_theoryjson",
        "generated_by": [
            "analysis/critmin/analysis/bond_valence_theory.py"
        ],
    },
    {
        "id": "master_oxygen_summary_theory",
        "path": "data/processed/theory/master_oxygen_summary_theory.json",
        "title": "Master oxygen summary theory payload",
        "category": "processed-dataset",
        "description": "Authoritative 103-species manuscript summary with characteristic pairs, branch assignment, and family-level summaries.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#master_oxygen_summary_theoryjson",
        "generated_by": [
            "analysis/critmin/analysis/bond_valence_theory.py",
            "analysis/scripts/export_theory_lambda_tables.py"
        ],
    },
    {
        "id": "manuscript_theory_manifest",
        "path": "data/processed/theory/manuscript_theory_manifest.json",
        "title": "Manuscript theory manifest",
        "category": "processed-dataset",
        "description": "Compact manifest for the manuscript theory exports, including file paths, source trackers, and aggregate counts.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#manuscript_theory_manifestjson",
        "generated_by": [
            "analysis/critmin/analysis/bond_valence_theory.py"
        ],
    },
    {
        "id": "lambda_mixed_cation_holdout_prediction_json",
        "path": "data/processed/theory/lambda_mixed_cation_holdout_prediction.json",
        "title": "Mixed-cation lambda holdout prediction payload",
        "category": "processed-dataset",
        "description": "JSON export for the manuscript's mixed-cation holdout prediction experiment.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#lambda_mixed_cation_holdout_predictionjson",
        "generated_by": [
            "analysis/critmin/analysis/shell_thermodynamics.py"
        ],
    },
    {
        "id": "lambda_mixed_cation_holdout_prediction_csv",
        "path": "data/processed/theory/lambda_mixed_cation_holdout_prediction.csv",
        "title": "Mixed-cation lambda holdout prediction table",
        "category": "processed-dataset",
        "description": "CSV summary of the material-level mixed-cation holdout prediction results.",
        "format": "csv",
        "media_type": "text/csv",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#lambda_mixed_cation_holdout_predictioncsv",
        "generated_by": [
            "analysis/critmin/analysis/shell_thermodynamics.py"
        ],
    },
    {
        "id": "charge_density_benchmark",
        "path": "data/processed/theory/charge_density_benchmark.json",
        "title": "Charge-density screening-centroid benchmark",
        "category": "processed-dataset",
        "description": "Checked-in point set and regression summary for PRL Fig. 2 and the SI null-model control.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#charge_density_benchmarkjson",
        "generated_by": [
            "analysis/notebooks/thomas_fermi_screening.ipynb",
            "analysis/scripts/make_charge_density_benchmark_figures.py"
        ],
    },
    {
        "id": "charge_density_mp_id_manifest",
        "path": "data/charge_density/mp_id_manifest.json",
        "title": "Charge-density Materials Project ID manifest",
        "category": "external-manifest",
        "description": "Materials Project identifier manifest for the charge-density workflow used to derive the manuscript's screening-centroid benchmark.",
        "format": "json",
        "media_type": "application/json",
        "license": MP_TERMS,
        "schema_reference": "metadata/DATA_DICTIONARY.md#mp_id_manifestjson",
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_tracker_candidates(payload: Any) -> int:
    def is_bucket_block(value: Any) -> bool:
        return isinstance(value, dict) and all(bucket in value for bucket in ("oxides", "hydroxides"))

    total = 0
    if isinstance(payload, dict) and is_bucket_block(payload):
        blocks = [payload]
    elif isinstance(payload, dict):
        blocks = [value for value in payload.values() if is_bucket_block(value)]
    else:
        blocks = []

    for block in blocks:
        for bucket in ("oxides", "hydroxides"):
            bucket_data = block.get(bucket, {})
            if isinstance(bucket_data, dict):
                candidates = bucket_data.get("candidates", [])
            else:
                candidates = bucket_data
            if isinstance(candidates, list):
                total += len(candidates)
    return total


def _csv_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    header = rows[0] if rows else []
    return {
        "header": header,
        "row_count": max(len(rows) - 1, 0),
    }


def _json_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"top_level_type": type(payload).__name__}
    if isinstance(payload, dict):
        summary["top_level_keys"] = list(payload.keys())
        if "_candidates" in path.name:
            summary["candidate_count"] = _count_tracker_candidates(payload)
        if "master_rows" in payload:
            summary["master_row_count"] = len(payload["master_rows"])
        if "excluded_rows" in payload:
            summary["excluded_row_count"] = len(payload["excluded_rows"])
        if "points" in payload:
            summary["point_count"] = len(payload["points"])
        if "records" in payload:
            summary["record_count"] = len(payload["records"])
        if "shell_prediction_rows" in payload:
            summary["shell_prediction_row_count"] = len(payload["shell_prediction_rows"])
        if "material_prediction_rows" in payload:
            summary["material_prediction_row_count"] = len(payload["material_prediction_rows"])
        if path.name == "consolidated_store.json":
            datasets = payload.get("datasets", {})
            oxygen_store = datasets.get("oxygen_authoritative", {}).get("payload", {})
            summary["dataset_keys"] = list(datasets.keys())
            summary["oxygen_authoritative_cation_count"] = len(oxygen_store)
            summary["oxygen_authoritative_record_count"] = sum(
                len(block.get("oxides", [])) + len(block.get("hydroxides", []))
                for block in oxygen_store.values()
            )
    return summary


def _artifact_entry(spec: dict[str, Any]) -> dict[str, Any]:
    rel_path = Path(spec["path"])
    abs_path = REPO_ROOT / rel_path
    entry = dict(spec)
    entry["path"] = rel_path.as_posix()
    entry["size_bytes"] = abs_path.stat().st_size
    entry["sha256"] = _sha256(abs_path)
    if rel_path.suffix == ".json":
        entry["content_summary"] = _json_summary(abs_path)
    elif rel_path.suffix == ".csv":
        entry["content_summary"] = _csv_summary(abs_path)
    return entry


def build_catalog() -> dict[str, Any]:
    return {
        "title": "YukawaScreening FAIR artifact catalog",
        "repository": {
            "name": "YukawaScreening",
            "url": REPOSITORY_URL,
            "license": "MIT for code and manuscript text; data terms are documented separately",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(ARTIFACT_SPECS),
        "artifacts": [_artifact_entry(spec) for spec in ARTIFACT_SPECS],
    }


def main() -> None:
    catalog = build_catalog()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
