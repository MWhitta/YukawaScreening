"""JSON payload schema validation and integrity verification for BV payloads.

Provides dataclass-based schema definitions for bond-valence payload records,
grouped payloads, and candidate tracker entries, plus SHA256 manifest
operations for payload integrity checking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Schema definitions ───────────────────────────────────────────────────────

_BUCKETS = ("oxides", "hydroxides")

_VALID_STATUSES = frozenset({
    "fitted", "high_uncertainty", "boundary_capped", "completed",
    "ready", "no_species",
})
_VALID_SCHEMA_ORIGINS = frozenset({"legacy", "adaptive"})


@dataclass(slots=True)
class BVRecordSchema:
    """Expected fields for a single bond-valence fit record."""

    mid: str
    formula: str
    R0: float
    B: float
    R0_std: float
    B_std: float
    n_algos: int
    status: str
    energy_above_hull: float | None = None
    cn: int | None = None
    cn_all: list[int] = field(default_factory=list)
    original_status: str | None = None
    oxi_state: int | None = None
    oxi_state_label: str | None = None
    fit_strategy: str | None = None
    fit_diagnostics: dict[str, Any] | None = None
    schema_origin: str | None = None


@dataclass(slots=True)
class CandidateSchema:
    """Expected fields for a candidate tracker entry."""

    mid: str
    status: str
    cn_mode: int | None = None
    cn_all: list[int] = field(default_factory=list)
    oxi_state: int | None = None
    oxi_state_label: str | None = None


# ── Validation functions ─────────────────────────────────────────────────────


def validate_bv_record(record: dict[str, Any]) -> list[str]:
    """Validate a single BV fit record dict, returning a list of error strings."""
    errors: list[str] = []

    # Required string fields
    for key in ("mid", "formula"):
        val = record.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"missing or empty required field '{key}'")

    # Required numeric fields
    for key in ("R0", "B", "R0_std", "B_std"):
        val = record.get(key)
        if val is None:
            errors.append(f"missing required numeric field '{key}'")
        elif not isinstance(val, (int, float)):
            errors.append(f"'{key}' must be numeric, got {type(val).__name__}")

    # n_algos
    n_algos = record.get("n_algos")
    if n_algos is None:
        errors.append("missing required field 'n_algos'")
    elif not isinstance(n_algos, int) or n_algos < 1:
        errors.append(f"'n_algos' must be a positive integer, got {n_algos!r}")

    # status
    status = record.get("status")
    if not isinstance(status, str) or not status.strip():
        errors.append("missing or empty 'status' field")
    elif status not in _VALID_STATUSES:
        errors.append(f"unknown status '{status}', expected one of {sorted(_VALID_STATUSES)}")

    # Optional cn fields
    cn = record.get("cn")
    if cn is not None and not isinstance(cn, int):
        errors.append(f"'cn' must be int or null, got {type(cn).__name__}")
    energy_above_hull = record.get("energy_above_hull")
    if energy_above_hull is not None and not isinstance(energy_above_hull, (int, float)):
        errors.append(
            f"'energy_above_hull' must be numeric or null, got {type(energy_above_hull).__name__}"
        )
    cn_all = record.get("cn_all")
    if cn_all is not None:
        if not isinstance(cn_all, list):
            errors.append(f"'cn_all' must be a list, got {type(cn_all).__name__}")
        elif not all(isinstance(v, int) for v in cn_all):
            errors.append("'cn_all' must contain only integers")

    # Uncertainty sanity checks
    for key in ("R0_std", "B_std"):
        val = record.get(key)
        if isinstance(val, (int, float)) and val < 0:
            errors.append(f"'{key}' must be non-negative, got {val}")

    # Optional oxidation-state fields
    oxi_state = record.get("oxi_state")
    if oxi_state is not None and not isinstance(oxi_state, int):
        errors.append(f"'oxi_state' must be int or null, got {type(oxi_state).__name__}")
    oxi_label = record.get("oxi_state_label")
    if oxi_label is not None and oxi_label not in ("pure", "mixed"):
        errors.append(
            f"'oxi_state_label' must be 'pure' or 'mixed', got {oxi_label!r}"
        )

    fit_strategy = record.get("fit_strategy")
    if fit_strategy is not None and not isinstance(fit_strategy, str):
        errors.append(
            f"'fit_strategy' must be str or null, got {type(fit_strategy).__name__}"
        )
    fit_diagnostics = record.get("fit_diagnostics")
    if fit_diagnostics is not None and not isinstance(fit_diagnostics, dict):
        errors.append(
            f"'fit_diagnostics' must be a dict or null, got {type(fit_diagnostics).__name__}"
        )
    schema_origin = record.get("schema_origin")
    if schema_origin is not None and schema_origin not in _VALID_SCHEMA_ORIGINS:
        errors.append(
            f"'schema_origin' must be one of {sorted(_VALID_SCHEMA_ORIGINS)}, got {schema_origin!r}"
        )

    return errors


def validate_cation_payload(
    payload: dict[str, Any],
) -> dict[str, list[str]]:
    """Validate a single-cation payload (``{"oxides": [...], "hydroxides": [...]}``).

    Returns a dict mapping ``"<bucket>/<index>"`` to error lists. Empty dict
    means the payload is valid.
    """
    all_errors: dict[str, list[str]] = {}

    for bucket in _BUCKETS:
        records = payload.get(bucket)
        if records is None:
            all_errors[bucket] = [f"missing required bucket '{bucket}'"]
            continue
        if not isinstance(records, list):
            all_errors[bucket] = [f"'{bucket}' must be a list, got {type(records).__name__}"]
            continue
        for i, record in enumerate(records):
            errs = validate_bv_record(record)
            if errs:
                all_errors[f"{bucket}/{i}"] = errs

    return all_errors


def validate_grouped_payload(
    payload: dict[str, Any],
) -> dict[str, dict[str, list[str]]]:
    """Validate a grouped payload (``{"Li": {"oxides": [...], ...}, ...}``).

    Returns a nested dict: ``{cation: {location: [errors]}}``.
    Empty dict means valid.
    """
    all_errors: dict[str, dict[str, list[str]]] = {}

    if not isinstance(payload, dict):
        return {"_root": {"_type": [f"payload must be a dict, got {type(payload).__name__}"]}}

    for cation, cation_payload in payload.items():
        if not isinstance(cation_payload, dict):
            all_errors[cation] = {
                "_type": [f"cation block must be a dict, got {type(cation_payload).__name__}"]
            }
            continue
        errs = validate_cation_payload(cation_payload)
        if errs:
            all_errors[cation] = errs

    return all_errors


def validate_candidate_tracker(
    tracker: dict[str, Any],
) -> list[str]:
    """Validate a candidate tracker JSON structure.

    Returns a list of error strings. Empty list means valid.
    """
    errors: list[str] = []

    if not isinstance(tracker, dict):
        return [f"tracker must be a dict, got {type(tracker).__name__}"]

    # Can be either single-cation (has "cation" key) or grouped
    if "cation" in tracker:
        # Single-cation tracker
        cation = tracker.get("cation")
        if not isinstance(cation, str) or not cation.strip():
            errors.append("'cation' must be a non-empty string")
        errors.extend(_validate_tracker_cation_block(tracker))
    else:
        # Grouped tracker — each key is a cation
        for cation, block in tracker.items():
            if not isinstance(cation, str) or not cation.strip():
                errors.append("grouped tracker cation keys must be non-empty strings")
            if not isinstance(block, dict):
                errors.append(f"cation '{cation}' block must be a dict")
                continue
            errors.extend(_validate_tracker_cation_block(block, prefix=f"{cation}/"))

    return errors


def _validate_tracker_cation_block(
    block: dict[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    errors: list[str] = []
    for bucket in _BUCKETS:
        if bucket not in block:
            errors.append(f"'{prefix}{bucket}' missing required bucket")
            continue

        bucket_block = block[bucket]
        if not isinstance(bucket_block, dict):
            errors.append(f"'{prefix}{bucket}' must be a dict")
            continue

        candidates = bucket_block.get("candidates", [])
        if not isinstance(candidates, list):
            errors.append(f"'{prefix}{bucket}/candidates' must be a list")
            continue

        for i, candidate in enumerate(candidates):
            candidate_prefix = f"'{prefix}{bucket}/candidates/{i}'"
            if not isinstance(candidate, dict):
                errors.append(f"{candidate_prefix} must be a dict")
                continue

            mid = candidate.get("mid")
            if not isinstance(mid, str) or not mid.strip():
                errors.append(f"{candidate_prefix} missing 'mid'")

            status = candidate.get("status")
            if not isinstance(status, str) or not status.strip():
                errors.append(f"{candidate_prefix} missing 'status'")
            elif status not in _VALID_STATUSES:
                errors.append(f"{candidate_prefix} unknown status '{status}'")

            cn_mode = candidate.get("cn_mode")
            if cn_mode is not None and not isinstance(cn_mode, int):
                errors.append(f"{candidate_prefix} 'cn_mode' must be an int or null")

            cn_all = candidate.get("cn_all")
            if cn_all is not None:
                if not isinstance(cn_all, list):
                    errors.append(f"{candidate_prefix} 'cn_all' must be a list or null")
                elif not all(isinstance(value, int) for value in cn_all):
                    errors.append(f"{candidate_prefix} 'cn_all' must contain only integers")

            oxi_state = candidate.get("oxi_state")
            if oxi_state is not None and not isinstance(oxi_state, int):
                errors.append(f"{candidate_prefix} 'oxi_state' must be an int or null")

            oxi_state_label = candidate.get("oxi_state_label")
            if oxi_state_label is not None and oxi_state_label not in ("pure", "mixed"):
                errors.append(
                    f"{candidate_prefix} 'oxi_state_label' must be 'pure' or 'mixed'"
                )

    return errors


# ── SHA256 payload manifest ──────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_payload_hash(path: str | Path) -> str:
    """Compute SHA256 hash of a payload file."""
    return _sha256(Path(path).read_bytes())


def load_payload_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load an existing payload manifest, or return empty structure."""
    p = Path(manifest_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"files": {}}


def save_payload_manifest(manifest: dict[str, Any], manifest_path: str | Path) -> None:
    """Write the payload manifest atomically."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def register_payload(
    manifest: dict[str, Any],
    payload_path: str | Path,
    *,
    description: str = "",
) -> dict[str, Any]:
    """Register a payload file in the manifest with its SHA256 hash.

    Returns the updated manifest (mutated in place).
    """
    p = Path(payload_path)
    manifest.setdefault("files", {})[p.name] = {
        "sha256": compute_payload_hash(p),
        "size_bytes": p.stat().st_size,
        "description": description,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    return manifest


def verify_payload_integrity(
    manifest: dict[str, Any],
    data_dir: str | Path,
) -> dict[str, str]:
    """Verify all registered payloads against their manifest hashes.

    Returns a dict mapping filename to status:
    ``"ok"``, ``"hash_mismatch"``, or ``"missing"``.
    """
    results: dict[str, str] = {}
    data_path = Path(data_dir)

    for filename, info in manifest.get("files", {}).items():
        filepath = data_path / filename
        if not filepath.exists():
            results[filename] = "missing"
            continue

        current_hash = compute_payload_hash(filepath)
        expected_hash = info.get("sha256", "")
        results[filename] = "ok" if current_hash == expected_hash else "hash_mismatch"

    return results
