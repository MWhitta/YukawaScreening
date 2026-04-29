"""Default payload source resolution helpers.

These helpers keep manuscript-facing workflows pointed at the newest
timestamped artifacts when a refreshed dataset exists, while preserving the
legacy checked-in payloads as a fallback.
"""

from __future__ import annotations

import glob
from pathlib import Path

LEGACY_DBLOCK_3D_OXYGEN_ACCEPTED_SOURCE = "data/raw/ima/dblock_3d_bv_params.json"
LEGACY_DBLOCK_3D_OXYGEN_HIGH_UNCERTAINTY_SOURCE = (
    "data/raw/ima/dblock_3d_bv_params_high_uncertainty.json"
)
DBLOCK_3D_OXYGEN_REMOTE_PREFIX = "data/processed/remote_jobs/dblock_3d_o_two_phase_"
LEGACY_DBLOCK_4D_POST_OXYGEN_ACCEPTED_SOURCE = (
    "data/processed/remote_jobs/dblock_4d_post_o_two_phase_20260412T155839Z.json"
)
LEGACY_DBLOCK_4D_POST_OXYGEN_HIGH_UNCERTAINTY_SOURCE = (
    "data/processed/remote_jobs/dblock_4d_post_o_two_phase_20260412T155839Z_high_uncertainty.json"
)
DBLOCK_4D_POST_OXYGEN_REMOTE_PREFIX = (
    "data/processed/remote_jobs/dblock_4d_post_o_two_phase_"
)
LEGACY_ALL_REMAINING_OXYGEN_ACCEPTED_SOURCE = (
    "data/processed/remote_jobs/all_remaining_o_two_phase_20260413T030533Z.json"
)
LEGACY_ALL_REMAINING_OXYGEN_HIGH_UNCERTAINTY_SOURCE = (
    "data/processed/remote_jobs/all_remaining_o_two_phase_20260413T030533Z_high_uncertainty.json"
)
ALL_REMAINING_OXYGEN_REMOTE_PREFIX = (
    "data/processed/remote_jobs/all_remaining_o_two_phase_"
)


def latest_timestamped_payload(prefix: str) -> str | None:
    """Return the latest timestamped JSON payload for ``prefix``.

    Companion high-uncertainty and phase-A checkpoint files are excluded.
    """
    matches = sorted(glob.glob(f"{prefix}*.json"))
    filtered = [
        match
        for match in matches
        if not match.endswith("_high_uncertainty.json")
        and not match.endswith(".phase_a.json")
    ]
    return filtered[-1] if filtered else None


def _high_uncertainty_companion(path: str) -> str:
    payload_path = Path(path)
    return str(payload_path.with_name(f"{payload_path.stem}_high_uncertainty{payload_path.suffix}"))


def resolve_timestamped_accepted_sources(
    *,
    payload_prefix: str,
    legacy_source: str,
) -> tuple[str, ...]:
    """Return the preferred accepted payload source for a timestamped series."""
    latest = latest_timestamped_payload(payload_prefix)
    return (latest or legacy_source,)


def resolve_timestamped_high_uncertainty_sources(
    *,
    payload_prefix: str,
    legacy_source: str,
) -> tuple[str, ...]:
    """Return the preferred high-uncertainty payload source for a series.

    If a refreshed accepted payload exists but its companion high-uncertainty
    file has not been written yet, return no preferred high-uncertainty source
    rather than silently mixing generations.
    """
    latest = latest_timestamped_payload(payload_prefix)
    if latest is None:
        return (legacy_source,)

    companion = _high_uncertainty_companion(latest)
    if Path(companion).exists():
        return (companion,)
    return ()


def resolve_dblock_3d_oxygen_accepted_sources(
    *,
    payload_prefix: str = DBLOCK_3D_OXYGEN_REMOTE_PREFIX,
    legacy_source: str = LEGACY_DBLOCK_3D_OXYGEN_ACCEPTED_SOURCE,
) -> tuple[str, ...]:
    """Return the preferred accepted 3d-oxygen payload source."""
    return resolve_timestamped_accepted_sources(
        payload_prefix=payload_prefix,
        legacy_source=legacy_source,
    )


def resolve_dblock_3d_oxygen_high_uncertainty_sources(
    *,
    payload_prefix: str = DBLOCK_3D_OXYGEN_REMOTE_PREFIX,
    legacy_source: str = LEGACY_DBLOCK_3D_OXYGEN_HIGH_UNCERTAINTY_SOURCE,
) -> tuple[str, ...]:
    """Return the preferred high-uncertainty 3d-oxygen payload source."""
    return resolve_timestamped_high_uncertainty_sources(
        payload_prefix=payload_prefix,
        legacy_source=legacy_source,
    )


def resolve_dblock_4d_post_oxygen_accepted_sources(
    *,
    payload_prefix: str = DBLOCK_4D_POST_OXYGEN_REMOTE_PREFIX,
    legacy_source: str = LEGACY_DBLOCK_4D_POST_OXYGEN_ACCEPTED_SOURCE,
) -> tuple[str, ...]:
    """Return the preferred accepted 4d/post-transition oxygen payload source."""
    return resolve_timestamped_accepted_sources(
        payload_prefix=payload_prefix,
        legacy_source=legacy_source,
    )


def resolve_dblock_4d_post_oxygen_high_uncertainty_sources(
    *,
    payload_prefix: str = DBLOCK_4D_POST_OXYGEN_REMOTE_PREFIX,
    legacy_source: str = LEGACY_DBLOCK_4D_POST_OXYGEN_HIGH_UNCERTAINTY_SOURCE,
) -> tuple[str, ...]:
    """Return the preferred high-uncertainty 4d/post-transition oxygen payload source."""
    return resolve_timestamped_high_uncertainty_sources(
        payload_prefix=payload_prefix,
        legacy_source=legacy_source,
    )


def resolve_all_remaining_oxygen_accepted_sources(
    *,
    payload_prefix: str = ALL_REMAINING_OXYGEN_REMOTE_PREFIX,
    legacy_source: str = LEGACY_ALL_REMAINING_OXYGEN_ACCEPTED_SOURCE,
) -> tuple[str, ...]:
    """Return the preferred accepted all-remaining oxygen payload source."""
    return resolve_timestamped_accepted_sources(
        payload_prefix=payload_prefix,
        legacy_source=legacy_source,
    )


def resolve_all_remaining_oxygen_high_uncertainty_sources(
    *,
    payload_prefix: str = ALL_REMAINING_OXYGEN_REMOTE_PREFIX,
    legacy_source: str = LEGACY_ALL_REMAINING_OXYGEN_HIGH_UNCERTAINTY_SOURCE,
) -> tuple[str, ...]:
    """Return the preferred high-uncertainty all-remaining oxygen payload source."""
    return resolve_timestamped_high_uncertainty_sources(
        payload_prefix=payload_prefix,
        legacy_source=legacy_source,
    )


__all__ = [
    "ALL_REMAINING_OXYGEN_REMOTE_PREFIX",
    "DBLOCK_4D_POST_OXYGEN_REMOTE_PREFIX",
    "DBLOCK_3D_OXYGEN_REMOTE_PREFIX",
    "LEGACY_ALL_REMAINING_OXYGEN_ACCEPTED_SOURCE",
    "LEGACY_ALL_REMAINING_OXYGEN_HIGH_UNCERTAINTY_SOURCE",
    "LEGACY_DBLOCK_4D_POST_OXYGEN_ACCEPTED_SOURCE",
    "LEGACY_DBLOCK_4D_POST_OXYGEN_HIGH_UNCERTAINTY_SOURCE",
    "LEGACY_DBLOCK_3D_OXYGEN_ACCEPTED_SOURCE",
    "LEGACY_DBLOCK_3D_OXYGEN_HIGH_UNCERTAINTY_SOURCE",
    "latest_timestamped_payload",
    "resolve_all_remaining_oxygen_accepted_sources",
    "resolve_all_remaining_oxygen_high_uncertainty_sources",
    "resolve_dblock_4d_post_oxygen_accepted_sources",
    "resolve_dblock_4d_post_oxygen_high_uncertainty_sources",
    "resolve_dblock_3d_oxygen_accepted_sources",
    "resolve_dblock_3d_oxygen_high_uncertainty_sources",
    "resolve_timestamped_accepted_sources",
    "resolve_timestamped_high_uncertainty_sources",
]
