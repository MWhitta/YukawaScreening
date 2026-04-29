"""Diagnostics for per-material bond-valence fits."""

from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Any

import numpy as np


def target_bond_lengths(result: Any) -> list[float]:
    """Return cation-anion bond lengths used by the per-material fit."""
    theoretical = getattr(result, "theoretical", None)
    material = getattr(result, "material", None)
    if theoretical is None or material is None:
        return []

    cation = getattr(material, "cation", None)
    anion = getattr(material, "anion", None)
    if not cation or not anion:
        return []

    lengths: list[float] = []
    for bond_type in getattr(theoretical, "bond_types", ()):
        parts = re.split(r"\d+", str(bond_type))
        if len(parts) < 2:
            continue
        if parts[0] != cation or parts[1] != anion:
            continue
        value = getattr(theoretical, "bond_lengths", {}).get(bond_type)
        if value is not None:
            lengths.append(float(value))
    return lengths


def summarize_target_bond_lengths(result: Any) -> dict[str, float | int | None]:
    """Return simple spread statistics for the fitted cation-anion bond lengths."""
    lengths = target_bond_lengths(result)
    if not lengths:
        return {
            "n_target_bonds": 0,
            "bond_length_mean": None,
            "bond_length_std": None,
            "bond_length_min": None,
            "bond_length_max": None,
            "bond_length_range": None,
            "bond_length_iqr": None,
            "bond_length_cv": None,
        }

    arr = np.asarray(lengths, dtype=float)
    bond_mean = float(mean(lengths))
    bond_std = float(pstdev(lengths))
    bond_range = float(np.max(arr) - np.min(arr))
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    return {
        "n_target_bonds": int(len(lengths)),
        "bond_length_mean": bond_mean,
        "bond_length_std": bond_std,
        "bond_length_min": float(np.min(arr)),
        "bond_length_max": float(np.max(arr)),
        "bond_length_range": bond_range,
        "bond_length_iqr": float(q3 - q1),
        "bond_length_cv": float(bond_std / bond_mean) if bond_mean else None,
    }


def min_gap_to_bounds(value: float, bounds: tuple[float, float]) -> float:
    """Return the minimum absolute distance from *value* to either bound."""
    return min(abs(float(value) - float(bounds[0])), abs(float(bounds[1]) - float(value)))
