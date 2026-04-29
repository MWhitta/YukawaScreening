"""Yukawa screened-Coulomb helpers for per-material bond-valence fits."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any, Literal

COULOMB_E2_OVER_4PI_EPS0_EV_ANG = 14.3996454784255
EV_TO_KJ_PER_MOL = 96.48533212331002
_BOND_PART_RE = re.compile(r"\d+")
_DEFAULT_ANION_FORMAL_CHARGES = {
    "O": -2.0,
    "S": -2.0,
    "Se": -2.0,
    "Te": -2.0,
    "F": -1.0,
    "Cl": -1.0,
    "Br": -1.0,
    "I": -1.0,
    "N": -3.0,
    "P": -3.0,
}

Branch = Literal["positive", "negative"]
RepresentativeRadiusMode = Literal["fit", "weighted_mean"]
CationChargeMode = Literal["empirical", "formal"]


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def fit_summary(result: Any) -> Any | None:
    """Return a summary-like object exposing ``r0`` and ``b`` when available."""
    if result is None:
        return None
    if hasattr(result, "r0") and hasattr(result, "b"):
        return result

    aggregate = getattr(result, "aggregate", None)
    if callable(aggregate):
        try:
            summary = aggregate()
        except TypeError:
            summary = aggregate(reducer=None)
        if summary is not None:
            return summary

    return getattr(result, "summary", None)


def target_bond_rows(result: Any) -> list[dict[str, float | str]]:
    """Return the cation-anion bond rows used by one material fit."""
    theoretical = getattr(result, "theoretical", None)
    material = getattr(result, "material", None)
    if theoretical is None or material is None:
        return []

    cation = getattr(material, "cation", None)
    anion = getattr(material, "anion", None)
    if not cation or not anion:
        return []

    bond_valences = dict(getattr(theoretical, "bond_valences", {}) or {})
    bond_lengths = dict(getattr(theoretical, "bond_lengths", {}) or {})
    rows: list[dict[str, float | str]] = []
    for bond_type in getattr(theoretical, "bond_types", ()):
        parts = _BOND_PART_RE.split(str(bond_type))
        if len(parts) < 2 or parts[0] != cation or parts[1] != anion:
            continue

        sij = _finite_float(bond_valences.get(bond_type))
        distance = _finite_float(bond_lengths.get(bond_type))
        if sij is None or distance is None or sij <= 0.0:
            continue
        rows.append(
            {
                "bond_type": str(bond_type),
                "sij": sij,
                "bond_length": distance,
            }
        )
    return rows


def shell_statistics(result: Any) -> dict[str, Any]:
    """Return shell statistics needed by the Yukawa and free-energy formulas."""
    rows = target_bond_rows(result)
    q_empirical = sum(float(row["sij"]) for row in rows)
    if q_empirical <= 0.0:
        return {
            "n_bonds": len(rows),
            "q_empirical": None,
            "weighted_mean_bond_length": None,
            "entropy": None,
            "bond_rows": [],
        }

    weighted_mean = 0.0
    entropy = 0.0
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        sij = float(row["sij"])
        p_ij = sij / q_empirical
        distance = float(row["bond_length"])
        weighted_mean += p_ij * distance
        entropy -= p_ij * math.log(p_ij)
        normalized_rows.append(
            {
                "bond_type": row["bond_type"],
                "sij": sij,
                "bond_length": distance,
                "p_ij": p_ij,
            }
        )

    return {
        "n_bonds": len(normalized_rows),
        "q_empirical": q_empirical,
        "weighted_mean_bond_length": weighted_mean,
        "entropy": entropy,
        "bond_rows": normalized_rows,
    }


def anion_formal_charge(anion: str | None) -> float | None:
    """Return the default formal charge for common monoatomic anions."""
    if anion is None:
        return None
    return _DEFAULT_ANION_FORMAL_CHARGES.get(str(anion))


def screening_length_branches(
    *,
    r0: float,
    beta: float,
    r0_star: float,
    b_star: float,
) -> dict[str, float | bool | None]:
    """Return the notebook-style positive/negative ``lambda(R0)`` branches."""
    r0_value = float(r0)
    predicted_b = float(beta) * (r0_value - float(r0_star)) + float(b_star)
    discriminant = r0_value * r0_value + 4.0 * predicted_b * r0_value
    payload: dict[str, float | bool | None] = {
        "r0_input": r0_value,
        "predicted_b": predicted_b,
        "discriminant": discriminant,
        "has_real_branches": discriminant >= 0.0,
        "positive": None,
        "negative": None,
    }
    if discriminant < 0.0:
        return payload

    sqrt_discriminant = math.sqrt(discriminant)
    payload["positive"] = (sqrt_discriminant - r0_value) / 2.0
    payload["negative"] = (-sqrt_discriminant - r0_value) / 2.0
    return payload


def screening_length_derivative(
    *,
    r0: float,
    beta: float,
    lambda_value: float,
) -> float | None:
    """Return ``d lambda / d R0`` for either quadratic branch."""
    r0_value = _finite_float(r0)
    lambda_cast = _finite_float(lambda_value)
    if r0_value is None or abs(r0_value) <= 1.0e-15 or lambda_cast is None:
        return None

    denominator = r0_value * (2.0 * lambda_cast + r0_value)
    if abs(denominator) <= 1.0e-15:
        return None
    numerator = float(beta) * r0_value * r0_value + lambda_cast * lambda_cast
    return numerator / denominator


def _shell_state_for_b(
    bond_lengths: Sequence[float],
    *,
    q: float,
    b_value: float,
) -> dict[str, Any]:
    """Return shell weights and entropy for one positive ``B`` value."""
    q_value = float(q)
    if q_value <= 0.0:
        raise ValueError("q must be positive")
    if b_value <= 0.0:
        raise ValueError("b_value must be positive")

    distances = [_finite_float(distance) for distance in bond_lengths]
    if not distances or any(distance is None for distance in distances):
        raise ValueError("bond_lengths must contain only finite values")

    cast_distances = [float(distance) for distance in distances if distance is not None]
    exponents = [(-distance / b_value) for distance in cast_distances]
    max_exponent = max(exponents)
    weights = [math.exp(exponent - max_exponent) for exponent in exponents]
    normalizer = sum(weights)
    if normalizer <= 0.0:
        raise ValueError("shell weights failed to normalize")

    probabilities = [weight / normalizer for weight in weights]
    weighted_mean = sum(probability * distance for probability, distance in zip(probabilities, cast_distances))
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    log_q_minus_h = math.log(q_value) - entropy
    r0_value = weighted_mean + b_value * log_q_minus_h
    return {
        "n_bonds": len(cast_distances),
        "bond_lengths": cast_distances,
        "p_ij": probabilities,
        "weighted_mean_bond_length": weighted_mean,
        "entropy": entropy,
        "log_q_minus_h": log_q_minus_h,
        "r0": r0_value,
    }


def solve_self_consistent_yukawa_state(
    bond_lengths: Sequence[float],
    *,
    q: float,
    beta: float,
    r0_star: float,
    b_star: float,
    initial_b: float | None = None,
    max_iter: int = 100,
    tol: float = 1.0e-12,
    damping: float = 0.5,
    min_b: float = 1.0e-9,
) -> dict[str, Any]:
    """Solve the shell-level self-consistent Yukawa state for one branch family."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0, 1]")

    q_value = float(q)
    if q_value <= 0.0:
        raise ValueError("q must be positive")

    distances = [_finite_float(distance) for distance in bond_lengths]
    if not distances or any(distance is None for distance in distances):
        raise ValueError("bond_lengths must contain at least one finite value")
    cast_distances = [float(distance) for distance in distances if distance is not None]

    weighted_guess = sum(cast_distances) / len(cast_distances)
    b_guess = _finite_float(initial_b)
    if b_guess is None or b_guess <= min_b:
        b_guess = _finite_float(b_star)
    if b_guess is None or b_guess <= min_b:
        b_guess = max(weighted_guess / 10.0, 0.1)
    b_value = max(float(b_guess), float(min_b))

    converged = False
    termination_reason = "max_iter"
    iterations = 0
    residual = math.nan
    predicted_b = math.nan

    for iterations in range(1, max_iter + 1):
        shell_state = _shell_state_for_b(cast_distances, q=q_value, b_value=b_value)
        r0_value = float(shell_state["r0"])
        predicted_b = float(beta) * (r0_value - float(r0_star)) + float(b_star)
        residual = predicted_b - b_value

        if predicted_b <= min_b:
            termination_reason = "non_positive_b"
            break

        scale = max(1.0, abs(b_value), abs(predicted_b))
        if abs(residual) <= float(tol) * scale:
            b_value = predicted_b
            converged = True
            termination_reason = "converged"
            break

        damped_b = b_value + float(damping) * residual
        if damped_b <= min_b:
            damped_b = (b_value + float(min_b)) / 2.0
        b_value = damped_b

    final_shell_state = _shell_state_for_b(cast_distances, q=q_value, b_value=b_value)
    final_r0 = float(final_shell_state["r0"])
    final_entropy = float(final_shell_state["entropy"])
    final_log_q_minus_h = float(final_shell_state["log_q_minus_h"])
    final_predicted_b = float(beta) * (final_r0 - float(r0_star)) + float(b_star)
    final_residual = final_predicted_b - b_value
    gamma = 1.0 - float(beta) * final_log_q_minus_h

    branch_payload = screening_length_branches(
        r0=final_r0,
        beta=float(beta),
        r0_star=float(r0_star),
        b_star=float(b_star),
    )
    branches: dict[str, dict[str, float | None]] = {}
    for branch_name in ("positive", "negative"):
        lambda_value = _finite_float(branch_payload[branch_name])
        branches[branch_name] = {
            "lambda_value": lambda_value,
            "dlambda_d_r0": screening_length_derivative(
                r0=final_r0,
                beta=float(beta),
                lambda_value=lambda_value,
            )
            if lambda_value is not None
            else None,
        }

    return {
        "q": q_value,
        "beta": float(beta),
        "r0_star": float(r0_star),
        "b_star": float(b_star),
        "bond_lengths": cast_distances,
        "p_ij": list(final_shell_state["p_ij"]),
        "weighted_mean_bond_length": float(final_shell_state["weighted_mean_bond_length"]),
        "entropy": final_entropy,
        "log_q_minus_h": final_log_q_minus_h,
        "gamma": gamma,
        "r0": final_r0,
        "b": b_value,
        "predicted_b": final_predicted_b,
        "residual": final_residual,
        "iterations": iterations,
        "converged": converged,
        "termination_reason": termination_reason,
        "discriminant": branch_payload["discriminant"],
        "has_real_branches": bool(branch_payload["has_real_branches"]),
        "branches": branches,
    }


def screening_factor(distance: float, lambda_value: float) -> float | None:
    """Return the explicit Yukawa factor ``exp(-R_ij / lambda)``."""
    distance_value = float(distance)
    lambda_cast = _finite_float(lambda_value)
    if lambda_cast is None or abs(lambda_cast) <= 1.0e-15:
        return None
    exponent = -distance_value / lambda_cast
    if exponent > 700.0:
        return math.inf
    if exponent < -745.0:
        return 0.0
    return math.exp(exponent)


def yukawa_pair_energy(
    *,
    cation_charge: float,
    anion_charge: float,
    bond_length: float,
    lambda_value: float,
) -> dict[str, float | None]:
    """Return the pair Yukawa energy in reduced, eV, and kJ/mol units."""
    distance = float(bond_length)
    factor = screening_factor(distance, lambda_value)
    if factor is None:
        return {
            "screening_factor": None,
            "reduced_energy": None,
            "u_ij_ev": None,
            "u_ij_kj_mol": None,
        }

    reduced_energy = float(cation_charge) * float(anion_charge) * factor / distance
    u_ij_ev = COULOMB_E2_OVER_4PI_EPS0_EV_ANG * reduced_energy
    return {
        "screening_factor": factor,
        "reduced_energy": reduced_energy,
        "u_ij_ev": u_ij_ev,
        "u_ij_kj_mol": u_ij_ev * EV_TO_KJ_PER_MOL,
    }


def compute_yukawa_shell(
    result: Any,
    *,
    beta: float,
    r0_star: float,
    b_star: float,
    branch: Branch = "positive",
    representative_radius_mode: RepresentativeRadiusMode = "fit",
    cation_charge_mode: CationChargeMode = "empirical",
    cation_charge: float | None = None,
    anion_charge: float | None = None,
) -> dict[str, Any]:
    """Compute the explicit screened-Coulomb shell sum for one fitted material."""
    if branch not in ("positive", "negative"):
        raise ValueError("branch must be 'positive' or 'negative'")
    if representative_radius_mode not in ("fit", "weighted_mean"):
        raise ValueError("representative_radius_mode must be 'fit' or 'weighted_mean'")
    if cation_charge_mode not in ("empirical", "formal"):
        raise ValueError("cation_charge_mode must be 'empirical' or 'formal'")

    material = getattr(result, "material", None)
    summary = fit_summary(result)
    shell = shell_statistics(result)
    r0_fit = _finite_float(getattr(summary, "r0", None))
    b_fit = _finite_float(getattr(summary, "b", None))
    q_empirical = _finite_float(shell["q_empirical"])
    weighted_mean = _finite_float(shell["weighted_mean_bond_length"])
    entropy = _finite_float(shell["entropy"])

    material_metadata = dict(getattr(material, "metadata", {}) or {})
    formal_cation_charge = _finite_float(material_metadata.get("oxi_state"))
    if cation_charge is not None:
        selected_cation_charge = float(cation_charge)
    elif cation_charge_mode == "formal" and formal_cation_charge is not None:
        selected_cation_charge = formal_cation_charge
    else:
        selected_cation_charge = q_empirical

    if anion_charge is not None:
        selected_anion_charge = float(anion_charge)
    else:
        selected_anion_charge = anion_formal_charge(getattr(material, "anion", None))

    if representative_radius_mode == "fit":
        r0_input = r0_fit
    else:
        r0_input = weighted_mean

    branch_payload = {
        "r0_input": r0_input,
        "predicted_b": None,
        "discriminant": None,
        "lambda_value": None,
        "has_real_branches": False,
    }
    if r0_input is not None:
        branch_data = screening_length_branches(
            r0=float(r0_input),
            beta=float(beta),
            r0_star=float(r0_star),
            b_star=float(b_star),
        )
        branch_payload.update(
            {
                "r0_input": branch_data["r0_input"],
                "predicted_b": branch_data["predicted_b"],
                "discriminant": branch_data["discriminant"],
                "lambda_value": branch_data[branch],
                "has_real_branches": bool(branch_data["has_real_branches"]),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    total_reduced = 0.0
    total_u_ev = 0.0
    total_u_kj_mol = 0.0
    has_pair_energies = False
    lambda_value = _finite_float(branch_payload["lambda_value"])
    for row in shell["bond_rows"]:
        pair = {
            "bond_type": row["bond_type"],
            "bond_length": row["bond_length"],
            "sij": row["sij"],
            "p_ij": row["p_ij"],
            "screening_factor": None,
            "reduced_energy": None,
            "u_ij_ev": None,
            "u_ij_kj_mol": None,
        }
        if (
            lambda_value is not None
            and selected_cation_charge is not None
            and selected_anion_charge is not None
        ):
            energy = yukawa_pair_energy(
                cation_charge=selected_cation_charge,
                anion_charge=selected_anion_charge,
                bond_length=float(row["bond_length"]),
                lambda_value=lambda_value,
            )
            pair.update(energy)
            if energy["reduced_energy"] is not None:
                total_reduced += float(energy["reduced_energy"])
                total_u_ev += float(energy["u_ij_ev"])
                total_u_kj_mol += float(energy["u_ij_kj_mol"])
                has_pair_energies = True
        pair_rows.append(pair)

    return {
        "material_id": getattr(material, "material_id", None),
        "formula_pretty": getattr(material, "formula_pretty", None),
        "cation": getattr(material, "cation", None),
        "anion": getattr(material, "anion", None),
        "branch": branch,
        "representative_radius_mode": representative_radius_mode,
        "cation_charge_mode": cation_charge_mode,
        "r0_fit": r0_fit,
        "b_fit": b_fit,
        "q_empirical": q_empirical,
        "formal_cation_charge": formal_cation_charge,
        "selected_cation_charge": selected_cation_charge,
        "selected_anion_charge": selected_anion_charge,
        "weighted_mean_bond_length": weighted_mean,
        "shell_entropy": entropy,
        "beta": float(beta),
        "r0_star": float(r0_star),
        "b_star": float(b_star),
        "r0_input": branch_payload["r0_input"],
        "predicted_b": branch_payload["predicted_b"],
        "discriminant": branch_payload["discriminant"],
        "has_real_branches": branch_payload["has_real_branches"],
        "lambda_value": lambda_value,
        "pair_rows": pair_rows,
        "u_shell_reduced": total_reduced if has_pair_energies else None,
        "u_shell_ev": total_u_ev if has_pair_energies else None,
        "u_shell_kj_mol": total_u_kj_mol if has_pair_energies else None,
    }


def compute_yukawa_shell_branches(
    result: Any,
    *,
    beta: float,
    r0_star: float,
    b_star: float,
    representative_radius_mode: RepresentativeRadiusMode = "fit",
    cation_charge_mode: CationChargeMode = "empirical",
    cation_charge: float | None = None,
    anion_charge: float | None = None,
) -> dict[str, Any]:
    """Return the positive and negative Yukawa shell branches together."""
    return {
        "positive": compute_yukawa_shell(
            result,
            beta=beta,
            r0_star=r0_star,
            b_star=b_star,
            branch="positive",
            representative_radius_mode=representative_radius_mode,
            cation_charge_mode=cation_charge_mode,
            cation_charge=cation_charge,
            anion_charge=anion_charge,
        ),
        "negative": compute_yukawa_shell(
            result,
            beta=beta,
            r0_star=r0_star,
            b_star=b_star,
            branch="negative",
            representative_radius_mode=representative_radius_mode,
            cation_charge_mode=cation_charge_mode,
            cation_charge=cation_charge,
            anion_charge=anion_charge,
        ),
    }
