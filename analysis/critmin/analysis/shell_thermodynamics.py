"""Direct bond-valence thermodynamics from coordination geometry.

Standalone module — not part of the fitting pipeline.  Computes (R₀, B) and
the full thermodynamic decomposition F = U − TS for a coordination shell
*without* solving the bond-network equations.  The only inputs are the
measured bond lengths, the cation formal charge z, the coordination number
n, and the element-level characteristic pair (R₀*, B*).

This module uses the **cation-centered** self-consistency relation between
the β-line and the entropy-corrected identity.  For the **anion-centered**
fitting approach (which breaks the z = n degeneracy), see
:mod:`critmin.analysis.anion_valence`.

The two theoretical constraints—the β-line B = B* + β(R₀ − R₀*) and the
entropy-corrected identity R₀ = m_i + B(ln z − H_i)—are not independent
at the arithmetic-mean level (β·ln(z/n) ≡ 1).  B is identifiable only
through the *entropy correction*, i.e. the departure of the Boltzmann-
weighted mean m_i from the arithmetic mean R̄.

Three identifiability regimes:

* **self-consistent** — enough bond-length spread to determine B exactly.
* **best-fit** — B weakly determined; min |g(B)| is small but nonzero.
* **degenerate** — all bonds identical; B set to B*.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from scipy.optimize import minimize_scalar

from critmin.analysis.bond_valence_yukawa import (
    _shell_state_for_b,
    screening_length_branches,
    COULOMB_E2_OVER_4PI_EPS0_EV_ANG,
    EV_TO_KJ_PER_MOL,
)


DEFAULT_MASTER_SUMMARY_PATH = Path(
    "data/processed/theory/master_oxygen_summary_theory.json"
)

Identifiability = Literal["self_consistent", "best_fit", "degenerate"]


@dataclass(frozen=True, slots=True)
class CharacteristicPair:
    """Element-level (R₀*, B*) reference pair."""

    element: str
    oxi_state: int | None
    r0_star: float
    b_star: float
    n_lines: int


@dataclass(frozen=True, slots=True)
class ShellThermodynamics:
    """Thermodynamic decomposition of one coordination shell.

    All length-unit quantities are in Å.  Physical energies are in eV
    and kJ/mol per shell.

    Attributes
    ----------
    r0, b : float
        Bond-valence intercept and softness.
    u : float
        Internal energy = weighted mean bond length m_i (Å).
    s : float
        Entropy = Shannon entropy H_i of bond-weight distribution.
    t : float
        Temperature = B (Å), conjugate to S.
    f : float
        Free energy F = U − T·S (Å).
    gamma : float
        Entropy correction Γ_i = 1 − β(ln q − H_i).
    p_ij : tuple[float, ...]
        Bond weights (Boltzmann probabilities).
    bond_lengths : tuple[float, ...]
        Input bond lengths (sorted).
    z, n : int
        Formal charge and coordination number.
    beta : float
        Pole-form slope 1/ln(z/n).
    r0_star, b_star : float
        Element-level characteristic pair.
    lambda_plus, lambda_minus : float | None
        Screening-length branches.
    f_ev, f_kj_mol : float | None
        Physical free energy per shell (eV, kJ/mol).
    identifiability : Identifiability
        How well B is determined by the shell geometry.
    g_residual : float
        Self-consistency residual g(B) = B − B_predicted.
    """

    r0: float
    b: float
    u: float
    s: float
    t: float
    f: float
    gamma: float
    p_ij: tuple[float, ...]
    bond_lengths: tuple[float, ...]
    z: int
    n: int
    beta: float
    r0_star: float
    b_star: float
    lambda_plus: float | None
    lambda_minus: float | None
    f_ev: float | None
    f_kj_mol: float | None
    identifiability: Identifiability
    g_residual: float


def load_characteristic_pairs(
    path: str | Path = DEFAULT_MASTER_SUMMARY_PATH,
) -> dict[tuple[str, int | None], CharacteristicPair]:
    """Load the (R₀*, B*) lookup table from the master oxygen summary.

    Returns a dict keyed by ``(element, oxi_state)`` where ``oxi_state``
    is ``None`` for species with a single dominant oxidation state (e.g.
    Group 1/2, lanthanides) and an integer for oxidation-state-resolved
    species (e.g. d-block).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("master_rows", [])

    table: dict[tuple[str, int | None], CharacteristicPair] = {}
    for row in rows:
        element = str(row["element"])
        raw_oxi = row.get("oxi_state")
        oxi_state = int(raw_oxi) if raw_oxi is not None else None
        pair = CharacteristicPair(
            element=element,
            oxi_state=oxi_state,
            r0_star=float(row["R0_star"]),
            b_star=float(row["B_star"]),
            n_lines=int(row["n_lines"]),
        )
        table[(element, oxi_state)] = pair
    return table


def _lookup_pair(
    table: dict[tuple[str, int | None], CharacteristicPair],
    element: str,
    z: int,
) -> CharacteristicPair | None:
    """Look up (R₀*, B*) for an element, falling back from (element, z) to (element, None)."""
    pair = table.get((element, z))
    if pair is not None:
        return pair
    return table.get((element, None))


def _bond_length_spread(bond_lengths: Sequence[float]) -> float:
    """Return the relative spread (max − min) / mean of the shell."""
    if len(bond_lengths) < 2:
        return 0.0
    mn = min(bond_lengths)
    mx = max(bond_lengths)
    mean = sum(bond_lengths) / len(bond_lengths)
    if mean <= 0.0:
        return 0.0
    return (mx - mn) / mean


def _g_residual(
    b: float,
    lengths: list[float],
    z: int,
    beta: float,
    r0_star: float,
    b_star: float,
) -> tuple[float, dict[str, Any]]:
    """Return g(B) = B − B_predicted and the shell state at this B."""
    state = _shell_state_for_b(lengths, q=float(z), b_value=b)
    r0 = float(state["r0"])
    b_predicted = b_star + beta * (r0 - r0_star)
    return b - b_predicted, state


def _build_result(
    b: float,
    state: dict[str, Any],
    g: float,
    *,
    lengths: list[float],
    z: int,
    n: int,
    beta: float,
    r0_star: float,
    b_star: float,
    anion_charge: float,
    identifiability: Identifiability,
) -> ShellThermodynamics:
    """Package a shell state into a ShellThermodynamics dataclass."""
    r0 = float(state["r0"])
    m_i = float(state["weighted_mean_bond_length"])
    h_i = float(state["entropy"])
    log_q_minus_h = float(state["log_q_minus_h"])
    gamma = 1.0 - beta * log_q_minus_h
    f_val = m_i - b * h_i

    branches = screening_length_branches(
        r0=r0, beta=beta, r0_star=r0_star, b_star=b_star,
    )
    lam_plus = branches.get("positive")
    lam_minus = branches.get("negative")

    f_phys = _physical_free_energy(
        r0=r0, b=b, n=n, z=z,
        anion_charge=anion_charge, f_length=f_val,
    )

    return ShellThermodynamics(
        r0=r0, b=b,
        u=m_i, s=h_i, t=b, f=f_val,
        gamma=gamma,
        p_ij=tuple(float(p) for p in state["p_ij"]),
        bond_lengths=tuple(lengths),
        z=z, n=n, beta=beta,
        r0_star=r0_star, b_star=b_star,
        lambda_plus=lam_plus,
        lambda_minus=lam_minus,
        f_ev=f_phys.get("f_ev"),
        f_kj_mol=f_phys.get("f_kj_mol"),
        identifiability=identifiability,
        g_residual=g,
    )


def shell_thermodynamics(
    bond_lengths: Sequence[float],
    *,
    z: int,
    r0_star: float,
    b_star: float,
    anion_charge: float = -2.0,
    b_min: float = 0.01,
    b_max: float = 10.0,
    self_consistent_threshold: float = 1.0e-4,
    spread_threshold: float = 1.0e-8,
) -> ShellThermodynamics:
    """Compute the thermodynamic decomposition for a single coordination shell.

    Parameters
    ----------
    bond_lengths
        Cation–anion distances (Å) for the coordination shell.
    z
        Formal cation oxidation state (positive integer).
    r0_star, b_star
        Element-level characteristic pair (Å).
    anion_charge
        Formal anion charge (default −2 for oxygen).
    b_min, b_max
        Bounds for the B optimizer (Å).
    self_consistent_threshold
        Maximum |g(B)| / B for the result to be classified as
        ``"self_consistent"`` rather than ``"best_fit"``.
    spread_threshold
        Relative bond-length spread below which the shell is flagged as
        degenerate (all bonds effectively identical).

    Returns
    -------
    ShellThermodynamics
        Dataclass with R₀, B, U, S, T, F, Γ, p_ij, screening branches,
        and physical energies.
    """
    lengths = sorted(float(r) for r in bond_lengths)
    n = len(lengths)
    if n < 1:
        raise ValueError("need at least one bond")
    if z <= 0:
        raise ValueError("formal charge z must be positive")
    if z == n:
        raise ValueError(
            f"z = n = {n}: at the pole β = 1/ln(z/n) → ±∞. "
            "The β-line is undefined at the covalent reference z = CN."
        )

    beta = 1.0 / math.log(z / n)
    spread = _bond_length_spread(lengths)
    common = dict(
        lengths=lengths, z=z, n=n, beta=beta,
        r0_star=r0_star, b_star=b_star, anion_charge=anion_charge,
    )

    # --- Degenerate: all bonds identical ---
    if spread < spread_threshold:
        r_bar = sum(lengths) / n
        h_i = math.log(n) if n > 1 else 0.0
        b_val = max(b_star, b_min)
        state = _shell_state_for_b(lengths, q=float(z), b_value=b_val)
        g, _ = _g_residual(b_val, lengths, z, beta, r0_star, b_star)
        return _build_result(
            b_val, state, g, identifiability="degenerate", **common,
        )

    # --- Non-degenerate: minimize g(B)² to find the best B ---
    #
    # Because β·ln(z/n) ≡ 1, the β-line and the bond-length identity
    # collapse at the arithmetic-mean level: g(B) → g_∞ (a small
    # constant) as B → ∞.  B is identifiable only through the entropy
    # correction.  A physically motivated upper bound prevents the
    # optimizer from running to infinity when g(B) ≠ 0 for all B.
    effective_b_max = min(b_max, max(5.0 * b_star, 1.0))

    def objective(log_b: float) -> float:
        b = math.exp(log_b)
        g, _ = _g_residual(b, lengths, z, beta, r0_star, b_star)
        return g * g

    result = minimize_scalar(
        objective,
        bounds=(math.log(b_min), math.log(effective_b_max)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    b_opt = math.exp(float(result.x))

    # If the optimizer hit the upper bound, B is not independently
    # determined by the shell.  Fall back to B* — the element-level
    # reference — and honestly report the residual.
    at_upper_bound = b_opt > 0.95 * effective_b_max
    if at_upper_bound:
        b_opt = max(b_star, b_min)

    g_opt, state_opt = _g_residual(b_opt, lengths, z, beta, r0_star, b_star)

    scale = max(b_opt, b_star, 0.01)
    if abs(g_opt) / scale < self_consistent_threshold:
        ident: Identifiability = "self_consistent"
    else:
        ident = "best_fit"

    return _build_result(b_opt, state_opt, g_opt, identifiability=ident, **common)


def _physical_free_energy(
    *,
    r0: float,
    b: float,
    n: int,
    z: int,
    anion_charge: float,
    f_length: float,
) -> dict[str, float | None]:
    """Convert the length-unit free energy to physical units.

    Uses the Coulomb envelope K_ij · n / R₀ as the energy prefactor.
    """
    if r0 <= 0.0 or b <= 0.0:
        return {"f_ev": None, "f_kj_mol": None}

    k_ij = z * abs(anion_charge) * COULOMB_E2_OVER_4PI_EPS0_EV_ANG
    coulomb_envelope = k_ij * n / r0
    f_ev = coulomb_envelope * f_length
    f_kj = f_ev * EV_TO_KJ_PER_MOL
    return {"f_ev": f_ev, "f_kj_mol": f_kj}


def decompose_structure(
    shells: Sequence[dict[str, Any]],
    *,
    table: dict[tuple[str, int | None], CharacteristicPair],
    anion_charge: float = -2.0,
    **kwargs: Any,
) -> list[ShellThermodynamics | dict[str, Any]]:
    """Compute thermodynamic decompositions for all cation sites in a structure.

    Parameters
    ----------
    shells
        List of dicts, each with keys ``"element"``, ``"z"``,
        ``"bond_lengths"`` (and optionally ``"site_index"``).
    table
        (R₀*, B*) lookup from :func:`load_characteristic_pairs`.
    anion_charge
        Default anion formal charge.

    Returns
    -------
    list
        One :class:`ShellThermodynamics` per shell, or a dict with
        ``"error"`` for shells that could not be processed.
    """
    results: list[ShellThermodynamics | dict[str, Any]] = []
    for shell in shells:
        element = str(shell["element"])
        z = int(shell["z"])
        bond_lengths = shell["bond_lengths"]

        pair = _lookup_pair(table, element, z)
        if pair is None:
            results.append({
                "element": element,
                "z": z,
                "error": f"no (R0*, B*) for ({element}, z={z})",
            })
            continue

        try:
            thermo = shell_thermodynamics(
                bond_lengths,
                z=z,
                r0_star=pair.r0_star,
                b_star=pair.b_star,
                anion_charge=anion_charge,
                **kwargs,
            )
            results.append(thermo)
        except (ValueError, ArithmeticError) as exc:
            results.append({
                "element": element,
                "z": z,
                "bond_lengths": list(bond_lengths),
                "error": str(exc),
            })

    return results
