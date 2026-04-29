"""Anion-centered bond-valence fitting.

Fit cation–anion (R₀, B) parameters from **anion** valence-sum constraints
instead of cation valence sums.  This breaks the z = n degeneracy that
makes B unidentifiable in the cation-centered approach (e.g. Si⁴⁺ CN4,
H⁺ CN1).

Why the cation approach degenerates at z = n
--------------------------------------------

The cation-centered fit residual is ``B · ln(s_ij) − R₀ + R_ij``.  When
``s_ij = z/n = 1`` for all bonds (the pole), ``ln(1) = 0`` and B drops
out — the design matrix is rank-1 regardless of the bond-length spread.

The anion valence-sum constraint ``Σ_k exp((R₀_k − R_kj) / B_k) = |z_O|``
retains the exponential nonlinearity even at z = n.  With two bonds at
different distances, the curvature of exp() constrains B through the
flux split across the anion coordination shell.

Per-material fitting cascade
-----------------------------

:func:`try_anion_centered_for_material` implements a three-level cascade,
accepting the first result that passes quality gates:

1. **De novo joint fit** (``fit_strategy="anion_de_novo"``) — fit all
   cation species present in the O shells simultaneously with no priors.
2. **Partial prior** (``fit_strategy="anion_partial_prior"``) — fix the
   sparsest species from the master table until the system is overdetermined.
3. **Target-only** (``fit_strategy="anion_centered"``) — fix all
   non-target species from the master table.

Quality gates (all levels): optimizer converged, B not at bounds,
RMS < 0.5 v.u., Jacobian full rank.

Angular-partition proxy
-----------------------

The module also supports an anion-side angular-partition proxy via
:func:`fit_single_species_from_anion_partition`.  When an anion site
carries normalized bond weights ``omega_k``, each bond receives a
pseudo-valence ``s = |z_anion| · omega`` and the usual linear
bond-valence relation ``R = R₀ − B ln s`` is fitted.

For hydrogen, :func:`extract_proton_oxygen_shells` provides the companion
proton-centered version: each H site is treated as an ``O-H...O`` shell,
the unit H valence is partitioned across nearby oxygen contacts, and the
same linear ``R = R₀ − B ln s`` relation is fitted from those H-centered
shares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True, slots=True)
class AnionSite:
    """One anion coordination environment.

    Attributes
    ----------
    site_index : int
        Index of the anion site in the structure.
    anion : str
        Element symbol of the anion (e.g. "O", "F").
    target_valence : float
        Expected anion valence magnitude (e.g. 2.0 for O²⁻).
    bonds : tuple[tuple[str, int, float], ...]
        Each entry is (cation_species_key, cation_site_index, bond_length).
        The species key encodes both element and oxidation state
        (e.g. "Si_4", "Mg_2", "H_1").
    bond_weights : tuple[float, ...] | None
        Optional normalized anion-side angular weights aligned with ``bonds``.
        When present, these weights should sum to one across the stored bonds.
    """

    site_index: int
    anion: str
    target_valence: float
    bonds: tuple[tuple[str, int, float], ...]
    bond_weights: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class AnionFitResult:
    """Result of anion-centered fitting for one or more cation species.

    Attributes
    ----------
    species_params : dict[str, tuple[float, float]]
        Fitted (R₀, B) per species key.
    fixed_species : dict[str, tuple[float, float]]
        Species held fixed during the fit.
    n_anion_sites : int
        Number of anion sites used.
    n_params : int
        Number of free parameters (2 per fitted species).
    rms_residual : float
        Root-mean-square anion-valence residual.
    max_residual : float
        Largest absolute anion-valence residual.
    jacobian_rank : int
        Rank of the Jacobian at the solution.
    converged : bool
        Whether the optimizer converged.
    """

    species_params: dict[str, tuple[float, float]]
    fixed_species: dict[str, tuple[float, float]]
    n_anion_sites: int
    n_params: int
    rms_residual: float
    max_residual: float
    jacobian_rank: int
    converged: bool


@dataclass(frozen=True, slots=True)
class ProtonShell:
    """One proton-centered oxygen environment.

    Attributes
    ----------
    site_index : int
        Index of the H site in the structure.
    oxygen_bonds : tuple[tuple[int, float], ...]
        Oxygen contacts as ``(oxygen_site_index, H-O distance)``.
    bond_weights : tuple[float, ...]
        Normalized proton-centered valence shares aligned with
        ``oxygen_bonds``.  These weights sum to one per H site.
    """

    site_index: int
    oxygen_bonds: tuple[tuple[int, float], ...]
    bond_weights: tuple[float, ...]


def species_key(element: str, oxi_state: int) -> str:
    """Canonical key for one cation species: ``"Si_4"``, ``"H_1"``."""
    return f"{element}_{oxi_state}"


def extract_anion_sites(
    structure_graph: Any,
    structure: Any,
    *,
    anion: str = "O",
    anion_valence: float = 2.0,
    possible_species: Sequence[Any] = (),
    cation_oxi_map: Mapping[str, int] | None = None,
    angular_weight_mode: str | None = None,
) -> list[AnionSite]:
    """Extract anion coordination shells from a pymatgen structure graph.

    Parameters
    ----------
    structure_graph
        A pymatgen ``StructureGraph``.
    structure
        The underlying ``Structure`` (from ``structure_graph.structure``).
    anion
        Element symbol of the anion to target.
    anion_valence
        Expected valence magnitude for the anion.
    possible_species
        Materials Project ``possible_species`` list (e.g. ``["Si4+", "O2-"]``).
        Used to assign oxidation states to cation neighbors.
    cation_oxi_map
        Optional explicit mapping ``{element: oxi_state}`` overriding
        ``possible_species``.
    angular_weight_mode
        Optional anion-side bond-weight extraction mode.  ``None`` stores no
        weights.  ``"voronoi_solid_angle"`` stores normalized oxygen-centered
        Voronoi solid-angle weights over the retained cation neighbors.

    Returns
    -------
    list[AnionSite]
        One entry per anion site with at least one cation neighbor.
    """
    # Build cation oxidation-state lookup
    oxi_lookup: dict[str, int] = dict(cation_oxi_map or {})
    if not oxi_lookup and possible_species:
        import re
        species_re = re.compile(r"^([A-Z][a-z]?)(\d*)([\+\-])$")
        for sp in possible_species:
            m = species_re.match(str(sp).strip())
            if m is None:
                continue
            elem, charge_str, sign = m.groups()
            charge = int(charge_str) if charge_str else 1
            if sign == "-":
                charge = -charge
            if charge > 0:
                oxi_lookup.setdefault(elem, charge)

    sites: list[AnionSite] = []
    for site_idx, site in enumerate(structure):
        sym = _symbol(site)
        if sym != anion:
            continue

        neighbors = structure_graph.get_connected_sites(site_idx)
        bonds: list[tuple[str, int, float]] = []
        for nbr in neighbors:
            nbr_site = getattr(nbr, "site", None)
            nbr_sym = _symbol(nbr_site)
            if nbr_sym is None or nbr_sym == anion:
                continue
            oxi = oxi_lookup.get(nbr_sym)
            if oxi is None:
                continue
            key = species_key(nbr_sym, oxi)
            bonds.append((key, int(nbr.index), float(nbr.dist)))

        if bonds:
            bond_weights = None
            if angular_weight_mode is not None:
                bond_weights = _extract_anion_bond_weights(
                    structure,
                    site_idx,
                    bonds,
                    mode=angular_weight_mode,
                )
            sites.append(AnionSite(
                site_index=site_idx,
                anion=anion,
                target_valence=anion_valence,
                bonds=tuple(bonds),
                bond_weights=bond_weights,
            ))

    return sites


def _symbol(site: Any) -> str | None:
    specie = getattr(site, "specie", None)
    if specie is None:
        return None
    return getattr(specie, "symbol", str(specie))


def _normalize_weights(weights: Sequence[float]) -> tuple[float, ...] | None:
    values = [float(value) for value in weights]
    total = float(sum(values))
    if total <= 0.0:
        return None
    return tuple(value / total for value in values)


def _voronoi_solid_angle_weights(
    structure: Any,
    site_index: int,
    neighbor_indices: Sequence[int],
) -> tuple[float, ...] | None:
    """Return normalized Voronoi solid-angle weights for one anion site.

    Voronoi neighbors may include multiple periodic images of the same
    crystallographic site.  We therefore aggregate raw facet solid angles by
    ``site_index`` and only then normalize over the requested neighbor set.
    """
    try:
        from pymatgen.analysis.local_env import VoronoiNN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pymatgen is required to compute Voronoi solid-angle weights"
        ) from exc

    voronoi = VoronoiNN(extra_nn_info=True)
    aggregated: dict[int, float] = {}
    for info in voronoi.get_nn_info(structure, int(site_index)):
        neighbor_index = int(info["site_index"])
        poly_info = dict(info.get("poly_info") or {})
        raw_weight = float(poly_info.get("solid_angle", info.get("weight", 0.0)))
        if raw_weight <= 0.0:
            continue
        aggregated[neighbor_index] = aggregated.get(neighbor_index, 0.0) + raw_weight

    requested = [aggregated.get(int(neighbor_index), 0.0) for neighbor_index in neighbor_indices]
    return _normalize_weights(requested)


def _distance_power_weights(
    distances: Sequence[float],
    *,
    power: float,
) -> tuple[float, ...] | None:
    values = [
        float(distance) ** (-float(power))
        for distance in distances
        if float(distance) > 0.0
    ]
    if len(values) != len(distances):
        return None
    return _normalize_weights(values)


def _screened_exponential_weights(
    distances: Sequence[float],
    *,
    screening_length: float,
) -> tuple[float, ...] | None:
    if screening_length <= 0.0:
        raise ValueError("screening_length must be positive")
    values = [math.exp(-float(distance) / float(screening_length)) for distance in distances]
    return _normalize_weights(values)


def _proton_oxygen_contacts(
    structure: Any,
    site_index: int,
    *,
    oxygen_cutoff: float,
    max_oxygen_neighbors: int,
) -> tuple[tuple[int, float], ...]:
    """Return the nearest oxygen contacts around one H site."""
    site = structure[int(site_index)]
    closest_by_index: dict[int, float] = {}
    for neighbor in structure.get_neighbors(site, float(oxygen_cutoff)):
        if _symbol(neighbor) != "O":
            continue
        neighbor_index = int(neighbor.index)
        distance = float(neighbor.nn_distance)
        current = closest_by_index.get(neighbor_index)
        if current is None or distance < current:
            closest_by_index[neighbor_index] = distance

    ordered = sorted(closest_by_index.items(), key=lambda item: item[1])
    if max_oxygen_neighbors > 0:
        ordered = ordered[: int(max_oxygen_neighbors)]
    return tuple((int(index), float(distance)) for index, distance in ordered)


def _proton_shell_weights(
    structure: Any,
    site_index: int,
    oxygen_bonds: Sequence[tuple[int, float]],
    *,
    weight_mode: str,
    distance_power: float,
    screening_length: float,
) -> tuple[float, ...] | None:
    distances = [distance for _oxygen_index, distance in oxygen_bonds]
    if weight_mode == "distance_power":
        return _distance_power_weights(distances, power=distance_power)
    if weight_mode == "screened_exponential":
        return _screened_exponential_weights(distances, screening_length=screening_length)
    if weight_mode == "voronoi_solid_angle":
        return _voronoi_solid_angle_weights(
            structure,
            site_index,
            [oxygen_index for oxygen_index, _distance in oxygen_bonds],
        )
    raise ValueError(f"unsupported proton weight_mode {weight_mode!r}")


def extract_proton_oxygen_shells(
    structure: Any,
    *,
    oxygen_cutoff: float = 2.2,
    min_oxygen_neighbors: int = 2,
    max_oxygen_neighbors: int = 2,
    weight_mode: str = "distance_power",
    distance_power: float = 4.0,
    screening_length: float = 0.37,
) -> list[ProtonShell]:
    """Extract proton-centered ``O-H...O`` shells from a structure.

    The default keeps the two nearest oxygens within ``oxygen_cutoff`` and
    partitions the unit H valence with an inverse-distance power law.  This
    deliberately does not use the MP ``StructureGraph`` bond list, because
    the acceptor oxygen in an ``O-H...O`` motif is often not represented as a
    formal H-O graph edge.
    """
    shells: list[ProtonShell] = []
    for site_index, site in enumerate(structure):
        if _symbol(site) != "H":
            continue
        oxygen_bonds = _proton_oxygen_contacts(
            structure,
            site_index,
            oxygen_cutoff=oxygen_cutoff,
            max_oxygen_neighbors=max_oxygen_neighbors,
        )
        if len(oxygen_bonds) < int(min_oxygen_neighbors):
            continue
        weights = _proton_shell_weights(
            structure,
            site_index,
            oxygen_bonds,
            weight_mode=weight_mode,
            distance_power=distance_power,
            screening_length=screening_length,
        )
        if weights is None:
            continue
        shells.append(
            ProtonShell(
                site_index=int(site_index),
                oxygen_bonds=tuple(oxygen_bonds),
                bond_weights=tuple(weights),
            )
        )
    return shells


def _extract_anion_bond_weights(
    structure: Any,
    site_index: int,
    bonds: Sequence[tuple[str, int, float]],
    *,
    mode: str,
) -> tuple[float, ...] | None:
    if mode == "voronoi_solid_angle":
        return _voronoi_solid_angle_weights(
            structure,
            site_index,
            [bond[1] for bond in bonds],
        )
    raise ValueError(f"unsupported angular_weight_mode {mode!r}")


def _site_partition_weights(
    site: AnionSite,
    *,
    weight_mode: str,
) -> tuple[float, ...]:
    if not site.bonds:
        return ()
    if weight_mode == "uniform":
        uniform = 1.0 / float(len(site.bonds))
        return tuple(uniform for _ in site.bonds)
    if weight_mode == "stored":
        if site.bond_weights is None:
            raise ValueError(
                f"anion site {site.site_index} does not carry stored bond weights"
            )
        if len(site.bond_weights) != len(site.bonds):
            raise ValueError(
                f"anion site {site.site_index} has {len(site.bond_weights)} weights "
                f"for {len(site.bonds)} bonds"
            )
        normalized = _normalize_weights(site.bond_weights)
        if normalized is None:
            raise ValueError(
                f"anion site {site.site_index} has non-positive total bond weight"
            )
        return normalized
    raise ValueError(f"unsupported weight_mode {weight_mode!r}")


def anion_valence_residuals(
    params_array: np.ndarray,
    sites: Sequence[AnionSite],
    fitted_keys: Sequence[str],
    fixed_params: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Compute per-site anion valence residuals.

    Parameters
    ----------
    params_array
        Flat array of [R₀₁, B₁, R₀₂, B₂, ...] for each fitted species,
        in the order given by *fitted_keys*.
    sites
        Anion sites to evaluate.
    fitted_keys
        Ordered species keys corresponding to params_array.
    fixed_params
        Species whose (R₀, B) are held constant.

    Returns
    -------
    np.ndarray
        One residual per site: predicted valence − target valence.
    """
    params: dict[str, tuple[float, float]] = dict(fixed_params)
    for i, key in enumerate(fitted_keys):
        r0 = float(params_array[2 * i])
        b = float(params_array[2 * i + 1])
        if b <= 1e-6:
            return np.full(len(sites), 1e8)
        params[key] = (r0, b)

    residuals = np.empty(len(sites), dtype=float)
    for j, site in enumerate(sites):
        valence = 0.0
        for sp_key, _site_idx, dist in site.bonds:
            pair = params.get(sp_key)
            if pair is None:
                continue
            r0, b = pair
            valence += math.exp((r0 - dist) / b)
        residuals[j] = valence - site.target_valence

    return residuals


def build_anion_partition_rows(
    sites: Sequence[AnionSite],
    target_species: str,
    *,
    weight_mode: str = "stored",
) -> list[dict[str, float | int]]:
    """Convert anion-site partitions into bond rows for one target species.

    Each retained bond receives an anion-side pseudo-valence
    ``s = |z_anion| * omega``, where ``omega`` is the normalized angular share
    of that bond around the anion site.
    """
    rows: list[dict[str, float | int]] = []
    for site in sites:
        weights = _site_partition_weights(site, weight_mode=weight_mode)
        n_target_bonds = sum(1 for species, _, _ in site.bonds if species == target_species)
        if n_target_bonds <= 0:
            continue
        per_row_weight = 1.0 / float(n_target_bonds)
        for (species, neighbor_index, bond_length), omega in zip(site.bonds, weights, strict=True):
            if species != target_species:
                continue
            assigned_valence = float(site.target_valence) * float(omega)
            if assigned_valence <= 0.0:
                continue
            rows.append(
                {
                    "site_index": int(site.site_index),
                    "neighbor_index": int(neighbor_index),
                    "bond_length": float(bond_length),
                    "omega": float(omega),
                    "sij": float(assigned_valence),
                    "log_sij": float(math.log(assigned_valence)),
                    "weight": float(per_row_weight),
                }
            )
    return rows


def build_proton_partition_rows(
    shells: Sequence[ProtonShell],
) -> list[dict[str, float | int]]:
    """Convert proton-centered shells into H-O bond-valence rows.

    The total cation valence is one for H⁺, so each contact receives
    ``s = omega`` directly.
    """
    rows: list[dict[str, float | int]] = []
    for shell in shells:
        n_bonds = len(shell.oxygen_bonds)
        if n_bonds == 0:
            continue
        if len(shell.bond_weights) != n_bonds:
            raise ValueError(
                f"proton shell {shell.site_index} has {len(shell.bond_weights)} "
                f"weights for {n_bonds} oxygen contacts"
            )
        normalized = _normalize_weights(shell.bond_weights)
        if normalized is None:
            continue
        per_row_weight = 1.0 / float(n_bonds)
        for (oxygen_index, distance), omega in zip(shell.oxygen_bonds, normalized, strict=True):
            if omega <= 0.0:
                continue
            rows.append(
                {
                    "site_index": int(shell.site_index),
                    "neighbor_index": int(oxygen_index),
                    "bond_length": float(distance),
                    "omega": float(omega),
                    "sij": float(omega),
                    "log_sij": float(math.log(float(omega))),
                    "weight": float(per_row_weight),
                }
            )
    return rows


def _anion_partition_residuals(
    params_array: np.ndarray,
    rows: Sequence[Mapping[str, float | int]],
) -> np.ndarray:
    r0 = float(params_array[0])
    b = float(params_array[1])
    if b <= 1e-6:
        return np.full(len(rows), 1e8)

    residuals = np.empty(len(rows), dtype=float)
    for index, row in enumerate(rows):
        predicted = r0 - b * float(row["log_sij"])
        bond_length = float(row["bond_length"])
        row_weight = float(row.get("weight", 1.0))
        residuals[index] = math.sqrt(row_weight) * (predicted - bond_length)
    return residuals


def _fit_single_species_from_partition_rows(
    rows: Sequence[Mapping[str, float | int]],
    target_species: str,
    *,
    r0_init: float,
    b_init: float,
    r0_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
) -> AnionFitResult:
    result = least_squares(
        _anion_partition_residuals,
        np.asarray([float(r0_init), float(b_init)], dtype=float),
        args=(rows,),
        bounds=(
            np.asarray([r0_bounds[0], b_bounds[0]], dtype=float),
            np.asarray([r0_bounds[1], b_bounds[1]], dtype=float),
        ),
        method="trf",
    )

    raw_residuals = np.asarray(
        [
            float(result.x[0]) - float(result.x[1]) * float(row["log_sij"]) - float(row["bond_length"])
            for row in rows
        ],
        dtype=float,
    )
    row_weights = np.asarray([float(row.get("weight", 1.0)) for row in rows], dtype=float)
    weight_sum = float(np.sum(row_weights))
    if weight_sum > 0.0:
        rms = float(np.sqrt(np.sum(row_weights * raw_residuals * raw_residuals) / weight_sum))
    else:
        rms = float(np.sqrt(np.mean(raw_residuals ** 2)))
    max_res = float(np.max(np.abs(raw_residuals)))
    jac_rank = int(np.linalg.matrix_rank(result.jac))

    return AnionFitResult(
        species_params={target_species: (float(result.x[0]), float(result.x[1]))},
        fixed_species={},
        n_anion_sites=len({int(row["site_index"]) for row in rows}),
        n_params=2,
        rms_residual=rms,
        max_residual=max_res,
        jacobian_rank=jac_rank,
        converged=bool(result.success),
    )


def fit_anion_centered(
    sites: Sequence[AnionSite],
    *,
    fit_species: Mapping[str, tuple[float, float]],
    fixed_species: Mapping[str, tuple[float, float]] | None = None,
    r0_bounds: tuple[float, float] = (0.3, 3.5),
    b_bounds: tuple[float, float] = (0.01, 3.0),
) -> AnionFitResult:
    """Fit (R₀, B) for one or more cation species from anion valence sums.

    Parameters
    ----------
    sites
        Anion coordination environments (from :func:`extract_anion_sites`).
    fit_species
        Species to fit, with initial (R₀, B) guesses.
        Keys are species keys like ``"Si_4"``.
    fixed_species
        Species whose (R₀, B) are held constant during the fit.
    r0_bounds, b_bounds
        Bounds for each R₀ and B parameter.

    Returns
    -------
    AnionFitResult
        Fitted parameters and diagnostics.
    """
    fixed = dict(fixed_species or {})
    fitted_keys = list(fit_species.keys())
    n_params = 2 * len(fitted_keys)

    x0 = np.empty(n_params, dtype=float)
    lower = np.empty(n_params, dtype=float)
    upper = np.empty(n_params, dtype=float)
    for i, key in enumerate(fitted_keys):
        r0_init, b_init = fit_species[key]
        x0[2 * i] = r0_init
        x0[2 * i + 1] = b_init
        lower[2 * i] = r0_bounds[0]
        lower[2 * i + 1] = b_bounds[0]
        upper[2 * i] = r0_bounds[1]
        upper[2 * i + 1] = b_bounds[1]

    result = least_squares(
        anion_valence_residuals,
        x0,
        args=(sites, fitted_keys, fixed),
        bounds=(lower, upper),
        method="trf",
    )

    species_params: dict[str, tuple[float, float]] = {}
    for i, key in enumerate(fitted_keys):
        species_params[key] = (float(result.x[2 * i]), float(result.x[2 * i + 1]))

    residuals = result.fun
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    max_res = float(np.max(np.abs(residuals)))
    jac_rank = int(np.linalg.matrix_rank(result.jac))

    return AnionFitResult(
        species_params=species_params,
        fixed_species=fixed,
        n_anion_sites=len(sites),
        n_params=n_params,
        rms_residual=rms,
        max_residual=max_res,
        jacobian_rank=jac_rank,
        converged=bool(result.success),
    )


def fit_single_species_from_anion_sites(
    sites: Sequence[AnionSite],
    target_species: str,
    *,
    known_species: Mapping[str, tuple[float, float]],
    r0_init: float = 1.6,
    b_init: float = 0.37,
    r0_bounds: tuple[float, float] = (0.3, 3.5),
    b_bounds: tuple[float, float] = (0.01, 3.0),
) -> AnionFitResult:
    """Fit one cation species while holding all others fixed.

    This is the typical use case: back-solve for a z = n species
    (e.g. Si⁴⁺ CN4, H⁺ CN1) by fixing all other cation parameters
    from the master table and fitting only the target from O valence sums.

    Parameters
    ----------
    sites
        Anion sites that include bonds to *target_species*.
    target_species
        Species key to fit (e.g. ``"Si_4"``).
    known_species
        (R₀, B) for all other cation species present in *sites*.
    r0_init, b_init
        Initial guesses for the target species.
    r0_bounds, b_bounds
        Parameter bounds.

    Returns
    -------
    AnionFitResult
    """
    relevant = [
        site for site in sites
        if any(sp == target_species for sp, _, _ in site.bonds)
    ]
    if not relevant:
        raise ValueError(f"no anion sites contain bonds to {target_species!r}")

    return fit_anion_centered(
        relevant,
        fit_species={target_species: (r0_init, b_init)},
        fixed_species=known_species,
        r0_bounds=r0_bounds,
        b_bounds=b_bounds,
    )


def _extract_sites_for_material(
    material: Any,
    *,
    anion: str = "O",
    anion_valence: float = 2.0,
) -> tuple[str, list[AnionSite]] | tuple[None, list[AnionSite]]:
    """Extract O sites and the target species key from a material object.

    Returns ``(target_key, sites)`` or ``(None, [])`` on failure.
    """
    structure_graph = getattr(material, "structure_graph", None)
    if structure_graph is None:
        return None, []
    structure = getattr(structure_graph, "structure",
                        getattr(material, "structure", None))
    if structure is None:
        return None, []

    cation = str(material.cation)
    metadata = dict(getattr(material, "metadata", {}) or {})
    oxi_state = metadata.get("oxi_state")
    if oxi_state is None:
        return None, []

    target_key = species_key(cation, int(oxi_state))

    possible_species = getattr(material, "possible_species", ())
    cation_oxi_map: dict[str, int] | None = None
    if possible_species:
        import re
        _re = re.compile(r"^([A-Z][a-z]?)(\d*)([\+\-])$")
        cation_oxi_map = {}
        for sp in possible_species:
            m = _re.match(str(sp).strip())
            if m:
                elem, charge_str, sign = m.groups()
                charge = int(charge_str) if charge_str else 1
                if sign == "-":
                    charge = -charge
                if charge > 0:
                    cation_oxi_map.setdefault(elem, charge)

    try:
        sites = extract_anion_sites(
            structure_graph, structure,
            anion=anion, anion_valence=anion_valence,
            cation_oxi_map=cation_oxi_map,
        )
    except Exception:
        return target_key, []

    return target_key, sites


def _accept_result(
    result: AnionFitResult,
    target_key: str,
    *,
    b_bounds: tuple[float, float],
    max_rms: float,
) -> bool:
    """Return True if the fit passes all quality gates."""
    if not result.converged:
        return False
    r0, b = result.species_params.get(target_key, (0.0, 0.0))
    if b >= b_bounds[1] - 0.01 or b <= b_bounds[0] + 0.001:
        return False
    if result.rms_residual > max_rms:
        return False
    if result.jacobian_rank < result.n_params:
        return False
    return True


def _result_to_payload(
    result: AnionFitResult,
    target_key: str,
    strategy: str,
) -> dict[str, Any]:
    """Package an accepted AnionFitResult as a pipeline-compatible dict."""
    r0, b = result.species_params[target_key]
    return {
        "R0": r0,
        "B": b,
        "R0_std": 0.0,
        "B_std": 0.0,
        "rss": result.rms_residual ** 2 * result.n_anion_sites,
        "fit_strategy": strategy,
        "anion_n_sites": result.n_anion_sites,
        "anion_n_species_fitted": len(result.species_params),
        "anion_n_species_fixed": len(result.fixed_species),
        "anion_rms": result.rms_residual,
        "anion_jac_rank": result.jacobian_rank,
    }


def try_anion_centered_for_material(
    material: Any,
    *,
    known_species: Mapping[str, tuple[float, float]],
    anion: str = "O",
    anion_valence: float = 2.0,
    r0_bounds: tuple[float, float] = (0.3, 3.5),
    b_bounds: tuple[float, float] = (0.01, 3.0),
    min_sites: int = 3,
    max_rms: float = 0.5,
    min_sites_per_species: int = 4,
) -> dict[str, Any] | None:
    """Attempt anion-centered fitting for one material's target cation.

    The fit proceeds through a three-level cascade, accepting the first
    result that passes quality gates:

    1. **De novo joint fit** — fit *all* cation species present in the O
       coordination shells simultaneously, with no priors from the master
       table.  Requires at least ``min_sites_per_species`` O sites per
       fitted parameter pair and full-rank Jacobian.

    2. **Partial prior** — if the joint fit is underdetermined, fix the
       species with the fewest O-site occurrences from ``known_species``
       until the system is overdetermined, then refit.

    3. **Target-only fit** — fix all non-target species from
       ``known_species`` and fit only the target.  This is the most
       constrained but always identifiable when clean O sites exist.

    Returns a dict with ``R0``, ``B``, ``rss``, ``fit_strategy`` on
    success, or ``None`` if all levels fail.
    """
    target_key, sites = _extract_sites_for_material(
        material, anion=anion, anion_valence=anion_valence,
    )
    if target_key is None or not sites:
        return None

    # Identify all cation species present in O shells that contain the target
    target_sites = [
        s for s in sites
        if any(sp == target_key for sp, _, _ in s.bonds)
    ]
    if len(target_sites) < min_sites:
        return None

    # Count O-site occurrences per species
    species_counts: dict[str, int] = {}
    for site in target_sites:
        for sp, _, _ in site.bonds:
            species_counts[sp] = species_counts.get(sp, 0) + 1
    all_species = sorted(species_counts.keys())

    # Default initial guesses: use known_species if available, else generic
    def _init(sp: str) -> tuple[float, float]:
        if sp in known_species:
            return known_species[sp]
        return (1.8, 0.37)

    # --- Level 1: De novo joint fit of ALL species ---
    n_species = len(all_species)
    n_params = 2 * n_species
    if len(target_sites) >= n_params + min_sites_per_species:
        try:
            joint_result = fit_anion_centered(
                target_sites,
                fit_species={sp: _init(sp) for sp in all_species},
                fixed_species={},
                r0_bounds=r0_bounds,
                b_bounds=b_bounds,
            )
            if _accept_result(joint_result, target_key,
                              b_bounds=b_bounds, max_rms=max_rms):
                return _result_to_payload(
                    joint_result, target_key, "anion_de_novo",
                )
        except (ValueError, ArithmeticError):
            pass

    # --- Level 2: Fix under-sampled species from master, fit the rest ---
    # Sort species by count (ascending) and fix the sparsest until the
    # remaining fitted params are well-overdetermined.
    species_by_count = sorted(all_species, key=lambda sp: species_counts[sp])
    fixed: dict[str, tuple[float, float]] = {}
    to_fit = list(all_species)

    for sp in species_by_count:
        if sp == target_key:
            continue  # never fix the target
        n_fit_params = 2 * len(to_fit)
        if len(target_sites) >= n_fit_params + min_sites_per_species:
            break  # system is overdetermined
        if sp in known_species:
            fixed[sp] = known_species[sp]
            to_fit.remove(sp)

    if len(to_fit) > 1 and len(to_fit) < n_species:
        # We fixed some species but still fitting more than just the target
        try:
            partial_result = fit_anion_centered(
                target_sites,
                fit_species={sp: _init(sp) for sp in to_fit},
                fixed_species=fixed,
                r0_bounds=r0_bounds,
                b_bounds=b_bounds,
            )
            if _accept_result(partial_result, target_key,
                              b_bounds=b_bounds, max_rms=max_rms):
                return _result_to_payload(
                    partial_result, target_key, "anion_partial_prior",
                )
        except (ValueError, ArithmeticError):
            pass

    # --- Level 3: Fix ALL non-target species from master, fit target only ---
    full_fixed: dict[str, tuple[float, float]] = {}
    all_known = True
    for sp in all_species:
        if sp == target_key:
            continue
        if sp in known_species:
            full_fixed[sp] = known_species[sp]
        else:
            all_known = False

    if not all_known:
        # Some species are unknown even in the master table — can't proceed
        return None

    clean = [
        s for s in target_sites
        if all(sp == target_key or sp in full_fixed for sp, _, _ in s.bonds)
    ]
    if len(clean) < min_sites:
        return None

    try:
        single_result = fit_single_species_from_anion_sites(
            clean,
            target_key,
            known_species=full_fixed,
            r0_init=_init(target_key)[0],
            b_init=_init(target_key)[1],
            r0_bounds=r0_bounds,
            b_bounds=b_bounds,
        )
        if _accept_result(single_result, target_key,
                          b_bounds=b_bounds, max_rms=max_rms):
            return _result_to_payload(
                single_result, target_key, "anion_centered",
            )
    except (ValueError, ArithmeticError):
        pass

    return None


def fit_single_species_from_anion_partition(
    sites: Sequence[AnionSite],
    target_species: str,
    *,
    weight_mode: str = "stored",
    r0_init: float = 1.6,
    b_init: float = 0.37,
    r0_bounds: tuple[float, float] = (0.3, 3.5),
    b_bounds: tuple[float, float] = (0.01, 3.0),
) -> AnionFitResult:
    """Fit one cation species from an anion-side angular partition proxy.

    The fit uses bond-level pseudo-valences ``s = |z_anion| * omega`` assigned
    from the anion-centered bond weights on each site.
    """
    rows = build_anion_partition_rows(
        sites,
        target_species,
        weight_mode=weight_mode,
    )
    if not rows:
        raise ValueError(f"no partition rows contain bonds to {target_species!r}")

    return _fit_single_species_from_partition_rows(
        rows,
        target_species,
        r0_init=r0_init,
        b_init=b_init,
        r0_bounds=r0_bounds,
        b_bounds=b_bounds,
    )


def fit_hydrogen_from_proton_partition(
    shells: Sequence[ProtonShell],
    *,
    r0_init: float = 1.029,
    b_init: float = 0.37,
    r0_bounds: tuple[float, float] = (0.3, 2.0),
    b_bounds: tuple[float, float] = (0.01, 3.0),
) -> AnionFitResult:
    """Fit H-O ``(R₀, B)`` from proton-centered partition rows."""
    rows = build_proton_partition_rows(shells)
    if not rows:
        raise ValueError("no proton partition rows")

    return _fit_single_species_from_partition_rows(
        rows,
        "H_1",
        r0_init=r0_init,
        b_init=b_init,
        r0_bounds=r0_bounds,
        b_bounds=b_bounds,
    )
