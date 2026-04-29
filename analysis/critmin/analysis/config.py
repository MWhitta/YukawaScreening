"""Central configuration constants for the CritMin analysis package.

Gathers element data, bond-valence fitting parameters, and plot styling
into one importable location so that values are defined once and shared
across classification, speciation, thermodynamics, bond-valence, and
visualization modules.
"""

from __future__ import annotations

# ── Element constants ────────────────────────────────────────────────────────

ELEMENT_Z: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}

Z_TO_SYMBOL: dict[int, str] = {v: k for k, v in ELEMENT_Z.items()}
Z_TO_SYMBOL[0] = "\u25fb"  # site vacancy

ANIONS: set[str] = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "C", "H", "REE"}

# Electronegativity-based anion classification.
# "always_anion": elements that are always the anion in mineral formulas.
# "amphoteric": elements that can act as either cation or anion depending on
#   the mineral.  When oxidation state data is available, negative states -> anion.
#   When unavailable, electronegativity relative to partners determines the role.
# Elements not in either set are always cations.
ALWAYS_ANION: set[str] = {"O", "F", "Cl", "Br", "I"}
AMPHOTERIC: set[str] = {"S", "Se", "Te", "N", "C", "H", "P", "As", "Sb"}

# Si is assumed to be Si4+ (cation) in all minerals.  This holds for
# tetrahedral silicates (CN=4), which account for the vast majority of
# Si-bearing minerals.  The rare silicide exceptions (Si4-, CN!=4) should
# be reclassified once coordination-number data is available.

# Pauling electronegativities for disambiguation
ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.20, "He": 0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 0,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16,
    "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Pm": 0, "Sm": 1.17, "Eu": 0, "Gd": 1.20, "Tb": 0, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 0, "Lu": 1.27,
    "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20,
    "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
    "Po": 2.00, "At": 2.20, "Rn": 0, "Fr": 0.70, "Ra": 0.90, "Ac": 1.10,
    "Th": 1.30, "Pa": 1.50, "U": 1.38,
}

# Periodic table block membership (elements present in mineral data)
PERIODIC_BLOCKS: dict[str, set[str]] = {
    "s": {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"},
    "d": {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    },
    "f": {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U",
    },
    "p": {
        "B", "Al", "Ga", "In", "Tl", "C", "Si", "Ge", "Sn", "Pb",
        "N", "P", "As", "Sb", "Bi", "O", "S", "Se", "Te",
        "F", "Cl", "Br", "I",
    },
}

BLOCK_COLORS: dict[str, str] = {
    "s": "#e74c3c",  # red -- alkali / alkaline earth
    "d": "#3498db",  # blue -- transition metals
    "f": "#2ecc71",  # green -- lanthanides / actinides
    "p": "#9b59b6",  # purple -- metalloids / post-transition
}


def get_block(elem: str) -> str:
    """Return the periodic-table block letter ('s', 'd', 'f', or 'p') for *elem*."""
    for block, members in PERIODIC_BLOCKS.items():
        if elem in members:
            return block
    return "p"


# ── Bond-valence fitting defaults ────────────────────────────────────────────

DEFAULT_R0_BOUNDS: tuple[float, float] = (-5.0, 10.0)
DEFAULT_MAX_R0_STD: float = 0.1
DEFAULT_MAX_B_STD: float = 0.1
DEFAULT_DEGENERATE_B_PRIOR: float = 0.37
DEFAULT_DEGENERATE_CONDITION_NUMBER: float = 1.0e8
DEFAULT_DEGENERATE_LOG_SIJ_RANGE: float = 1.0e-6
DEFAULT_DEGENERATE_REGULARIZATION: float = 10.0
DEFAULT_DEGENERATE_CN_LINE_REGULARIZATION: float = 10.0

# ── D-block cation groups for BV fitting ────────────────────────────────────

DBLOCK_3D_CATIONS: tuple[str, ...] = (
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
)

DBLOCK_4D_CATIONS: tuple[str, ...] = (
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
)

POST_TRANSITION_CATIONS: tuple[str, ...] = (
    "In", "Sn", "Sb", "Te",
)

POST_TRANSITION_REMAINING_O_CATIONS: tuple[str, ...] = (
    "B", "Al", "Ga", "Tl",
    "Si", "Ge", "Pb",
    "P", "As", "Bi",
    "S", "Se",
)

POST_TRANSITION_OXYGEN_CATIONS: tuple[str, ...] = (
    "B", "Al", "Ga", "In", "Tl",
    "Si", "Ge", "Sn", "Pb",
    "P", "As", "Sb", "Bi",
    "S", "Se", "Te",
)

NONMETAL_HALOGEN_OXYGEN_CATIONS: tuple[str, ...] = (
    "H", "C", "N", "F", "Cl", "Br", "I",
)

DBLOCK_4D_POST_CATIONS: tuple[str, ...] = DBLOCK_4D_CATIONS + POST_TRANSITION_CATIONS

DBLOCK_5D_CATIONS: tuple[str, ...] = (
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
)

ACTINIDE_CATIONS: tuple[str, ...] = (
    "Ac", "Th", "Pa", "U",
)

DBLOCK_COMMON_OXI_STATES: dict[str, tuple[int, ...]] = {
    "Sc": (3,),
    "Ti": (3, 4),
    "V": (2, 3, 4, 5),
    "Cr": (2, 3, 6),
    "Mn": (2, 3, 4, 7),
    "Fe": (2, 3),
    "Co": (2, 3),
    "Ni": (2, 3),
    "Cu": (1, 2),
    "Zn": (2,),
}
