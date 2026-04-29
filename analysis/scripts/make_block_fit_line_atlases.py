"""Generate coordination-number-resolved fit-line atlases for every block.

This script complements ``make_dblock_beta_prl_figure.py``: it uses the same
authoritative 103-species oxygen dataset and ``build_unified_oxygen_cn_fits``
pipeline, then groups species by family and renders one or more atlas pages
per family using ``export_species_fit_line_atlas`` (the same routine that
produces the d-block atlas already in the Supplemental Material).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, F401  (used implicitly)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.exceptions import UndefinedMetricWarning  # noqa: E402

from critmin.analysis.bond_valence_store import (  # noqa: E402
    DEFAULT_OXYGEN_AUTHORITATIVE_SOURCE_SPEC,
)
from critmin.analysis.bond_valence_theory import (  # noqa: E402
    build_unified_oxygen_cn_fits,
    load_grouped_payload_json,
    observed_coordination_numbers,
    oxidation_state_species_label,
)
from critmin.viz.notebook_families import (  # noqa: E402
    export_species_fit_line_atlas,
)
from critmin.viz.notebook_setup import init_notebook  # noqa: E402


FIGURES_DIR = ROOT / "theory" / "figures"
THEORY_FIGURES_DIR = ROOT / "theory" / "figures"
MASTER_SUMMARY_PATH = (
    ROOT / "data" / "processed" / "theory" / "master_oxygen_summary_theory.json"
)

_SUPERSCRIPT_TO_DIGIT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


# Tab10-derived palette for non-d-block elements; falls back to a fixed gray.
_PALETTE: dict[str, str] = {
    # s-block
    "H":  "#888888", "Li": "#1f77b4", "Na": "#ff7f0e", "K":  "#2ca02c",
    "Rb": "#d62728", "Cs": "#9467bd",
    "Be": "#8c564b", "Mg": "#e377c2", "Ca": "#7f7f7f", "Sr": "#bcbd22",
    "Ba": "#17becf",
    # p-block
    "B":  "#332288", "C":  "#117733", "N":  "#44AA99", "O":  "#88CCEE",
    "Al": "#DDCC77", "Si": "#999933", "P":  "#CC6677", "S":  "#882255",
    "Cl": "#AA4499",
    "Ga": "#1B9E77", "Ge": "#D95F02", "As": "#7570B3", "In": "#E7298A",
    "Sn": "#66A61E", "Sb": "#E6AB02", "Te": "#A6761D", "I":  "#666666",
    "Tl": "#1F78B4", "Pb": "#FB9A99", "Bi": "#FDBF6F",
    # f-block (lanthanides)
    "La": "#332288", "Ce": "#88CCEE", "Pr": "#44AA99", "Nd": "#117733",
    "Sm": "#999933", "Eu": "#DDCC77", "Gd": "#CC6677", "Tb": "#882255",
    "Dy": "#AA4499", "Ho": "#332288", "Er": "#88CCEE", "Tm": "#44AA99",
    "Yb": "#117733", "Lu": "#999933",
    # actinides
    "Th": "#332288", "U":  "#882255",
}


def parse_charge_from_label(label: str) -> tuple[str, int | None]:
    plain = str(label).translate(_SUPERSCRIPT_TO_DIGIT)
    element = "".join(char for char in plain if char.isalpha())
    suffix = plain[len(element):]
    sign = -1 if "-" in suffix else 1
    digits = "".join(char for char in suffix if char.isdigit())
    charge = sign * int(digits) if digits else None
    return element, charge


def master_species_rows() -> list[dict]:
    return json.loads(MASTER_SUMMARY_PATH.read_text(encoding="utf-8"))["master_rows"]


def build_authoritative_species_payload(
    species_rows: list[dict],
) -> tuple[dict[str, dict[str, list[dict]]], list[tuple[str, str, int, str]]]:
    """Return (species_payload, species_index) where species_index also carries
    the family_key for each species so the caller can re-key by family without
    relying on master-row label conventions (which differ across families)."""
    grouped = load_grouped_payload_json(DEFAULT_OXYGEN_AUTHORITATIVE_SOURCE_SPEC)
    species_payload: dict[str, dict[str, list[dict]]] = {}
    species_index: list[tuple[str, str, int, str]] = []
    for row in species_rows:
        family_key = str(row.get("family_key") or "")
        label = str(row.get("species") or row.get("cation") or row.get("label") or "")
        element, charge = parse_charge_from_label(label)
        if charge is None:
            if family_key == "group1":
                charge = 1
            elif family_key == "group2":
                charge = 2
            elif family_key == "lanthanides":
                charge = 3
            else:
                raise ValueError(
                    f"Cannot infer oxidation state for master row: {row}"
                )
            label = oxidation_state_species_label(element, charge)
        species_index.append((label, element, charge, family_key))
        cat_data = grouped.get(element, {"oxides": [], "hydroxides": []})
        bucket_payload: dict[str, list[dict]] = {"oxides": [], "hydroxides": []}
        for bucket in ("oxides", "hydroxides"):
            for record in cat_data.get(bucket, []):
                if record.get("status", "fitted") != "fitted":
                    continue
                if record.get("oxi_state_label") == "mixed":
                    continue
                record_charge = record.get("oxi_state")
                if record_charge is None:
                    continue
                if int(record_charge) == charge:
                    bucket_payload[bucket].append(record)
        species_payload[label] = bucket_payload
    return species_payload, species_index


def _color_for(element: str) -> str:
    return _PALETTE.get(element, "#444444")


# Block groupings for the SI.  Lanthanides and actinides are combined into a
# single f-block atlas because together they only fill two atlas pages.
BLOCK_GROUPS: dict[str, dict[str, list[str]]] = {
    "group2_oxygen_cn_fit_lines": {
        "title": "Group 2",
        "family_keys": ["group2"],
    },
    "pblock_oxygen_cn_fit_lines": {
        "title": "p-block",
        "family_keys": ["post_transition_pblock", "nonmetal_halogen"],
    },
    "fblock_oxygen_cn_fit_lines": {
        "title": "f-block",
        "family_keys": ["lanthanides", "actinides"],
    },
}


def build_block_atlases() -> dict[str, list[str]]:
    init_notebook()
    rows = master_species_rows()
    species_payload, species_index = build_authoritative_species_payload(rows)
    species_labels = [label for label, _e, _c, _f in species_index]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        cn_fits, _outliers = build_unified_oxygen_cn_fits(
            species_payload,
            species_labels,
            target_cns=observed_coordination_numbers(species_payload, species_labels),
        )

    outputs: dict[str, list[str]] = {}
    for prefix, spec in BLOCK_GROUPS.items():
        family_keys = set(spec["family_keys"])
        block_labels = [
            label for label, _e, _c, family_key in species_index
            if family_key in family_keys
        ]
        if not block_labels:
            continue
        block_target_cns = sorted({
            int(cn)
            for label in block_labels
            for cn in cn_fits.get(label, {}).keys()
            if isinstance(cn, int)
        })
        if not block_target_cns:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            files = export_species_fit_line_atlas(
                cn_fits,
                block_labels,
                block_target_cns,
                filename_prefix=prefix,
                figures_dir=FIGURES_DIR,
                theory_figures_dir=THEORY_FIGURES_DIR,
                color_fn=lambda label: _color_for(parse_charge_from_label(label)[0]),
            )
        outputs[prefix] = list(files)
        print(f"{spec['title']}: {len(block_labels)} species -> {files}")
    return outputs


def main() -> None:
    outputs = build_block_atlases()
    if not outputs:
        raise RuntimeError("No block atlases were produced.")


if __name__ == "__main__":
    main()
