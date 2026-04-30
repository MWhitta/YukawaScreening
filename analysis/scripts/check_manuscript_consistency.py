#!/usr/bin/env python3
"""Check that the checked-in manuscript assets agree with their structured sources."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

README_PATH = REPO_ROOT / "README.md"
MAIN_TEX_PATH = REPO_ROOT / "theory" / "prl_main.tex"
SI_TEX_PATH = REPO_ROOT / "theory" / "prl_supplemental.tex"
LAMBDA_TABLE_PATH = REPO_ROOT / "theory" / "si_lambda_table.tex"
FIGURES_DOC_PATH = REPO_ROOT / "theory" / "FIGURE_SOURCES.md"
FIGURES_DIR = REPO_ROOT / "theory" / "figures"
MASTER_SUMMARY_PATH = REPO_ROOT / "data" / "processed" / "theory" / "master_oxygen_summary_theory.json"
BENCHMARK_PATH = REPO_ROOT / "data" / "processed" / "theory" / "charge_density_benchmark.json"
CITATION_PATH = REPO_ROOT / "CITATION.cff"
ZENODO_PATH = REPO_ROOT / ".zenodo.json"
METADATA_README_PATH = REPO_ROOT / "metadata" / "README.md"
DATA_DICTIONARY_PATH = REPO_ROOT / "metadata" / "DATA_DICTIONARY.md"
DATA_LICENSES_PATH = REPO_ROOT / "metadata" / "DATA_LICENSES.md"
ARTIFACT_CATALOG_PATH = REPO_ROOT / "metadata" / "artifact_catalog.json"


def _expect(condition: bool, message: str, *, errors: list[str], successes: list[str]) -> None:
    if condition:
        successes.append(message)
    else:
        errors.append(message)


def _parse_figure_refs(*tex_paths: Path) -> list[str]:
    pattern = re.compile(r"includegraphics\[[^\]]*\]\{([^}]+)\}")
    refs: list[str] = []
    for path in tex_paths:
        refs.extend(pattern.findall(path.read_text(encoding="utf-8")))
    return refs


def _count_lambda_rows(text: str) -> int:
    pattern = re.compile(r"^\s+[A-Z][A-Za-z]?\$\^\{[^}]+\}\$.*\\\\$", re.MULTILINE)
    return len(pattern.findall(text))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def main() -> int:
    errors: list[str] = []
    successes: list[str] = []

    main_tex = MAIN_TEX_PATH.read_text(encoding="utf-8")
    si_tex = SI_TEX_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    figure_refs = _parse_figure_refs(MAIN_TEX_PATH, SI_TEX_PATH)
    for ref in figure_refs:
        _expect(
            (FIGURES_DIR / ref).is_file(),
            f"figure asset exists: {ref}",
            errors=errors,
            successes=successes,
        )

    _expect(FIGURES_DOC_PATH.is_file(), "figure source map exists", errors=errors, successes=successes)
    _expect(CITATION_PATH.is_file(), "CITATION.cff exists", errors=errors, successes=successes)
    _expect(ZENODO_PATH.is_file(), ".zenodo.json exists", errors=errors, successes=successes)
    _expect(METADATA_README_PATH.is_file(), "metadata README exists", errors=errors, successes=successes)
    _expect(DATA_DICTIONARY_PATH.is_file(), "data dictionary exists", errors=errors, successes=successes)
    _expect(DATA_LICENSES_PATH.is_file(), "data license notes exist", errors=errors, successes=successes)
    _expect(ARTIFACT_CATALOG_PATH.is_file(), "artifact catalog exists", errors=errors, successes=successes)

    if ARTIFACT_CATALOG_PATH.is_file():
        catalog = json.loads(ARTIFACT_CATALOG_PATH.read_text(encoding="utf-8"))
        artifacts = catalog.get("artifacts", [])
        _expect(len(artifacts) == 17, "artifact catalog covers 17 manuscript-critical data artifacts", errors=errors, successes=successes)
        required_artifact_paths = {
            "data/processed/bond_valence/consolidated_store.json",
            "data/processed/theory/master_oxygen_summary_theory.json",
            "data/processed/theory/charge_density_benchmark.json",
            "data/charge_density/mp_id_manifest.json",
        }
        catalog_paths = {str(entry.get("path")) for entry in artifacts}
        for required_path in sorted(required_artifact_paths):
            _expect(
                required_path in catalog_paths,
                f"artifact catalog includes {required_path}",
                errors=errors,
                successes=successes,
            )
        for entry in artifacts:
            rel_path = Path(str(entry["path"]))
            abs_path = REPO_ROOT / rel_path
            _expect(
                abs_path.is_file(),
                f"catalog entry exists on disk: {rel_path.as_posix()}",
                errors=errors,
                successes=successes,
            )
            if abs_path.is_file():
                _expect(
                    _sha256(abs_path) == str(entry.get("sha256")),
                    f"catalog hash matches {rel_path.as_posix()}",
                    errors=errors,
                    successes=successes,
                )

    master_payload = json.loads(MASTER_SUMMARY_PATH.read_text(encoding="utf-8"))
    master_rows = master_payload["master_rows"]
    positive_b = sum(1 for row in master_rows if float(row["B_star"]) > 0.0)
    negative_b = sum(1 for row in master_rows if float(row["B_star"]) < 0.0)
    _expect(len(master_rows) == 103, "master summary has 103 species rows", errors=errors, successes=successes)
    _expect(positive_b == 101, "master summary has 101 screening-branch species", errors=errors, successes=successes)
    _expect(negative_b == 2, "master summary has 2 anti-screening species", errors=errors, successes=successes)

    lambda_table = LAMBDA_TABLE_PATH.read_text(encoding="utf-8")
    _expect(
        _count_lambda_rows(lambda_table) == len(master_rows),
        "SI lambda table row count matches master summary",
        errors=errors,
        successes=successes,
    )

    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    benchmark_points = benchmark["points"]
    counts = benchmark["summary"]["counts"]
    _expect(counts["total_points"] == 35, "charge-density benchmark has 35 plotted oxides", errors=errors, successes=successes)
    _expect(counts["panel_a_points"] == 10, "charge-density panel a has 10 oxides", errors=errors, successes=successes)
    _expect(counts["panel_b_points"] == 25, "charge-density panel b has 25 oxides", errors=errors, successes=successes)
    _expect(counts["panel_b_fit_points"] == 21, "charge-density panel b fit subset has 21 oxides", errors=errors, successes=successes)
    _expect(counts["panel_b_outliers"] == 4, "charge-density panel b has 4 explicit outliers", errors=errors, successes=successes)
    _expect(
        all("block" in point for point in benchmark_points),
        "charge-density benchmark stores an explicit block label for every point",
        errors=errors,
        successes=successes,
    )
    explicit_block_counts = Counter(str(point["block"]) for point in benchmark_points)
    _expect(
        explicit_block_counts == Counter({"s": 10, "d": 13, "p": 9, "f": 3}),
        "charge-density benchmark stores the expected explicit block counts",
        errors=errors,
        successes=successes,
    )
    panel_a_block_counts = Counter(
        str(point["block"])
        for point in benchmark_points
        if str(point["panel"]) == "a"
    )
    _expect(
        panel_a_block_counts == Counter({"s": 10}),
        "charge-density panel a explicit block tags are all s-block",
        errors=errors,
        successes=successes,
    )
    panel_b_block_counts = Counter(
        str(point["block"])
        for point in benchmark_points
        if str(point["panel"]) == "b"
    )
    _expect(
        panel_b_block_counts == Counter({"d": 13, "p": 9, "f": 3}),
        "charge-density panel b explicit block coverage matches the manuscript fit set",
        errors=errors,
        successes=successes,
    )
    outlier_elements = {
        str(point["element"])
        for point in benchmark_points
        if bool(point["is_outlier"])
    }
    _expect(
        outlier_elements == {"As", "Au", "B", "P"},
        "charge-density benchmark retains the four explicit panel-b outliers",
        errors=errors,
        successes=successes,
    )

    panel_a_fit = benchmark["summary"]["panel_a_fit"]
    panel_b_fit = benchmark["summary"]["panel_b_fit"]
    null_models = benchmark["summary"]["panel_a_null_models"]
    shift = benchmark["summary"]["panel_a_oxygen_shift"]

    prl_figure_size = _png_size(FIGURES_DIR / "thomas_fermi_reff_prl.png")
    prl_aspect = prl_figure_size[0] / prl_figure_size[1]
    _expect(
        3.2 <= prl_aspect <= 4.2,
        "PRL Fig. 2 export preserves the original flat two-panel aspect",
        errors=errors,
        successes=successes,
    )

    _expect(
        re.search(r"31\s+binary oxides", readme) is not None,
        "README still reports the 31-oxide validation subset",
        errors=errors,
        successes=successes,
    )
    _expect("R^2=0.9998" in main_tex, "main text retains the panel-a rounded R^2", errors=errors, successes=successes)
    _expect("R^2=0.967" in main_tex, "main text retains the panel-b rounded R^2", errors=errors, successes=successes)
    _expect(
        f"{float(panel_a_fit['slope']):.4f}" in si_tex and f"{float(panel_a_fit['intercept']):.4f}" in si_tex,
        "SI text contains the panel-a fit coefficients from the benchmark JSON",
        errors=errors,
        successes=successes,
    )
    _expect(
        f"{float(panel_b_fit['slope']):.4f}" in MAIN_TEX_PATH.read_text(encoding="utf-8") or "0.98" in main_tex,
        "main text contains the panel-b fit summary",
        errors=errors,
        successes=successes,
    )
    _expect(
        f"{float(shift['mean']):.4f}" in si_tex and f"{float(shift['std']):.4f}" in si_tex,
        "SI text contains the oxygen-shift mean and spread",
        errors=errors,
        successes=successes,
    )
    _expect(
        f"{float(null_models['mae_m_i']):.3f}" in si_tex
        and f"{float(null_models['mae_R0']):.3f}" in si_tex
        and f"{float(null_models['mae_shannon_radius']):.3f}" in si_tex,
        "SI text contains the null-model MAE triplet",
        errors=errors,
        successes=successes,
    )

    _expect(
        "44{,}844" in si_tex,
        "SI text reports the bond-length-identity sample size from the exporter",
        errors=errors,
        successes=successes,
    )
    _expect(
        "$r=0.976$" in si_tex and "MAE$\\,=0.019$" in si_tex,
        "SI text reports the bond-length-identity Pearson r and MAE from the exporter",
        errors=errors,
        successes=successes,
    )

    for message in successes:
        print(f"OK: {message}")
    for message in errors:
        print(f"ERROR: {message}", file=sys.stderr)

    if errors:
        print(f"\nConsistency check failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print(f"\nConsistency check passed with {len(successes)} checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
