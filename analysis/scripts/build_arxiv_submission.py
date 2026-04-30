#!/usr/bin/env python3
"""Build a clean arXiv submission bundle for the PRL manuscript."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THEORY_DIR = REPO_ROOT / "theory"
OUTPUT_DIR = THEORY_DIR / "arxiv_submission"
ARCHIVE_PATH = THEORY_DIR / "arxiv_submission.tar.gz"

MAIN_SOURCE = THEORY_DIR / "prl_main.tex"
MAIN_BBL = THEORY_DIR / "prl_main.bbl"
MAIN_BIB = THEORY_DIR / "references.bib"
MAIN_FIGURES = [
    THEORY_DIR / "figures" / "prl_oxygen_beta_vs_charge_cn4_cn6.png",
    THEORY_DIR / "figures" / "thomas_fermi_reff_prl.png",
]
SUPPLEMENTAL_PDF = THEORY_DIR / "prl_supplemental.pdf"


README_TEXT = """This directory is a clean arXiv upload bundle for the PRL manuscript.

Upload `arxiv_submission.tar.gz` to arXiv rather than the full repository or the full `theory/` directory.

Bundle contents:
- `ms.tex`: main manuscript source
- `ms.bbl`: frozen bibliography for the main manuscript
- `references.bib`: included as a fallback if arXiv invokes BibTeX
- `figures/`: only the two figures used by the main manuscript
- `supplemental_material.pdf`: upload this as an ancillary file

This bundle intentionally excludes:
- `cover_letter.tex`
- `prl_supplemental.tex`
- auxiliary TeX build files
- manuscript-source figures used only in the Supplemental Material

The bundle is validated locally by running `pdflatex` twice on `ms.tex` in a clean temporary directory.
"""


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_readme(dst: Path) -> None:
    dst.write_text(README_TEXT, encoding="utf-8")


def build_bundle_tree() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    copy_file(MAIN_SOURCE, OUTPUT_DIR / "ms.tex")
    copy_file(MAIN_BBL, OUTPUT_DIR / "ms.bbl")
    copy_file(MAIN_BIB, OUTPUT_DIR / "references.bib")
    copy_file(SUPPLEMENTAL_PDF, OUTPUT_DIR / "supplemental_material.pdf")

    for figure in MAIN_FIGURES:
        copy_file(figure, OUTPUT_DIR / "figures" / figure.name)

    write_readme(OUTPUT_DIR / "README.txt")


def verify_bundle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        shutil.copytree(OUTPUT_DIR, tmp_path / "bundle", dirs_exist_ok=True)
        bundle_dir = tmp_path / "bundle"
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "ms.tex"],
                cwd=bundle_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "arXiv bundle verification failed:\n"
                    f"{result.stdout}\n{result.stderr}"
                )


def write_archive() -> None:
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(OUTPUT_DIR))


def main() -> None:
    build_bundle_tree()
    verify_bundle()
    write_archive()
    print(f"Built {OUTPUT_DIR}")
    print(f"Built {ARCHIVE_PATH}")


if __name__ == "__main__":
    main()
