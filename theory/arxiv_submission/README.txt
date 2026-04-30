This directory is a clean arXiv upload bundle for the PRL manuscript.

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
