# YukawaScreening

Code, data, and manuscript artifacts for:

> **Yukawa screening derivation of the bond-valence rule**
> M. L. Whittaker, P. Wang, C. Li, N. Katyal, P. Zarzycki
> *Physical Review Letters* (submitted, 2026)

The bond-valence model has been the standard tool for crystal-structure
validation since Brown and Altermatt (1985), but its exponential
distance dependence had never been derived from a specific physical
interaction. This work shows that a first-order Yukawa expansion
recovers the exponential, predicts the parameter correlation
$\beta = 1/\ln(z/n)$ in closed form with no adjustable constants, and
ties the fitted softness $B$ to a measurable electronic screening
length, validated against first-principles charge densities for 31
binary oxides.

Repository-level FAIR metadata lives in `metadata/` and includes a
machine-readable artifact catalog (`metadata/artifact_catalog.json`), a
data dictionary, and license notes for the manuscript-critical data
products.

## Repository layout

```
YukawaScreening/
├── metadata/      FAIR metadata, data dictionary, and license notes
├── theory/        Manuscript files (LaTeX, figures, bib, compiled PDFs)
├── analysis/      Python package and scripts that produce the figures and tables
└── data/          Inputs and outputs of the analysis pipeline
```

### `theory/`

- `prl_main.tex`, `prl_supplemental.tex`, `cover_letter.tex`,
  `si_lambda_table.tex` — manuscript LaTeX source.
- `prl_main.pdf`, `prl_supplemental.pdf`, `cover_letter.pdf` — current
  compiled output.
- `references.bib` — bibliography.
- `figures/` — 20 PNG figures referenced by the manuscript.
- `FIGURE_SOURCES.md` — source map for every manuscript figure asset.

To rebuild:

```bash
cd theory
latexmk -pdf prl_main.tex
latexmk -pdf prl_supplemental.tex
latexmk -pdf cover_letter.tex
```

### `analysis/`

- `critmin/analysis/` — the bond-valence theory modules used by the
  PRL: shell-thermodynamics partition function, Yukawa weight
  inversion, characteristic-pair fitting, valence-sum rule solver.
- `critmin/viz/` — figure-style and family-plot helpers.
- `scripts/` — manuscript-facing exporters and checks:
  - `make_dblock_beta_prl_figure.py` — Fig. 1 ($\beta$ vs. $z$).
  - `make_block_fit_line_atlases.py` — coordination-resolved $B$–$R_0$
    atlas figures used in the Supplemental Material (Group 1, Group 2,
    $d$ block, $p$ block, and $f$ block).
  - `make_characteristic_pair_summary_figure.py` — the Group 1 / Group
    2 / lanthanide $(R_0^*, B^*)$ summary panel.
  - `make_charge_density_benchmark_figures.py` — PRL Fig. 2 and the SI
    null-model control figure from the checked-in benchmark JSON.
  - `make_bond_length_identity_parity_figure.py` — SI bond-length
    identity parity figure (`r0_bond_length_identity_parity.png`).
  - `export_theory_lambda_tables.py` — `si_lambda_table.tex` and
    related $\lambda^*$ exports.
  - `build_fair_metadata.py` — regenerates the machine-readable FAIR
    artifact catalog with SHA256 hashes and content summaries.
  - `check_manuscript_consistency.py` — validates the figure inventory,
    master-summary counts, lambda-table row count, and the structured
    charge-density benchmark claims used in the manuscript, plus the
    FAIR metadata layer.
- `notebooks/thomas_fermi_screening.ipynb` — exploratory charge-density
  analysis notebook. It is no longer the manuscript-facing source of
  record for Fig. 2; the checked-in
  `data/processed/theory/charge_density_benchmark.json` file is.

To install:

```bash
cd analysis
python -m venv .venv && source .venv/bin/activate
pip install -e .
# To regenerate charge-density data:
pip install -e ".[charge-density]"
```

To regenerate figures and tables (run from the repo root so relative
data paths resolve):

```bash
PYTHONPATH=analysis python analysis/scripts/make_dblock_beta_prl_figure.py
PYTHONPATH=analysis python analysis/scripts/make_block_fit_line_atlases.py
python analysis/scripts/make_characteristic_pair_summary_figure.py
python analysis/scripts/make_charge_density_benchmark_figures.py
PYTHONPATH=analysis python analysis/scripts/make_bond_length_identity_parity_figure.py
PYTHONPATH=analysis python analysis/scripts/export_theory_lambda_tables.py
python analysis/scripts/build_fair_metadata.py
python analysis/scripts/check_manuscript_consistency.py
```

### `data/`

- `data/processed/theory/` — per-species summary records used by the
  figure scripts. `master_oxygen_summary_theory.json` contains the
  103-species characteristic pairs $(R_0^*, B^*, \lambda^*)$ tabulated
  in the SI; `manuscript_theory_manifest.json` lists the source
  trackers and aggregate statistics. `charge_density_benchmark.json`
  is the authoritative checked-in point set and regression summary for
  PRL Fig. 2 and the SI null-model control.
- `data/processed/bond_valence/consolidated_store.json` — the
  oxygen-authoritative bond-valence fit store (per-structure $(R_0, B)$
  fits across all 103 species, ~20K records). This is the input the
  figure scripts read.
- `data/raw/ima/*.json` — IMA-derived candidate trackers identifying
  which Materials Project entries were queried for each cation
  family. These are referenced from
  `data/processed/theory/manuscript_theory_manifest.json`.
- `data/charge_density/mp_id_manifest.json` — Materials Project ID
  manifest used by the charge-density workflow. **Charge-density files
  are not bundled in this repository.** The checked-in manuscript
  source of record is
  `data/processed/theory/charge_density_benchmark.json`; to re-derive
  that benchmark from CHGCAR objects, install the optional
  `charge-density` extra and use
  `MPRester().get_charge_density_from_material_id(mpid)` for each
  entry. The exploratory notebook
  `analysis/notebooks/thomas_fermi_screening.ipynb` documents that
  workflow and consumes the resulting CHGCAR objects.

All Materials Project structures are publicly available at
[https://materialsproject.org](https://materialsproject.org).

## FAIR Metadata

- `CITATION.cff` and `.zenodo.json` provide repository-level citation
  and archival metadata.
- `metadata/artifact_catalog.json` is the machine-readable catalog of
  manuscript-critical data artifacts with SHA256 hashes, sizes,
  descriptions, and content summaries.
- `metadata/DATA_DICTIONARY.md` documents the main JSON/CSV schemas and
  field meanings.
- `metadata/DATA_LICENSES.md` explains the split between the MIT
  license for code/text and the Materials Project terms governing the
  redistributed derived data.

## Citation

If you use this code or data, please cite:

```bibtex
@article{Whittaker2026YukawaBV,
  author  = {Whittaker, Michael L. and Wang, Pan and Li, Chunhui and
             Katyal, Naman and Zarzycki, Piotr},
  title   = {Yukawa screening derivation of the bond-valence rule},
  journal = {Physical Review Letters},
  year    = {2026},
  note    = {submitted}
}
```

## License

Code and manuscript text are released under the MIT License (see
`LICENSE`). Materials Project data redistributed via summary files is
covered by the Materials Project terms of use.
