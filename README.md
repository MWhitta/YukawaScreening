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

## Repository layout

```
YukawaScreening/
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
- `scripts/` — three figure / table generators:
  - `make_dblock_beta_prl_figure.py` — Fig. 1 ($\beta$ vs. $z$).
  - `make_block_fit_line_atlases.py` — coordination-resolved $B$–$R_0$
    atlas figures used in the Supplemental Material.
  - `export_theory_lambda_tables.py` — `si_lambda_table.tex` and
    related $\lambda^*$ exports.
- `notebooks/thomas_fermi_screening.ipynb` — the charge-density
  centroid analysis (PRL Fig. 2 and the SI null-model controls).

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
PYTHONPATH=analysis python analysis/scripts/export_theory_lambda_tables.py
```

### `data/`

- `data/processed/theory/` — per-species summary records used by the
  figure scripts. `master_oxygen_summary_theory.json` contains the
  103-species characteristic pairs $(R_0^*, B^*, \lambda^*)$ tabulated
  in the SI; `manuscript_theory_manifest.json` lists the source
  trackers and aggregate statistics.
- `data/processed/bond_valence/consolidated_store.json` — the
  oxygen-authoritative bond-valence fit store (per-structure $(R_0, B)$
  fits across all 103 species, ~20K records). This is the input the
  figure scripts read.
- `data/raw/ima/*.json` — IMA-derived candidate trackers identifying
  which Materials Project entries were queried for each cation
  family. These are referenced from
  `data/processed/theory/manuscript_theory_manifest.json`.
- `data/charge_density/mp_id_manifest.json` — Materials Project IDs
  for the structures whose DFT-relaxed CHGCAR files were used to
  compute the screening centroid $r_{\mathrm{eff}}$ in Fig. 2 and the
  SI null-model controls. **Charge-density files are not bundled in
  this repository.** To regenerate, install the optional
  `charge-density` extra and use
  `MPRester().get_charge_density_from_material_id(mpid)` for each
  entry; the notebook
  `analysis/notebooks/thomas_fermi_screening.ipynb` consumes the
  resulting CHGCAR objects.

All Materials Project structures are publicly available at
[https://materialsproject.org](https://materialsproject.org).

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
