# Figure Sources

This file maps every manuscript figure asset in [`theory/figures`](./figures) to the checked-in source that drives it.

## Main Text

| Figure file | Source of record | Regeneration path |
| --- | --- | --- |
| `prl_oxygen_beta_vs_charge_cn4_cn6.png` | `data/processed/bond_valence/consolidated_store.json::oxygen_authoritative` + `data/processed/theory/master_oxygen_summary_theory.json` | `PYTHONPATH=analysis python analysis/scripts/make_dblock_beta_prl_figure.py` |
| `thomas_fermi_reff_prl.png` | `data/processed/theory/charge_density_benchmark.json` | `python analysis/scripts/make_charge_density_benchmark_figures.py` |

## Supplemental Material

| Figure file(s) | Source of record | Regeneration path |
| --- | --- | --- |
| `group1_oxygen_cn_fit_lines.png` | `data/processed/bond_valence/consolidated_store.json::oxygen_authoritative` + `data/processed/theory/master_oxygen_summary_theory.json` | `PYTHONPATH=analysis python analysis/scripts/make_block_fit_line_atlases.py` |
| `group2_oxygen_cn_fit_lines_part01.png` | same as above | `PYTHONPATH=analysis python analysis/scripts/make_block_fit_line_atlases.py` |
| `dblock_oxi_oxygen_cn_fit_lines_part01.png` ... `part08.png` | same as above | `PYTHONPATH=analysis python analysis/scripts/make_block_fit_line_atlases.py` |
| `pblock_oxygen_cn_fit_lines_part01.png` ... `part03.png` | same as above | `PYTHONPATH=analysis python analysis/scripts/make_block_fit_line_atlases.py` |
| `fblock_oxygen_cn_fit_lines_part01.png` ... `part02.png` | same as above | `PYTHONPATH=analysis python analysis/scripts/make_block_fit_line_atlases.py` |
| `group1_group2_lanthanide_r0_star_vs_b_star.png` | `data/processed/theory/master_oxygen_summary_theory.json` | `python analysis/scripts/make_characteristic_pair_summary_figure.py` |
| `thomas_fermi_reff_null_models_si.png` | `data/processed/theory/charge_density_benchmark.json` | `python analysis/scripts/make_charge_density_benchmark_figures.py` |
| `r0_bond_length_identity_parity.png` | `data/processed/bond_valence/consolidated_store.json::oxygen_authoritative` | `PYTHONPATH=analysis python analysis/scripts/make_bond_length_identity_parity_figure.py` |

## Notes

- `analysis/notebooks/thomas_fermi_screening.ipynb` remains as the exploratory notebook used to derive the charge-density benchmark, but the manuscript-facing source of record is now the checked-in `data/processed/theory/charge_density_benchmark.json`.
- `analysis/scripts/make_charge_density_benchmark_figures.py` is expected to preserve the original PRL Fig. 2 layout and styling: flat two-panel geometry, block-colored panel-b symbols, and open markers for the four explicit outliers.
- The consistency checker at `analysis/scripts/check_manuscript_consistency.py` validates the figure inventory plus the manuscript counts and rounded benchmark summaries that are sourced from structured JSON/TeX artifacts.
