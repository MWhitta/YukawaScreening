# Data Dictionary

Units are angstroms unless noted otherwise. JSON files are UTF-8 encoded.

## `consolidated_store.json`

Top-level structure:

- `schema_version`: store schema version string.
- `generated_at`: UTC timestamp for store generation.
- `record_fields`: canonical ordered list of record field names.
- `datasets`: named grouped payload datasets. The manuscript uses `datasets.oxygen_authoritative`.

`datasets.oxygen_authoritative.payload` is a grouped payload keyed by cation symbol. Each cation maps to:

- `oxides`: list of oxygen-containing fit records.
- `hydroxides`: list of hydroxide fit records.

Each fit record contains:

- `mid`: Materials Project material identifier.
- `formula`: reduced formula string.
- `R0`, `B`: fitted bond-valence parameters.
- `R0_std`, `B_std`: uncertainty estimates from the fitting workflow.
- `n_algos`: number of fitting algorithms or passes contributing to the final record.
- `cn`: selected coordination number.
- `cn_all`: all candidate coordination numbers carried by the record.
- `oxi_state`: formal oxidation state used for the record.
- `oxi_state_label`: `pure` or `mixed`.
- `fit_strategy`: name of the fit workflow variant that produced the record.
- `fit_diagnostics`: nested diagnostic summary for that fit.
- `status`: final record status; manuscript-facing exports use `fitted`.
- `schema_origin`: `legacy` or `adaptive`.

Validation reference: `analysis/critmin/analysis/payload_schema.py::validate_grouped_payload`.

## `group12_unified_oxygen_theory.json`

Top-level keys:

- `sources`: accepted source payloads and tracker files used to build the export.
- `workflow`: workflow label; this file uses `unified_oxygen`.
- `groups`: dictionary with `group1` and `group2` entries.

Each group entry contains:

- `group_label`
- `cations`
- `target_cns`
- `cn_fits`: per-cation, per-coordination-number linear fit summaries.
- `outliers`: records rejected by the RANSAC fit variant.
- `alpha_lines`: coordination-resolved line summaries.
- `intersections`: least-spread characteristic-pair summaries.
- `group_convergence`: aggregate convergence diagnostics for the family.

## `dblock_oxi_unified_oxygen_theory.json`

Top-level keys:

- `sources`
- `workflow`
- `group_label`
- `elements`
- `species`: oxidation-state-resolved species labels such as `Fe³⁺`.
- `mixed_counts`: mixed-oxidation-state bookkeeping.
- `total_pure_fitted_records`
- `target_cns`
- `cn_fits`
- `outliers`
- `alpha_lines`
- `intersections`
- `cluster_summary`

This file is the d-block analogue of the Group 1/2 unified oxygen summary, but keyed by oxidation-state-resolved species instead of element alone.

## `master_oxygen_summary_theory.json`

Top-level keys:

- `workflow`
- `sources`
- `families`
- `master_rows`
- `excluded_rows`
- `family_summary`
- `global_summary`

`master_rows` is the manuscript’s authoritative 103-species table. Important fields per row:

- `group`: manuscript family label.
- `family_key`: normalized family identifier.
- `element`: cation element symbol.
- `species` or `label`: species label with oxidation state where needed.
- `oxi_state`: integer oxidation state when resolved explicitly, else `null` for fixed-valence families such as Group 1/2 and lanthanides.
- `n_lines`: number of coordination-number lines used in the characteristic-point intersection.
- `R0_star`, `B_star`: characteristic pair.
- `sigma_B_at_R0_star`: weighted spread of the contributing lines at the characteristic point.
- `pct_sigma_B_at_R0_star`
- `B_range_at_R0_star`
- `lines_used`: list of coordination-number labels such as `CN4`.

`excluded_rows` holds sparse or otherwise excluded species that were not admitted to the 103-row master table.

## `manuscript_theory_manifest.json`

Compact provenance manifest for the theory exports.

Top-level keys:

- `output_dir`
- `files`: relative paths to the main processed exports.
- `sources`: family-by-family source payloads and tracker paths.
- `summary`: aggregate counts used by the manuscript workflow.

## `lambda_mixed_cation_holdout_prediction.json`

Top-level keys:

- `workflow`
- `inputs`
- `parameters`
- `proxy_rankings`
- `summary`
- `species`
- `shell_prediction_rows`
- `material_prediction_rows`

`shell_prediction_rows` holds shell-level prediction diagnostics. `material_prediction_rows` is the material-level holdout table that is also exported as CSV.

## `lambda_mixed_cation_holdout_prediction.csv`

Flat table version of the material-level holdout predictions. The first row contains column names; each subsequent row is one material-level benchmark case.

## `charge_density_benchmark.json`

This is the manuscript-facing source of record for PRL Fig. 2 and the SI null-model control.

Top-level keys:

- `description`
- `provenance`
- `points`
- `summary`

Each `points` entry contains:

- `formula`
- `element`
- `z`: formal cation oxidation state.
- `family`: `Group 1`, `Group 2`, or `Other`.
- `block`: explicit periodic-table block label frozen into the benchmark source
  (`s`, `d`, `p`, or `f`).
- `panel`: manuscript panel label, `a` or `b`.
- `included_in_regression`: whether the point is part of the panel regression.
- `is_outlier`: explicit panel-b outlier flag.
- `m_i`: Boltzmann shell centroid.
- `r_eff`: charge-density screening centroid.
- `r_eff_over_m_i`
- `mpids`: supporting Materials Project identifiers recorded in the repo-level manifest when available.

The benchmark is intentionally self-contained: Fig. 2 styling reads `block`
directly from this checked-in file instead of re-deriving it from shared
periodic-table helpers. Panel `a` points are tagged `s`; panel `b` uses
explicit `d`/`p`/`f` tags for blue circles, green squares, and mauve diamonds.
The four `is_outlier=true` cases are rendered as open symbols within those
same block classes.

`summary` contains the rounded manuscript-facing fit coefficients and null-model
MAE values.

## `mp_id_manifest.json`

Top-level keys:

- `description`
- `fetch_instructions`
- `mp_dataset_doi`
- `count`
- `records`

Each record contains:

- `oxide`: benchmark oxide formula.
- `mpid`: Materials Project identifier.
- `cation`: cation element symbol.
- optional `z`
- optional `n`

This manifest supports the charge-density workflow more broadly than the final 35-point checked-in benchmark file.

## `*_candidates.json`

The files in `data/raw/ima/` are candidate trackers. They record which Materials Project entries were selected for each family-level search stage before bond-valence fitting.

Common structure:

- one cation block or a mapping of cation blocks
- each cation block contains `oxides` and `hydroxides`
- each bucket contains a `candidates` list

Each candidate typically records:

- `mid`
- `status`
- `cn_mode`
- `cn_all`
- `oxi_state`
- `oxi_state_label`

Validation reference: `analysis/critmin/analysis/payload_schema.py::validate_candidate_tracker`.
