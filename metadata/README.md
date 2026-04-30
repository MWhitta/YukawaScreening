# FAIR Metadata

This directory contains repository-level metadata intended to make the manuscript source package easier to discover, validate, and reuse without reading the codebase first.

## Files

- `artifact_catalog.json` — machine-readable catalog of the manuscript-critical data artifacts, including file paths, media types, descriptions, licenses, content summaries, file sizes, and SHA256 hashes.
- `DATA_DICTIONARY.md` — human-readable schema notes for the main JSON/CSV data products used by the manuscript.
- `DATA_LICENSES.md` — license and terms-of-use guidance for code, manuscript text, derived data, and external-source identifiers.

## FAIR Mapping

### Findable

- Root-level `README.md`, `CITATION.cff`, and `.zenodo.json` provide repository discovery and citation metadata.
- `artifact_catalog.json` assigns stable artifact identifiers inside the repository and records the exact relative path for each manuscript-critical data file.

### Accessible

- All checked-in manuscript data are distributed as plain UTF-8 JSON or CSV files under version control.
- The only non-redistributed inputs are raw CHGCAR charge-density objects from the Materials Project; their reuse path is documented through `data/charge_density/mp_id_manifest.json` and the benchmark source file `data/processed/theory/charge_density_benchmark.json`.

### Interoperable

- Machine-readable artifacts use open formats: JSON, CSV, BibTeX, and LaTeX.
- `DATA_DICTIONARY.md` documents field meanings, units, and the relationship between the processed exports.
- `analysis/critmin/analysis/payload_schema.py` contains validation logic for grouped bond-valence payloads and candidate trackers.

### Reusable

- `artifact_catalog.json` records SHA256 hashes so downstream users can verify file integrity after download.
- `theory/FIGURE_SOURCES.md` maps manuscript figures to their source datasets and regeneration scripts.
- `analysis/scripts/check_manuscript_consistency.py` validates the core manuscript counts, figure inventory, and benchmark summaries against the checked-in structured artifacts.

## Important Limitation

The repository is FAIR-aligned as a versioned source package, but a globally persistent identifier for a specific release still depends on archival deposition (for example via Zenodo). The checked-in `.zenodo.json` file prepares that step, but it does not mint a DOI by itself.

Every manuscript figure now has a checked-in standalone exporter; see `theory/FIGURE_SOURCES.md` for the figure-to-script map.
