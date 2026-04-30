# Data And License Notes

## Code And Manuscript Text

- Repository source code, analysis scripts, and manuscript text are released under the MIT License in [`LICENSE`](../LICENSE).

## Materials Project-Derived Data

- The numerical summary files in `data/processed/` and the identifier manifests and candidate trackers in `data/raw/` and `data/charge_density/` are derived from Materials Project structures or workflows that depend on Materials Project content.
- Those artifacts should therefore be treated as redistributed derived data governed by the Materials Project terms of use in addition to any local repository permissions.
- The repository README states this explicitly; the machine-readable artifact catalog repeats that classification on each manuscript-critical data file.

## Charge-Density Inputs

- Raw CHGCAR objects are not redistributed in this repository.
- To reconstruct those inputs, users must obtain access through the Materials Project API and follow the workflow documented in `README.md` and `analysis/notebooks/thomas_fermi_screening.ipynb`.

## Reuse Guidance

- If you reuse repository code only, cite the repository and follow the MIT license.
- If you reuse manuscript data products or regenerate them from Materials Project content, also follow the Materials Project terms of use and cite the Materials Project appropriately.
