# PRM revision directory

Manuscript LR20628MR, "Yukawa screening derivation of the bond-valence rule".

**Current state: second revision (round 2).** The files in this directory are
the live second-revision package. Everything from the first revision is frozen
in [`round1/`](round1/) and the PRL baseline is in
[`../prl_submission/`](../prl_submission/).

| File | Role |
| --- | --- |
| `reviewer_report_round2.txt` | Second-round reports (First Referee recommends publication, Second Referee is new) |
| `response_to_referees_round2.tex/.pdf` | Second-revision response letter (draft, `\todo{}` markers flag open items) |
| `prm_main.tex/.pdf` | Clean second revision of the manuscript |
| `prm_supplemental.tex/.pdf` | Clean second revision of the Supplemental Material |
| `prm_main_diff_round2.tex/.pdf` | latexdiff of `prm_main.tex` against `round1/prm_main.tex` |
| `prm_supplemental_diff_round2.tex/.pdf` | latexdiff of `prm_supplemental.tex` against `round1/prm_supplemental.tex` |
| `make_diff_round2.sh` | Regenerates and compiles both diff documents (`-n` for a dry run) |
| `response_figures/` | Figures made for the letter (per-structure λ histogram, Co–O lines with points); generating scripts listed in `FIGURE_SOURCES.md` |
| `../../reports/2026-10-05-*.md`, `2026-10-06-*.md` | Verification reports behind the numbers quoted in the letter |
| `REVISION_CHANGELOG.md` | Running record of every change since the PRL baseline, including round 2 |
| `FIGURE_SOURCES.md`, `si_description.md`, `si_lambda_table.tex`, `references.bib` | Figure provenance, SI description, SI table, bibliography |

Build:

```
latexmk -pdf prm_main.tex
latexmk -pdf prm_supplemental.tex
latexmk -pdf response_to_referees_round2.tex
./make_diff_round2.sh
```
