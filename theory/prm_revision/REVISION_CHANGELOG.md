# PRM Revision Changelog

The manuscript was submitted to PRL, transferred to Physical Review Materials,
and returned with revisions requested. This file is the running record of
**every difference between this directory and the as-submitted baseline in
[`theory/prl_submission/`](../prl_submission/)**, including the code and figure
assets behind the manuscript. Any future change to the manuscript, its
figures, or the scripts that generate them must be logged here.

Baseline: `theory/prl_submission/` as transferred on 2026-07-22 (identical
content, file-for-file, at the time of duplication).

---

## 2026-07-22 — Directory setup

- Duplicated all source and output files from `theory/prl_submission/`
  (LaTeX build intermediates excluded).
- Renamed `prl_main.*` → `prm_main.*` and `prl_supplemental.*` →
  `prm_supplemental.*` (tex, pdf, bbl, and RevTeX `*Notes.bib`).
- Added symlink `figures -> ../figures` so both documents compile from inside
  this directory (the PRL originals were compiled from `theory/`).
- Not yet changed: `\documentclass` options still say `prl`; RevTeX 4.2's
  option for Physical Review Materials is `prmaterials`.

## 2026-07-22 — Fig. 1 redesign (referee issue: R² not visually supported)

Referee concern: Fig. 1 does not look like R² = 0.986 because coincident
points sit on top of each other and hide the redundancy.

Manuscript (`prm_main.tex`):
- Caption of Fig. 1 (`fig:pole`): added one sentence — markers are colored by
  cation block and drawn semi-transparent, so darker points mark loci where
  the fitted slopes of multiple species coincide. This is the **only tex
  delta** vs `prl_submission/prl_main.tex` so far.

Figure asset (`theory/figures/prl_oxygen_beta_vs_charge_cn4_cn6.png`,
regenerated in place; the as-submitted raster survives in git history and in
`theory/arxiv_submission/`):
- Markers render at 45 % opacity with a thin white separation ring, so
  coincident species accumulate toward full saturation while lone species
  stay pale (darkness now encodes multiplicity). Open z > n markers have
  transparent faces and no longer occlude points beneath them.
- Replaced the ~60-hue per-element palette (decorative; no legend) with four
  block colors keyed off `family_key` in
  `data/processed/theory/master_oxygen_summary_theory.json`:
  s `#E6A000` (amber), d `#2353C4` (royal blue), p `#26943A` (green),
  f `#A81254` (crimson). The four hues are staggered in lightness and chosen
  to maximize pairwise separation *after* alpha compositing over white, so
  lone semi-transparent points remain tellable apart; the full-strength set
  passes all-pairs colorblind-separation (ΔE ≥ 10 under protan/deutan/tritan
  simulation) and 3:1 surface contrast. (An earlier revision pass used muted
  Fig.-2-style hues, s `#D07830` / d `#4A76B0` / p `#6F9A45` / f `#9C55A8`;
  superseded because their 45 %-alpha tints were too similar, especially
  blue vs purple.)
- Marker opacity raised from 45 % to 55 % as part of the same
  distinguishability fix.
- Added a second legend row identifying the four blocks; chips reuse the
  exact rendering of a lone data point (same alpha fill, white ring, size) so
  plotted colors map one-to-one onto the legend.
- Coverage and statistics are unchanged: 94 species, 150 fitted slopes,
  weighted R² = 0.986, MAE = 0.151.

Follow-up refinements (same date):
- Removed the per-element lines connecting data points across z; markers,
  error bars, and the model curve are the only remaining ink in the panels.
- Increased vertical space between the two legend rows (marker/model row vs
  block-color row).

## 2026-07-22 — Fig. 1 replaced by box-and-whisker summary; scatter moved to SI

- New main-text Fig. 1: `theory/figures/prm_oxygen_beta_vs_charge_box_cn4_cn6.png`,
  a box-and-whisker summary of the per-species slopes at each formal charge
  (boxes = interquartile range, black point = median, whiskers = 1.5×IQR,
  open circles = points beyond), overlaid on the parameter-free curve
  $\beta = 1/\ln(z/n)$ with the pole marked. Rendered by
  `render_box_figure()` in `analysis/scripts/make_dblock_beta_prl_figure.py`
  (same data pipeline; the script now writes both figures). Both panels
  share one $\beta$ axis with the top pinned at 6 (plus marker headroom) and
  the bottom clamped to the data range, so the diverging model curve does
  not set the scale; the $z$ axis extends to 8 in both panels so the
  $(n{=}6, z{=}7)$ group and the model curve descending past it are clearly
  resolved. Panels are labeled (a)/(b) outside the frame, above each
  panel's top-left corner; the $\beta$ axis label and tick labels appear
  on the left panel only.
- The per-species scatter (`prl_oxygen_beta_vs_charge_cn4_cn6.png`, the
  revised PRL Fig. 1) moved to the Supplemental Material: new section
  "Per-species slopes at fixed coordination" with figure
  `fig:pole_species`, inserted before "Characteristic pairs and bond-length
  identity". Its block-color/transparency description moved from the main
  caption to the SI caption.
- Main-text Fig. 1 caption rewritten accordingly (box conventions; pointer
  to the per-species SI figure). Species/slope counts and stats unchanged.
- `FIGURE_SOURCES.md` updated: box figure listed under Main Text; scatter
  listed under Supplemental Material.
- Main text contains no hard-coded SI figure numbers (checked), so the
  SI insertion renumbers nothing that is referenced.

Code (shared analysis library — changes are opt-in and leave all other
figures untouched):
- `analysis/critmin/viz/notebook_families.py`: `plot_fixed_cn_beta_panel`
  gained `marker_alpha` and `show_series_lines` keywords (defaults `None` /
  `True` preserve the old rendering).
- `analysis/scripts/make_dblock_beta_prl_figure.py`: block palette +
  element→block mapping, `marker_alpha=0.55`, `show_series_lines=False`,
  second legend row; regeneration command unchanged
  (`PYTHONPATH=analysis python analysis/scripts/make_dblock_beta_prl_figure.py`).

## 2026-07-22 — Screened-flux derivation elaborated (referee issue: derivation too abrupt)

`prm_main.tex`, derivation section and partition-thermodynamics paragraph:
- Motivated the flux-ratio postulate $S_{ij}/S_{ik}=w_{ij}/w_{ik}$ before
  imposing it: the valence-sum rule fixes only the total, the split must
  come from a physical prescription, and the postulate carries the
  Gauss's-law flux–valence proportionality of Preiser et al. over to the
  screened field. Since the sum rule renormalizes the total, the screened
  field determines only the relative apportionment, which the ratio
  condition preserves. Noted it is the minimal closure (no parameter
  beyond λ, nonnegativity guaranteed). The ratio condition now appears as
  its own display equation.
- Removed the premature definition of $R^*$ as the Boltzmann mean $m_i$;
  the derivation now expands about an unspecified shell-scale reference
  length, and the identification $R^*=m_i$ is made in the
  partition-thermodynamics paragraph where the Boltzmann weight arises
  (its closing sentence rewritten accordingly). The later species-level
  sentence now says the operational $R^*$ equals $R_0^*$ without invoking
  $m_i$ early.
- Explained screening vs anti-screening by the sign of λ at
  Eq. (B-composite): λ>0 → weight decays with bond length, shorter bonds
  carry more valence, B>0 (conventional parameter); −R*<λ<0 → weight
  grows with bond length, longer bonds carry more valence, B<0; tied the
  anti-screening branch to z>n and the sign reversal of β across the
  pole.

## 2026-07-22 — Limitation (ii) recast as isotropic flux capture

`prm_main.tex`, limitations paragraph: the derivation no longer defines
solid-angle fractions Ω_ij (dropped from the flux expression during the
derivation elaboration), so limitation (ii) — previously "solid-angle
fractions approximately uniform," with ln Ω_ij absorbed into C_i — was
recast as the isotropic flux-capture assumption: the ansatz
w_ij ∝ e^{−R_ij/λ}(1+R_ij/λ) assigns each bond a weight depending on its
length alone; anisotropic environments contribute a bond-dependent
geometric term to Eq. (2) (a j-dependent constant not absorbable into
C_i) and decorrelate the weights from λ. The later Fig. 2b sentence now
says "isotropic flux-capture assumption" in place of "isotropic-shell
assumption." No Ω_ij references remain in main text or SI.

## 2026-07-22 — Screening-collapse paragraph corrected (degenerate vs nearly degenerate shells)

Physics correction (main text + SI): the baseline claimed that at z=n the
valence-sum rule enforces S_ij = 1 for every bond, so any shell with
unequal bond lengths would force B → ∞. That is wrong — the sum rule
fixes only the mean bond valence; unequal bond lengths force unequal
S_ij, which finite B accommodates.

- `prm_main.tex`, "Screening collapse at z=n": rewritten to distinguish
  (i) truly degenerate shells (all bond lengths equal by symmetry, e.g.
  Li⁺ in antifluorite Li₂O), for which a single structure supplies only
  the uniform-limit relation and B is unconstrained (at z=n, R₀ = R̄ at
  any B), from (ii) nearly degenerate shells (e.g. α-quartz, two
  experimental Si–O distances ~0.01 Å apart), for which S_ij differ,
  finite B satisfies the sum rule, and no divergence of B or λ is
  required. What degenerates at z=n is the leading-order link between
  fitted parameters: R₀ ceases to constrain B, which is then fixed only
  by second-order spread terms — exactly degenerate for symmetric
  shells, ill-conditioned for nearly symmetric ones.
- Added corpus counts (computed from
  `data/processed/bond_valence/consolidated_store.json::oxygen_authoritative`,
  status=fitted records with diagnostics): 79,009 fitted shells total;
  6,158 degenerate within 1e-3 Å bond-length range; 14,511 more with
  range < 0.05 Å; among the 7,565 z=n shells: 959 and 2,960
  respectively. (Note: the DFT-relaxed α-quartz mp-7000 in the corpus
  symmetrizes to equal bonds — hence the parenthetical about relaxation
  symmetrizing shells; the ~0.01 Å figure cited is the experimental
  α-quartz splitting. Li₂O itself is absent from the corpus and is cited
  as a crystal-chemical example only.)
- ε-scaling paragraph reframed: the |B| ~ n|R₀−R₀*|/ε growth is what the
  leading-order relation requires to sustain a finite intercept offset
  ("inferred screening length grows"), not a per-shell physical
  divergence.
- `prm_supplemental.tex`, "Exclusion of z=n species from Fig. 1":
  same correction (mean valence, not per-bond; degenerate → β formally
  unconstrained; nearly degenerate → finite B but ill-conditioned fit),
  now pointing to the main-text screening-collapse discussion.

## 2026-07-22 — Truly-degenerate example changed from Li₂O to zircon

Li₂O (z=1, n=4) does not sit at the z=n pole, so it was the wrong
exemplar for the screening-collapse paragraph. Replaced with Si⁴⁺ in
zircon (ZrSiO₄, corpus record mp-4820): z=n=4 with all four Si–O bonds
symmetry-equivalent (1.625 Å in the relaxed structure) — and a
same-species contrast with the nearly degenerate α-quartz example.
Restored the truly-degenerate count alongside the near-degenerate one
(of 7,565 z=n shells: 959 degenerate within 1e-3 Å, 2,960 more within
0.05 Å). Other z=n identical-bond candidates verified in the corpus:
hafnon HfSiO₄ (mp-4609), stetindite CeSiO₄ (mp-10523), thorite ThSiO₄
(mp-5836), ReO₃ (mp-190, six equivalent Re–O — metallic, so not used in
text), and ordered double perovskites Ba₂CaWO₆/Ba₂SrWO₆ (W⁶⁺, six
equivalent W–O).

## 2026-07-22 — Characteristic-intercepts paragraph expanded

`prm_main.tex`: rewrote the "Characteristic intercepts" paragraph
(which had been left with a dangling splice after the examples sentence
was moved up) in the requested order:
1. Leads with the near-pole species examples — B³⁺ (B₂O₃, 6 eV), C⁴⁺ at
   n=4, Si⁴⁺ (SiO₂, 9 eV) — in the wide-gap covalent limit.
2. Draws the implication from the preceding screening-collapse
   paragraph: the pole is the limit of the first-order approximation,
   so the B–R₀ line at the pole coordination cannot by itself constrain
   the softness.
3. States the central empirical fact: the n-resolved B–R₀ lines of a
   given cation (slopes β⁽ⁿ⁾) intersect at a common point in the SI
   data, defining the characteristic points (R₀*, B*) — species-level
   constants fixed by the family of lines, hence well defined even when
   one line steepens toward the pole; the quantitative intersection
   criterion remains in the empirical-validation paragraph.
4. Retains the shell-level λ inversion and the tabulated λ* formula,
   now phrased as evaluation at the characteristic point.
Also updated the ε-scaling paragraph's parenthetical: R₀* is now
"introduced above" rather than "below" (ordering changed).

## 2026-07-22 — Intercept equations added to "Characteristic intersections"

`prm_main.tex`: inserted, before "These intersections define
characteristic points…", the derivation the reader previously had to
assemble: because the slopes are fixed by z/n, the intersection is set
entirely by the line intercepts. New numbered equations:
- eq:intercept — β₀⁽ⁿ⁾ = −R̄⁽ⁿ⁾/ln(z/n), from the uniform-limit
  relation, with R̄⁽ⁿ⁾ the characteristic shell radius at coordination n
  (made precise as the Boltzmann centroid below): each line crosses
  B = 0 at the shell radius itself.
- eq:bstar_dilation — B* = (R̄⁽ⁿ²⁾ − R̄⁽ⁿ¹⁾)/ln(n₂/n₁), with
  R₀* = R̄⁽ⁿ¹⁾ + B* ln(z/n₁): the characteristic softness is the
  logarithmic dilation rate of the shell with coordination number, and
  B* > 0 follows on the screening branch because bond lengths grow with
  n. (Note: paragraph retitled "Characteristic intersections" and
  "nearly common point" softening are the user's own edits.)
Subsequent equation numbers shift by two; all references are by \eqref
so they resolve automatically.

## 2026-07-22 — Partition thermodynamics restructured stepwise

`prm_main.tex`:
- The unique screened-flux partition S_ij = z_i w_ij / Σ_k w_ik is now a
  numbered equation (eq:partition) and is cited where the normalized
  weights are formed.
- "Partition thermodynamics" rewritten as three explicit steps:
  (1) eq:pweights — p_ij := S_ij/Σ_k S_ik = w_ij/Σ_k w_ik =
  e^{−R_ij/B}/Z_i, defining p_ij relative to w_ij (middle form from
  eq:partition, whose shell total is z_i; last form from eq:bv, in
  which R_0/B cancels), identified as a Boltzmann distribution with
  bond length as energy and B as the temperature-like scale;
  (2) eq:mean_entropy — explicit displayed definitions of the Boltzmann
  mean m_i (valence-weighted shell centroid) and Shannon entropy H_i;
  (3) eq:R0_identity — the exact identity R_0 = m_i + B(ln q_i − H_i),
  now derived in-line by inverting eq:bv bond by bond and averaging
  over p_ij (using ln S_ij = ln q_i + ln p_ij).
- Geometric interpretation of H_i and the R* = m_i closure sentence
  retained unchanged. Later equation numbers shift (all \eqref, so
  references resolve); no overfull boxes introduced.

## 2026-07-22 — Gibbs framing generalized to both signs of B

`prm_main.tex`, partition-thermodynamics step 1: the sentence framing
p_ij as "a canonical partition function with B > 0 as the
temperature-like scale on the screening branch" was generalized so the
anti-screening branch is included rather than excluded:
- Eq. (pweights) is now called a Gibbs distribution with B the
  temperature-like scale, both signs admissible because a finite shell
  has a bounded bond-length spectrum (the standard condition for
  negative temperatures in bounded statistical systems).
- B > 0 (screening): ordinary positive temperature, weight decays with
  bond length, shorter bonds dominate.
- B < 0 (anti-screening): negative temperature, population-inverted
  shell, weight grows with bond length; bonds longer than R_0 carry
  more than one valence unit (S_ij > 1 requires R_ij > R_0 when B < 0).
- The branches meet at the pole: |B| → ∞ is the infinite-temperature
  limit (p_ij → 1/n, the screening collapse of z = n), crossed through
  1/B = 0 exactly as a bounded system passes from positive to negative
  temperature.

## 2026-07-22 — Style pass: grammatical colons removed from revision prose

Standing style rule from the user going forward. No grammatical colons
(the "claim: elaboration" construction); prefer shorter sentences or
clauses. Applied to all passages authored in this revision of
`prm_main.tex` — the sign-of-λ branch discussion, the screening-collapse
paragraph, the characteristic-intersections dilation sentence, the
partition-thermodynamics steps (Gibbs framing, negative-temperature
passage, identity derivation, R* = m_i closure), and limitations
item (ii) ("isotropic flux capture, in which…"). Colons that introduce
displayed equations were kept, as were colon constructions inherited
from the PRL baseline prose (e.g. "cast this idea as the bond-valence
model:", "This is parameter-free:").

## 2026-07-22 — Anti-screening λ* added to SI Table I

- `analysis/scripts/export_theory_lambda_tables.py` now computes λ* from
  the branch-continuous "+" root of B = λ(λ+R*)/R* for either sign of
  B*, real for B* ≥ −R₀*/4 (previously λ* was emitted only for B* > 0).
  Output path repointed from the pre-reorg `theory/si_lambda_table.tex`
  to `theory/prm_revision/si_lambda_table.tex`.
- SI Table I regenerated. The two anti-screening species now carry
  λ* values as signed Yukawa-branch diagnostics rather than "---".
  Mo³⁺ has λ* = −0.268(0) Å and Sb³⁺ has λ* = −0.410(131) Å, with
  uncertainties propagated by the same σ_λ = |∂λ*/∂B*| σ_B formula.
  Table caption rewritten accordingly (also colon-free per the style
  rule).
- `prm_supplemental.tex` species-level paragraph updated. The
  reality condition B* ≥ −R₀*/4 is stated and both anti-screening
  entries satisfy it.
- `prm_main.tex` consistency edits. The λ* sentence in the
  characteristic-intersections paragraph now states the root is real
  for B* ≥ −R₀*/4, positive on the screening branch and negative on
  the anti-screening branch. The empirical-validation sentence about
  Mo³⁺/Sb³⁺ notes their negative λ* values are tabulated.

## 2026-07-22 — Second root λ*₋ added to SI Table I for all species

Follow-up to the anti-screening λ* addition. The table now has a
$\lambda^*_-$ column reporting the second root of
B = λ(λ+R*)/R*, namely λ*₋ = (−R₀* − √(R₀*² + 4B*R₀*))/2, for every
species (initially added for the anti-screening pair only; extended to
the screening branch at the user's request). Uncertainties use the same
σ_λ = |∂λ/∂B*| σ_B, identical in magnitude for both roots. On the
anti-screening branch both roots lie in the admissible interval
(−R₀*, 0) — Mo³⁺ has λ*₋ = −2.057(0) Å and Sb³⁺ has λ*₋ = −2.025(131) Å.
On the screening branch λ*₋ = −R₀* − λ* lies below −R₀*, outside the
admissible interval, and is reported for completeness. Table caption
and the SI species-level paragraph state the geometry of both cases.
Table generator updated accordingly (7-column longtable).

## 2026-08-07 — SI content description updated (user request)

New file `si_description.md` replacing the original submission's SI
description (which lived only in the submission form). Same
one-paragraph-per-section format, updated for the revised SI: the two
new sections (Per-species slopes; Comparison with published softness
parameters), the new paragraphs folded into their host sections
(line-fitting procedure, sign census + Fig. S1, slope standard
errors, near-coincidence, second-root anchoring, both-branch λ*),
the S-numbering note, and the λ*₋ table column. Numbers refreshed
against the current SI — notably the parity check (44,844 structures,
r = 0.976, MAE 0.019 Å; the old description's 50,706 / 0.969 / 0.018
predate the completed inclusion criteria) and the Shannon-control MAE
(0.095 Å, not 0.094).

## 2026-08-07 — Marked-up diff manuscript generated (user request)

New derived artifacts `prm_main_diff.tex` / `prm_main_diff.pdf` (8
pages): latexdiff markup of the current draft against the original
submission baseline `theory/prl_submission/prl_main.tex`. Additions
render blue-underlined, deletions red-struck. Options:
`--math-markup=whole` (changed displays marked as wholes, which is
what let the equation-dense diff compile cleanly) and
`--append-textcmd=runinsec` (section-head arguments diffed as text).
The diff document uses the revised preamble, so it carries the
prmaterials class options, \raggedbottom, and the \runinsec macro.
Compiles with zero errors; markup verified on a rendered page.
Regeneration: `latexdiff --math-markup=whole --append-textcmd=runinsec
../prl_submission/prl_main.tex prm_main.tex > prm_main_diff.tex`
followed by latexmk. Note these are derived files — regenerate after
any further manuscript edit rather than editing them directly.

## 2026-08-07 — Response letter restructured around the full referee report (user request)

`response_to_referee.tex` rewritten so the report's complete text
appears verbatim, in its original order, divided at paragraph
boundaries into eight quoted blocks, each followed by its response —
so the editor can confirm from the letter alone that every point is
addressed and none were selectively chosen. A sentence before the
blocks states this explicitly. Changes from the point-by-point
version:

- The preamble paragraphs (praise + the referee's stated perspective)
  and the overall-assessment/encouragement paragraphs are now quoted
  and answered too; the "three named emphases" of the assessment
  paragraph are mapped to their blocks, with the
  validation-figure-interpretation emphasis answered by the Δ_O,val
  offset mechanism and the Fig. 1 redesign.
- Block 5's response now concedes explicitly that some intersections
  are weakly constrained ("the revised documents say so rather than
  leaving it as an impression").
- Report text reproduced verbatim including its own typo
  ("bond-valance community"); no [sic] added.
- Additional-changes list updated (band-gap sign census Fig. S1;
  Fig. 2 label grouping).

Verified programmatically that every sentence of the report appears in
the letter (zero missing). Compiles to 6 pages with zero errors.

## 2026-08-07 — Fig. 2a labels regrouped: Group 1 below the fit line, Group 2 above (user request)

`analysis/scripts/make_charge_density_benchmark_figures.py`
(PANEL_A_LABEL_OFFSETS): Li and Cs moved below the fit line, Mg and Sr
moved above, so that all Group 1 labels (Li, Na, K, Cs, Rb) sit below
and all Group 2 labels (Be, Mg, Ca, Sr, Ba) above — a systematic
visual grouping replacing the earlier purely collision-driven
placement. Note: the request said "Na label above", but Na is Group 1
and the stated rule puts it below (where it already was); interpreted
as a slip for Mg, its Group 2 neighbor in the adjacent Li/Mg pair.
`thomas_fermi_reff_prl.png` regenerated (null-models PNG restored as
usual); verified in the render — no label collisions, K/Cs/Rb
staircase clean. prm_main.pdf rebuilt with zero errors.

## 2026-08-07 — SI Fig. S1 added: band-gap-resolved sign census (user request)

New SI figure `theory/figures/si_band_gap_sign_census.png` — stacked
histogram of the Materials Project band gap for all 62,552 clean
fitted shells (single-valence, unregularized linear fits), stacked by
the sign of the fitted B, with metallic hosts (E_g = 0) as a separate
bar. Headline result: 20% of shells in metallic hosts fit B < 0
versus 12% across gapped hosts, consistent with over-screening by
free carriers (the correlation survives a bond-length-spread control
in session analysis: 21% vs 10% among well-spread shells). Placed
after the sign-census paragraph with a pointer sentence; label
fig:gap_sign_census renders as FIG. S1 (later SI figures renumber
automatically via \ref; the main text hard-codes no SI figure
numbers).

Supporting assets:
- `analysis/scripts/make_gap_sign_census_figure.py` — generator; also
  rebuilds the gap map if absent using formula-unique batches (the MP
  API masks material_id in responses, so per-batch formula_pretty is
  the join key; same-formula polymorphs are dealt to different
  batches).
- `data/processed/theory/mp_band_gaps.json` — mid → band gap for
  31,533 of the corpus's 31,779 unique materials (99.2%; the
  remainder are deprecated MP entries), fetched 2026-08-07.
- Two-color palette (B>0 blue #2B5FAC, B<0 orange #C96A00) validated
  for CVD separation and surface contrast.
- `FIGURE_SOURCES.md` updated. SI compiles with zero errors.

## 2026-08-06 — Response letter: explicit manuscript/SI locations (user request)

`response_to_referee.tex`. Vague locators — "both documents", "the
revision", "a new section", unanchored "now appears" — replaced with
explicit named locations throughout: "the manuscript
(Parameter-free slope / Empirical validation / First-principles
confirmation / Characteristic intersections / Partition
thermodynamics)" and "the Supplemental Material ('Line-fitting
procedure' / 'Per-species slopes at fixed coordination' / 'Slope
standard errors' / 'Near-coincidence of same-branch lines')". Zero
occurrences of "both documents" remain; the two surviving "revision"
mentions refer to the revision record, not to a document. Compiles
with zero errors.

## 2026-08-06 — B–R₀ line provenance attributed to Li et al. (user request)

Per the user, the coordination-resolved B–R₀ lines were first
introduced in the Li et al. American Mineralogist paper (Li2025, Ref.
[19]), and the original manuscript's citation of that work was
intended to supply the construction context the referee found
missing. Three coordinated edits:

- `prm_main.tex`, Empirical validation: "The coordination-resolved
  lines of Fig. 1, first introduced in Ref. [19], are then obtained
  by regressing…" — attribution at the point of use (no renumbering,
  Li2025 already cited in the same section).
- `response_to_referee.tex`, point 1: opens with the provenance — the
  lines were first introduced in Li et al., which the original
  manuscript cited intending it to supply this context — before
  describing the now-explicit in-manuscript construction.
- `response_to_referee.tex`, point 2 (per user: this bears on the
  prior-work concern too): notes that referencing of previous work
  has been improved throughout, including the explicit Li et al.
  attribution.

Verified Li2025 remains Ref. [19] after recompilation (the letter
hardcodes the number — re-check if the bibliography ever changes).
Both documents compile with zero errors.

## 2026-08-06 — Response-to-referee letter drafted

New file `response_to_referee.tex` (compiles, 4 pages). Opens with the
framing the user specified — the referee's paired concerns about
abruptness and available length prompted an expansion throughout the
document, with specific changes listed in detail but additional
context added beyond them, plus the two new results (the
characteristic-intersection relations giving B* its dilation-rate
meaning, and the second-root contact-distance law anchoring λ* in
Shannon radii). Seven point-by-point responses quoting the report
(line construction with the joint-fit/either-or answer, prior work,
exact-vs-approximation bookkeeping, z=3 spread + slope errors,
near-coincident lines — crediting the referee's correct surmise with
the explicit slope-difference formula, SI captions, Letter format at
~4,200 word-equivalents). An "Additional changes" section discloses
the screening-collapse physics correction and branch-labeling fix,
the both-branch λ*/λ*₋ tabulation, the new SI analyses with
reproduction scripts, and the figure/terminology work. All claims in
the letter cross-checked against the final documents.

## 2026-08-06 — Joint-fit clause restored in Empirical validation (user request)

`prm_main.tex`: "(R₀,B) were fitted by linear least squares" →
"fitted jointly by linear least squares, not by re-optimizing B at
fixed R₀" — answering the referee's either/or question verbatim in
the manuscript (the full procedure remains in the SI line-fitting
paragraph). Also recording the user's terminology ruling: the
abstract's "150 fitted relationships" is deliberate — "slopes" is
ambiguous in a way "relationships" is not — and should not be
"corrected." Compiles with zero errors.

## 2026-08-06 — REVISION_SUMMARY.md created (net initial-vs-final comparison)

New file `REVISION_SUMMARY.md`. Clean comparison of the resubmission
against the true baseline in `theory/prl_submission/` (both main and
SI read in full against the current documents; diffs 625/356 changed
lines). Describes only net changes — no intermediate history — and
opens with a referee-concern → resolution table. Intended to feed the
response-to-referee letter. This changelog remains the intermediate
record.

## 2026-08-06 — Pole/Thomas–Fermi passage rewritten as an explicit argument (user request: non sequitur)

`prm_main.tex`, end of the screening-length paragraph in
"Characteristic intersections". The passage previously stated three
facts (ε^(−1/2) scaling, metals-vs-insulators contrast, "consistent
with" conclusion) without writing the connecting argument. Rewritten
in logical order — topic sentence ("The pole itself has a screening
interpretation"), empirical anchor tying back to the section's
wide-gap opening ("The species at z=n are wide-gap insulators (B₂O₃
and SiO₂ above)"), the textbook weak-screening expectation for gapped
systems, the model's matching behavior ("The fitted family behaves
the same way" + the ε^(−1/2) requirement), the explicit consistency
statement ("the species where weak screening is physically expected
are exactly those for which the model demands an unbounded screening
length"), and the scope caveat. Back-reference verified (B₂O₃/SiO₂
still named at the section opening). ~20 words longer than the
replaced text. Compiles with zero errors.

## 2026-08-05 — Vertical space added above run-in section heads (user request)

`prm_main.tex`. New preamble macro
`\newcommand{\runinsec}[1]{\medskip\emph{#1}}` and all nine
paragraph-initial `\emph{...}` section heads converted to
`\runinsec{...}` (Screened-flux derivation, Parameter-free slope,
Origin of the near-pole spread, Screening collapse, Characteristic
intersections, Partition thermodynamics, Empirical validation,
First-principles confirmation, Discussion). Mid-paragraph `\emph`
(e.g., `\emph{et al.}`) untouched — conversion keyed on
paragraph-initial position. The `\medskip` glue is discardable, so no
spurious space when a head lands at a column top; amount tunable in
one place. Verified on the rendered page 2 — sections separate
cleanly while intra-section paragraphs stay tight. Compiles with
zero errors.

## 2026-08-04 — \raggedbottom added (stretched paragraph gaps in the Discussion column)

`prm_main.tex` preamble. Diagnosis: REVTeX 4.2 sets
\parskip = 0pt plus 1pt (verified with a class probe) and runs
\flushbottom, so any column whose natural content falls short of the
page break gets the deficit distributed across its paragraph-break
glue. The Discussion is the manuscript's only long equation-free
column (page 5, right), so it alone showed large inter-paragraph
gaps — every other column hides the stretch in displayed-equation
glue. \raggedbottom keeps natural paragraph spacing and pools the
slack at the column bottom. Verified on the rendered page — the
Discussion column now matches the rest of the paper. No effect on
APS production (they retypeset). Compiles with zero errors.

## 2026-08-04 — Identity-line offset explained mechanistically (user request)

`prm_main.tex`, Discussion paragraph 3. "The small deviation from the
identity line is explained by the oxygen-side valence centroid shift"
told the reader a name, not a mechanism. Rewritten: the two centroids
reference different markers — m_i is built from internuclear M–O
distances (references the oxygen nucleus) while r_eff tracks the
oxygen-side screening charge, whose centroid is displaced from the
nucleus into the bond — and the same charge-density paths measure the
shift directly, Δ_O,val = 0.39±0.03 Å, matching the m_i − r_eff gap
of the fitted lines across the shell radii. Values verified against
the SI (Δ_O,val = 0.0613 m_i + 0.2394 Å, mean 0.3910±0.0269 Å, the
exact complement of the fitted main-text relation). Null-model
sentence retained. Compiles with zero errors.

## 2026-08-04 — Discussion updated with the Shannon-radius–λ* interpretation; formatting normalized (user request)

`prm_main.tex`, Discussion closing paragraph. The contact-distance
reframe now states the crystallographic anchoring explicitly. Through
|λ*₋| = R₀* + λ* = r_cr + c(z), the outer reach of the screened bond
coincides with the cation's Shannon crystal radius plus the
valence-dependent effective oxygen size, and λ* is read as the length
that carries the bond-valence shell radius R₀* out to the
crystallographic contact — before the existing tabulated-descriptor
sentence. Paragraphs 1–3 of the Discussion re-wrapped to a uniform
line width with content unchanged.

Formatting normalization (whole document): runs of two-plus blank
lines collapsed to one (two sites) and trailing whitespace stripped
from 11 lines — no typeset change except removing accidental extra
paragraph spacing. Compiles with zero errors.

## 2026-08-04 — Boltzmann → Gibbs terminology unified (user request)

Eq. (8) is called a Gibbs distribution, but the derived quantities
still carried "Boltzmann" names. All five sites renamed for
consistency — Fig. 2 caption ("Gibbs shell centroid m_i"), the
definition at Eq. (9) ("the Gibbs mean—the valence-weighted
centroid"), the R' = m_i closure sentence, the first-principles
opening, and the SI species-level paragraph ("chosen operationally as
the Gibbs shell centroid"). Deliberately unchanged: "Z_i is the
canonical partition function" (the standard companion term of a
Gibbs/canonical distribution). No figure assets affected — the
figures label the centroid only by its symbol. Zero "Boltzmann"
occurrences remain in either document; both compile cleanly.

## 2026-08-04 — "Characteristic intersections" streamlined; segue to "Partition thermodynamics" repaired (user request)

`prm_main.tex`, whole section restructured into four paragraphs with
one narrative each — geometry of the intersections (wide-gap opening
through the dilation-rate meaning of Eq. (7)), the characteristic pair
(definition, near-coincidence caveat, weighted-variance construction,
B*-vs-0.37 pointer), the screening length (inversion, λ*, reality
condition, pole limit with the Thomas–Fermi comparison), and the
second root (tabulation, contact-distance law, segue to the
first-principles test). Specific changes:

- Removed the duplicated characteristic-pair definition ("These
  intersections define…" and "converge to characteristic pairs …
  defined as the intersection that minimizes…" said the same thing in
  consecutive paragraphs — merged into one statement plus an
  "Operationally, …" sentence).
- Cut "Within each (z,n) group, the slope β⁽ⁿ⁾ confirms Eq. (5)
  across all families." — redundant with the Fig. 1 statistics
  already given in the parameter-free-slope section.
- Moved the near-coincidence caveat next to the characteristic-pair
  definition it qualifies (it sat mid-derivation between the two
  numbered equations and the definition).
- Repaired the dangling "tabulated there / diagnosed there" (its
  Supplemental-Material antecedent was lost in a reshuffle) —
  first reference now explicit, and the tabulation sentence now opens
  the two-roots paragraph so the pole/TF material no longer strands
  between λ* and the second root.
- "the intercept defines the unscreened shell radius" → "records
  where the shell sits" (B = 0 corresponds to λ → 0, the
  strong-screening limit, so "unscreened" read backwards).
- Partition-thermodynamics opening rewritten as a functional segue —
  "That comparison requires a shell radius that the bond-valence
  partition itself defines…" — fixing the truncated "compared with
  first-principles charge." sentence, the grammatical colon, and the
  dangling "We" that ran straight into Eq. (8) (the "three steps"
  lead-in had been lost).

Net ~40 words shorter. Compiles with zero errors.

## 2026-08-04 — Contact-distance relationships defined and explained (user request)

`prm_main.tex`, second-root paragraph. The unit-slope/offset passage
was rewritten so each relationship is defined at the point of use:

- $r_{\mathrm{cr}}$ now defined at its symbol (cation Shannon crystal
  radius averaged over observed coordination numbers, the geometric
  baseline for the contact).
- "Unit-slope fits ... hold" replaced by the operational statement —
  within each formal-charge class the slope of $|\lambda^*_-|$
  against $r_{\mathrm{cr}}$ is fixed at one and the single offset
  $c(z)$ is fitted — with the relation promoted to an unnumbered
  display and $c(z)$ glossed as how far the screened bond's outer
  reach extends beyond the cation radius.
- The prior "We assume that the same (unspecified) physics ... so
  that unit-slope fits hold" sentence was replaced by a cleaner hedge
  ("The comparison is empirical, with no assumption about the physics
  that sets the crystallographic radii") — the old construction read
  as if an assumption made the fits hold, when the unit slope is
  imposed and the offset fitted, the empirical content being that the
  one-parameter form describes each class.
- The two terms of $c(z)\approx 1.05+0.09\,z$ Å now each get their
  reading — the 1.05 Å constant at the oxygen-crystal-radius scale
  (low-charge limit reproduces the geometric contact) and the
  0.09 Å-per-charge increment, closing with $c(z)$ as an effective
  oxygen size, which the retained following sentences then develop.

Adds ~50 words plus one display. Compiles with zero errors.

## 2026-08-04 — Pole-decorrelation and skewness tests of the screening-collapse claim (analysis only, no manuscript changes yet)

User request: test the claim that at z=n "B is then fixed only by the
second-order spread terms δ_ij". Two new analysis scripts; no tex
changes in this entry.

`analysis/scripts/bv_pole_decorrelation.py` — expanded decorrelation
test. Within every (element, z, n) cell with ≥30 clean fits
(fit_strategy linear_ls, no degenerate fallback — the regularized
modes pull B toward a 0.37 Å prior and would fake decorrelation),
|r(B,R₀)| across structures: median 0.996/0.996/0.974 in the
|ln(z/n)| bands ≥0.75 / [0.35,0.75) / (0,0.35) (96+70+28 cells,
minimum 0.601), versus median 0.264 over the 14 z=n cells. All large
pole cells decorrelate (Si⁴⁺ N=2314 r=0.048, B³⁺ 737/0.079, Ge⁴⁺
368/0.122, W⁶⁺ 455/0.288, Te⁶⁺ 183/0.008); the only high-r pole
cells are small-N (P⁴⁺ 30/0.966, Ti⁴⁺ 36/0.890). Independent
corroboration: 32% of z=n shells forced the regularized fallback vs
16% off-pole. VERDICT — the degeneracy of the leading-order link at
the pole is strongly confirmed.

`analysis/scripts/bv_skewness_test.py` — direct test of the δ_ij
attribution. Carrying the second-order Yukawa expansion through the
per-structure OLS predicts B_fit ≈ B₀[1 − B₀/(2(λ+R̄)²)·m₃/m₂], i.e.
a small negative slope (−0.005 to −0.05) of B against bond-length
skewness γ=m₃/m₂ within z=n cells. Per-shell bond lists were rebuilt
from the MP bonds endpoint (single-mid requests — the API masks
material_id and does not preserve batch order; fetch cached, 794 of
1,412 requested docs still resolvable) and validated record-by-record
against the stored bond_length_mean (essentially 100% of resolvable
docs match, so the reconstruction reproduces the pipeline's bond
sets exactly). RESULT — 9 of 12 regressable cells have slope CIs
spanning zero; the three significant cells disagree in sign (Te⁶⁺
+8.7, W⁶⁺ +8.2, U⁶⁺ −5.6); pooled within-cell-centered slope 4.1
[−1.8, 14.1], r = 0.11. The observed B scatter at the pole
(±0.1–0.3 Å) exceeds the isotropic second-order prediction
(~4×10⁻⁴ Å over the observed γ range) by roughly three orders of
magnitude. VERDICT — the *negative* half of the claim holds (leading
order does not fix B) but the positive attribution to the isotropic
δ_ij term is not supported; at the pole, structure-to-structure B
variation is dominated by effects outside the single-λ isotropic
model (chemistry/anisotropy), and the predicted δ_ij coefficient is
fundamentally below detectability. Main-text wording implication
flagged to the user (the sentence is exact as a statement about the
model's information content, but should not be read as an empirical
account of what sets fitted B values at z=n).

## 2026-07-28 — SI sign census of per-shell fits added (precedence for main-text anti-screening statistics)

User request. New SI paragraph "Sign census of the per-shell fits" at
the end of the line-fitting-procedure block, documenting the B<0
census that the rewritten main-text sign-of-λ passage draws on:

- Population and filters stated in the SI text. All 79,009
  status=fitted records (oxides + hydroxides) of the
  oxygen_authoritative payload, restricted to single-valence
  assignments with defined (B, z, n) → 63,783 shells; 16.4% return
  B<0; split 20.4% (z<n) / 5.8% (z=n) / 10.2% (z>n). A parenthetical
  reconciles the main text's 7,565 z=n count (which includes
  mixed-valence records) with the census's 7,386.
- Near-degeneracy statistics. Median bond-length range 0.059 Å for
  B<0 shells vs 0.111 Å for B>0; 47% of B<0 shells within 0.05 Å vs
  29% of B>0 — the negative sign is dominated by barely-constrained
  near-degenerate shells.
- Chemically systematic fractions. Square-planar/linear late
  transition metals (Pd²⁺ 47%, Au³⁺ 43%, Cu¹⁺ 37%) and large soft
  high-CN cations (Eu²⁺ 51%, Rb⁺ 40%, Sr²⁺ 31%).
- New script `analysis/scripts/bv_sign_census.py` reproduces every
  quoted number from the consolidated store (verified by running it);
  the SI cites it via the repository reference.

Note: these supersede the oxides-only exploratory numbers discussed in
session (60,919 shells, 16.1%, splits 20.0/5.7/10.0) — the census now
includes the hydroxides group for consistency with the corpus totals
already cited in the manuscript (79,009 / 7,565). Per user direction,
species-level anti-screening (Mo³⁺/Sb³⁺) is deliberately not tied to
this passage in the main text, since those species-level B*<0 values
are likely sparse-data artifacts. SI compiles with zero errors.

## 2026-07-28 — References added for bond valence as an ML structural descriptor

User request: support "and increasingly as a structural descriptor" in
the opening paragraph, primarily in machine-learning contexts. Three
references found, read (abstract-level verification), and integrated;
the sentence now ends "…and increasingly as a structural descriptor in
machine-learning models \cite{Li2021,Zhang2023,Miller2023}."

New `references.bib` entries (author lists verified against Crossref;
two given names corrected from my initial-expansion guesses):

- Li2021 — C. Li, H. Hao, B. Xu, Z. Shen, E. Zhou, D. Jiang, H. Liu,
  "Improved physics-based structural descriptors of perovskite
  materials enable higher accuracy of machine learning," Comput.
  Mater. Sci. 198, 110714 (2021). Uses bond-valence vector sum, global
  instability index, and a bond-valence tolerance factor as explicit
  ML features.
- Zhang2023 — L. Zhang, Z. Zhuang, Q. Fang, X. Wang, "Study on the
  automatic identification of ABX₃ perovskite crystal structure based
  on the bond-valence vector sum," Materials 16, 334 (2023). BVVS as a
  Random-Forest feature for crystal-system/space-group identification
  (verified by reading the open-access text via PMC).
- Miller2023 — K. D. Miller, J. M. Rondinelli, "Testing the limits of
  the global instability index," APL Mater. 11, 101108 (2023).
  Documents the GII's popularity as a data-driven screening feature
  and critically assesses its limits; entry matches the vetted
  MillerRondinelli2023 entry in ../bv-methods-review (rekeyed to this
  repo's first-author-year convention).

Candidates screened out: an RSC Adv. 2024 interpretable-ML perovskite
paper (mentions bond-valence tolerance factors only in its literature
review, uses none as features — verified via PMC) and a 2026 JPCL
descriptors paper (abstract not accessible, left uncited unread).
Compiles with zero errors; all three keys resolve in prm_main.bbl.

## 2026-07-28 — Fig. 2 follow-up: Na label raised, As³⁺ label moved above its point

User request, same script (`render_prl_figure` offset tables):

- Panel a: Na offset (9, −8) → (9, −5), lifting the label clear of the
  legend box below while staying adjacent to the Na point and off the
  fit line.
- Panel b: As³⁺ moved from the square's lower-left corner to directly
  above it, (0, 5, center, bottom). As is the only outlier below the
  identity line, so the label necessarily meets the dashed guide; in
  the render the text sits just above the dashed line and draws over
  it where they touch — verified legible in a zoomed crop, with no
  marker collisions.
- `thomas_fermi_reff_prl.png` regenerated; the untouched SI
  null-models PNG restored again (same one-pixel bbox jitter);
  prm_main.pdf rebuilt cleanly.

## 2026-07-28 — Fig. 2 markers halved; data labels placed adjacent to their points

User request. `analysis/scripts/make_charge_density_benchmark_figures.py`
(`render_prl_figure` only; the SI null-models renderer untouched):

- Marker sizes halved in linear dimension (scatter areas /4): panel a
  Group 1 s=88 → 22, Group 2 s=98 → 24; panel b included s=86 → 21.5,
  outliers s=106 → 26.5 (inner ring keeps its 0.82 area ratio). Legend
  chips unchanged — at the new sizes they match the data markers.
- Panel a label offsets (PANEL_A_LABEL_OFFSETS) redesigned for
  adjacency. Every label sits directly beside its marker,
  perpendicular to the fit line (upper-left or lower-right), and the
  two nearly coincident pairs are split across the line — Ca
  upper-left / Na lower-right at (2.420, 2.036)/(2.423, 2.027), and
  Ba upper-left / K lower-right at (2.807, 2.401)/(2.809, 2.395) —
  which is what had forced the old scattered placements. Cs goes
  directly above, Rb lower-right. Verified against the render: no
  label overlaps another label, a marker, either line, or the legend.
- Panel b outlier labels (OUTLIER_LABEL_OFFSETS) tightened to the
  nearest clear space per point: B³⁺ below, P⁵⁺ above, Au³⁺ right,
  As³⁺ at the square's lower-left corner (iterated visually; the
  straight-below slot collides with the Mo circle, the left slot with
  the identity line).
- `theory/figures/thomas_fermi_reff_prl.png` regenerated in place.
  Data, fits, and stats unchanged. The script also rewrites
  `thomas_fermi_reff_null_models_si.png`; it differed only by
  one-pixel tight-bbox jitter and was restored to the committed
  version. prm_main.pdf rebuilt cleanly with the new figure.

## 2026-07-27 — B*-vs-literature paragraph moved from main text to SI (user request: fitting methods, not physics)

`prm_main.tex`: the paragraph comparing B* with published softness
values (Brown–Altermatt convention, softBV, Gagné–Hawthorne;
distribution stats, well-conditioned subset, per-species
decorrelation, valley interpretation) was removed and replaced by a
single pointer sentence merged into the head of the λ* paragraph —
the screening-branch B* values are not universal but cluster near the
conventional 0.37 Å (Brown1985), with the comparison
(Adams2001, Chen2017, Gagn2015) deferred to the SI. The λ*
sentence's own "in the Supplemental Material" was shortened to
"there" to avoid back-to-back \cite{suppmat} sentences.

`prm_supplemental.tex`, "Comparison with published softness
parameters": absorbed the paragraph's three elements not already
present — the comparability framing (B* of main-text Eq. (7) is a
single species-level constant from aggregated per-structure fits,
hence directly comparable to pooled per-pair sets), the full B* range
0.004–0.918 Å with the cluster-near-0.37 statement, and the explicit
verdict clause (consistent with the Gagné–Hawthorne spread, lower
than softBV). Both documents compile with zero errors.

## 2026-07-27 — SI section "Comparison with published softness parameters" added

`prm_supplemental.tex`, new section between "Characteristic pairs and
bond-length identity" and the s-block controls. Backs the main text's
new B*-vs-literature paragraph (the \cite{suppmat} pointer for the
per-species correlation claim now resolves). Two paragraphs:

- Distribution-level agreement. B* (101 screening-branch species)
  mean 0.38±0.15 Å, median 0.37 Å, IQR 0.31–0.44 Å, vs softBV
  (155 cation–O pairs, 0.45±0.05 Å, range 0.34–0.62 Å) and
  Gagné–Hawthorne 2015 (135 pairs, 0.40±0.06 Å, range 0.25–0.66 Å);
  matched-species mean offsets −0.01 Å (GH2015, 92 species, MAD
  0.11 Å) and −0.06 Å (softBV, 99 species, MAD 0.13 Å);
  well-conditioned subset (≥3 lines, σ_B/B* ≤ 20%) 78 species,
  0.39±0.12 Å, median 0.37 Å, range 0.14–0.79 Å.
- Species-level decorrelation. r = 0.09 (vs softBV), −0.01 (vs
  GH2015); the two published sets themselves correlate only r = 0.26
  (MAD 0.065 Å, 130 common pairs). Interpreted via the pooled-fit vs
  per-material estimand distinction (pooled fits select a
  corpus-dependent point in the shallow (R0,B) valley; B* is the
  least-spread intersection of the coordination-resolved lines).

Numbers computed by matching data/processed/theory/
master_oxygen_summary_theory.json (element + oxidation state) against
the published-parameter catalog in
../bv-methods-review/methods_comparison/comparison.csv (BA1985 /
softBV / GH2015 columns; sources documented in that repo). Verified
2026-07-27; analysis commands in the session transcript. Citation
keys Brown1985 / Adams2001 / Chen2017 / Gagn2015 all resolve. SI
compiles with zero errors.

## 2026-07-27 — Fig. 1 x-axis label spacing fixed (label overlapped tick labels)

The shared "z" supxlabel sat flush against the panels' tick-label row
(label top at y=0.19 in figure coordinates, tick labels ending at the
same height). Three layout parameters changed in `render_box_figure()`
(`analysis/scripts/make_dblock_beta_prl_figure.py`); the scatter
figure's function is untouched:

- `panel_height_ratio` 0.78 → 0.85 (taller canvas gives the bottom
  region absolute room; canvas width unchanged at 1764 px, so the
  on-page text-size sync with Fig. 2 documented in the code comment is
  preserved).
- `subplots_adjust` bottom 0.27 → 0.30.
- `supxlabel` y 0.09 → 0.085.

`theory/figures/prm_oxygen_beta_vs_charge_box_cn4_cn6.png` regenerated
in place (1764×788 → 1764×844); the label now has clear separation
from both the tick row and the legend. Data and stats unchanged (94
species, 150 slopes, weighted R² = 0.986, MAE = 0.151). The script
also rewrites the SI scatter PNG on every run; the rerun differed only
by one pixel of tight-bbox jitter, so it was restored to the committed
version (no change intended or made to that figure).

## 2026-07-27 — ε-scaling and Thomas–Fermi paragraphs merged (user request)

`prm_main.tex`. The standalone ε-scaling paragraph (with its unnumbered
|B|, |λ| display) and the Thomas–Fermi paragraph were merged into one
~70-word paragraph. Rationale discussed with the user: the scaling laws
were consumed nowhere except by the TF paragraph's first sentence, the
|B| ~ 1/ε form invited the misreading (a per-shell divergence of B at
the pole) that the screening-collapse paragraph explicitly denies, and
the closing clause restated the screening collapse a third time. Kept:
the conditional ε^(−1/2) growth of the inferred λ as one inline
sentence (no display), the metals-vs-gapped-insulators contrast with
its citations (Thomas1927, Fermi1928, Debye1923, AshcroftMermin1976),
and the scope caveat that the coarse-grained λ is not the microscopic
lengths but is consistent with the same weak-screening limit. Dropped:
the |B| scaling form and the redundant final clause. Recovers roughly
100 word-equivalents.

## 2026-07-27 — Reference length R* relabeled R′ (user request: distinguish from R₀*)

The Taylor-expansion reference length was written R* and was easily
confused with the characteristic intercept R₀*. Renamed R* → R′
everywhere; R₀*, B*, λ*, and λ*₋ are unchanged.

- `prm_main.tex` (12 sites): eq:expansion intro and coefficient
  −R′/[λ(λ+R′)], the δ_ij order (λ+R′)², the R_ij=R′ statement,
  eq:B_composite B = λ(λ+R′)/R′, the anti-screening interval
  −R′<λ<0, the shell-level inversion λ=(−R′+√(R′²+4BR′))/2 and the
  "operational choice of R′ equals R₀*" sentence, the near-pole
  |λ| scaling, and the partition-thermodynamics R′=m_i closure.
- `prm_supplemental.tex` (2 sites): the species-level paragraph's
  representative radius and the R′=R₀* substitution sentence.
- Patterns replaced exactly (R^* and R^{*2}); R_0^* cannot match
  either, verified by grep (0 remaining R^*, 12 R_0^* intact in the
  main text). Both documents recompile with zero errors.

## 2026-07-27 — SI figures and tables renumbered with the S prefix

`prm_supplemental.tex` preamble: added
`\renewcommand{\thefigure}{S\arabic{figure}}` and
`\renewcommand{\thetable}{S\arabic{table}}`, so SI figures render as
FIG. S1–S20 and the λ* table as TABLE S1 (APS supplemental
convention). All internal `\ref`s pick up the prefix automatically,
including the longtable continuation header, which uses `\thetable`.
The SI has no numbered equations, so no equation prefix is needed.
Verified in the aux file (fig:pole_species → S16, tab:lambda_star →
S1). Two remaining bare main-text figure references disambiguated
while renumbering ("Exclusion of z=n species from main-text Fig. 1"
paragraph title, and "Main-text Fig. 1 therefore shows..."), so no
bare "Fig. N" in the SI can be mistaken for an S-numbered figure.
Main text hard-codes no SI figure numbers (checked previously), so
nothing else changes.

## 2026-07-23 — Second root elevated; full referee-response edit batch (length cap relaxed to 4,500 by user)

User direction. The second-root contact-distance law is the physically
important finding relative to the λ* tabulation, and the word budget
may extend to 4,500 (no unnecessary filling). All drafted text below
was verified by the review workflow before application (numbers
re-derived from the pipeline; procedural claims checked against code).

Second-root reweighting (`prm_main.tex`):
- λ*-tabulation paragraph compressed to a corollary (~100 → ~60 words).
  One sentence covers the tabulation for all 103 species with signs by
  branch, pointing to the SI for the Mo³⁺/Sb³⁺ diagnoses (stereoactive
  lone pair, two-shell sparsity); the non-universal B* sentence
  retained; "λ* provides a single-parameter species descriptor"
  dropped here (the Discussion already says it). The second-root
  paragraph is kept at full strength (earlier plan to condense it is
  abandoned).
- Discussion closing reframed. It now leads with the second root
  reproducing cation–oxygen contact distances with a valence-dependent
  effective oxygen size, then presents λ* as the tabulated descriptor.
- Abstract. "150 fitted valences" corrected to "150 fitted slopes"
  (they are slopes of B–R₀ lines), and one sentence added on the
  contact-distance law (~0.09 Å growth per unit cation charge vs
  tabulated crystal radii).

Referee-response additions (`prm_main.tex`):
- Introduction prior-work paragraph rewritten. Preiser's point-charge
  flux result and Brown's capacitor framework characterized; Adams
  distinguished as a bond-stiffness (not screening) route; the
  documented near-degeneracy of jointly fitted (R₀,B) and the fixed
  B≈0.37 Å convention added as the missing context (cites Brown1985,
  Gagn2015); fixes "Gauss's-law" hyphen and the Adams tense mismatch.
  Closing roadmap sentences added. The flux-ratio partition is a
  postulate, the isotropic Yukawa weight and its linearization are the
  two controlled approximations, the partition-thermodynamic relations
  are exact identities, and the centroid comparison is an independent
  test.
- Screened-flux opening now derives the Yukawa flux factor
  e^(−R/λ)(1+R/λ) from Gauss's law applied to the screened potential,
  with Debye/Thomas–Fermi/Yukawa citations moved up to first use, and
  names the single-λ weight as the physical ansatz.
- Status sentence after Eq. (2). The partition is an exact consequence
  of the two conditions; the physical assumption is the flux ansatz;
  the first mathematical approximation is the linearization.
- Anti-screening validity caveat. On the λ<0 branch the flux amplitude
  vanishes and changes sign at R=|λ|, so the expansion requires every
  bond beyond that node (holds for the tabulated species).
- Slope uncertainties (referee request). One sentence in the
  parameter-free-slope paragraph: median relative bootstrap standard
  error 0.2%, 90th percentile 4% across the 150 slopes, so the scatter
  reflects real inter-species differences (verified by re-running the
  Fig. 1 pipeline: 0.1942% / 3.93%).
- New paragraph "Origin of the near-pole spread" (referee request re
  z=3). Data-verified content: perturbation u in 1/β shifts β by
  −β²u; per-class spread grows as |β|^1.8; same cations at n=6 match
  to median |Δβ|=0.03; largest deviations are rare-CN4 species
  (trivalent lanthanides and Y, median 17 structures/fit) vs
  Ga³⁺ 209 / B³⁺ 214 / Fe³⁺ 160 on the curve; deviant slopes almost
  all shallower (δ_ij attenuation). A Jahn–Teller explanation is NOT
  used — Mn³⁺ lies on the curve (residual 0.26).
- Near-coincidence comment (referee request). Same-branch lines differ
  in slope by ln(n₂/n₁)/[ln(z/n₁)ln(z/n₂)], small when z is well
  separated from both coordination numbers; the intersection is
  constrained by the shell dilation of Eq. (7); construction pools all
  lines and reports σ_B.
- "pole is the limit of the first-order approximation" → "pole marks
  the breakdown of the first-order link between R₀ and B".
- "final fit population" → "number of structures retained in its final
  fit" (RANSAC-inlier meaning, plain words).
- Partition-thermodynamics opening rewritten with motivation (supplies
  R* and the shell radius compared with charge densities below) and
  status declaration (exact identities; thermodynamic language is
  interpretation). Added the first-order caveat that the middle and
  last forms of Eq. (8) coincide to first order of Eq. (3), and the
  note that q_i = z_i when the sum rule is enforced exactly.
- First-principles-confirmation opening now motivates r_eff physically
  (m_i should *track* the screening-charge radius — deliberately not
  "equal", consistent with the fitted non-zero intercepts) before the
  operational definition.
- Discussion limitation (ii) repaired. The orphaned "as in stereoactive
  lone-pair…" fragment regained its connective clause (directional
  bonding adds a bond-dependent geometric term that Eq. (3) cannot
  absorb into C_i), matching the limitation-(ii) recast entry.
- Typos. "for a the screened field", doubled "the" in acknowledgments.
- Stray blank line before eq:intercept removed (spurious paragraph
  break in typeset output).
- `\documentclass` option prl → prmaterials (both documents; flagged
  since directory setup).

SI (`prm_supplemental.tex`):
- New paragraph "Slope standard errors" after the weighting scheme
  (σ_β from the 400-resample bootstrap; median 0.19%, p90 3.9%), and
  fig:pole_species caption now explains its error bars.
- New paragraph "Near-coincidence of same-branch lines" after the
  characteristic-pair construction, with the slope-difference formula,
  the sharp-crossing case n₁<z<n₂, and conditioning stats verified
  from master_oxygen_summary_theory.json (85 species with ≥3 lines,
  median σ_B/B* = 5.8%, 62 below 10%, worst case Ag¹⁺ from near-zero
  B*=0.011 Å; 18 two-line species have σ_B=0, marked provisional).
- Characteristic-pairs text states the per-structure pairs and robust
  fit define each line, that no R² screening applies at the
  intersection stage (true at all production call sites), and that N_n
  counts inliers; weighting-scheme N_i likewise corrected to the
  final-refit inlier count.
- Parity-check inclusion rule completed (fitted single-valence records
  with a defined mean bond length) — with those two criteria the
  script reproduces exactly N=44,844, r=0.976, MAE=0.019.
- Six bare "Fig.~2" references disambiguated to main-text Fig. 2
  (SI has its own Fig. 2).
- R₀ bounds (−10,10) Å VERIFIED correct against the consolidated
  store: all 85,661 fitted records carry r0_bounds=[−10,10]. The local
  code default (−5,10) in analysis/critmin/analysis/config.py differs
  from the production run; SI text unchanged.

## 2026-07-22 — Main-text B(R₀) construction stated (referee issue: fitting procedure must be in the manuscript)

`prm_main.tex`, two additions answering the referee's direct question
("Are they obtained by fixing R₀ and re-optimizing B, or by some other
procedure?") in the manuscript itself, complementing the new SI
line-fitting paragraph:

- Parameter-free-slope paragraph: appended one sentence at first use of
  the fitted slopes — each fitted slope is the regression slope of one
  species' per-structure (R₀,B) fits at fixed n, constructed as
  described in the empirical validation below.
- Empirical-validation paragraph, after the Supplemental Material
  sentence: each structure contributes one (R₀,B) pair with both
  parameters fitted jointly (not B re-optimized at fixed R₀); the
  coordination-resolved lines of Fig. 1 are regressions of B on R₀
  across the per-structure pairs of each species at fixed n (≥5
  structures per cell), via an outlier-robust RANSAC estimator refit by
  least squares on its inliers, with slopes accepted only at ≥4
  inliers, σ_β ≤ 5, and R² ≥ 0.3. All numeric criteria verified against
  `analysis/scripts/make_dblock_beta_prl_figure.py:151-156`,
  `analysis/critmin/analysis/bond_valence_theory.py:389,407`, and
  `analysis/critmin/viz/bond_valence.py:229-253`.

Adds ~100 words; the manuscript was at ~3,638 of the 3,750
word-equivalent APS Letter budget before this entry, so the
compensating cuts identified in review (condense second-root
paragraph, trim negative-temperature exposition, compress Discussion
recap) are now required before any further additions.

## 2026-07-22 — SI line-fitting procedure documented (referee issue: B(R₀) construction unexplained)

`prm_supplemental.tex`, new `\paragraph*{Line-fitting procedure.}` at the
top of "Coordination-number-resolved B–R₀ fit lines" (after the intro
paragraph, before the Group-1 atlas figure). Documents, verified
statement-by-statement against the analysis code: per-structure joint
two-parameter OLS on R_ij = R₀ − B ln S_ij (neither parameter held
fixed; R₀ fixed with B re-optimized only in the rare bound-projection
fallback — `analysis/critmin/analysis/bond_valence.py:815-912`);
unambiguous-CN binning with mixed-coordination structures left unbinned
(`bond_valence.py:314-340,426-449`); per-(species, n) line fits with
minimum five structures (`bond_valence_theory.py:389,407`), OLS and
RANSAC (residual threshold 2× median absolute OLS residual, floored at
0.01 Å) with the RANSAC line preferred
(`analysis/critmin/viz/bond_valence.py:229-253`,
`bond_valence_theory.py:461,590`); slope/intercept from the OLS refit on
RANSAC inliers with 400-resample bootstrap errors
(`viz/bond_valence.py:43-88`); line population N_n = final inlier count
supplying the weights w = max(N_n, 1)
(`bond_valence_theory.py:487,633`); Fig. 1 acceptance filter ≥4
inliers, σ_β ≤ 5, R² ≥ 0.3
(`analysis/scripts/make_dblock_beta_prl_figure.py:151-156`). Cosmetic
deltas from the verified draft: OLS/RANSAC acronyms expanded at first
use; bare "Fig.~1" in the weights sentence written as "main-text
Fig.~1".

## 2026-07-22 — SI atlas captions corrected (referee note: many SI captions incorrect)

The atlas PNGs order panels alphabetically by element symbol within
each block, but the captions assumed atomic-number ordering; all eight
d-block captions misnamed their pages (even naming species absent from
the 103-species set — Ru, Rh, Os, Ta⁴⁺). Every correction below was
verified against the rendered PNGs. All in `prm_supplemental.tex`:

- Atlas intro: added the convention sentence — captions quote panels in
  reading order (in full or as a first–last range); panels alphabetical
  by element symbol and oxidation state within each block.
- fig:dblock_fits_1: "(Sc–Cr)" → explicit list Co¹⁺, Co²⁺, Co³⁺, Co⁴⁺,
  Cr²⁺, Cr³⁺.
- fig:dblock_fits_2: "(Cr⁵⁺–Fe⁴⁺)" → Cr⁴⁺, Cr⁵⁺, Cu¹⁺, Cu²⁺, Cu³⁺, Fe²⁺.
- fig:dblock_fits_3: "(Co–Cu²⁺)" → Fe³⁺, Fe⁴⁺, Mn²⁺, Mn³⁺, Mn⁴⁺, Ni²⁺.
- fig:dblock_fits_4: "3d/4d (Cu³⁺–Nb)" → 3d only; Ni³⁺, Sc³⁺, Ti³⁺,
  Ti⁴⁺, V³⁺, V⁴⁺.
- fig:dblock_fits_5: "4d (Mo–Ru)" → 3d/4d; V⁵⁺, Zn²⁺, Ag¹⁺, Ag²⁺, Cd²⁺,
  Mo³⁺.
- fig:dblock_fits_6: "4d/5d (Rh–Ta⁴⁺)" → 4d only; Mo⁵⁺, Mo⁶⁺, Nb⁵⁺,
  Pd²⁺, Y³⁺, Zr⁴⁺.
- fig:dblock_fits_7: "5d (Ta⁵⁺–Os)" → Au³⁺, Hf⁴⁺, Hg¹⁺, Hg²⁺, Ir⁴⁺, Pt²⁺.
- fig:dblock_fits_8: "5d (Ir–Hg)" → Re⁶⁺, Re⁷⁺, Ta⁵⁺, W⁶⁺.
- fig:group1_fits: species list put in panel reading order (Cs, K, Li,
  Na, Rb, H); "Each colored line is a different coordination number"
  (false — color encodes species panel, line style encodes n) → "Line
  styles distinguish coordination numbers n (legend)"; the open-circle
  least-spread intersection marker is now explained.
- fig:group2_fits: species list in reading order (Ba, Be, Ca, Mg, Sr).
- fig:pblock_fits_2: "(S⁶⁺–Cl)" hid the C⁴⁺ panel and quoted Cl without
  its charge → explicit 12-species list ending C⁴⁺, Cl¹⁻ (the
  nonmetal/halogen panels are appended after Tl³⁺, out of alphabetical
  order).
- fig:pblock_fits_3: species list in panel reading order (H¹⁺, I⁵⁺,
  I⁷⁺, N³⁺, N⁵⁺).
- Unchanged as correct: fig:pblock_fits_1 "(Al³⁺–S⁴⁺)" and both f-block
  range captions (their ranges read correctly under the alphabetical
  convention now stated in the intro).

## 2026-07-22 — Physical consistency: z>n no longer equated with anti-screening (B<0)

The manuscript equated the z>n regime with the anti-screening branch
(λ<0, B<0), which its own data contradict. Roughly 90% of fitted z>n
records in the corpus have B>0, the median B is positive in every z>n
cell, every z>n species in SI Table I has B*>0, and the only two B*<0
species (Mo³⁺, Sb³⁺) have z=3<n. What reverses across the pole is the
slope β = 1/ln(z/n), not the sign of B (the sum rule fixes no sign;
the screening-collapse paragraph already says B is fixed by the
second-order spread terms). Five coordinated replacements:

- `prm_main.tex` sign-of-λ paragraph: "This anti-screening branch is
  the regime that the valence-sum rule selects whenever the formal
  charge exceeds the coordination number (z>n)…" → the branch is
  realized only rarely (two of the 103 fitted species), and the branch
  sign of B is distinct from the sign reversal of the slope across the
  pole, which reverses the sense of the B–R₀ correlation while B
  itself remains positive for nearly all species on both sides.
- `prm_main.tex` parameter-free-slope paragraph: dropped ", marking
  the screening/anti-screening crossover" from the pole sentence.
- `prm_main.tex` screening-collapse paragraph: "with B>0 for z<n and
  B<0 for z>n" → the slope β is negative for z<n and positive for z>n
  (no claim about the sign of B).
- `prm_main.tex` Fig. 1 caption: pole separates the negative-slope
  (z<n) and positive-slope (z>n) branches of Eq. (5), not
  screening/anti-screening branches.
- `prm_supplemental.tex` fig:pole_species caption: filled/open markers
  now labeled by z<n (negative β) / z>n (positive β) instead of
  screening/anti-screening branch.

Two related physics corrections in `prm_main.tex`:

- Expansion remainder order corrected. d²(ln w)/dR² = −1/(λ+R)², so
  δ_ij = O[(R_ij−R*)²/(λ+R*)²]; the previous denominator λ(λ+R*) is a
  loose bound on the screening branch and understates the error on the
  anti-screening branch for λ < −R*/2.
- "H_i is a purely geometric quantity" removed — H_i depends on B
  through the weights p_ij, not on shell geometry alone. Sentence
  split (also removes a grammatical colon per the style rule); the
  ln n bound and distortion statement retained unchanged.

## 2026-07-22 — Second-root radius comparison added to SI; validation sections expanded and revised

New SI figure (`theory/figures/lambda_minus_vs_weighted_radius.png`,
FIG. 20, label fig:lambda_minus_radius, placed before Table I):
|λ*₋| = R₀* + λ* versus population-weighted Shannon crystal radius for
96 screening-branch cations, unit-slope fits per formal charge, inset
c(z) = 1.05 + 0.092 z Å (r = 0.97). Generated by
`theory/exploratory/build_weighted_radii.py` +
`theory/exploratory/lambda_minus_vs_shannon.py` from the λ table,
Shannon 1976 radii (via pymatgen, provenance-stamped JSONs in
theory/exploratory/), and corpus CN populations; FIGURE_SOURCES.md
updated. New SI paragraph "Crystal-chemical anchoring of the second
root" documents the radius construction (CN-interpolated, population
weighted, high spin), the fits, and the exclusions (H⁺, P⁴⁺, Co⁺, Sn²⁺
lack Shannon entries; Cl⁻ and the anti-screening pair excluded).

`prm_main.tex`, "Empirical validation" — new closing paragraph with the
three requested elements. (1) Physical significance — the roots sum to
−R₀*, so |λ*₋| = R₀* + λ* is the characteristic shell radius extended
by one screening length, the outer reach of the screened bond.
(2) Contact-distance law — unit-slope fits per charge class with
c(z) ≈ 1.05 + 0.09 z Å; the constant sits at the oxygen crystal-radius
scale and the increment grows the effective contact by ~0.09 Å per unit
charge, so the effective oxygen size is valence-dependent, not a fixed
sphere. (3) Segue — this is the claim that the bond-valence shell
follows the electronic screening cloud, which the first-principles
comparison then tests directly.

Accuracy/clarity revisions during review of both sections:
- Repaired the dangling fragment left from restructuring ("for these
  the screening length…" → "For the 101 screening-branch species, λ* is
  a tabulated constant…"), removing a duplicated λ* formula.
- "First-principles confirmation" opening now cites m_i via
  eq:mean_entropy instead of re-defining it inline.
- Closing sentence de-duplicated (R² = 0.967 stated once; "extends"
  repetition removed) and stray blank lines cleaned.

## 2026-07-22 — Fig. 1 text enlarged 25%, panel letters bolded

`analysis/scripts/make_dblock_beta_prl_figure.py`, `render_box_figure` —
the main-text box figure now scales the shared manuscript style's five
font sizes (title, axis label, tick label, legend, annotation) by 1.25
via `dataclasses.replace` before rendering, and the (a)/(b) panel
letters are set in bold. Line widths and marker sizes unchanged. The SI
per-species scatter is untouched. Regenerated
`theory/figures/prm_oxygen_beta_vs_charge_box_cn4_cn6.png` and verified
no clipping of the panel letters, tick labels, or the one-row legend.

## 2026-07-22 — Fig. 2(b) legend order and outlier-label placement

`analysis/scripts/make_charge_density_benchmark_figures.py` — panel (b)
legend entries reordered to p, d, f, the order the blocks first occur
on the periodic table (previously d, p, f). The four outlier labels
were repositioned next to their own markers with per-label offsets and
anchors (`OUTLIER_LABEL_OFFSETS` now carries dx, dy, ha, va). B³⁺ sits
below its marker, P⁵⁺ above, As³⁺ below-left (clear of the identity
line and the V⁵⁺ point), and Au³⁺ to the right of its open circle
(previously the As³⁺ and Au³⁺ labels sat on the central cluster and
the B³⁺ label sat next to the P⁵⁺ marker). Regenerated
`theory/figures/thomas_fermi_reff_prl.png` (panel (a) unchanged) and
rebuilt `prm_main.pdf`.

## 2026-07-23 — Fig. 2 text reduced 20%

`analysis/scripts/make_charge_density_benchmark_figures.py`,
`render_prl_figure` — all text (axis labels, tick labels, legends,
panel letters, element and outlier labels) now scales by 0.80 via a
`text_scale` factor. The tick locations are pinned to the original
0.5 Å layout (x at 1.5–3.0, y at 1.5–2.5), since the auto-locator
otherwise densifies the ticks to match the smaller tick-label font.
Marker sizes, line widths, and label placements unchanged. Regenerated
`theory/figures/thomas_fermi_reff_prl.png` and rebuilt `prm_main.pdf`;
the SI null-model figure is rewritten by the same script but its
rendering is untouched.

## 2026-07-23 — Fig. 2 text scales adjusted

`analysis/scripts/make_charge_density_benchmark_figures.py`,
`render_prl_figure` — legend text restored to its original size
(8.1 pt), while axis and tick labels now scale by 0.75 (25% smaller
than original, superseding the previous 0.80). Panel letters and the
element/outlier data labels keep the 0.80 scale from the previous
entry, and the pinned 0.5 Å tick layout is unchanged. Regenerated
`theory/figures/thomas_fermi_reff_prl.png` and rebuilt `prm_main.pdf`.

## 2026-07-23 — Fig. 2 axis/tick text reduction made effective (25%)

`analysis/scripts/make_charge_density_benchmark_figures.py`,
`render_prl_figure` — correction to the two preceding entries: the
axis-label and tick-label reductions recorded there never rendered.
The figure's axes are created before the `rc_context` is entered, so
the tick and axis label text objects kept their creation-time sizes
(10.6/8.3 pt) and the rcParams overrides only reached the legends,
the explicit-fontsize annotations, and the tick auto-locator. The
sizes are now passed explicitly (`tick_params(labelsize=...)`,
`set_xlabel/set_ylabel(fontsize=...)`) at 0.75x the originals, i.e.
axis labels 7.95 pt and tick labels 6.2 pt. Legend text stays at the
original 8.1 pt; panel letters and element/outlier labels stay at
0.80x; the pinned 0.5 Å tick layout is unchanged. Regenerated
`theory/figures/thomas_fermi_reff_prl.png` (tight-crop canvas shrank
1906x488 -> 1874x467, confirming the change took effect) and rebuilt
`prm_main.pdf`.

## 2026-07-23 — Fig. 2 axis/tick text up 15% from the 25%-reduced size

`analysis/scripts/make_charge_density_benchmark_figures.py`,
`render_prl_figure` — `axis_scale` raised from 0.75 to 0.75 x 1.15
= 0.8625, so axis labels render at 9.1 pt and tick labels at 7.2 pt.
Legends (8.1 pt), panel letters and data labels (0.80x), and the
pinned tick layout are unchanged. Regenerated
`theory/figures/thomas_fermi_reff_prl.png` and rebuilt `prm_main.pdf`.

## 2026-07-23 — All semicolons removed from prm_main.tex

Ten prose semicolons (the manuscript's only ones — none in math or
macros) rewritten as separate sentences or joined clauses:
- anti-screening rarity, sum-rule-at-unity, p_ij middle/last forms,
  RANSAC acceptance criteria, block-coverage/Fig. 1 pointer (now
  "Figure~\ref{fig:pole}" at sentence start), and SI-pointer after the
  four excluded oxides — each split into two sentences;
- Fig. 1 stats parenthetical unpacked ("(68 at n=4, 82 at n=6, 150
  slopes in total). The weighted R^2 is 0.986, with mean absolute
  error 0.15 under the weighting scheme...");
- Fig. 2(a) caption "Solid line: shared fit; dashed line: ..." →
  "The solid line is the shared fit and the dashed line is
  r_eff = m_i." (also clears two grammatical colons);
- oxygen-side-window sentence joined with ", and"; "(Li2O through
  BaO; Fig. 2a)" → comma.
Verified zero ";" in source and in pdftotext of the rebuilt
`prm_main.pdf`.

## 2026-07-23 — All semicolons removed from prm_supplemental.tex

Fifteen prose semicolons (all in the SI — none in math or macros)
rewritten in the same style as the main text:
- twelve split into two sentences (anion-centered reparameterization,
  open-symbol outliers, fixed exclusion list, atlas panel ordering,
  mixed-coordination binning, line population/weights, degenerate vs
  nearly degenerate shells, pole-divergence plotting choice, BCP proxy
  definition, Thomas--Fermi tag note, excess-density weight fallback);
- per-panel metrics recast without the "Per panel:" colon ("The n=4
  panel alone gives R^2_w=0.973 with MAE_w=0.23, and the n=6 panel
  gives R^2_w=0.979 with MAE_w=0.089.");
- the z=n species parenthetical ("...for n=4, and Mo6+ and W6+ for
  n=6") and the "(r=0.97, inset)" tag now use commas, and the
  bond-length-identity stats moved out of the parenthetical into a
  ", with Pearson r=0.976 and MAE=0.019 A" clause.
Verified zero ";" in source and in pdftotext of the rebuilt
`prm_supplemental.pdf`.

## 2026-07-23 — Fig. 1 and Fig. 2 text matched on the page

`analysis/scripts/make_dblock_beta_prl_figure.py`, `render_box_figure`
— Fig. 1 is typeset at \columnwidth (3.40 in) from a ~5.9 in
tight-cropped canvas while Fig. 2 spans \textwidth (7.06 in) from a
6.28 in canvas, so equal in-figure point sizes render ~2x different on
the page. Fig. 1's fonts replace the previous flat 1.25x scaling with
per-class sizes chosen so the on-page rendered sizes match Fig. 2's
current ones (axis 17.9 pt -> 10.3 pt on page, ticks 14.0 -> 8.0,
legend 15.9 -> 9.1, panel letters 16.9 -> 9.7, n-labels 12.2 -> 7.0,
all within ~1% of Fig. 2). The bottom legend's handle spacing was
tightened (columnspacing 0.7, handlelength 1.2, handletextpad 0.3) so
the legend row no longer drives the tight-crop width, which had made
uniform font scaling self-defeating. Regenerated
`theory/figures/prm_oxygen_beta_vs_charge_box_cn4_cn6.png` (canvas
1764x788) and rebuilt `prm_main.pdf`; visual page-level comparison of
pages 2-3 confirms matched text sizes. A comment in the script records
the derivation and warns to re-sync if Fig. 2's sizes or either
typeset width changes.
