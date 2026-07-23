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
