# Revision Summary

Net comparison of the resubmission against the as-submitted baseline
(`theory/prl_submission/`, transferred 2026-07-22). This document
describes only the difference between the initial and final documents.
The running edit history lives in `REVISION_CHANGELOG.md`.

## Response to the referee's concerns

| Referee concern | Where addressed |
|---|---|
| Fig. 1 does not visually support R² = 0.986 | Fig. 1 redesigned as a box-and-whisker summary of per-species slopes; the full per-species scatter moved to the SI with block coloring and semi-transparency so coincident species are visible |
| How are the coordination-resolved B(R₀) lines constructed? Fixing R₀ and re-optimizing B? | Answered explicitly in the main text (Empirical validation): each structure contributes one (R₀,B) pair with both parameters fitted jointly; the lines are RANSAC regressions of B on R₀ across per-structure pairs at fixed species and n, with stated acceptance criteria. Full procedure documented in a new SI "Line-fitting procedure" paragraph |
| Theory introduction abrupt; distinguish exact reformulation, approximation, and interpretation | The Yukawa flux factor is now derived from Gauss's law at first use; the introduction closes with a roadmap (flux partition = postulate, isotropic Yukawa weight and its linearization = the two controlled approximations, partition-thermodynamic relations = exact identities); status statements accompany Eq. (2), the linearization, and the Partition-thermodynamics block |
| Broader discussion of prior work | Introduction rewritten: Preiser's point-charge flux result stated concretely, Brown's capacitor framework characterized, Adams distinguished as a bond-stiffness route, and the documented near-degeneracy of jointly fitted (R₀,B) with the fixed-B ≈ 0.37 Å convention added as context |
| Origin of the scatter near z = 3; slope uncertainties | New "Origin of the near-pole spread" section (quadratic pole amplification of fixed chemical scatter, −β²u; per-class spread grows as \|β\|^1.8; the same cations match at n = 6 to median \|Δβ\| = 0.03; deviants are sparse-CN4 species). Bootstrap slope standard errors now reported (median 0.2%, 90th percentile 4%) with error bars in the SI per-species figure and a new SI paragraph |
| Nearly coincident coordination lines; delicate intersections | Main-text comment (same-branch slope difference ln(n₂/n₁)/[ln(z/n₁)ln(z/n₂)]; intersection conditioned on the shell dilation of Eq. (7)) plus a new SI paragraph with the conditioning statistics (85 species with ≥3 lines, median σ_B/B* = 5.8%) |
| Many SI figure captions incorrect | All eight d-block atlas captions corrected against the rendered pages (several named species absent from the corpus); Group-1/Group-2 and p-block captions fixed; a caption-ordering convention stated in the atlas introduction |
| Letter format | Kept as a Letter; explanations added were offset by condensation elsewhere |

## Main text

**Format and conventions.** Document class option `prl` → `prmaterials`.
The reference length is now R′ (formerly R\*, which collided visually
with the characteristic intercept R₀\*). Ensemble terminology unified
to "Gibbs" (distribution, mean, shell centroid). Run-in section heads
get vertical separation via a preamble macro; `\raggedbottom` added.

**Abstract.** "150 fitted valences" → "150 fitted relationships"; the
abstract-level R² was removed. A new sentence states the second-root
result: the screening length sets a valence-dependent cation–oxygen
contact distance reproducing tabulated crystal radii, with an
effective oxygen size growing ~0.09 Å per unit cation charge.

**Introduction.** Opening now notes bond valence's growing use as a
structural descriptor in machine-learning models (three new
references). Prior-work paragraph rewritten (see referee table).
Closing paragraph adds the postulate/approximation/identity roadmap
and now states that first-principles screened-interaction methods are
shown to correlate with the Yukawa screening length.

**Screened-flux derivation.** The Yukawa flux factor
e^(−R/λ)(1 + R/λ) is derived by applying Gauss's law to the screened
potential, with the screening citations moved to first use. The
solid-angle factors Ω_ij of the initial ansatz are gone — the weight
depends on bond length alone, and this single-λ isotropic ansatz is
named as the derivation's physical postulate (its failure mode,
directional bonding, is handled in the Discussion as approximation
(ii)). The flux-ratio condition is motivated (screening attenuates,
the sum rule renormalizes, so only the relative apportionment is
physical) and the partition is now a numbered equation with an
exactness statement. The expansion is taken about an unspecified R′,
whose identification with the Gibbs mean is deferred to the
partition-thermodynamics section; the remainder order is corrected to
O[(R_ij−R′)²/(λ+R′)²]. A new sign-of-λ passage reports that 16% of
corpus shells fit B < 0, that roughly half of those are near-degenerate
shells with barely constrained B, and that the chemically systematic
anti-screening cases are square-planar/linear late transition metals
and large soft cations in high coordination (SI sign census).

**Parameter-free slope.** Unchanged in substance; adds a pointer to
the line-construction procedure and the bootstrap slope-uncertainty
statement.

**Origin of the near-pole spread.** New section (see referee table).

**Screening collapse at z = n.** Physics corrected. The initial
version claimed the sum rule forces S_ij = 1 bond-by-bond at z = n, so
any shell with unequal bonds requires B → ∞ (and λ → ∞ as √(BR\*)).
The final version states what actually degenerates: the sum rule fixes
only the mean bond valence; truly degenerate shells (Si⁴⁺ in zircon)
leave B unconstrained by a single structure, while nearly degenerate
shells (α-quartz, ~0.01 Å splitting) are satisfied at finite B. Corpus
counts added (7,565 z = n shells; 959 degenerate within 10⁻³ Å; 2,960
more within 0.05 Å). The branch change across the pole is carried by
the sign reversal of the slope β — the initial claim that z > n
selects B < 0 was removed as contradicted by the corpus (nearly all
z > n species fit B* > 0).

**Characteristic intersections (new section).** The wide-gap pole
species now motivate the construction. Two new numbered equations
give the line intercepts (β₀⁽ⁿ⁾ = −R̄⁽ⁿ⁾/ln(z/n)) and the
intersection ordinate as the shell-dilation rate
(B* = [R̄⁽ⁿ²⁾−R̄⁽ⁿ¹⁾]/ln(n₂/n₁)), giving the characteristic softness
a geometric meaning. The near-coincidence conditioning comment and
the weighted-variance definition of (R₀\*, B\*) follow, with the note
that screening-branch B\* values cluster near the conventional
0.37 Å (comparison with published parameterizations in the SI). The
λ\* inversion now covers both branches (real for B\* ≥ −R₀\*/4), so
the anti-screening pair carries signed λ\* values instead of being
excluded. The pole is given its screening interpretation as an
explicit argument (pole species are wide-gap insulators; gapped
systems screen weakly; the model concurs by demanding
|λ| ~ ε^(−1/2)), with the scope caveat retained. A new closing
passage derives the second root |λ\*₋| = R₀\* + λ\* — the outer reach
of the screened bond — and its contact-distance law against Shannon
crystal radii, |λ\*₋| = r_cr + c(z) with c(z) ≈ 1.05 + 0.09z Å, read
as a valence-dependent effective oxygen size.

**Partition thermodynamics.** Expanded from a compact aside into
three explicit steps with all three relations now numbered equations
(the Gibbs distribution, the mean/entropy definitions, and the exact
identity R₀ = m_i + B(ln q_i − H_i)). Status declared (exact
consequences; thermodynamic language is interpretation). Adds the
first-order coincidence caveat relating the screened-flux and
bond-valence forms, the q_i = z_i note, the reduction to Brown's
single-bond unit-valence limit, and the R′ = m_i closure. The section
now sits after the characteristic-intersections material, introduced
as supplying the shell radius the first-principles comparison needs.

**Empirical validation.** Now states the full line-construction
procedure in the manuscript (joint per-structure fits — not B
re-optimized at fixed R₀ — RANSAC regression across per-structure
pairs, ≥5 structures per cell, acceptance at ≥4 inliers, σ_β ≤ 5,
R² ≥ 0.3). Species-level λ\* material moved into Characteristic
intersections.

**First-principles confirmation.** Adds the physical motivation
before the operational definition: if bond valence is captured
screened flux, the Gibbs shell centroid should track the radius where
screening charge accumulates. Numbers unchanged.

**Discussion.** Approximation (ii) is now isotropic flux capture
(with its failure mechanism: directional bonding adds a
bond-dependent geometric term the expansion cannot absorb into C_i),
replacing the initial Ω_ij-uniformity statement. The identity-line
offset is now explained mechanistically — m_i references the oxygen
nucleus while r_eff tracks the screening charge displaced into the
bond — with the directly measured shift Δ_O,val = 0.39 ± 0.03 Å
matching the fitted-line gap. The closing paragraph anchors λ\* in
crystallographic systematics (|λ\*₋| = R₀\* + λ\* = r_cr + c(z); λ\*
is the length carrying the shell radius out to the crystallographic
contact) before the tabulated-descriptor claim.

## Supplemental Material

- Figures and tables renumbered with the S prefix (FIG. S1–, TABLE S1).
- New section "Per-species slopes at fixed coordination" — the full
  per-species scatter (formerly main-text Fig. 1) with block coloring,
  semi-transparency, and σ_β error bars.
- New section "Comparison with published softness parameters" —
  distribution-level agreement of B\* with softBV and
  Gagné–Hawthorne values and the species-level decorrelation, read
  through the pooled-fit vs per-material estimand distinction.
- New paragraphs: "Line-fitting procedure" (full pipeline: joint OLS,
  CN binning, RANSAC with thresholds, bootstrap errors, inlier-count
  weights, Fig. 1 acceptance filter); "Sign census of the per-shell
  fits" (63,783 single-valence shells, 16.4% B < 0, regime split,
  near-degeneracy statistics, reproduction script); "Slope standard
  errors"; "Near-coincidence of same-branch lines" (slope-difference
  formula and conditioning statistics); "Crystal-chemical anchoring
  of the second root" (with a new figure of |λ\*₋| against
  population-weighted Shannon radii).
- Corrections: all eight d-block atlas captions (panel contents were
  misidentified; some named species absent from the corpus), Group-1
  and p-block caption fixes, a stated caption-ordering convention,
  completed inclusion criteria for the 44,844-structure parity check,
  N_i defined as the final-refit inlier count, and the z = n exclusion
  discussion aligned with the corrected screening-collapse physics.
- λ\* table extended: λ\* computed for both branches (signed values
  for Mo³⁺/Sb³⁺) and a second-root λ\*₋ column added for all species.

## Figures

- **Fig. 1 (main)**: new box-and-whisker asset summarizing per-species
  slopes at each formal charge; per-species scatter moved to the SI.
- **Fig. 2 (main)**: markers reduced to half linear size; all data
  labels placed adjacent to their points (the coincident pairs Ca/Na
  and Ba/K split across the fit line); caption updated to the Gibbs
  terminology.
- **New SI figure**: second-root magnitude vs population-weighted
  Shannon crystal radius with per-charge unit-slope fits and the
  c(z) inset.

## References and supporting material

- Three references added on bond-valence descriptors in machine
  learning (Li et al. 2021; Zhang et al. 2023; Miller & Rondinelli
  2023).
- New analysis scripts in the repository reproduce the quoted
  statistics: `bv_sign_census.py` (sign census),
  `bv_pole_decorrelation.py` (B–R₀ decorrelation at the pole), and
  `bv_skewness_test.py` (second-order skewness test), alongside the
  updated figure and table generators.
- The SI's stated R₀ fit bounds were verified against the corpus
  store (all 85,661 fitted records) and left unchanged.
