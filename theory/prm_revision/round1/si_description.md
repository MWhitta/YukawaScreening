# Description of Supplemental Material

All supplemental figures and tables carry S-prefixed numbering
(FIG. S1-S21, TABLE S1).

**Data provenance.** The 103 cation-oxygen species, the per-structure
fit procedure (joint linear least squares with anion-centered
reparameterization for rank-deficient bond graphs and an R0-bounded
nonlinear fallback), and the explicit lists of s-block (10) and
non-s-block (25, with 21 included) oxides used in the Fig. 2 centroid
comparison, with the a priori directional-bonding exclusion rule.

**Coordination-number-resolved B-R0 fit lines.** Atlas of n-resolved
fit lines for every species, grouped by block: Group 1, Group 2,
d-block (8 panels), p-block (3 panels), f-block (2 panels), with the
panel-ordering convention stated. Now prefaced by the complete
line-fitting procedure (per-structure joint OLS, unambiguous-CN
binning, OLS+RANSAC line fits with inlier refit and 400-resample
bootstrap errors, inlier-count weights, and the Fig. 1 acceptance
filter: >= 4 inliers, sigma_beta <= 5, R2 >= 0.3); a sign census of
the per-shell fits (63,783 single-valence shells, 16.4% with B < 0,
resolved by z/n regime and by near-degeneracy, with the band-gap
resolution in Fig. S1: 20% B < 0 in metallic hosts versus 12% in
gapped hosts); the weighted-R2 / MAE definition with abundance
weights and per-panel goodness-of-fit metrics (pooled 0.986 / 0.15;
n=4 0.973 / 0.23; n=6 0.979 / 0.089); per-slope bootstrap standard
errors (median 0.19%, 90th percentile 3.9%); and the formal exclusion
rationale for z=n species with the named species.

**Per-species slopes at fixed coordination.** The per-species beta
versus z scatter underlying main-text Fig. 1 (block-colored,
semi-transparent so coincident species accumulate visibly), with
per-slope standard-error bars.

**Characteristic pairs and bond-length identity.** Closed-form
weighted-intersection formulas for (R0*, B*) and sigma_B (all
coordination lines pooled with inlier-count weights, no R2 screening
at this stage), an explicit analysis of the near-coincidence of
same-branch lines (slope-difference formula, conditioning statistics:
85 species with >= 3 lines, median sigma_B/B* = 5.8%, two-line
species flagged provisional), the (R0*, B*) convergence plot for
Group 1, Group 2, and the lanthanides, and the bond-length-identity
parity plot (R0 ~ Rbar + B ln(z/n)) across 44,844 screening-branch
structures (Pearson r = 0.976, MAE 0.019 A).

**Comparison with published softness parameters.** Distribution-level
comparison of B* (mean 0.38 +/- 0.15 A, median 0.37 A) with the
softBV (0.45 +/- 0.05 A, 155 pairs) and Gagne-Hawthorne
(0.40 +/- 0.06 A, 135 pairs) parameter sets, matched-species offsets,
and the species-level decorrelation among all tabulated softness
sets, interpreted through the pooled-per-pair versus per-material
estimand distinction.

**s-block screening-centroid controls.** Leave-one-out null-model bar
chart (MAE: m_i 0.006 A versus R0 0.078 A versus Shannon radius
0.095 A), full discrete construction of r_eff from 200-point
straight-segment density samples with the degenerate-weight fallback
rule, the directly measured oxygen-side valence shift fit
(Delta_O,val = 0.0613 m_i + 0.2394 A, mean 0.391 +/- 0.027 A), the
Thomas-Fermi positivity-filter disclaimer, the species-level
derivation of lambda* from the characteristic-pair fixed point (both
branches, real for B* >= -R0*/4), and the crystal-chemical anchoring
of the second root: |lambda*_-| = R0* + lambda* versus
population-weighted Shannon crystal radii for 96 screening-branch
species, unit-slope fits per formal charge with offsets
c(z) = 1.05 + 0.092 z A (r = 0.97).

**Screening-length table.** Full table of (R0*, B*, lambda*,
lambda*_-, n_CN) for all 103 species, atomic-number-ordered, with
propagated uncertainties; lambda* is assigned on both branches (101
screening-branch entries positive, the two anti-screening cases
signed), the second root lambda*_- is tabulated for every species,
and two-coordination-line entries are flagged provisional.
