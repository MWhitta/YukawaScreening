# Co-O response figure (2026-10-05)

Command (repo root): `analysis/.venv/bin/python analysis/scripts/make_response_co_o_figure.py` (about 39 s).
Files created: analysis/scripts/make_response_co_o_figure.py; theory/prm_revision/response_figures/co_o_fit_lines_with_points.{png,pdf}. No other file modified. No tests run (no existing code touched).

Method: same calls as make_block_fit_line_atlases.build_block_atlases (all species fitted, same target CNs). Line drawing and axis padding copied from export_species_fit_line_atlas. Line styles: cn_linestyle_map over the cns present on the six panels of Supplemental part01 (Co1-4+, Cr2+, Cr3+), so styles match the page (n=6 dash-dot-dot, n=5 dashed, n=4 dotted, n=3 dash-dot, n=2 solid-dashed per map). Colors: tab10 per cn (the atlas uses a single gray per element, which cannot separate cn in a points plot), so colors differ from the SI page by design. Inliers = filled circles (alpha 0.45), outliers = x, least-spread point = open circle.

Key facts
- fit_bv_regression_ransac is seeded (RANSACRegressor random_state=42, bootstrap rng seeded) so the fits are reproducible.
- cn_fits serialization (_serialize_fit_summary) does NOT keep inlier_mask. The script re-runs fit_bv_regression_ransac on the identical fit_input and checks beta/beta0 equal the stored ones (refitOK True for all 13 rows) and len(mask)==len(fit_records) (maskOK True for all 13).
- Lines match the Supplemental PNG visually (viewed both): same slopes, same extents (Co1+ to R0~253, Co2+ to about +-10, Co3+ to about +-1000, Co4+ 1.0-2.8), same least-spread points, same styles.
- Top row has ranges to R0 = 253 and +-1000 Å; they come from real inliers (Co1+ cn6, Co3+ cn6), not from extrapolation.

Table (physical window 0<R0<4, -1<B<2; frac_out = fraction of inliers outside window; medrng = median fit_diagnostics.bond_length_range of inliers inside / outside window, nan = none)
```
label cn N n_in n_out beta beta0 R0min R0max n_inwin frac_out medrng_in medrng_out maskOK refitOK maskstored
Co¹⁺ 2 5 5 0 -1.337e-14 0.37 1.492 1.513 5 0.000 0 nan True True False
Co¹⁺ 6 5 5 0 -0.5588 1.255 1.132 253.3 3 0.400 0.02809 0 True True False
Co²⁺ 3 14 13 1 -2.456 4.555 0.4713 2.307 9 0.308 0.115 0.1328 True True False
Co²⁺ 4 127 81 46 -1.247 2.517 0.7893 2.457 81 0.000 0.008573 nan True True False
Co²⁺ 5 41 36 5 -1.114 2.307 -0.2229 3.535 32 0.111 0.1663 0.2465 True True False
Co²⁺ 6 493 372 121 -0.9187 1.953 -10.64 8.987 323 0.132 0.1364 0.02503 True True False
Co³⁺ 2 5 5 0 1.022e-15 0.37 1.673 1.785 5 0.000 0.2562 nan True True False
Co³⁺ 4 46 42 4 -3.433 6.447 1.159 3.313 34 0.190 0.06089 0.1453 True True False
Co³⁺ 5 9 8 1 -1.954 3.858 0.08041 4.47 5 0.375 0.2391 0.2611 True True False
Co³⁺ 6 262 165 97 -1.442 2.914 -870.4 1140 82 0.503 0.2828 0.02342 True True False
Co⁴⁺ 4 19 17 2 6.444 -11.13 1.78 1.825 17 0.000 0.06325 nan True True False
Co⁴⁺ 5 6 5 1 -6.708 12.06 1.576 1.666 5 0.000 0.1771 nan True True False
Co⁴⁺ 6 124 92 32 -2.301 4.426 1.048 2.849 87 0.054 0.02499 0.03781 True True False
```

Window annotations (all cn pooled): Co1+ 2 of 10 inliers outside (0 outliers); Co2+ 57 of 502 (173 outliers); Co3+ 94 of 220 (102 outliers); Co4+ 5 of 114 (35 outliers).

Degeneracy: the far-out points are the near-degenerate shells. Co1+ cn6: in-window median range 0.028, out-of-window median exactly 0 (all bond lengths identical). Co3+ cn6: 0.283 inside vs 0.023 outside. Co2+ cn6: 0.136 vs 0.025. Co4+ cn6: 0.025 vs 0.038 (only 5 outside, weaker contrast). Exceptions: Co2+ cn3 and cn5, Co3+ cn4 and cn5 show outside >= inside (small n, mostly mild window clipping at B>2, not huge R0). So the +-1000 Å extents come from cn=6 shells with near-zero bond-length spread (R0 and B poorly determined).

Surprises/caveats
- Co1+ cn2, Co3+ cn2 slopes are about 0 (1e-14) with R0 span 0.02-0.11 Å, drawn as short segments, nearly invisible, as in the SI page. Legend lists n=2 because it appears on Co panels, although it is barely visible.
- Co1+ cn6 has only 3 of 5 inliers in window; Co1+ n=6 line ends at R0=253 so bottom-left panel line is a clipped segment.
- Colors differ from the SI page (see above). Legend shows n values 2-6 only (cn 1 absent from Co).
