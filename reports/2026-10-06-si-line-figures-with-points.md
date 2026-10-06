# SI fit-line figures with points (2026-10-06)

Changes: analysis/critmin/analysis/bond_valence_theory.py (_serialize_fit_summary keeps `points`; new `_points_entry`; `points` attached to `oxygen` and `oxygen_ransac` in build_unified_oxygen_cn_fits). analysis/critmin/viz/notebook_families.py (export_species_fit_line_atlas: kwargs show_points, window; scatter s=7 alpha .55 no edge, inliers in cn color, outliers gray 0.45; fixed window axes; annotation above panel, set_in_layout(False); module list ANNOT_LOG).
Regeneration: PYTHONPATH=analysis analysis/.venv/bin/python analysis/scripts/make_block_fit_line_atlases.py. File name set identical (16 files, diffed before/after).
Lines unchanged: beta, beta0, R0_min, R0_max, n_inliers compared for 4302 values over 103 species, max |diff| = 0 (before dump from unmodified code).
Consumers: only key-wise readers (make_dblock_beta_prl_figure.py, notebook_setup.py, make_response_co_o_figure.py use .get("oxygen_ransac")/beta); no JSON dump of cn_fits found. viz/bond_valence.py has its own separate cn_fits builder, untouched.
Tests: no pytest in analysis/.venv, no test files under analysis; tests/ with system python collects 0 tests. check_manuscript_consistency.py fails pre-existing (FileNotFoundError theory/prl_main.tex), unrelated.
Viewed dblock part01 and group1 page: points visible, window 0-4 x -1-2 applied, annotation legible two-line left-aligned above panel, legend intact. Note: first attempt squeezed axes (annotation in tight_layout), fixed with set_in_layout(False). Group1 view was before that fix only; dblock part01 viewed after.
Caveat: d-block cn colors are grays, so inliers vs gray outliers are hard to tell apart in d-block pages (existing color scheme).

## Per-panel out-of-window counts (inlier structures outside window / inliers / RANSAC outliers)
| page | panel | k outside | N inliers | RANSAC outliers |
|---|---|---|---|---|
| group1_oxygen_cn_fit_lines_part01 | Cs¹⁺ | 12 | 234 | 54 |
| group1_oxygen_cn_fit_lines_part01 | K¹⁺ | 27 | 650 | 160 |
| group1_oxygen_cn_fit_lines_part01 | Li¹⁺ | 54 | 2990 | 749 |
| group1_oxygen_cn_fit_lines_part01 | Na¹⁺ | 45 | 1515 | 347 |
| group1_oxygen_cn_fit_lines_part01 | Rb¹⁺ | 17 | 245 | 42 |
| group1_oxygen_cn_fit_lines_part01 | H¹⁺ | 177 | 1015 | 144 |
| group2_oxygen_cn_fit_lines_part01 | Ba²⁺ | 42 | 1011 | 217 |
| group2_oxygen_cn_fit_lines_part01 | Be²⁺ | 3 | 107 | 21 |
| group2_oxygen_cn_fit_lines_part01 | Ca²⁺ | 10 | 1157 | 208 |
| group2_oxygen_cn_fit_lines_part01 | Mg²⁺ | 15 | 969 | 219 |
| group2_oxygen_cn_fit_lines_part01 | Sr²⁺ | 47 | 880 | 152 |
| dblock_oxi_oxygen_cn_fit_lines_part01 | Co¹⁺ | 2 | 10 | 0 |
| dblock_oxi_oxygen_cn_fit_lines_part01 | Co²⁺ | 57 | 502 | 173 |
| dblock_oxi_oxygen_cn_fit_lines_part01 | Co³⁺ | 94 | 220 | 102 |
| dblock_oxi_oxygen_cn_fit_lines_part01 | Co⁴⁺ | 5 | 114 | 35 |
| dblock_oxi_oxygen_cn_fit_lines_part01 | Cr²⁺ | 22 | 118 | 34 |
| dblock_oxi_oxygen_cn_fit_lines_part01 | Cr³⁺ | 104 | 447 | 194 |
| dblock_oxi_oxygen_cn_fit_lines_part02 | Cr⁴⁺ | 0 | 76 | 29 |
| dblock_oxi_oxygen_cn_fit_lines_part02 | Cr⁵⁺ | 22 | 49 | 26 |
| dblock_oxi_oxygen_cn_fit_lines_part02 | Cu¹⁺ | 21 | 155 | 25 |
| dblock_oxi_oxygen_cn_fit_lines_part02 | Cu²⁺ | 153 | 765 | 193 |
| dblock_oxi_oxygen_cn_fit_lines_part02 | Cu³⁺ | 19 | 94 | 20 |
| dblock_oxi_oxygen_cn_fit_lines_part02 | Fe²⁺ | 47 | 602 | 168 |
| dblock_oxi_oxygen_cn_fit_lines_part03 | Fe³⁺ | 92 | 982 | 317 |
| dblock_oxi_oxygen_cn_fit_lines_part03 | Fe⁴⁺ | 9 | 45 | 17 |
| dblock_oxi_oxygen_cn_fit_lines_part03 | Mn²⁺ | 94 | 817 | 218 |
| dblock_oxi_oxygen_cn_fit_lines_part03 | Mn³⁺ | 125 | 572 | 177 |
| dblock_oxi_oxygen_cn_fit_lines_part03 | Mn⁴⁺ | 21 | 280 | 110 |
| dblock_oxi_oxygen_cn_fit_lines_part03 | Ni²⁺ | 90 | 532 | 204 |
| dblock_oxi_oxygen_cn_fit_lines_part04 | Ni³⁺ | 74 | 190 | 60 |
| dblock_oxi_oxygen_cn_fit_lines_part04 | Sc³⁺ | 68 | 362 | 158 |
| dblock_oxi_oxygen_cn_fit_lines_part04 | Ti³⁺ | 18 | 80 | 26 |
| dblock_oxi_oxygen_cn_fit_lines_part04 | Ti⁴⁺ | 140 | 1322 | 290 |
| dblock_oxi_oxygen_cn_fit_lines_part04 | V³⁺ | 44 | 410 | 165 |
| dblock_oxi_oxygen_cn_fit_lines_part04 | V⁴⁺ | 5 | 489 | 92 |
| dblock_oxi_oxygen_cn_fit_lines_part05 | V⁵⁺ | 46 | 746 | 144 |
| dblock_oxi_oxygen_cn_fit_lines_part05 | Zn²⁺ | 82 | 718 | 245 |
| dblock_oxi_oxygen_cn_fit_lines_part05 | Ag¹⁺ | 32 | 176 | 43 |
| dblock_oxi_oxygen_cn_fit_lines_part05 | Ag²⁺ | 2 | 13 | 5 |
| dblock_oxi_oxygen_cn_fit_lines_part05 | Cd²⁺ | 63 | 225 | 98 |
| dblock_oxi_oxygen_cn_fit_lines_part05 | Mo³⁺ | 0 | 51 | 15 |
| dblock_oxi_oxygen_cn_fit_lines_part06 | Mo⁵⁺ | 4 | 89 | 35 |
| dblock_oxi_oxygen_cn_fit_lines_part06 | Mo⁶⁺ | 37 | 628 | 165 |
| dblock_oxi_oxygen_cn_fit_lines_part06 | Nb⁵⁺ | 44 | 1045 | 196 |
| dblock_oxi_oxygen_cn_fit_lines_part06 | Pd²⁺ | 33 | 69 | 21 |
| dblock_oxi_oxygen_cn_fit_lines_part06 | Y³⁺ | 31 | 592 | 222 |
| dblock_oxi_oxygen_cn_fit_lines_part06 | Zr⁴⁺ | 34 | 511 | 134 |
| dblock_oxi_oxygen_cn_fit_lines_part07 | Au³⁺ | 20 | 41 | 23 |
| dblock_oxi_oxygen_cn_fit_lines_part07 | Hf⁴⁺ | 53 | 326 | 94 |
| dblock_oxi_oxygen_cn_fit_lines_part07 | Hg¹⁺ | 2 | 7 | 5 |
| dblock_oxi_oxygen_cn_fit_lines_part07 | Hg²⁺ | 25 | 77 | 19 |
| dblock_oxi_oxygen_cn_fit_lines_part07 | Ir⁴⁺ | 0 | 77 | 27 |
| dblock_oxi_oxygen_cn_fit_lines_part07 | Pt²⁺ | 0 | 18 | 1 |
| dblock_oxi_oxygen_cn_fit_lines_part08 | Re⁶⁺ | 0 | 37 | 10 |
| dblock_oxi_oxygen_cn_fit_lines_part08 | Re⁷⁺ | 7 | 99 | 19 |
| dblock_oxi_oxygen_cn_fit_lines_part08 | Ta⁵⁺ | 73 | 782 | 139 |
| dblock_oxi_oxygen_cn_fit_lines_part08 | W⁶⁺ | 3 | 621 | 166 |
| pblock_oxygen_cn_fit_lines_part01 | Al³⁺ | 18 | 900 | 168 |
| pblock_oxygen_cn_fit_lines_part01 | As⁵⁺ | 42 | 465 | 138 |
| pblock_oxygen_cn_fit_lines_part01 | B³⁺ | 16 | 931 | 354 |
| pblock_oxygen_cn_fit_lines_part01 | Bi³⁺ | 107 | 473 | 118 |
| pblock_oxygen_cn_fit_lines_part01 | Ga³⁺ | 59 | 482 | 107 |
| pblock_oxygen_cn_fit_lines_part01 | Ge⁴⁺ | 4 | 461 | 192 |
| pblock_oxygen_cn_fit_lines_part01 | In³⁺ | 20 | 426 | 86 |
| pblock_oxygen_cn_fit_lines_part01 | P⁴⁺ | 0 | 44 | 6 |
| pblock_oxygen_cn_fit_lines_part01 | P⁵⁺ | 40 | 5957 | 1247 |
| pblock_oxygen_cn_fit_lines_part01 | Pb²⁺ | 43 | 333 | 81 |
| pblock_oxygen_cn_fit_lines_part01 | Pb⁴⁺ | 3 | 43 | 19 |
| pblock_oxygen_cn_fit_lines_part01 | S⁴⁺ | 2 | 34 | 2 |
| pblock_oxygen_cn_fit_lines_part02 | S⁶⁺ | 83 | 1500 | 374 |
| pblock_oxygen_cn_fit_lines_part02 | Sb³⁺ | 13 | 126 | 39 |
| pblock_oxygen_cn_fit_lines_part02 | Sb⁵⁺ | 9 | 398 | 171 |
| pblock_oxygen_cn_fit_lines_part02 | Si⁴⁺ | 1 | 2336 | 567 |
| pblock_oxygen_cn_fit_lines_part02 | Sn²⁺ | 15 | 72 | 17 |
| pblock_oxygen_cn_fit_lines_part02 | Sn⁴⁺ | 45 | 436 | 125 |
| pblock_oxygen_cn_fit_lines_part02 | Te⁴⁺ | 26 | 236 | 42 |
| pblock_oxygen_cn_fit_lines_part02 | Te⁶⁺ | 0 | 199 | 132 |
| pblock_oxygen_cn_fit_lines_part02 | Tl¹⁺ | 24 | 113 | 24 |
| pblock_oxygen_cn_fit_lines_part02 | Tl³⁺ | 28 | 76 | 14 |
| pblock_oxygen_cn_fit_lines_part02 | C⁴⁺ | 63 | 790 | 151 |
| pblock_oxygen_cn_fit_lines_part02 | Cl¹⁻ | 0 | 5 | 0 |
| pblock_oxygen_cn_fit_lines_part03 | H¹⁺ | 177 | 1015 | 144 |
| pblock_oxygen_cn_fit_lines_part03 | I⁵⁺ | 2 | 113 | 12 |
| pblock_oxygen_cn_fit_lines_part03 | I⁷⁺ | 4 | 37 | 17 |
| pblock_oxygen_cn_fit_lines_part03 | N³⁺ | 0 | 25 | 3 |
| pblock_oxygen_cn_fit_lines_part03 | N⁵⁺ | 34 | 200 | 56 |
| fblock_oxygen_cn_fit_lines_part01 | Ce³⁺ | 20 | 146 | 25 |
| fblock_oxygen_cn_fit_lines_part01 | Dy³⁺ | 19 | 220 | 64 |
| fblock_oxygen_cn_fit_lines_part01 | Er³⁺ | 7 | 235 | 41 |
| fblock_oxygen_cn_fit_lines_part01 | Eu³⁺ | 20 | 107 | 29 |
| fblock_oxygen_cn_fit_lines_part01 | Gd³⁺ | 8 | 286 | 90 |
| fblock_oxygen_cn_fit_lines_part01 | Ho³⁺ | 11 | 212 | 67 |
| fblock_oxygen_cn_fit_lines_part01 | La³⁺ | 78 | 677 | 199 |
| fblock_oxygen_cn_fit_lines_part01 | Lu³⁺ | 51 | 186 | 61 |
| fblock_oxygen_cn_fit_lines_part01 | Nd³⁺ | 69 | 424 | 115 |
| fblock_oxygen_cn_fit_lines_part01 | Pr³⁺ | 27 | 296 | 57 |
| fblock_oxygen_cn_fit_lines_part01 | Sm³⁺ | 48 | 275 | 91 |
| fblock_oxygen_cn_fit_lines_part01 | Tb³⁺ | 21 | 182 | 44 |
| fblock_oxygen_cn_fit_lines_part02 | Tm³⁺ | 10 | 179 | 29 |
| fblock_oxygen_cn_fit_lines_part02 | Yb³⁺ | 1 | 60 | 9 |
| fblock_oxygen_cn_fit_lines_part02 | Th⁴⁺ | 17 | 63 | 21 |
| fblock_oxygen_cn_fit_lines_part02 | U⁴⁺ | 15 | 30 | 5 |
| fblock_oxygen_cn_fit_lines_part02 | U⁵⁺ | 21 | 39 | 17 |
| fblock_oxygen_cn_fit_lines_part02 | U⁶⁺ | 0 | 241 | 48 |
