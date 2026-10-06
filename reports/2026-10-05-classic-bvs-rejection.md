# Classic BVS rejection estimate (2026-10-05)

Command: `cd /Users/mwhittaker/Projects/github/YukawaScreening && python3 analysis/scripts/classic_bvs_rejection_estimate.py` (numpy 2.4.3, runtime ~12 s). Script prints everything below (verbatim output follows the notes). Not committed.

## Filters and definitions
- Store records: `datasets.oxygen_authoritative.payload[el][oxides|hydroxides]`, status == 'fitted', oxi_state_label == 'pure', finite R0, B, cn, oxi_state. Exactly 63,783 selected (of 85,661 scanned).
- Match: fixture sites with same mid, element, round(z) == round(oxi_state). Unmatched records are excluded from all fractions. cn_mismatch = any matched site with len(R) != cn (kept).
- dev = |BVS_P - z|/z with BVS = sum exp((R0_P - R_j)/B_P), averaged over a record's matched sites. Published value looked up by (element, z) in comparison.csv (softBV uses softBV_b). Uncovered records are excluded from the denominator.
- Extreme = |R0|>4 or |B|>2. bond_length_range taken from fit_diagnostics.
- Equal weight per record. Fractions are k/n with n = matched and covered records.

## Notes, surprises, unverified
- Only 52,252 of 63,783 (81.9%) records match a fixture site, so all rates are on the matched subset. Unmatched rate is not uniform: Si4+ has only 13 matched of 2,903, so the fixture likely labels Si differently (e.g. oxi_label or z) or excludes it. Not investigated.
- Sanity check is only partly clean. Own-fit median |dev| = 0.0086 (small), but p90 = 0.31, 26.7% of matched records have own-fit dev > 0.05, and the mean is inf (overflow in exp for extreme fits, e.g. negative B). So the fitted (R0,B) do not reproduce z for a sizeable minority, meaning the fit target (store) and fixture shell (R lists) are not fully consistent. Likely causes (unchecked): fits use a different shell/cutoff than the fixture R lists, a different z-constraint, or the cn_mismatch subset (859). This caveat applies to interpreting the rejection rates.
- Moderate rejection (dev>0.10) is high (about 29-31% for GH2015/BA1985) but dev>0.20 is 8%. Classic parameters are themselves not exact for these DFT-relaxed shells, so a 0.10 threshold is a stringent test.
- Extreme fits are only modestly more likely to be rejected (GH2015 dev>0.20: 13.0% vs 7.4%; median dev 0.081 vs 0.063). Extreme fits are mostly NOT rejected: 1,491 of 15,721 GH2015 rejects at 0.10 (9.5%) and 462 of 3,975 at 0.20 (11.6%) are extreme. Extreme fits are 3,672 of 52,252 matched (7.0%).
- z==cn stratum has only n=7, not meaningful. z>cn stratum has the lowest rejection rates (GH2015>0.20: 0.3%); z<cn the highest.
- BO1991 covers only 13,049 matched records, so its numbers are not comparable to others.
- Not verified: independent recomputation of any number, correctness of the fixture-vs-store species labelling, bond_length_range definition (assumed scalar or [min,max]).

## Verbatim script output
```
# Classic BVS rejection estimate (raw script output)
store raw records scanned: 85661; selected (fitted, pure, finite R0,B,cn,z): 63783
fixture structures: 28799; fixture (mid,element,z) keys: 80050
matched 52252/63783; unmatched 11531/63783; cn_mismatch (any matched site len(R)!=cn) 859/52252 matched
SANITY own-fit |dev|: median 0.008643, mean inf, p90 0.3089, p99 1.01e+04, frac>0.05 0.2669 (n=52252)
published species available: GH2015=135, BA1985=71, BO1991=46, softBV=155

## Overall (denominator = matched records covered by P)
GH2015: covered 51237, uncovered 1015 (of 52252 matched)
  dev>0.05: 0.6110 (31304/51237)
  dev>0.1: 0.3068 (15721/51237)
  dev>0.2: 0.0776 (3975/51237)
  dev>0.3: 0.0327 (1676/51237)
BA1985: covered 45461, uncovered 6791 (of 52252 matched)
  dev>0.05: 0.6064 (27566/45461)
  dev>0.1: 0.2927 (13307/45461)
  dev>0.2: 0.0826 (3756/45461)
  dev>0.3: 0.0379 (1725/45461)
BO1991: covered 13049, uncovered 39203 (of 52252 matched)
  dev>0.05: 0.7699 (10047/13049)
  dev>0.1: 0.1827 (2384/13049)
  dev>0.2: 0.0489 (638/13049)
  dev>0.3: 0.0180 (235/13049)
softBV: covered 51863, uncovered 389 (of 52252 matched)
  dev>0.05: 0.8247 (42774/51863)
  dev>0.1: 0.4548 (23586/51863)
  dev>0.2: 0.1184 (6142/51863)
  dev>0.3: 0.0392 (2032/51863)

## Stratified (matched records; fraction covered-by-P with dev>t, with k/n)
### bond_length_range <0.05 vs >=0.05
  <0.05 (n matched=14247): GH2015>0.1: 0.2659 (3659/13761) | GH2015>0.2: 0.0498 (685/13761) | BA1985>0.1: 0.2394 (2712/11327) | BA1985>0.2: 0.0465 (527/11327)
  >=0.05 (n matched=38005): GH2015>0.1: 0.3219 (12062/37476) | GH2015>0.2: 0.0878 (3290/37476) | BA1985>0.1: 0.3104 (10595/34134) | BA1985>0.2: 0.0946 (3229/34134)
  range missing (n matched=0): GH2015>0.1: n/a (0) | GH2015>0.2: n/a (0) | BA1985>0.1: n/a (0) | BA1985>0.2: n/a (0)
### extreme fit (|R0|>4 or |B|>2)
  extreme (n matched=3672): GH2015>0.1: 0.4204 (1491/3547) | GH2015>0.2: 0.1303 (462/3547) | BA1985>0.1: 0.4122 (1106/2683) | BA1985>0.2: 0.1211 (325/2683)
  non-extreme (n matched=48580): GH2015>0.1: 0.2984 (14230/47690) | GH2015>0.2: 0.0737 (3513/47690) | BA1985>0.1: 0.2852 (12201/42778) | BA1985>0.2: 0.0802 (3431/42778)
### sign of B
  B<0 (n matched=9255): GH2015>0.1: 0.3830 (3454/9019) | GH2015>0.2: 0.1212 (1093/9019) | BA1985>0.1: 0.3779 (2899/7672) | BA1985>0.2: 0.1273 (977/7672)
  B>0 (n matched=42997): GH2015>0.1: 0.2906 (12267/42218) | GH2015>0.2: 0.0683 (2882/42218) | BA1985>0.1: 0.2754 (10408/37789) | BA1985>0.2: 0.0735 (2779/37789)
  B==0 (n matched=0): GH2015>0.1: n/a (0) | GH2015>0.2: n/a (0) | BA1985>0.1: n/a (0) | BA1985>0.2: n/a (0)
### z vs cn
  z<cn (n matched=39614): GH2015>0.1: 0.3709 (14359/38712) | GH2015>0.2: 0.1016 (3935/38712) | BA1985>0.1: 0.3699 (12277/33187) | BA1985>0.2: 0.1125 (3735/33187)
  z==cn (n matched=7): GH2015>0.1: 0.8571 (6/7) | GH2015>0.2: 0.1429 (1/7) | BA1985>0.1: 0.8571 (6/7) | BA1985>0.2: 0.1429 (1/7)
  z>cn (n matched=12631): GH2015>0.1: 0.1083 (1356/12518) | GH2015>0.2: 0.0031 (39/12518) | BA1985>0.1: 0.0835 (1024/12267) | BA1985>0.2: 0.0016 (20/12267)

## dev distribution under GH2015
extreme: n=3547 median 0.08108 IQR [0.03564, 0.1476]
non-extreme: n=47690 median 0.06274 IQR [0.03674, 0.1103]
(BA1985) extreme: n=2683 median 0.08183 IQR [0.03627, 0.1401]
(BA1985) non-extreme: n=42778 median 0.06096 IQR [0.03643, 0.1079]
GH2015>0.1: rejects 15721, of which extreme 1491; extreme covered 3547, of which rejected 1491
GH2015>0.2: rejects 3975, of which extreme 462; extreme covered 3547, of which rejected 462
BA1985>0.1: rejects 13307, of which extreme 1106; extreme covered 2683, of which rejected 1106
BA1985>0.2: rejects 3756, of which extreme 325; extreme covered 2683, of which rejected 325
extreme records: total 3978 of 63783 selected; 3672 matched

## Per-species (all selected records; "n" includes unmatched; fractions over matched covered)
species | n | n matched | n GH2015-covered | GH2015>0.10 | BA1985>0.10
Li1+ | 3743 | 3620 | 3620 | 0.4262 (1543/3620) | 0.5039 (1824/3620)
Na1+ | 1864 | 1762 | 1762 | 0.7111 (1253/1762) | 0.7372 (1299/1762)
K1+ | 813 | 739 | 739 | 0.7212 (533/739) | 0.6563 (485/739)
Mg2+ | 1189 | 1143 | 1143 | 0.2432 (278/1143) | 0.2870 (328/1143)
Ca2+ | 1373 | 1307 | 1307 | 0.4858 (635/1307) | 0.4652 (608/1307)
Al3+ | 1068 | 955 | 955 | 0.1403 (134/955) | 0.0984 (94/955)
Si4+ | 2903 | 13 | 13 | 0.1538 (2/13) | 0.1538 (2/13)
Ti4+ | 1615 | 1491 | 1491 | 0.1140 (170/1491) | 0.0503 (75/1491)
Mn2+ | 1037 | 968 | 968 | 0.3543 (343/968) | 0.3130 (303/968)
Fe3+ | 1307 | 1232 | 1232 | 0.2394 (295/1232) | 0.2565 (316/1232)

runtime 11.9 s
```

## Verification by the main session (2026-10-05)

Independent recomputation from the two sources (separate script, same filters):
- Matching reproduced exactly: 52,252 of 63,783 matched. The unmatched
  11,531 are 7,379 z = n records (fixture contains only 7 z = n sites, since it
  stores network-solved sites and the z = n solve is degenerate) and 4,152
  z ≠ n records spread thinly across species (P5+ 428, S6+ 127, C4+ 114, ...).
  So the pole species cannot be tested with this fixture; Fig. 1 species (z ≠ n)
  are matched at 52,245 / 56,397 = 92.6 %.
- Same bond set: 46,284 of 52,252 matched records (88.6 %) have identical
  min and max bond length in fixture and store (|Δ| < 1e-3 Å). Mean bond
  lengths differ more often (55 % agree) because the store averages
  symmetry-distinct bonds and the fixture lists bonds with multiplicity.
- The own-fit "sanity failure" is NOT a data mismatch. On same-bond-set
  records it persists (p90 0.205, 21 % > 0.05), and it tracks the fit's own
  residual: records with own-fit dev > 0.05 have median
  linear_residual_rms/|B| = 0.51 versus 0.039 for the rest (dev > 0.2: 1.19 vs
  0.047). The per-structure (R0, B) is a least-squares fit to per-bond ln s_ij,
  not a constraint that Σ exp((R0 − R)/B) = z, so ill-conditioned (large or
  near-zero B) shells reproduce the sum poorly. The published-parameter
  rejection rates are unaffected by this.
- Context from the primary sources (Chen & Adams 2017 criteria (i)–(viii),
  Adams 2001 p. 283, G&H 2015 §2, B&A 1985): none of the classic fits applied a
  per-structure BVS-deviation rejection. Rejections were on experimental
  quality (R-factor, disorder, temperature, modulation, anion purity), with the
  exception of ~20 Li–O environments Adams 2001 removed by hand for "strange
  bond-valence sums" (of n = 96, i.e. ~20 %). Those criteria do not apply to
  DFT-relaxed structures, so the BVS-deviation screen here is a proxy, not a
  reproduction.
