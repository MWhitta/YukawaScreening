"""Population-weighted Shannon crystal radii for the lambda-table species.

For each (element, oxidation state) species, the coordination-number
population is counted from fitted records in the authoritative corpus
(same filters as the manuscript pipeline). The Shannon crystal radius is
interpolated linearly in CN between tabulated values (clamped at the
ends) and averaged with the CN populations as weights, mirroring how the
characteristic pair is a population-weighted intersection over shells.
High spin is preferred when both spin states are tabulated; geometry
suffixes on Shannon CN labels (e.g. IVSQ) are reduced to their numeral.
"""
import json
import re
from collections import Counter

import numpy as np

HERE = "/Users/mwhittaker/Projects/github/YukawaScreening/theory/exploratory"
STORE = "/Users/mwhittaker/Projects/github/YukawaScreening/data/processed/bond_valence/consolidated_store.json"

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
         "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}


def shannon_crystal_by_cn(entry: dict) -> dict[int, float]:
    by_cn: dict[int, float] = {}
    for cn_label, spins in entry.items():
        m = re.match(r"^(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)", cn_label)
        if not m:
            continue
        cn = ROMAN[m.group(1)]
        spin = "High Spin" if "High Spin" in spins else ("" if "" in spins else sorted(spins)[0])
        radius = float(spins[spin]["crystal_radius"])
        if radius <= 0:
            continue
        # plain CN label wins over geometry-suffixed variants
        if cn_label in ROMAN or cn not in by_cn:
            by_cn[cn] = radius
    return by_cn


shannon = json.load(open(f"{HERE}/shannon_radii_full.json"))["radii"]
store = json.load(open(STORE))["datasets"]["oxygen_authoritative"]["payload"]

cn_pops: dict[tuple[str, int], Counter] = {}
for el, buckets in store.items():
    for recs in buckets.values():
        for r in recs:
            if r.get("status") != "fitted" or r.get("oxi_state") is None or r.get("cn") is None:
                continue
            cn_pops.setdefault((el, int(r["oxi_state"])), Counter())[int(r["cn"])] += 1

weighted, skipped = {}, []
for (el, oxi), pops in sorted(cn_pops.items()):
    entry = shannon.get(el, {}).get(str(oxi))
    if not entry:
        skipped.append(f"{el}+{oxi} (no Shannon entry)")
        continue
    by_cn = shannon_crystal_by_cn(entry)
    if not by_cn:
        skipped.append(f"{el}+{oxi} (no usable CN values)")
        continue
    cns = np.array(sorted(by_cn))
    radii = np.array([by_cn[c] for c in cns])
    obs_cns = np.array(sorted(pops))
    weights = np.array([pops[c] for c in obs_cns], dtype=float)
    interp = np.interp(obs_cns, cns, radii)  # clamps outside tabulated range
    weighted[f"{el}+{oxi}"] = {
        "radius_angstrom": float(np.sum(weights * interp) / weights.sum()),
        "mean_cn": float(np.sum(weights * obs_cns) / weights.sum()),
        "cn_populations": {int(c): int(pops[c]) for c in obs_cns},
        "shannon_cns_available": [int(c) for c in cns],
        # radius at the characteristic effective coordination number CN = z
        # (at the characteristic reference shell H = ln q, so e^H = q ~ z)
        "radius_at_cn_z": float(np.interp(oxi, cns, radii)),
        "cn_z_clamped": bool(oxi < cns.min() or oxi > cns.max()),
    }

doc = {
    "description": "Fit-population-weighted Shannon crystal radii per species. "
                   "Weights are fitted-record counts per coordination number in "
                   "the authoritative corpus; radii interpolated linearly in CN "
                   "and clamped at the tabulated range.",
    "sources": ["shannon_radii_full.json", "consolidated_store.json::oxygen_authoritative"],
    "generated": "2026-07-22",
    "species": weighted,
}
json.dump(doc, open(f"{HERE}/species_weighted_radii.json", "w"), indent=1, sort_keys=True)
print(f"{len(weighted)} species with weighted radii; {len(skipped)} skipped")
for s in skipped:
    print(" ", s)
for k in ["Li+1", "Fe+3", "U+6", "K+1", "Pd+2"]:
    if k in weighted:
        w = weighted[k]
        print(f"  {k}: r={w['radius_angstrom']:.3f} mean_cn={w['mean_cn']:.2f} "
              f"pops={w['cn_populations']}")
