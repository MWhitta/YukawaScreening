#!/usr/bin/env python3
"""Estimate fraction of fitted cation-O shells rejected by classic BVS tests.
Run: python3 analysis/scripts/classic_bvs_rejection_estimate.py
"""
import json, gzip, csv, math, time, sys
from collections import defaultdict
import numpy as np

t0 = time.time()
STORE = 'data/processed/bond_valence/consolidated_store.json'
FIX = '/Users/mwhittaker/Projects/github/yukawa-bond-valence/data/shells/all_element_oxide_shells.json.gz'
CSV = '/Users/mwhittaker/Projects/github/bv-methods-review/methods_comparison/comparison.csv'
THR = [0.05, 0.10, 0.20, 0.30]
SETS = {'GH2015': ('GH2015_R0', 'GH2015_B'), 'BA1985': ('BA1985_R0', 'BA1985_B'),
        'BO1991': ('BO1991_R0', 'BO1991_B'), 'softBV': ('softBV_R0', 'softBV_b')}

def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

# published
pub = {P: {} for P in SETS}
with open(CSV) as fh:
    for row in csv.DictReader(fh):
        for P, (a, b) in SETS.items():
            if row[a].strip() and row[b].strip():
                pub[P][(row['cation'], int(float(row['oxi_state'])))] = (float(row[a]), float(row[b]))

# fixture
with gzip.open(FIX, 'rt') as fh:
    f = json.load(fh)
fix = defaultdict(list)
for s in f['structures']:
    for st in s['sites']:
        fix[(s['mid'], st['element'], int(round(st['z'])))].append(np.array(st['R'], float))
nstruct = len(f['structures'])

# store
with open(STORE) as fh:
    d = json.load(fh)
payload = d['datasets']['oxygen_authoritative']['payload']
recs = []
nraw = 0
for el, byk in payload.items():
    for k, lst in byk.items():
        for r in lst:
            nraw += 1
            if r.get('status') != 'fitted' or r.get('oxi_state_label') != 'pure':
                continue
            if not all(fin(r.get(x)) for x in ('R0', 'B', 'cn', 'oxi_state')):
                continue
            fd = r.get('fit_diagnostics') or {}
            rng = fd.get('bond_length_range')
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                rng = rng[1] - rng[0]
            recs.append(dict(el=el, kind=k, mid=r['mid'], R0=float(r['R0']), B=float(r['B']),
                             cn=r['cn'], z=r['oxi_state'], rng=rng if fin(rng) else None))
del d
N = len(recs)

def bvs(R, R0, B):
    return float(np.exp((R0 - R) / B).sum())

matched = unmatched = cnmm = 0
np.seterr(over='ignore')
for r in recs:
    r['ext'] = abs(r['R0']) > 4 or abs(r['B']) > 2
for r in recs:
    sites = fix.get((r['mid'], r['el'], int(round(r['z']))), [])
    r['matched'] = bool(sites)
    if not sites:
        unmatched += 1
        continue
    matched += 1
    if any(len(R) != r['cn'] for R in sites):
        cnmm += 1
    z = r['z']
    r['own'] = float(np.mean([abs(bvs(R, r['R0'], r['B']) - z) / z for R in sites]))
    r['dev'] = {}
    for P in SETS:
        p = pub[P].get((r['el'], int(round(z))))
        if p:
            r['dev'][P] = float(np.mean([abs(bvs(R, p[0], p[1]) - z) / z for R in sites]))

M = [r for r in recs if r['matched']]
out = []
def pr(s=''):
    out.append(s); print(s)

pr('# Classic BVS rejection estimate (raw script output)')
pr(f'store raw records scanned: {nraw}; selected (fitted, pure, finite R0,B,cn,z): {N}')
pr(f'fixture structures: {nstruct}; fixture (mid,element,z) keys: {len(fix)}')
pr(f'matched {matched}/{N}; unmatched {unmatched}/{N}; cn_mismatch (any matched site len(R)!=cn) {cnmm}/{matched} matched')
own = np.array([r['own'] for r in M])
pr(f'SANITY own-fit |dev|: median {np.median(own):.4g}, mean {own.mean():.4g}, p90 {np.percentile(own,90):.4g}, p99 {np.percentile(own,99):.4g}, frac>0.05 {np.mean(own>0.05):.4f} (n={len(M)})')
pr('published species available: ' + ', '.join(f'{P}={len(v)}' for P, v in pub.items()))
pr()

def frac(rs, P, t):
    c = [r for r in rs if P in r['dev']]
    if not c: return 'n/a (0)'
    k = sum(r['dev'][P] > t for r in c)
    return f'{k/len(c):.4f} ({k}/{len(c)})'

pr('## Overall (denominator = matched records covered by P)')
for P in SETS:
    cov = sum(P in r['dev'] for r in M)
    pr(f'{P}: covered {cov}, uncovered {len(M)-cov} (of {len(M)} matched)')
    for t in THR:
        pr(f'  dev>{t}: {frac(M,P,t)}')
pr()

strata = {
 'bond_length_range <0.05 vs >=0.05': [('<0.05', lambda r: r['rng'] is not None and r['rng'] < 0.05),
                                      ('>=0.05', lambda r: r['rng'] is not None and r['rng'] >= 0.05),
                                      ('range missing', lambda r: r['rng'] is None)],
 'extreme fit (|R0|>4 or |B|>2)': [('extreme', lambda r: r['ext']), ('non-extreme', lambda r: not r['ext'])],
 'sign of B': [('B<0', lambda r: r['B'] < 0), ('B>0', lambda r: r['B'] > 0), ('B==0', lambda r: r['B'] == 0)],
 'z vs cn': [('z<cn', lambda r: r['z'] < r['cn']), ('z==cn', lambda r: r['z'] == r['cn']), ('z>cn', lambda r: r['z'] > r['cn'])],
}
pr('## Stratified (matched records; fraction covered-by-P with dev>t, with k/n)')
for sname, groups in strata.items():
    pr(f'### {sname}')
    for gname, fn in groups:
        rs = [r for r in M if fn(r)]
        cells = []
        for P in ('GH2015', 'BA1985'):
            for t in (0.10, 0.20):
                cells.append(f'{P}>{t}: {frac(rs,P,t)}')
        pr(f'  {gname} (n matched={len(rs)}): ' + ' | '.join(cells))
pr()

def q(rs, P):
    a = np.array([r['dev'][P] for r in rs if P in r['dev']])
    if len(a) == 0: return 'n/a'
    return f'n={len(a)} median {np.median(a):.4g} IQR [{np.percentile(a,25):.4g}, {np.percentile(a,75):.4g}]'
pr('## dev distribution under GH2015')
pr('extreme: ' + q([r for r in M if r['ext']], 'GH2015'))
pr('non-extreme: ' + q([r for r in M if not r['ext']], 'GH2015'))
pr('(BA1985) extreme: ' + q([r for r in M if r['ext']], 'BA1985'))
pr('(BA1985) non-extreme: ' + q([r for r in M if not r['ext']], 'BA1985'))
# overlap: share of extremes among rejects
for P in ('GH2015', 'BA1985'):
    for t in (0.10, 0.20):
        rej = [r for r in M if P in r['dev'] and r['dev'][P] > t]
        ex = [r for r in M if P in r['dev'] and r['ext']]
        pr(f'{P}>{t}: rejects {len(rej)}, of which extreme {sum(r["ext"] for r in rej)}; extreme covered {len(ex)}, of which rejected {sum(r["dev"][P]>t for r in ex)}')
pr(f'extreme records: total {sum(r["ext"] for r in recs)} of {N} selected; {sum(r["ext"] for r in M)} matched')
pr()

pr('## Per-species (all selected records; "n" includes unmatched; fractions over matched covered)')
pr('species | n | n matched | n GH2015-covered | GH2015>0.10 | BA1985>0.10')
for el, z in [('Li',1),('Na',1),('K',1),('Mg',2),('Ca',2),('Al',3),('Si',4),('Ti',4),('Mn',2),('Fe',3)]:
    rs = [r for r in recs if r['el'] == el and int(round(r['z'])) == z]
    mm = [r for r in rs if r['matched']]
    pr(f'{el}{z}+ | {len(rs)} | {len(mm)} | {sum("GH2015" in r["dev"] for r in mm)} | {frac(mm,"GH2015",0.10)} | {frac(mm,"BA1985",0.10)}')
pr(f'\nruntime {time.time()-t0:.1f} s')
