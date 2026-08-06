"""Skewness test at the z = n pole: do the second-order spread terms fix B?

The screening-collapse argument says that at z = n the leading-order
relation no longer constrains the fitted softness, which is then fixed
by the second-order terms delta_ij of the log-weight expansion.
Carrying the expansion of the exact Yukawa weight through the
per-structure least-squares fit gives, for a nearly uniform shell,

    B_fit ~= B0 [ 1 + (b/a) * m3/m2 ],   a = 1/B0,
    b = -1 / [2 (lambda + Rbar)^2],

so within a z = n cell the fitted B should depend linearly on the
bond-length skewness ratio gamma = m3/m2 (units of Angstrom), with the
negative slope

    dB/dgamma = -B0^2 / [2 (lambda + Rbar)^2],

set by the screening length through B0 = lambda(lambda + Rbar)/Rbar.

Procedure: clean (linear_ls, non-degenerate) z = n records are grouped
into (element, z, n) cells; per-shell bond lists are rebuilt from the
Materials Project bonds endpoint (the same structure_graph source the
fitting pipeline consumed), validated record-by-record against the
stored bond_length_mean; gamma is computed from central moments of the
bond lengths; and B is regressed on gamma within each cell, with a
bootstrap confidence interval on the slope, for comparison with the
predicted slope.

Requires MP_API_KEY in the environment. Fetched documents are cached in
the JSON file named by BV_SKEW_CACHE (default: bv_skew_cache.json in
the working directory).

Usage: python analysis/scripts/bv_skewness_test.py
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

STORE_PATH = Path("data/processed/bond_valence/consolidated_store.json")
CACHE_PATH = Path(os.environ.get("BV_SKEW_CACHE", "bv_skew_cache.json"))
MIN_CELL = 30
MAX_PER_CELL = 150
BATCH = 40
MEAN_TOL = 2e-3  # Angstrom, tolerance for matching the stored bond_length_mean
BOOT = 2000


def clean_pole_records() -> dict[tuple[str, int, int], list[dict]]:
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    payload = store["datasets"]["oxygen_authoritative"]["payload"]
    cells: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for element, groups in payload.items():
        for records in groups.values():
            for rec in records:
                if rec.get("status") != "fitted" or rec.get("oxi_state_label") == "mixed":
                    continue
                b, cn, z = rec.get("B"), rec.get("cn"), rec.get("oxi_state")
                if None in (b, cn, z) or int(z) != int(cn):
                    continue
                diag = rec.get("fit_diagnostics") or {}
                strategy = str(diag.get("fit_strategy") or rec.get("fit_strategy") or "")
                if strategy != "linear_ls" or diag.get("degenerate_fit_applied"):
                    continue
                cells[(element, int(z), int(cn))].append({
                    "mid": str(rec["mid"]),
                    "B": float(b),
                    "mean": diag.get("bond_length_mean"),
                })
    return {
        key: sorted(recs, key=lambda r: r["mid"])[:MAX_PER_CELL]
        for key, recs in cells.items()
        if len(recs) >= MIN_CELL
    }


def _fetch_one(mid: str, key: str) -> tuple[str, dict | None]:
    # One mid per request: the API masks material_id in responses and does
    # not preserve request order, so multi-id batches cannot be mapped back
    # to the requested ids reliably (formulas collide across polymorphs).
    url = ("https://api.materialsproject.org/materials/bonds/"
           f"?material_ids={mid}&_fields=structure_graph&_limit=1")
    try:
        out = subprocess.run(
            ["curl", "-s", "--max-time", "60", "-H", f"X-API-KEY: {key}", url],
            capture_output=True, text=True, check=True,
        ).stdout
        data = json.loads(out).get("data", [])
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return mid, None
    if len(data) != 1:
        return mid, None
    return mid, data[0]["structure_graph"]


def fetch_bonds_docs(mids: list[str]) -> dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor

    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    missing = [m for m in mids if m not in cache]
    key = os.environ["MP_API_KEY"]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            done = 0
            for mid, sg in pool.map(lambda m: _fetch_one(m, key), missing):
                if sg is not None:
                    cache[mid] = sg
                done += 1
                if done % 200 == 0:
                    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
                    print(f"  fetched {done}/{len(missing)}", flush=True)
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def shell_bond_lengths(sg: dict, element: str, cn: int) -> list[float] | None:
    """All cation(element)-O bond lengths for sites of coordination cn."""
    sites = sg["structure"]["sites"]
    adjacency = sg["graphs"]["adjacency"]
    elem = [s["species"][0]["element"] for s in sites]
    per_site: dict[int, list[float]] = defaultdict(list)
    for i, edges in enumerate(adjacency):
        for edge in edges:
            j = int(edge["id"])
            w = float(edge["weight"])
            if elem[i] == element and elem[j] == "O":
                per_site[i].append(w)
            elif elem[i] == "O" and elem[j] == element:
                per_site[j].append(w)
    pooled: list[float] = []
    for i, lengths in per_site.items():
        if len(lengths) == cn:
            pooled.extend(lengths)
    return pooled or None


def moments(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs) / n
    m3 = sum((x - mean) ** 3 for x in xs) / n
    return mean, m2, m3


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0
    return slope, r


def main() -> None:
    cells = clean_pole_records()
    all_mids = sorted({r["mid"] for recs in cells.values() for r in recs})
    print(f"cells: {len(cells)}, structures to fetch: {len(all_mids)}")
    docs = fetch_bonds_docs(all_mids)
    print(f"bonds documents available: {sum(m in docs for m in all_mids)}/{len(all_mids)}")

    rng = random.Random(20260728)
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    print("\nper-cell regression of B on gamma = m3/m2 (clean z=n fits):")
    print(f"{'cell':>10s} {'N':>4s} {'matched':>7s} {'slope':>8s} {'95% CI':>18s} "
          f"{'pred':>8s} {'r':>6s}")
    for (element, z, cn), recs in sorted(cells.items(), key=lambda kv: -len(kv[1])):
        xs, ys, means = [], [], []
        n_mean_match = 0
        for rec in recs:
            sg = docs.get(rec["mid"])
            if sg is None:
                continue
            lengths = shell_bond_lengths(sg, element, cn)
            if not lengths or len(lengths) < 2:
                continue
            mean, m2, m3 = moments(lengths)
            stored = rec.get("mean")
            if stored is not None and abs(mean - float(stored)) > MEAN_TOL:
                continue
            n_mean_match += 1
            if m2 <= 0:
                continue
            xs.append(m3 / m2)
            ys.append(rec["B"])
            means.append(mean)
        if len(xs) < 15:
            print(f"{element}{z}+@{cn}: matched {n_mean_match}, too few for regression")
            continue
        slope, r = ols(xs, ys)
        boots = []
        idx = list(range(len(xs)))
        for _ in range(BOOT):
            sample = [rng.choice(idx) for _ in idx]
            s, _ = ols([xs[i] for i in sample], [ys[i] for i in sample])
            boots.append(s)
        boots.sort()
        lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)]
        b0 = sorted(ys)[len(ys) // 2]
        rbar = sorted(means)[len(means) // 2]
        lam = (-rbar + math.sqrt(rbar * rbar + 4 * b0 * rbar)) / 2 if b0 > 0 else float("nan")
        pred = -b0 * b0 / (2 * (lam + rbar) ** 2) if b0 > 0 else float("nan")
        label = f"{element}{z}+@{cn}"
        print(f"{label:>10s} {len(xs):4d} {n_mean_match:7d} {slope:8.3f} "
              f"[{lo:7.3f},{hi:7.3f}] {pred:8.3f} {r:6.2f}")
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        pooled_x.extend(x - mx for x in xs)
        pooled_y.extend(y - my for y in ys)

    slope, r = ols(pooled_x, pooled_y)
    boots = []
    idx = list(range(len(pooled_x)))
    for _ in range(BOOT):
        sample = [rng.choice(idx) for _ in idx]
        s, _ = ols([pooled_x[i] for i in sample], [pooled_y[i] for i in sample])
        boots.append(s)
    boots.sort()
    lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)]
    print(f"\npooled (within-cell centered): N = {len(pooled_x)}, "
          f"slope = {slope:.3f} [{lo:.3f}, {hi:.3f}], r = {r:.2f}")


if __name__ == "__main__":
    main()
