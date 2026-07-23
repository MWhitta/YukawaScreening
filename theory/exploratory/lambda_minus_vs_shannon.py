"""Exploratory figures: |lambda*_-| = R0* + lambda* vs Shannon crystal radius.

Two radius conventions, one figure each:
  1. lambda_minus_vs_weighted_radius.png — population-weighted radius (Shannon
     crystal radius interpolated in CN, averaged over each species'
     fitted-record CN populations; see build_weighted_radii.py).
  2. lambda_minus_vs_radius_at_z.png — radius evaluated at the characteristic
     effective coordination number CN = z (at the characteristic reference
     shell H = ln q, so e^H = q ~ z). Species whose charge falls outside
     Shannon's tabulated CN range are clamped and drawn as open markers.
Screening-branch species only.
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = "/Users/mwhittaker/Projects/github/YukawaScreening/theory/exploratory"
TABLE = "/Users/mwhittaker/Projects/github/YukawaScreening/theory/prm_revision/si_lambda_table.tex"

weighted = json.load(open(f"{HERE}/species_weighted_radii.json"))["species"]

pat = re.compile(
    r'^\s*([A-Z][a-z]?)\$\^\{(\d)([+-])\}\$\s*&.*?&\s*([\d.]+)\s*&\s*'
    r'(\$-\$)?[\d.]+\(\d+\)\s*&\s*(\$-\$)?([\d.]+)\((\d+)\)'
)
rows = []
unmatched = []
for line in open(TABLE):
    m = pat.match(line)
    if not m:
        continue
    el, oxi, sign, r0, _bneg, lneg, lam, sig = m.groups()
    if sign != "+" or lneg:  # cations on the screening branch only
        continue
    key = f"{el}+{oxi}"
    if key not in weighted:
        unmatched.append(key)
        continue
    rows.append({
        "label": f"{el}{oxi}+",
        "z": int(oxi),
        "y": float(r0) + float(lam),
        "yerr": int(sig) / 1000.0,
        "entry": weighted[key],
    })
print(f"{len(rows)} species matched; unmatched:",
      ", ".join(unmatched) if unmatched else "none")


def make_figure(radius_key, xlabel, outname, clamp_key=None, fix_slope=False):
    x = np.array([r["entry"][radius_key] for r in rows])
    y = np.array([r["y"] for r in rows])
    yerr = np.array([r["yerr"] for r in rows])
    charge = np.array([r["z"] for r in rows])
    labels = [r["label"] for r in rows]
    clamped = (np.array([r["entry"].get(clamp_key, False) for r in rows])
               if clamp_key else np.zeros(len(rows), dtype=bool))

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    cmap = plt.get_cmap("viridis")
    zmin, zmax = int(charge.min()), int(charge.max())
    resid = np.full_like(y, np.nan)
    intercepts = []
    mode = "slope fixed at 1" if fix_slope else "free slope"
    print(f"\n{outname}: per-charge fits ({mode})")
    for z in range(zmin, zmax + 1):
        sel = charge == z
        n = int(sel.sum())
        if n == 0:
            continue
        color = cmap((z - zmin) / max(zmax - zmin, 1) * 0.92)
        label = fr"$z={z}$ ($n={n}$)"
        if fix_slope and n >= 2:
            c = float(np.mean(y[sel] - x[sel]))
            sem = float(np.std(y[sel] - x[sel], ddof=1) / np.sqrt(n))
            resid[sel] = y[sel] - (x[sel] + c)
            intercepts.append((z, c, sem, n))
            xz = np.linspace(x[sel].min() - 0.06, x[sel].max() + 0.06, 10)
            ax.plot(xz, xz + c, "-", color=color, linewidth=1.2,
                    alpha=0.85, zorder=2)
            label = fr"$z={z}$  $r+{c:.2f}$ ($n={n}$)"
            print(f"  z={z}: c={c:.3f} +/- {sem:.3f} n={n} "
                  f"rms={np.sqrt(np.mean(resid[sel]**2)):.3f} "
                  f"clamped={int(clamped[sel].sum())}")
        elif not fix_slope and n >= 3:
            A = np.vstack([x[sel], np.ones(n)]).T
            (slope, intercept), *_ = np.linalg.lstsq(A, y[sel], rcond=None)
            rr = np.corrcoef(x[sel], y[sel])[0, 1]
            resid[sel] = y[sel] - (slope * x[sel] + intercept)
            xz = np.linspace(x[sel].min() - 0.06, x[sel].max() + 0.06, 10)
            ax.plot(xz, slope * xz + intercept, "-", color=color,
                    linewidth=1.2, alpha=0.85, zorder=2)
            label = (fr"$z={z}$  ${slope:.2f}\,r+{intercept:.2f}$ "
                     fr"($r={rr:.2f}$, $n={n}$)")
            print(f"  z={z}: slope={slope:.3f} intercept={intercept:.3f} "
                  f"r={rr:.3f} n={n} "
                  f"rms={np.sqrt(np.mean(resid[sel]**2)):.3f} "
                  f"clamped={int(clamped[sel].sum())}")
        else:
            print(f"  z={z}: n={n}, too few points to fit")
        for is_clamped, marker_face in ((False, None), (True, "none")):
            sub = sel & (clamped == is_clamped)
            if not sub.any():
                continue
            ax.errorbar(x[sub], y[sub], yerr=yerr[sub], fmt="o",
                        markersize=4.8, color=color,
                        markerfacecolor=marker_face or color,
                        markeredgecolor=color if is_clamped else "white",
                        markeredgewidth=0.8 if is_clamped else 0.5,
                        ecolor=color, elinewidth=0.8, capsize=1.6,
                        linestyle="", label=label, zorder=3)
            label = None  # first drawn sub-group carries the legend entry

    order = np.argsort(-np.abs(np.nan_to_num(resid)))
    for i in order[:8]:
        ax.annotate(labels[i], (x[i], y[i]), textcoords="offset points",
                    xytext=(5, 4), fontsize=7, color="0.2")

    if fix_slope and len(intercepts) >= 3:
        zs = np.array([t[0] for t in intercepts], dtype=float)
        cs = np.array([t[1] for t in intercepts])
        sems = np.array([t[2] for t in intercepts])
        (b, a), *_ = np.linalg.lstsq(np.vstack([zs, np.ones_like(zs)]).T,
                                     cs, rcond=None)
        rr = np.corrcoef(zs, cs)[0, 1]
        print(f"  intercept law: c(z) = {a:.3f} + {b:.3f} z  (r={rr:.3f})")
        axin = ax.inset_axes([0.60, 0.07, 0.36, 0.30])
        axin.errorbar(zs, cs, yerr=sems, fmt="o", markersize=4,
                      color="0.25", capsize=2, linewidth=1)
        zline = np.linspace(zs.min() - 0.3, zs.max() + 0.3, 5)
        axin.plot(zline, a + b * zline, "-", color="0.45", linewidth=1)
        axin.set_xlabel(r"$z$", fontsize=7, labelpad=1)
        axin.set_ylabel(r"$c(z)$ (Å)", fontsize=7, labelpad=1)
        axin.tick_params(labelsize=6.5)
        axin.set_title(fr"$c(z)={a:.2f}+{b:.3f}\,z$ ($r={rr:.2f}$)",
                       fontsize=6.5, pad=2)

    if clamp_key and clamped.any():
        ax.plot([], [], "o", markersize=4.8, markerfacecolor="none",
                color="0.4", label=fr"open: CN$=z$ outside Shannon range ($n={int(clamped.sum())}$)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$|\lambda^*_-| = R_0^* + \lambda^*$" + " (Å)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6.8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/{outname}")
    plt.close(fig)
    print(f"wrote {HERE}/{outname}")


make_figure("radius_angstrom",
            "Population-weighted Shannon crystal radius (Å)",
            "lambda_minus_vs_weighted_radius.png",
            fix_slope=True)
make_figure("radius_at_cn_z",
            "Shannon crystal radius at CN = z (Å)",
            "lambda_minus_vs_radius_at_z.png",
            clamp_key="cn_z_clamped")
