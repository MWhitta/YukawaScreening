"""Response-letter figure: Co-O CN-resolved fit lines with per-structure (R0, B) points.

Reuses the exact pipeline of make_block_fit_line_atlases.py so the lines are
identical to the Supplemental atlas page. The serialized cn_fits entries do not
carry ``inlier_mask``, so the (seeded, random_state=42) RANSAC fit is re-run on
the same input and checked against the stored beta/beta0.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "scripts"))

from sklearn.exceptions import UndefinedMetricWarning  # noqa: E402

from critmin.analysis.bond_valence_theory import (  # noqa: E402
    _valid_fit_record,
    build_unified_oxygen_cn_fits,
    collect_oxygen_fit_lines,
    merge_oxygen_records,
    minimum_spread_intersection,
    observed_coordination_numbers,
)
from critmin.viz.bond_valence import fit_bv_regression_ransac  # noqa: E402
from critmin.viz.notebook_families import _species_cn_colors  # noqa: E402
from critmin.viz.notebook_setup import cn_linestyle_map, init_notebook  # noqa: E402
from make_block_fit_line_atlases import (  # noqa: E402
    _color_for,
    build_authoritative_species_payload,
    master_species_rows,
    parse_charge_from_label,
)

OUT_DIR = ROOT / "theory" / "prm_revision" / "response_figures"
STEM = "co_o_fit_lines_with_points"
XWIN, YWIN = (0.0, 4.0), (-1.0, 2.0)
SUP = str.maketrans("0123456789+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺")


def main() -> None:
    init_notebook()
    rows = master_species_rows()
    payload, index = build_authoritative_species_payload(rows)
    labels = [l for l, *_ in index]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        cn_fits, _ = build_unified_oxygen_cn_fits(
            payload, labels, target_cns=observed_coordination_numbers(payload, labels)
        )
    co = []
    for l in labels:
        el, q = parse_charge_from_label(l)
        if el == "Co" and q is not None and 1 <= q <= 4:
            co.append((q, l))
    co.sort()
    co_labels = [l for _, l in co]
    print("Co labels:", co_labels)

    all_cns = sorted({int(cn) for l in co_labels for cn in cn_fits[l]
                      if cn_fits[l][cn].get("oxygen_ransac") or cn_fits[l][cn].get("oxygen")})
    # The Supplemental page part01 holds Co1-4+ and Cr2+, Cr3+, so its legend/style map is
    # built from the cns present on those six panels. Reproduce that so styles match.
    atlas_labels = co_labels + [l for l in labels if parse_charge_from_label(l) in (("Cr", 2), ("Cr", 3))]
    atlas_cns = sorted({int(cn) for l in atlas_labels for cn in cn_fits[l]
                        if cn_fits[l][cn].get("oxygen_ransac") or cn_fits[l][cn].get("oxygen")})
    cn_styles = cn_linestyle_map(atlas_cns)
    CN_COL = {cn: plt.get_cmap("tab10")(i) for i, cn in enumerate(atlas_cns)}

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.2), dpi=300, facecolor="white")
    fig.patch.set_facecolor("white")
    table = []
    for j, label in enumerate(co_labels):
        lines = collect_oxygen_fit_lines(cn_fits, label, all_cns)
        ordered = sorted({int(l["cn"]) for l in lines})
        colors = _species_cn_colors(label, ordered, color_fn=lambda s: _color_for("Co"))
        base = _color_for("Co")
        pts = {}
        xs, ys = [], []
        for ln in lines:
            cn = ln["cn"]
            recs = [r for r in merge_oxygen_records(payload, label) if r.get("cn") == cn]
            fit_recs = [r for r in recs if _valid_fit_record(r)]
            fit = cn_fits[label][cn].get("oxygen_ransac") or cn_fits[label][cn].get("oxygen")
            fin = [{"R0": float(r["R0"]), "B": float(r["B"])} for r in fit_recs]
            rr = fit_bv_regression_ransac(fin)
            mask = np.asarray(fit.get("inlier_mask", rr["inlier_mask"]), dtype=bool)
            ok_len = len(mask) == len(fit_recs)
            same = np.isclose(rr["beta"], fit["beta"]) and np.isclose(rr["beta0"], fit["beta0"])
            R0 = np.array([f["R0"] for f in fin]); B = np.array([f["B"] for f in fin])
            rng_ = np.array([(r.get("fit_diagnostics") or {}).get("bond_length_range", np.nan)
                             for r in fit_recs], dtype=float)
            pts[cn] = (R0, B, mask)
            inl = mask
            inwin = inl & (R0 > XWIN[0]) & (R0 < XWIN[1]) & (B > YWIN[0]) & (B < YWIN[1])
            out_in = inl & ~inwin
            table.append(dict(
                label=label, cn=cn, N=len(fit_recs), n_in=int(inl.sum()), n_out=int((~inl).sum()),
                beta=fit["beta"], beta0=fit["beta0"], r0min=R0[inl].min(), r0max=R0[inl].max(),
                n_inwin=int(inwin.sum()), frac_out=float(out_in.sum() / max(inl.sum(), 1)),
                med_rng_in=float(np.nanmedian(rng_[inwin])) if inwin.any() else float("nan"),
                med_rng_out=float(np.nanmedian(rng_[out_in])) if out_in.any() else float("nan"),
                mask_len_ok=ok_len, refit_matches=bool(same), mask_stored="inlier_mask" in fit,
            ))
            xs.extend(R0[inl]); ys.extend(B[inl])
        inter = minimum_spread_intersection(lines)
        # extents of the atlas panel: line ends + intersection (as in the SI page)
        lx, ly = [], []
        for ln in lines:
            lo, hi = float(ln["R0_min"]), float(ln["R0_max"])
            if np.isclose(lo, hi):
                lo -= 0.05; hi += 0.05
            xx = np.linspace(lo, hi, 120)
            lx += [xx.min(), xx.max()]
            yy = ln["beta"] * xx + ln["beta0"]
            ly += [yy.min(), yy.max()]
        if inter is not None and np.isfinite(inter["R0_star"]):
            lx.append(inter["R0_star"]); ly.append(inter["B_star"])
        ax_lims = []
        for vals_x, vals_y in [(lx + xs, ly + ys)]:
            px = max(0.03, 0.08 * (max(vals_x) - min(vals_x) or 1.0))
            py = max(0.03, 0.08 * (max(vals_y) - min(vals_y) or 1.0))
            ax_lims = (min(vals_x) - px, max(vals_x) + px, min(vals_y) - py, max(vals_y) + py)

        for row in (0, 1):
            ax = axes[row, j]
            for ln in lines:
                cn = ln["cn"]
                R0, B, mask = pts[cn]
                c = CN_COL[cn]
                # All per-structure points are transparent filled circles; inliers take the
                # coordination color, RANSAC outliers are gray.
                ax.scatter(R0[mask], B[mask], s=7, color=c, alpha=0.55, linewidths=0, zorder=1)
                ax.scatter(R0[~mask], B[~mask], s=7, color="0.45", alpha=0.55, linewidths=0,
                           zorder=1)
                lo, hi = float(ln["R0_min"]), float(ln["R0_max"])
                if np.isclose(lo, hi):
                    lo -= 0.05; hi += 0.05
                xx = np.linspace(lo, hi, 120)
                # Lines drawn faint and beneath the points so the raw data stay resolved.
                ax.plot(xx, ln["beta"] * xx + ln["beta0"], color=c, ls=cn_styles[cn],
                        lw=0.6, alpha=0.3, zorder=0.5)
            if inter is not None and np.isfinite(inter["R0_star"]):
                ax.scatter(inter["R0_star"], inter["B_star"], marker="o", s=30,
                           facecolors="white", edgecolors=base, linewidths=1.0, zorder=4)
            if row == 0:
                ax.set_xlim(ax_lims[0], ax_lims[1]); ax.set_ylim(ax_lims[2], ax_lims[3])
                ax.set_title(f"Co{str(label).translate(SUP) if False else ''}".strip(), fontsize=7)
            else:
                ax.set_xlim(*XWIN); ax.set_ylim(*YWIN)
                k = n_in_tot = n_out = 0
                for cn, (R0, B, mask) in pts.items():
                    n_in_tot += int(mask.sum()); n_out += int((~mask).sum())
                    k += int((mask & ~((R0 > XWIN[0]) & (R0 < XWIN[1]) & (B > YWIN[0]) & (B < YWIN[1]))).sum())
                # Annotation sits above the panel so it never overlaps the data.
                ax.set_title(f"{k} of {n_in_tot} inlier structures outside window\n"
                             f"({n_out} RANSAC outliers)", fontsize=4.8, loc="left", pad=3)
                ax.set_xlabel("R₀ (Å)", fontsize=7)
            if j == 0:
                ax.set_ylabel("B (Å)", fontsize=7)
            ax.tick_params(labelsize=5.5, length=2)
            for s in ax.spines.values():
                s.set_linewidth(0.6)
        axes[0, j].set_title(f"Co{str(parse_charge_from_label(label)[1]).translate(SUP)}⁺–O"
                             .replace("⁺⁺", "⁺"), fontsize=8)
    # titles: use the label itself if it already carries superscripts
    for j, label in enumerate(co_labels):
        axes[0, j].set_title(f"{label}–O", fontsize=8)

    # shared legend
    cn_present = all_cns  # only cns shown in these four panels
    handles = [Line2D([], [], color=CN_COL[cn], ls=cn_styles[cn], lw=1.0, alpha=0.6, label=f"n = {cn}")
               for cn in cn_present]
    handles += [
        Line2D([], [], marker="o", ls="", color="tab:brown", ms=3.5, alpha=0.5, mew=0,
               label="inlier (coordination color)"),
        Line2D([], [], marker="o", ls="", color="0.45", ms=3.5, alpha=0.5, mew=0,
               label="RANSAC outlier (gray)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="0.25", ms=5, label="least-spread point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=5.5,
               frameon=False, handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6, h_pad=1.2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{STEM}.{ext}", dpi=300, facecolor="white")
    print("saved", OUT_DIR / f"{STEM}.png")

    print("label cn N n_in n_out beta beta0 R0min R0max n_inwin frac_out medrng_in medrng_out maskOK refitOK maskstored")
    for t in table:
        print(f"{t['label']} {t['cn']} {t['N']} {t['n_in']} {t['n_out']} {t['beta']:.4g} {t['beta0']:.4g} "
              f"{t['r0min']:.4g} {t['r0max']:.4g} {t['n_inwin']} {t['frac_out']:.3f} "
              f"{t['med_rng_in']:.4g} {t['med_rng_out']:.4g} {t['mask_len_ok']} {t['refit_matches']} {t['mask_stored']}")


if __name__ == "__main__":
    main()
