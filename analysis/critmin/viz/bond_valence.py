"""Plotting functions for screened bond valence analysis."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def _linear_fit_stats(R0: np.ndarray, B: np.ndarray) -> dict[str, Any] | None:
    """Return OLS fit statistics for ``B = β·R₀ + β₀``."""
    if len(R0) < 3:
        return None
    if np.allclose(R0, R0[0]):
        return None

    fit = linregress(R0, B)
    residuals = B - (fit.slope * R0 + fit.intercept)
    sxx = float(np.sum((R0 - R0.mean()) ** 2))
    dof = len(R0) - 2

    if sxx <= 0 or dof <= 0:
        beta_stderr = float("nan")
        beta0_stderr = float("nan")
    else:
        residual_std = float(np.sqrt(np.sum(residuals ** 2) / dof))
        beta_stderr = residual_std / np.sqrt(sxx)
        beta0_stderr = residual_std * np.sqrt((1.0 / len(R0)) + ((R0.mean() ** 2) / sxx))

    return {
        "beta": float(fit.slope),
        "beta0": float(fit.intercept),
        "r2": float(fit.rvalue ** 2),
        "p": float(fit.pvalue),
        "n": int(len(R0)),
        "beta_stderr": float(beta_stderr),
        "beta0_stderr": float(beta0_stderr),
    }


def _bootstrap_fit_uncertainty(
    R0: np.ndarray,
    B: np.ndarray,
    *,
    n_bootstrap: int = 400,
    random_state: int = 42,
) -> dict[str, Any] | None:
    """Bootstrap fit uncertainty for ``B = β·R₀ + β₀`` on a fixed inlier set."""
    if len(R0) < 3 or n_bootstrap < 2:
        return None

    rng = np.random.default_rng(random_state)
    betas: list[float] = []
    beta0s: list[float] = []

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(R0), size=len(R0))
        sample_R0 = R0[sample_idx]
        sample_B = B[sample_idx]
        fit = _linear_fit_stats(sample_R0, sample_B)
        if fit is None or not np.isfinite(fit["beta"]) or not np.isfinite(fit["beta0"]):
            continue
        betas.append(float(fit["beta"]))
        beta0s.append(float(fit["beta0"]))

    if len(betas) < 2:
        return None

    beta_samples = np.array(betas, dtype=float)
    beta0_samples = np.array(beta0s, dtype=float)
    return {
        "beta_samples": beta_samples,
        "beta0_samples": beta0_samples,
        "beta_stderr": float(np.std(beta_samples, ddof=1)),
        "beta0_stderr": float(np.std(beta0_samples, ddof=1)),
        "beta_ci95": (
            float(np.quantile(beta_samples, 0.025)),
            float(np.quantile(beta_samples, 0.975)),
        ),
        "beta0_ci95": (
            float(np.quantile(beta0_samples, 0.025)),
            float(np.quantile(beta0_samples, 0.975)),
        ),
        "uncertainty_method": "bootstrap_inliers",
        "uncertainty_n_bootstrap": int(len(beta_samples)),
    }


def fit_bv_regression(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fit B = β·R₀ + β₀ from a list of BV result dicts with 'R0' and 'B' keys."""
    if len(results) < 5:
        return None
    R0 = np.array([r["R0"] for r in results])
    B = np.array([r["B"] for r in results])
    fit = _linear_fit_stats(R0, B)
    if fit is None:
        return None
    fit["uncertainty_method"] = "ols_analytic"
    return fit


def fit_bv_regression_huber(
    results: list[dict[str, Any]],
    *,
    epsilon: float = 1.35,
) -> dict[str, Any] | None:
    """Huber-robust fit of B = β·R₀ + β₀.

    Huber regression down-weights outliers rather than discarding them
    (as RANSAC does), giving a smooth compromise between OLS and RANSAC.
    The *epsilon* parameter controls the transition between squared and
    linear loss: smaller values are more resistant to outliers, with 1.35
    (the default) corresponding to ~95% asymptotic efficiency relative
    to OLS on Gaussian data.

    Returns the same base keys as :func:`fit_bv_regression` plus
    ``n_outliers`` (points whose sample weight was reduced by the Huber
    loss), ``outlier_mask`` (bool array, True for down-weighted points),
    and ``sample_weights`` (the per-point weights from the Huber fit).

    Falls back to OLS if sklearn is unavailable or too few points.
    """
    if len(results) < 5:
        return None

    R0 = np.array([r["R0"] for r in results])
    B = np.array([r["B"] for r in results])

    if np.allclose(R0, R0[0]):
        return None

    try:
        from sklearn.linear_model import HuberRegressor
    except ImportError:
        fit = fit_bv_regression(results)
        if fit is not None:
            fit["outlier_mask"] = np.zeros(len(results), dtype=bool)
            fit["n_outliers"] = 0
            fit["sample_weights"] = np.ones(len(results), dtype=float)
        return fit

    huber = HuberRegressor(epsilon=epsilon, max_iter=1000)
    huber.fit(R0.reshape(-1, 1), B)

    beta = float(huber.coef_[0])
    beta0 = float(huber.intercept_)
    residuals = B - (beta * R0 + beta0)
    scale = float(huber.scale_)

    # Identify outliers: points whose absolute residual exceeds epsilon * scale
    outlier_mask = np.abs(residuals) > epsilon * scale
    n_outliers = int(outlier_mask.sum())

    # Compute R² on the full dataset
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((B - B.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors from the inlier subset (Huber-weighted)
    inlier_R0 = R0[~outlier_mask]
    inlier_B = B[~outlier_mask]
    n_in = int((~outlier_mask).sum())
    dof = max(n_in - 2, 1)
    inlier_resid = inlier_B - (beta * inlier_R0 + beta0)
    residual_std = float(np.sqrt(np.sum(inlier_resid ** 2) / dof)) if dof > 0 else 0.0
    sxx = float(np.sum((inlier_R0 - inlier_R0.mean()) ** 2)) if n_in > 0 else 0.0
    if sxx > 0 and n_in > 0:
        beta_stderr = residual_std / np.sqrt(sxx)
        beta0_stderr = residual_std * np.sqrt(1.0 / n_in + (inlier_R0.mean() ** 2) / sxx)
    else:
        beta_stderr = float("nan")
        beta0_stderr = float("nan")

    return {
        "beta": beta,
        "beta0": beta0,
        "r2": float(r2),
        "p": float("nan"),  # Huber doesn't produce a p-value
        "n": int(len(R0)),
        "n_inliers": n_in,
        "n_outliers": n_outliers,
        "beta_stderr": float(beta_stderr),
        "beta0_stderr": float(beta0_stderr),
        "outlier_mask": outlier_mask,
        "sample_weights": np.where(outlier_mask, epsilon * scale / np.abs(residuals), 1.0),
        "huber_scale": scale,
        "huber_epsilon": epsilon,
        "uncertainty_method": "huber_inlier_ols",
    }


def fit_bv_regression_ransac(
    results: list[dict[str, Any]],
    *,
    residual_threshold: float | None = None,
    min_samples: int = 5,
    uncertainty_bootstrap_samples: int = 400,
) -> dict[str, Any] | None:
    """RANSAC-robust fit of B = β·R₀ + β₀.

    Returns the same keys as :func:`fit_bv_regression` plus ``n_inliers``,
    ``n_outliers``, and ``inlier_mask`` (bool array, same order as *results*).

    Uncertainty is estimated from the final RANSAC inlier set. When possible, a
    bootstrap over the inliers is used; otherwise the analytic OLS standard
    errors on the inlier refit are returned.

    Falls back to OLS if sklearn is unavailable or too few points.
    """
    if len(results) < max(min_samples, 5):
        return None

    R0 = np.array([r["R0"] for r in results])
    B = np.array([r["B"] for r in results])

    try:
        from sklearn.linear_model import RANSACRegressor, LinearRegression
    except ImportError:
        # Fallback: OLS with no outlier info
        fit = fit_bv_regression(results)
        if fit is not None:
            fit["inlier_mask"] = np.ones(len(results), dtype=bool)
            fit["n_inliers"] = fit["n"]
            fit["n_outliers"] = 0
        return fit

    if residual_threshold is None:
        # Default: 2× MAD of OLS residuals
        ols_s, ols_i, _, _, _ = linregress(R0, B)
        residuals = np.abs(B - (ols_s * R0 + ols_i))
        residual_threshold = 2.0 * np.median(residuals)
        # Floor to avoid pathologically tight thresholds
        residual_threshold = max(residual_threshold, 0.01)

    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=min(min_samples, len(results)),
        residual_threshold=residual_threshold,
        random_state=42,
    )
    ransac.fit(R0.reshape(-1, 1), B)
    inlier_mask = ransac.inlier_mask_

    # Refit OLS on inliers for clean stats
    R0_in = R0[inlier_mask]
    B_in = B[inlier_mask]
    if len(R0_in) < 3:
        return None
    fit = _linear_fit_stats(R0_in, B_in)
    if fit is None:
        return None

    uncertainty = _bootstrap_fit_uncertainty(
        R0_in,
        B_in,
        n_bootstrap=uncertainty_bootstrap_samples,
    )
    if uncertainty is None:
        uncertainty = {
            "beta_ci95": (
                float(fit["beta"] - 1.96 * fit["beta_stderr"]),
                float(fit["beta"] + 1.96 * fit["beta_stderr"]),
            ),
            "beta0_ci95": (
                float(fit["beta0"] - 1.96 * fit["beta0_stderr"]),
                float(fit["beta0"] + 1.96 * fit["beta0_stderr"]),
            ),
            "uncertainty_method": "ols_inliers_analytic",
            "uncertainty_n_bootstrap": 0,
        }

    fit.update(uncertainty)
    fit.update({
        "n": len(results),
        "n_inliers": int(inlier_mask.sum()),
        "n_outliers": int((~inlier_mask).sum()),
        "inlier_mask": inlier_mask,
    })
    return fit


def bv_summary_table(
    data: dict[str, dict[str, list[dict[str, Any]]]],
    cations: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Print summary table and return fit results for each cation.

    Parameters
    ----------
    data : dict mapping cation -> {"oxides": [...], "hydroxides": [...]}
    cations : ordered list of cation symbols
    """
    print(
        f"{'Cation':<6s}  {'n_ox':>4s} {'n_OH':>4s}  "
        f"{'β ox':>7s} {'β₀ ox':>7s} {'σβ ox':>7s} {'σβ₀ ox':>7s} {'R² ox':>5s}  "
        f"{'β OH':>7s} {'β₀ OH':>7s} {'σβ OH':>7s} {'σβ₀ OH':>7s} {'R² OH':>5s}  "
        f"{'Δβ':>7s} {'Δβ₀':>7s}"
    )
    print("-" * 122)

    fits = {}
    for cat in cations:
        d = data.get(cat, {"oxides": [], "hydroxides": []})
        ox, oh = d["oxides"], d["hydroxides"]
        f_ox = fit_bv_regression(ox)
        f_oh = fit_bv_regression(oh)
        if f_ox and f_oh:
            fits[cat] = {"ox": f_ox, "oh": f_oh}
            print(
                f"{cat:<6s}  {f_ox['n']:>4d} {f_oh['n']:>4d}  "
                f"{f_ox['beta']:>7.4f} {f_ox['beta0']:>7.4f} "
                f"{f_ox['beta_stderr']:>7.4f} {f_ox['beta0_stderr']:>7.4f} {f_ox['r2']:>5.3f}  "
                f"{f_oh['beta']:>7.4f} {f_oh['beta0']:>7.4f} "
                f"{f_oh['beta_stderr']:>7.4f} {f_oh['beta0_stderr']:>7.4f} {f_oh['r2']:>5.3f}  "
                f"{f_oh['beta'] - f_ox['beta']:>7.4f} "
                f"{f_oh['beta0'] - f_ox['beta0']:>7.4f}"
            )
        else:
            print(
                f"{cat:<6s}  {len(ox):>4d} {len(oh):>4d}  "
                "(too few for regression)"
            )
    return fits


def plot_bv_scatter(
    data: dict[str, dict[str, list[dict[str, Any]]]],
    cations: list[str],
    colors: dict[str, str],
    fits: dict[str, dict[str, dict[str, Any]]],
    charge_label: str = "+",
) -> plt.Figure:
    """R₀ vs B scatter with regression lines, split oxide/hydroxide."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for cat in cations:
        d = data.get(cat, {"oxides": [], "hydroxides": []})
        c = colors[cat]

        for ax, typ, marker in [(axes[0], "oxides", "o"), (axes[1], "hydroxides", "s")]:
            vals = d[typ]
            if not vals:
                continue
            R0 = np.array([r["R0"] for r in vals])
            B = np.array([r["B"] for r in vals])
            ax.scatter(
                R0, B, alpha=0.35, s=25, c=c, edgecolors="black",
                linewidths=0.2, marker=marker,
                label=f"{cat}{charge_label} (n={len(vals)})",
            )
            f = fits.get(cat, {}).get("ox" if typ == "oxides" else "oh")
            if f:
                x_line = np.linspace(R0.min(), R0.max(), 50)
                ax.plot(x_line, f["beta"] * x_line + f["beta0"],
                        c=c, linewidth=1.5, alpha=0.8)

    for ax, title in zip(axes, ["Oxides", "Hydroxides"]):
        ax.set_xlabel("R₀ (Å)", fontsize=12)
        ax.set_ylabel("B (Å)", fontsize=12)
        ax.set_title(f"Cation–O: {title}\nB = β·R₀ + β₀", fontsize=13)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    return fig


def plot_beta_bars(
    fits: dict[str, dict[str, dict[str, Any]]],
    cation_list: list[str],
    colors: list[str] | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing β and β₀ between oxides and hydroxides."""
    valid = [c for c in cation_list if c in fits]
    x = np.arange(len(valid))
    w = 0.35

    if colors is None:
        colors = ["#666666"] * len(valid)
    else:
        colors = [colors[i] for i in range(len(valid))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ox_betas = [fits[c]["ox"]["beta"] for c in valid]
    oh_betas = [fits[c]["oh"]["beta"] for c in valid]
    ox_beta0s = [fits[c]["ox"]["beta0"] for c in valid]
    oh_beta0s = [fits[c]["oh"]["beta0"] for c in valid]

    axes[0].bar(x - w / 2, ox_betas, w, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.7, label="Oxides")
    axes[0].bar(x + w / 2, oh_betas, w, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.4, hatch="//", label="Hydroxides")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(valid, fontsize=11)
    axes[0].set_ylabel("β (slope)", fontsize=12)
    axes[0].set_title("BV coefficient β", fontsize=13)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend(fontsize=10)

    axes[1].bar(x - w / 2, ox_beta0s, w, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.7, label="Oxides")
    axes[1].bar(x + w / 2, oh_beta0s, w, color=colors, edgecolor="black",
                linewidth=0.5, alpha=0.4, hatch="//", label="Hydroxides")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(valid, fontsize=11)
    axes[1].set_ylabel("β₀ (intercept, Å)", fontsize=12)
    axes[1].set_title("BV intercept β₀", fontsize=13)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    return fig


def _get_cn_records(results: list[dict[str, Any]], target_cn: int) -> list[dict[str, Any]]:
    """Return the subset of result dicts matching *target_cn*."""
    return [r for r in results if r.get("cn") == target_cn]


def _get_cn_subset(results: list[dict[str, Any]], target_cn: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract R₀, B arrays for structures with a given coordination number."""
    records = _get_cn_records(results, target_cn)
    R0s = [r["R0"] for r in records]
    Bs = [r["B"] for r in records]
    return np.array(R0s), np.array(Bs)


def cn_summary_table(
    data: dict[str, dict[str, list[dict[str, Any]]]],
    cations: list[str],
    target_cns: list[int] = (4, 6),
    ransac: bool = True,
) -> tuple[dict[str, dict[int, dict[str, dict[str, Any]]]], list[dict[str, Any]]]:
    """Print CN-partitioned β/β₀ table and return fit results plus outliers.

    When *ransac* is True, both OLS and RANSAC fits are shown and the returned
    dict stores ``"ransac"`` sub-keys alongside OLS results for each type.

    Returns
    -------
    cn_fits : dict
        Nested fit results keyed by cation → CN → type.
    outliers : list[dict]
        Each entry has ``cation``, ``cn``, ``bucket``, ``mid``,
        ``formula_pretty``, ``R0``, ``B``, and ``residual``.
    """
    header = (
        f"{'Cation':<6s}  {'CN':>3s}  {'Type':<4s}  {'Fit':<7s}  {'n':>4s}  "
        f"{'β':>8s}  {'β₀':>8s}  {'σβ':>8s}  {'σβ₀':>8s}  {'R²':>6s}"
    )
    if ransac:
        header += f"  {'outliers':>8s}"
    print(header)
    print("-" * len(header))

    cn_fits: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    all_outliers: list[dict[str, Any]] = []

    for cat in cations:
        d = data.get(cat, {"oxides": [], "hydroxides": []})
        cn_fits[cat] = {}
        for cn in target_cns:
            cn_fits[cat][cn] = {}
            for typ, label in [("oxides", "Ox"), ("hydroxides", "OH")]:
                cn_records = _get_cn_records(d[typ], cn)
                R0 = np.array([r["R0"] for r in cn_records])
                B = np.array([r["B"] for r in cn_records])
                fit_records = [{"R0": r, "B": b} for r, b in zip(R0, B)]

                # OLS
                f_ols = fit_bv_regression(fit_records) if len(R0) >= 5 else None
                if f_ols:
                    cn_fits[cat][cn][typ] = f_ols
                    print(
                        f"{cat:<6s}  {cn:>3d}  {label:<4s}  {'OLS':<7s}  {f_ols['n']:>4d}  "
                        f"{f_ols['beta']:>8.4f}  {f_ols['beta0']:>8.4f}  "
                        f"{f_ols['beta_stderr']:>8.4f}  {f_ols['beta0_stderr']:>8.4f}  "
                        f"{f_ols['r2']:>6.3f}"
                        + ("" if not ransac else "          ")
                    )
                else:
                    print(
                        f"{cat:<6s}  {cn:>3d}  {label:<4s}  {'OLS':<7s}  {len(R0):>4d}  "
                        "(too few)"
                    )

                # RANSAC
                if ransac:
                    f_ran = fit_bv_regression_ransac(fit_records) if len(R0) >= 5 else None
                    if f_ran:
                        cn_fits[cat][cn][f"{typ}_ransac"] = f_ran
                        print(
                            f"{'':6s}  {'':>3s}  {'':4s}  {'RANSAC':<7s}  "
                            f"{f_ran['n_inliers']:>4d}  "
                            f"{f_ran['beta']:>8.4f}  {f_ran['beta0']:>8.4f}  "
                            f"{f_ran['beta_stderr']:>8.4f}  {f_ran['beta0_stderr']:>8.4f}  "
                            f"{f_ran['r2']:>6.3f}  "
                            f"{f_ran['n_outliers']:>4d}/{f_ran['n']:>4d}"
                        )

                        # Collect outlier materials
                        mask = f_ran["inlier_mask"]
                        for idx_out in np.where(~mask)[0]:
                            rec = cn_records[idx_out]
                            residual = rec["B"] - (f_ran["beta"] * rec["R0"] + f_ran["beta0"])
                            all_outliers.append({
                                "cation": cat,
                                "cn": cn,
                                "bucket": typ,
                                "mid": rec.get("mid", ""),
                                "formula_pretty": rec.get("formula_pretty", ""),
                                "R0": rec["R0"],
                                "B": rec["B"],
                                "residual": round(residual, 4),
                            })

    return cn_fits, all_outliers


CN_STYLE = {
    (4, "oxides"): {"c": "#e74c3c", "m": "o", "ls": "-", "label": "CN4 oxide"},
    (4, "hydroxides"): {"c": "#f39c12", "m": "s", "ls": "--", "label": "CN4 hydroxide"},
    (6, "oxides"): {"c": "#3498db", "m": "o", "ls": "-", "label": "CN6 oxide"},
    (6, "hydroxides"): {"c": "#2ecc71", "m": "s", "ls": "--", "label": "CN6 hydroxide"},
}

_CN_FALLBACK_COLORS = [
    "#8e44ad",
    "#16a085",
    "#d35400",
    "#7f8c8d",
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#f1c40f",
]


def _style_for_cn(cn: int, bucket: str) -> dict[str, str]:
    """Return plot styling for a CN/bucket pair, with sensible defaults."""
    if (cn, bucket) in CN_STYLE:
        return CN_STYLE[(cn, bucket)]

    palette_idx = max(0, int(cn) - 1) % len(_CN_FALLBACK_COLORS)
    color = _CN_FALLBACK_COLORS[palette_idx]
    bucket_label = "oxide" if bucket == "oxides" else "hydroxide"
    return {
        "c": color,
        "m": "o" if bucket == "oxides" else "s",
        "ls": "-" if bucket == "oxides" else "--",
        "label": f"CN{cn} {bucket_label}",
    }


def _padded_axis_range(
    values: list[float] | np.ndarray,
    *,
    pad: float = 0.3,
    min_span: float = 0.2,
) -> tuple[float, float]:
    """Return a padded axis range that remains usable for nearly constant data."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (-2.0, 5.0)

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        half_span = max(min_span / 2.0, pad)
        return (lo - half_span, hi + half_span)
    return (lo - pad, hi + pad)


def plot_cn_panels(
    data: dict[str, dict[str, list[dict[str, Any]]]],
    cn_fits: dict[str, dict[int, dict[str, dict[str, Any]]]],
    cations: list[str],
    target_cns: list[int] = (4, 6),
    title_suffix: str = "",
    hist_bins: int = 20,
) -> plt.Figure:
    """One panel per cation with marginal histograms for the underlying R₀ and B data."""
    from matplotlib.gridspec import GridSpec

    n = len(cations)
    fig = plt.figure(figsize=(5 * n, 5.5))

    # Each cation gets a 2×2 grid:
    # top-left = R0 histogram
    # bottom-left = scatter
    # bottom-right = B histogram
    outer_gs = GridSpec(1, n, figure=fig, wspace=0.35)
    _BOUNDARY_TOL = 0.1  # values within 0.1 of ±5 are boundary-capped

    for idx, cat in enumerate(cations):
        inner = outer_gs[idx].subgridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_main = fig.add_subplot(inner[1, 0])
        ax_histx = fig.add_subplot(inner[0, 0], sharex=ax_main)
        ax_histy = fig.add_subplot(inner[1, 1], sharey=ax_main)

        ax_histx.tick_params(labelbottom=False)
        ax_histy.tick_params(labelleft=False, labelright=False)

        d = data.get(cat, {"oxides": [], "hydroxides": []})

        # Collect data by CN (merge oxide+hydroxide for marginal color)
        cn_R0 = {cn: [] for cn in target_cns}
        cn_B = {cn: [] for cn in target_cns}
        panel_inlier_R0, panel_inlier_B = [], []
        panel_all_R0, panel_all_B = [], []

        for cn in target_cns:
            for typ in ["oxides", "hydroxides"]:
                R0, B = _get_cn_subset(d[typ], cn)
                if len(R0) == 0:
                    continue
                st = _style_for_cn(cn, typ)
                panel_all_R0.extend(R0.tolist())
                panel_all_B.extend(B.tolist())

                # Check for RANSAC outlier mask
                f_ran = cn_fits.get(cat, {}).get(cn, {}).get(f"{typ}_ransac")
                if f_ran and "inlier_mask" in f_ran:
                    inlier = f_ran["inlier_mask"]
                    interior = np.abs(R0[inlier]) < 5.0 - _BOUNDARY_TOL
                    panel_inlier_R0.extend(R0[inlier][interior].tolist())
                    panel_inlier_B.extend(B[inlier][interior].tolist())
                    # Inliers: normal style
                    ax_main.scatter(R0[inlier], B[inlier], alpha=0.4, s=20,
                                    c=st["c"], marker=st["m"],
                                    edgecolors="black", linewidths=0.2,
                                    label=st["label"])
                    # Outliers: red X markers
                    if (~inlier).any():
                        ax_main.scatter(R0[~inlier], B[~inlier], alpha=0.7, s=35,
                                        c=st["c"], marker="x", linewidths=1.0,
                                        label=f"outlier ({(~inlier).sum()})" if cn == target_cns[0] and typ == "oxides" else None,
                                        zorder=5)
                else:
                    interior = np.abs(R0) < 5.0 - _BOUNDARY_TOL
                    panel_inlier_R0.extend(R0[interior].tolist())
                    panel_inlier_B.extend(B[interior].tolist())
                    ax_main.scatter(R0, B, alpha=0.4, s=20, c=st["c"], marker=st["m"],
                                    edgecolors="black", linewidths=0.2,
                                    label=st["label"])

                # OLS line (thin, translucent)
                f_ols = cn_fits.get(cat, {}).get(cn, {}).get(typ)
                if f_ols:
                    x_line = np.linspace(R0.min(), R0.max(), 50)
                    ax_main.plot(x_line, f_ols["beta"] * x_line + f_ols["beta0"],
                                c=st["c"], linewidth=0.8, linestyle=st["ls"],
                                alpha=0.35)

                # RANSAC line (bold)
                if f_ran:
                    x_line = np.linspace(R0.min(), R0.max(), 50)
                    ax_main.plot(x_line, f_ran["beta"] * x_line + f_ran["beta0"],
                                c=st["c"], linewidth=2.0, linestyle=st["ls"],
                                alpha=0.9)

                cn_R0[cn].extend(R0.tolist())
                cn_B[cn].extend(B.tolist())

        if panel_inlier_R0 and panel_inlier_B:
            r0_range = _padded_axis_range(panel_inlier_R0)
            b_range = _padded_axis_range(panel_inlier_B)
        elif panel_all_R0 and panel_all_B:
            r0_range = _padded_axis_range(panel_all_R0)
            b_range = _padded_axis_range(panel_all_B)
        else:
            r0_range, b_range = (-2, 5), (-3, 3)

        # Marginal histograms colored by CN
        # Use first color for each CN (oxide color) for the marginals
        cn_colors = {cn: _style_for_cn(cn, "oxides")["c"] for cn in target_cns}

        for cn in target_cns:
            if cn_R0[cn]:
                ax_histx.hist(cn_R0[cn], bins=hist_bins, range=r0_range, alpha=0.5,
                              color=cn_colors[cn], edgecolor="none",
                              label=f"CN{cn}")
            if cn_B[cn]:
                ax_histy.hist(cn_B[cn], bins=hist_bins, range=b_range, alpha=0.5,
                              color=cn_colors[cn], edgecolor="none",
                              orientation="horizontal")

        ax_main.set_xlim(r0_range)
        ax_main.set_ylim(b_range)
        ax_main.set_xlabel("R₀ (Å)", fontsize=10)
        if idx == 0:
            ax_main.set_ylabel("B (Å)", fontsize=10)

        ax_main.text(
            0.04, 0.06, f"{cat}–O",
            transform=ax_main.transAxes,
            fontsize=12, fontweight="bold",
            va="bottom", ha="left",
        )
        if ax_main.get_legend_handles_labels()[1]:
            ax_main.legend(fontsize=5.5, loc="upper right")

        ax_histx.set_ylabel("n", fontsize=8)
        ax_histy.set_xlabel("n", fontsize=8)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=13, y=1.02)

    return fig


def plot_cn_beta_bars(
    cn_fits: dict[str, dict[int, dict[str, dict[str, Any]]]],
    cation_list: list[str],
    target_cns: list[int] = (4, 6),
) -> plt.Figure:
    """Grouped bar chart of β and β₀ partitioned by CN and mineral type.

    Error bars show the one-sided shift between the robust RANSAC fit and the
    corresponding OLS fit with outliers left in.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(cation_list))

    series = [(cn, typ) for cn in target_cns for typ in ["oxides", "hydroxides"]]
    if not series:
        return fig

    w = min(0.18, 0.8 / len(series))
    center = (len(series) - 1) / 2.0
    offsets = {
        series_idx_pair: (idx - center) * w
        for idx, series_idx_pair in enumerate(series)
    }

    for (cn, typ), offset in offsets.items():
        st = _style_for_cn(cn, typ)
        betas, beta0s, beta_errs_low, beta_errs_high = [], [], [], []
        beta0_errs_low, beta0_errs_high, mask = [], [], []
        for cat in cation_list:
            f_ols = cn_fits.get(cat, {}).get(cn, {}).get(typ)
            f_ransac = cn_fits.get(cat, {}).get(cn, {}).get(f"{typ}_ransac")
            f = f_ransac if f_ransac is not None else f_ols
            betas.append(f["beta"] if f else 0)
            beta0s.append(f["beta0"] if f else 0)
            if f is None or f_ols is None or f_ransac is None:
                beta_errs_low.append(0.0)
                beta_errs_high.append(0.0)
                beta0_errs_low.append(0.0)
                beta0_errs_high.append(0.0)
            else:
                beta_delta = float(f_ols["beta"] - f_ransac["beta"])
                beta0_delta = float(f_ols["beta0"] - f_ransac["beta0"])
                beta_errs_low.append(max(0.0, -beta_delta))
                beta_errs_high.append(max(0.0, beta_delta))
                beta0_errs_low.append(max(0.0, -beta0_delta))
                beta0_errs_high.append(max(0.0, beta0_delta))
            mask.append(f is not None)

        x_valid = x[np.array(mask)]
        b_valid = [b for b, m in zip(betas, mask) if m]
        b0_valid = [b for b, m in zip(beta0s, mask) if m]
        be_low_valid = [e for e, m in zip(beta_errs_low, mask) if m]
        be_high_valid = [e for e, m in zip(beta_errs_high, mask) if m]
        b0e_low_valid = [e for e, m in zip(beta0_errs_low, mask) if m]
        b0e_high_valid = [e for e, m in zip(beta0_errs_high, mask) if m]

        axes[0].bar(x_valid + offset, b_valid, w, color=st["c"],
                    edgecolor="black", linewidth=0.5, label=st["label"],
                    yerr=np.array([be_low_valid, be_high_valid]), capsize=2,
                    error_kw={"elinewidth": 0.8, "ecolor": "black"})
        axes[1].bar(x_valid + offset, b0_valid, w, color=st["c"],
                    edgecolor="black", linewidth=0.5, label=st["label"],
                    yerr=np.array([b0e_low_valid, b0e_high_valid]), capsize=2,
                    error_kw={"elinewidth": 0.8, "ecolor": "black"})

    for ax, ylabel, title in [
        (axes[0], "β (slope)", "BV coefficient β by CN"),
        (axes[1], "β₀ (intercept, Å)", "BV intercept β₀ by CN"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(cation_list, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=8)

    axes[0].text(
        0.98,
        0.02,
        "Error bars: OLS - RANSAC shift",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    plt.tight_layout()
    return fig
