"""Numerical check of the expansion-point argument (round 2 letter, Block 8).

For model coordination shells with the exact Yukawa weight
w(R) = exp(-R/lambda) (1 + R/lambda), compare the exact shell normalization
with the first-order (Gibbs) one for a range of expansion points R'.
Checks: (i) the normalization error is second order for every R';
(ii) its magnitude is minimized at R' = m_i (the Gibbs mean);
(iii) the closed form -sum_j p_j (R_j - R')^2 / [2 (lambda + R')^2]
reproduces it; (iv) R' = m_i(B(R')) has a fixed point inside [Rmin, Rmax].

Run:  analysis/.venv/bin/python analysis/scripts/check_expansion_point.py
"""
from __future__ import annotations

import numpy as np

SHELLS = {
    "distorted octahedron": np.array([1.95, 1.95, 2.05, 2.05, 2.10, 2.20]),
    "asymmetric (one long bond)": np.array([1.90, 1.95, 1.97, 2.00, 2.02, 2.35]),
    "near-degenerate": np.array([2.00, 2.00, 2.00, 2.01, 2.01, 2.02]),
}
LAM = 0.9  # screening length, A


def yukawa(R: np.ndarray) -> np.ndarray:
    return np.exp(-R / LAM) * (1 + R / LAM)


def main() -> None:
    for name, R in SHELLS.items():
        W = yukawa(R)
        p_exact = W / W.sum()
        print(f"\n{name}: R = {R}, spread = {R.max() - R.min():.2f} A")
        print("   R'      m_i(B(R'))   lnZ error    closed form   rms per-bond ln p error")
        rows = []
        for Rp in np.linspace(R.min(), R.max(), 9):
            B = LAM * (LAM + Rp) / Rp                      # Eq. (4)
            pt = np.exp(-R / B); pt /= pt.sum()           # Gibbs weights at this B
            m = float((pt * R).sum())
            Z = W.sum(); Zt = (yukawa(Rp) * np.exp(-(R - Rp) / B)).sum()
            err = float(np.log(Z / Zt))
            pred = float(-(pt * (R - Rp) ** 2).sum() / (2 * (LAM + Rp) ** 2))
            perbond = float(np.sqrt((pt * (np.log(p_exact) - np.log(pt)) ** 2).sum()))
            rows.append((Rp, m, err, pred, perbond))
            print(f"  {Rp:.3f}    {m:.4f}      {err:+.2e}    {pred:+.2e}    {perbond:.2e}")
        best = min(rows, key=lambda r: abs(r[2]))
        Rp = float(R.mean())
        for _ in range(100):
            B = LAM * (LAM + Rp) / Rp; pt = np.exp(-R / B); pt /= pt.sum(); Rp = float((pt * R).sum())
        print(f"  |lnZ error| minimal at R' = {best[0]:.3f} (m_i = {best[1]:.4f}); "
              f"fixed point R' = m_i: {Rp:.4f}; exact Yukawa-weighted mean {(p_exact * R).sum():.4f}")


if __name__ == "__main__":
    main()
