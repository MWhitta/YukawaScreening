#!/usr/bin/env python3
"""Render manuscript SI lambda tables from the authoritative master summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

from critmin.analysis.config import ELEMENT_Z, get_block

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_SUMMARY_PATH = REPO_ROOT / "data/processed/theory/master_oxygen_summary_theory.json"

OUTPUT_SPECS = {
    "prm": {
        "path": REPO_ROOT / "theory/prm_revision/si_lambda_table.tex",
        "caption": (
            "\\caption{Characteristic pairs $(R_0^*,B^*)$ for all 103 cation--oxygen\n"
            "species and Yukawa screening lengths $\\lambda^*$, ordered by atomic\n"
            "number.  All values are in \\AA.  $\\lambda^*$ is computed from the\n"
            "branch-continuous inversion\n"
            "$\\lambda^*=(-R_0^*+\\sqrt{R_0^{*2}+4B^*R_0^*})/2$ in the main text,\n"
            "which is real for $B^*\\ge -R_0^*/4$.  For the 101 screening-branch\n"
            "species ($B^*>0$) it is positive and physical.  For the two\n"
            "anti-screening species (Mo$^{3+}$ and Sb$^{3+}$, $B^*<0$) the same\n"
            "root is negative and is tabulated as a signed Yukawa-branch\n"
            "diagnostic rather than a physical screening length, because the\n"
            "first-order expansion does not adequately describe their bonding.\n"
            "The second root of the same quadratic,\n"
            "$\\lambda^*_-=(-R_0^*-\\sqrt{R_0^{*2}+4B^*R_0^*})/2$, is listed for\n"
            "every species in the $\\lambda^*_-$ column.  On the anti-screening\n"
            "branch it also lies in the admissible interval $(-R_0^*,0)$.  On\n"
            "the screening branch it lies below $-R_0^*$, outside the\n"
            "admissible interval, and is reported for completeness.\n"
            "$n_{\\mathrm{CN}}$ is the number of distinct coordination shells used\n"
            "to determine $(R_0^*,B^*)$.\n"
            "Parenthetical uncertainties on $B^*$ are the weighted standard deviation\n"
            "$\\sigma_B$ of the $n$-resolved lines at $R_0^*$.  Uncertainties on\n"
            "$\\lambda^*$ are propagated via\n"
            "$\\sigma_\\lambda=|\\partial\\lambda^*/\\partial B^*|\\,\\sigma_B$.\n"
            "Entries with $n_{\\mathrm{CN}}=2$ have $\\sigma_B=0$ by construction\n"
            "and should be regarded as provisional.}"
        ),
        "b_label": "$B^*$",
    },
}


def _effective_oxi_state(row: dict[str, object]) -> int:
    oxi_state = row.get("oxi_state")
    if oxi_state is not None:
        return int(oxi_state)
    family_key = str(row.get("family_key") or "")
    if family_key == "group1":
        return 1
    if family_key == "group2":
        return 2
    if family_key == "lanthanides":
        return 3
    raise ValueError(f"Missing oxidation state for row: {row}")


def _species_label(row: dict[str, object]) -> str:
    element = str(row["element"])
    oxi_state = _effective_oxi_state(row)
    sign = "+" if oxi_state >= 0 else "-"
    magnitude = abs(oxi_state)
    return f"{element}$^{{{magnitude}{sign}}}$"


def _row_sort_key(row: dict[str, object]) -> tuple[int, int, str]:
    return (ELEMENT_Z.get(str(row["element"]), 999), _effective_oxi_state(row), str(row["element"]))


def _lambda_from_b(r0_star: float, b_star: float) -> float | None:
    # Branch-continuous "+" root of B = lambda(lambda + R*)/R*. Positive on
    # the screening branch (B > 0), negative on the anti-screening branch
    # (-R*/4 <= B < 0), and undefined (complex) for B < -R*/4.
    if r0_star <= 0.0:
        return None
    discriminant = r0_star * r0_star + 4.0 * b_star * r0_star
    if discriminant < 0.0:
        return None
    return (math.sqrt(discriminant) - r0_star) / 2.0


def _lambda_minus_from_b(r0_star: float, b_star: float) -> float | None:
    # Second root of the same quadratic, reported for every species.
    # Inside the admissible anti-screening interval (-R*, 0) when B < 0;
    # below -R* (outside the admissible interval) when B > 0.
    if r0_star <= 0.0:
        return None
    discriminant = r0_star * r0_star + 4.0 * b_star * r0_star
    if discriminant < 0.0:
        return None
    return (-math.sqrt(discriminant) - r0_star) / 2.0


def _lambda_sigma(r0_star: float, b_star: float, sigma_b: float) -> float | None:
    # |d lambda / d B| = R*/sqrt(discriminant) for either root.
    lambda_star = _lambda_from_b(r0_star, b_star)
    if lambda_star is None:
        return None
    denom = math.sqrt(r0_star * r0_star + 4.0 * b_star * r0_star)
    if denom <= 0.0:
        return None
    return abs(r0_star / denom) * sigma_b


def _format_uncertainty(value: float, sigma: float) -> str:
    sigma_digits = int(round(abs(sigma) * 1000.0))
    if value < 0.0:
        return f"$-${abs(value):.3f}({sigma_digits})"
    return f"{value:.3f}({sigma_digits})"


def _render_table(rows: list[dict[str, object]], *, caption: str, b_label: str) -> str:
    header = (
        f"  Species & Block & $R_0^*$ & {b_label} & $\\lambda^*$ & "
        f"$\\lambda^*_-$ & $n_{{\\mathrm{{CN}}}}$ \\\\"
    )
    lines = [
        "\\begin{longtable}{l l r r r r r}",
        caption,
        "\\label{tab:lambda_star} \\\\",
        "\\hline\\hline",
        header,
        "\\hline",
        "\\endfirsthead",
        "\\multicolumn{7}{c}{\\tablename\\ \\thetable{} -- continued from previous page} \\\\",
        "\\hline",
        header,
        "\\hline",
        "\\endhead",
        "\\hline",
        "\\multicolumn{7}{r}{Continued on next page} \\\\",
        "\\endfoot",
        "\\hline\\hline",
        "\\endlastfoot",
    ]
    for row in rows:
        r0_star = float(row["R0_star"])
        b_star = float(row["B_star"])
        sigma_b = float(row["sigma_B_at_R0_star"])
        lambda_star = _lambda_from_b(r0_star, b_star)
        lambda_minus = _lambda_minus_from_b(r0_star, b_star)
        lambda_sigma = _lambda_sigma(r0_star, b_star, sigma_b)
        lambda_text = (
            "---"
            if lambda_star is None or lambda_sigma is None
            else _format_uncertainty(lambda_star, lambda_sigma)
        )
        lambda_minus_text = (
            "---"
            if lambda_minus is None or lambda_sigma is None
            else _format_uncertainty(lambda_minus, lambda_sigma)
        )
        lines.append(
            f"  {_species_label(row):<20} & ${'s' if str(row['element']) == 'H' else get_block(str(row['element']))}$   & "
            f"{r0_star:>5.3f} & {_format_uncertainty(b_star, sigma_b):>14} & "
            f"{lambda_text:>14} & {lambda_minus_text:>14} & {int(row['n_lines'])} \\\\"
        )
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> None:
    payload = json.loads(MASTER_SUMMARY_PATH.read_text(encoding="utf-8"))
    rows = sorted(payload["master_rows"], key=_row_sort_key)
    for spec in OUTPUT_SPECS.values():
        content = _render_table(rows, caption=spec["caption"], b_label=spec["b_label"])
        output_path = Path(spec["path"])
        output_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
