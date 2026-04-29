"""Manuscript figure sizing and typography presets.

These helpers are calibrated to the theory manuscript in ``docs/theory/main.tex``,
which currently uses US letter paper with 3 cm left/right margins. That yields
an effective text width of roughly 6.14 inches. The core rule is simple:
export manuscript figures at their final inclusion width, not at an oversized
notebook width that will later be shrunk in LaTeX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend

LETTER_WIDTH_IN = 8.5
THEORY_SIDE_MARGIN_CM = 3.0
CM_PER_IN = 2.54
MANUSCRIPT_TEXT_WIDTH_IN = LETTER_WIDTH_IN - 2.0 * (THEORY_SIDE_MARGIN_CM / CM_PER_IN)
MANUSCRIPT_HALF_WIDTH_IN = 0.49 * MANUSCRIPT_TEXT_WIDTH_IN
MANUSCRIPT_PANEL_GUTTER_IN = 0.18
MANUSCRIPT_ROW_GUTTER_IN = 0.20

MIN_TITLE_PT = 11.0
MIN_AXIS_LABEL_PT = 11.0
MIN_TICK_LABEL_PT = 9.0
MIN_LEGEND_PT = 9.0
MIN_COMPACT_LEGEND_PT = 7.5
MIN_ANNOTATION_PT = 9.0


@dataclass(frozen=True)
class ManuscriptFigureStyle:
    """Concrete size and font settings for one manuscript figure layout."""

    preset: str
    figure_width_in: float
    figure_height_in: float
    panel_width_in: float
    panel_height_in: float
    title_pt: float
    axis_label_pt: float
    tick_label_pt: float
    legend_pt: float
    annotation_pt: float
    line_width_pt: float = 1.15
    marker_size_pt: float = 6.5
    grid_line_width_pt: float = 0.6
    annotation_box_alpha: float = 0.72

    def rcparams(self) -> dict[str, float | str | bool]:
        """Return rcParams that align a figure with this style."""
        return {
            "axes.titlesize": self.title_pt,
            "axes.labelsize": self.axis_label_pt,
            "xtick.labelsize": self.tick_label_pt,
            "ytick.labelsize": self.tick_label_pt,
            "legend.fontsize": self.legend_pt,
            "axes.linewidth": 0.8,
            "lines.linewidth": self.line_width_pt,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }


_PRESET_SPECS: dict[str, dict[str, float | int]] = {
    "full_width_1panel": {
        "figure_width_in": MANUSCRIPT_TEXT_WIDTH_IN,
        "ncols": 1,
        "title_pt": 12.5,
        "axis_label_pt": 12.0,
        "tick_label_pt": 10.0,
        "legend_pt": 10.0,
        "annotation_pt": 9.5,
    },
    "full_width_2panel": {
        "figure_width_in": MANUSCRIPT_TEXT_WIDTH_IN,
        "ncols": 2,
        "title_pt": 12.0,
        "axis_label_pt": 11.5,
        "tick_label_pt": 9.5,
        "legend_pt": 9.5,
        "annotation_pt": 9.0,
    },
    "full_width_3panel": {
        "figure_width_in": MANUSCRIPT_TEXT_WIDTH_IN,
        "ncols": 3,
        "title_pt": 11.5,
        "axis_label_pt": 11.0,
        "tick_label_pt": 9.0,
        "legend_pt": 9.0,
        "annotation_pt": 9.0,
    },
    "half_width_1panel": {
        "figure_width_in": MANUSCRIPT_HALF_WIDTH_IN,
        "ncols": 1,
        "title_pt": 11.0,
        "axis_label_pt": 11.0,
        "tick_label_pt": 9.0,
        "legend_pt": 9.0,
        "annotation_pt": 9.0,
    },
}


def validate_manuscript_style(style: ManuscriptFigureStyle) -> None:
    """Raise when a style falls below the manuscript legibility floor."""
    violations: list[str] = []
    if style.title_pt < MIN_TITLE_PT:
        violations.append(f"title {style.title_pt:.1f} pt < {MIN_TITLE_PT:.1f} pt")
    if style.axis_label_pt < MIN_AXIS_LABEL_PT:
        violations.append(
            f"axis labels {style.axis_label_pt:.1f} pt < {MIN_AXIS_LABEL_PT:.1f} pt"
        )
    if style.tick_label_pt < MIN_TICK_LABEL_PT:
        violations.append(
            f"tick labels {style.tick_label_pt:.1f} pt < {MIN_TICK_LABEL_PT:.1f} pt"
        )
    if style.legend_pt < MIN_LEGEND_PT:
        violations.append(f"legends {style.legend_pt:.1f} pt < {MIN_LEGEND_PT:.1f} pt")
    if style.annotation_pt < MIN_ANNOTATION_PT:
        violations.append(
            f"data labels {style.annotation_pt:.1f} pt < {MIN_ANNOTATION_PT:.1f} pt"
        )
    if violations:
        joined = "; ".join(violations)
        raise ValueError(f"Manuscript figure style below legibility floor: {joined}")


def get_manuscript_figure_style(
    preset: str,
    *,
    nrows: int = 1,
    panel_height_ratio: float = 0.88,
) -> ManuscriptFigureStyle:
    """Return a validated manuscript figure preset.

    ``panel_height_ratio`` is the final panel height divided by the final panel
    width. Values near 0.8--1.0 work well for most scatter and regression plots.
    """
    if preset not in _PRESET_SPECS:
        known = ", ".join(sorted(_PRESET_SPECS))
        raise ValueError(f"Unknown manuscript preset {preset!r}. Known presets: {known}")
    if nrows < 1:
        raise ValueError("nrows must be at least 1")
    if panel_height_ratio <= 0:
        raise ValueError("panel_height_ratio must be positive")

    spec = _PRESET_SPECS[preset]
    figure_width_in = float(spec["figure_width_in"])
    ncols = int(spec["ncols"])
    panel_width_in = (
        figure_width_in - MANUSCRIPT_PANEL_GUTTER_IN * max(ncols - 1, 0)
    ) / ncols
    panel_height_in = panel_width_in * panel_height_ratio
    figure_height_in = (
        panel_height_in * nrows + MANUSCRIPT_ROW_GUTTER_IN * max(nrows - 1, 0)
    )

    style = ManuscriptFigureStyle(
        preset=preset,
        figure_width_in=figure_width_in,
        figure_height_in=figure_height_in,
        panel_width_in=panel_width_in,
        panel_height_in=panel_height_in,
        title_pt=float(spec["title_pt"]),
        axis_label_pt=float(spec["axis_label_pt"]),
        tick_label_pt=float(spec["tick_label_pt"]),
        legend_pt=float(spec["legend_pt"]),
        annotation_pt=float(spec["annotation_pt"]),
    )
    validate_manuscript_style(style)
    return style


def manuscript_subplots(
    preset: str,
    *,
    nrows: int = 1,
    ncols: int | None = None,
    panel_height_ratio: float = 0.88,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    constrained_layout: bool = True,
    **kwargs: Any,
):
    """Create subplots at final manuscript size and return ``(fig, axes, style)``."""
    style = get_manuscript_figure_style(
        preset,
        nrows=nrows,
        panel_height_ratio=panel_height_ratio,
    )
    expected_ncols = int(_PRESET_SPECS[preset]["ncols"])
    if ncols is None:
        ncols = expected_ncols
    elif ncols != expected_ncols:
        raise ValueError(
            f"Preset {preset!r} expects {expected_ncols} columns, got {ncols}"
        )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(style.figure_width_in, style.figure_height_in),
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        constrained_layout=constrained_layout,
        **kwargs,
    )
    fig.set_dpi(300)
    return fig, axes, style


def apply_manuscript_axis_style(
    ax: Axes,
    style: ManuscriptFigureStyle,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    """Apply the standard manuscript font sizes and frame treatment to one axis."""
    if title is not None:
        ax.set_title(title, fontsize=style.title_pt, pad=6)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=style.axis_label_pt)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=style.axis_label_pt)
    ax.tick_params(axis="both", labelsize=style.tick_label_pt)
    if grid:
        ax.grid(alpha=0.22, linewidth=style.grid_line_width_pt)
    for spine in ax.spines.values():
        spine.set_alpha(0.45)


def style_manuscript_legend(
    target: Axes | Legend,
    style: ManuscriptFigureStyle,
    **kwargs: Any,
) -> Legend | None:
    """Create or normalize a legend at a manuscript-safe legend font size."""
    legend_fontsize = float(kwargs.pop("fontsize", style.legend_pt))
    if legend_fontsize < MIN_COMPACT_LEGEND_PT:
        raise ValueError(
            "Legend font below legibility floor: "
            f"{legend_fontsize:.1f} pt < {MIN_COMPACT_LEGEND_PT:.1f} pt"
        )
    title_fontsize = float(kwargs.pop("title_fontsize", legend_fontsize))
    if title_fontsize < MIN_COMPACT_LEGEND_PT:
        raise ValueError(
            "Legend title font below legibility floor: "
            f"{title_fontsize:.1f} pt < {MIN_COMPACT_LEGEND_PT:.1f} pt"
        )
    if isinstance(target, Legend):
        legend = target
    else:
        kwargs.setdefault("frameon", False)
        kwargs.setdefault("fontsize", legend_fontsize)
        kwargs.setdefault("title_fontsize", title_fontsize)
        legend = target.legend(**kwargs)
    if legend is None:
        return None
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)
    legend_title = legend.get_title()
    if legend_title is not None:
        legend_title.set_fontsize(title_fontsize)
    return legend


def manuscript_annotation_kwargs(
    style: ManuscriptFigureStyle,
    *,
    dx: float = 5,
    dy: float = 4,
    with_box: bool = True,
) -> dict[str, Any]:
    """Return standard kwargs for manuscript point labels."""
    kwargs: dict[str, Any] = {
        "textcoords": "offset points",
        "xytext": (dx, dy),
        "fontsize": style.annotation_pt,
    }
    if with_box:
        kwargs["bbox"] = {
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": style.annotation_box_alpha,
        }
    return kwargs


__all__ = [
    "LETTER_WIDTH_IN",
    "MANUSCRIPT_TEXT_WIDTH_IN",
    "MANUSCRIPT_HALF_WIDTH_IN",
    "MANUSCRIPT_PANEL_GUTTER_IN",
    "MIN_TITLE_PT",
    "MIN_AXIS_LABEL_PT",
    "MIN_TICK_LABEL_PT",
    "MIN_LEGEND_PT",
    "MIN_COMPACT_LEGEND_PT",
    "MIN_ANNOTATION_PT",
    "ManuscriptFigureStyle",
    "validate_manuscript_style",
    "get_manuscript_figure_style",
    "manuscript_subplots",
    "apply_manuscript_axis_style",
    "style_manuscript_legend",
    "manuscript_annotation_kwargs",
]
