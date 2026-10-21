"""
fig_style.py — Figure style for HydroSuiteAnalysis
================================================================
Implements the figures4papers conventions (ChenLiu-1996/figures4papers)
proven in Nature Machine Intelligence, ICML, NeurIPS, and ECCV.

Usage
-----
    from pipeline.fig_style import (
        apply_publication_style, finalize_figure,
        CA, CN, PALETTE, SEASON_COLORS, VAR_COLORS, VAR_LABELS,
        annotate_bars, make_legend_panel,
    )
    apply_publication_style()          # call once before creating figures
    fig, ax = plt.subplots(...)
    ...
    finalize_figure(fig, 'figures/my_plot.png')
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
#  PALETTE — figures4papers canonical colors
# ═══════════════════════════════════════════════════════════════════════

PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1":        "#DDF3DE",
    "green_2":        "#AADCA9",
    "green_3":        "#8BCF8B",
    "red_1":          "#F6CFCB",
    "red_2":          "#E9A6A1",
    "red_strong":     "#B64342",
    "neutral":        "#CFCECE",
    "highlight":      "#FFD700",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
}

# ── Semantic dataset colors ───────────────────────────────────────────
CA = PALETTE["blue_main"]       # AORC  — reference / primary
CN = PALETTE["red_strong"]      # NLDAS — evaluated / contrast

# ── Seasonal sub-palette ─────────────────────────────────────────────
SEASON_COLORS = {
    "DJF": PALETTE["blue_secondary"],   # winter
    "MAM": PALETTE["green_3"],          # spring
    "JJA": PALETTE["teal"],             # summer
    "SON": PALETTE["violet"],           # autumn
}

# ── 7-variable palette (Taylor, per-variable plots) ──────────────────
VAR_COLORS = {
    "precip":            PALETTE["blue_main"],
    "temp":              PALETTE["red_strong"],
    "shortwave":         PALETTE["green_3"],
    "longwave":          PALETTE["teal"],
    "specific_humidity":  PALETTE["violet"],
    "wind_2m":           "#DAA87C",                  # tan accent
    "pressure":          "#4D4D4D",                  # dark gray
    "surface_pressure":  "#4D4D4D",                  # alias
    "et0":               PALETTE["blue_secondary"],  # ET₀
}

# ── Human-readable variable labels ───────────────────────────────────
VAR_LABELS = {
    "precip":            "Precipitation",
    "temp":              "Temperature",
    "shortwave":         "Shortwave",
    "longwave":          "Longwave",
    "specific_humidity": "Sp. Humidity",
    "wind_2m":           "2-m Wind",
    "pressure":          "Pressure",
    "surface_pressure":  "Pressure",
    "et0":               r"ET$_0$",
}

# ── Variable display order ───────────────────────────────────────────
VAR_ORDER = [
    "precip", "temp", "shortwave", "longwave",
    "specific_humidity", "wind_2m", "pressure",
]

# ── Dataset color dict (for loop-based plotting) ─────────────────────
DATASET_COLORS = {"aorc": CA, "nldas": CN}


# ═══════════════════════════════════════════════════════════════════════
#  STYLE APPLICATION
# ═══════════════════════════════════════════════════════════════════════

def apply_publication_style(font_size: int = 16, axes_linewidth: float = 2.5,
                            use_tex: bool = False):
    """
    Configure matplotlib rcParams to match figures4papers conventions.
    Call **once** before creating any figures.

    Parameters
    ----------
    font_size : int
        Base font size.  Use 24 for large comparison bars, 16 for standard
        multi-panel, 14 for compact/heatmaps.
    axes_linewidth : float
        Spine/axis line width.  Use 3 for large panels, 2.5 standard, 2 compact.
    use_tex : bool
        Enable LaTeX rendering (requires a TeX installation).
    """
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size":          font_size,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     axes_linewidth,
        "axes.labelsize":     font_size,
        "xtick.labelsize":    font_size - 2,
        "ytick.labelsize":    font_size - 2,
        "legend.frameon":     False,
        "legend.fontsize":    font_size - 2,
        "svg.fonttype":       "none",
        "figure.dpi":         100,
        "savefig.dpi":        300,
    })
    if use_tex:
        plt.rcParams["text.usetex"] = True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE EXPORT
# ═══════════════════════════════════════════════════════════════════════

def finalize_figure(fig, out_path, dpi: int = 300, pad: float = 2.0,
                    close: bool = True, bbox_inches=None, **kwargs):
    """
    Apply tight_layout, save, and close.

    Parameters
    ----------
    fig : matplotlib Figure
    out_path : str or Path
    dpi : int
    pad : float
        Padding for tight_layout.
    close : bool
        Whether to close figure after saving.
    bbox_inches : str or None
        Pass 'tight' for polar/overflow plots.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout(pad=pad)
    except Exception:
        pass  # some polar/gridspec layouts don't support tight_layout
    save_kw = dict(dpi=dpi, facecolor='white', edgecolor='none')
    if bbox_inches:
        save_kw["bbox_inches"] = bbox_inches
    save_kw.update(kwargs)
    fig.savefig(str(out_path), **save_kw)
    if close:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=11, padding=3,
                  color="black", bold=False):
    """
    Add value text above each bar in a BarContainer.
    """
    weight = "bold" if bold else "normal"
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + padding,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight=weight,
        )


def make_legend_panel(ax, handles, labels, fontsize=18, loc="center",
                      ncols=1, **kwargs):
    """
    Turn *ax* into a dedicated legend-only panel.
    """
    ax.set_axis_off()
    ax.legend(handles, labels, loc=loc, fontsize=fontsize,
              frameon=False, ncol=ncols, **kwargs)


def style_ticks(ax, length=6, width=1.5, labelsize=None):
    """Uniform tick styling."""
    kw = dict(length=length, width=width, direction="out")
    if labelsize is not None:
        kw["labelsize"] = labelsize
    ax.tick_params(**kw)


def vlab(var: str) -> str:
    """Return human-readable label for a variable key."""
    return VAR_LABELS.get(var, var.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════════════
#  BAR DEFAULTS
# ═══════════════════════════════════════════════════════════════════════

BAR_KW = dict(edgecolor="black", linewidth=1.5, alpha=0.88)
BAR_WIDTH = 0.32
