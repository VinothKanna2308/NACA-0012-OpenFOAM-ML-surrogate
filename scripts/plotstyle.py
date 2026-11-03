"""Shared matplotlib style and airfoil-specific helpers for all exported figures.

I keep the look of every figure in this one file so the plots all match without me touching each
script. If you want different colours or fonts, change them here and re-run make_plots.py.
"""
import matplotlib.pyplot as plt
import numpy as np

# red/blue so suction and pressure side are the same colours in every plot
SUCTION = "#c0392b"
PRESSURE = "#1f4e9c"
CASE_TITLE = "NACA 0012, Re = 6×10$^6$, Spalart–Allmaras"


def apply():
    # I dropped the top/right spines and kept the grid faint so the data stands out, not the frame
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "dejavusans",
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.prop_cycle": plt.cycler(
            color=["#1f4e9c", "#d9822b", "#2e8b57", "#c0392b", "#7d5ba6", "#555555"]),
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "0.8",
        "legend.fancybox": False, "legend.fontsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
    })


def split_sides(y, cp):
    """Boolean masks (suction, pressure) for surface points.

    I label by suction/pressure side instead of upper/lower. At negative alpha the lower surface
    is the suction side, and calling it "upper" would make those plots misleading. The side with
    the lower mean Cp is the suction side.
    """
    up = np.asarray(y) >= 0
    if np.mean(cp[up]) <= np.mean(cp[~up]):
        return up, ~up
    return ~up, up


def plot_cp_sides(ax, x, y, cp, marker=False, ms=3, label_suffix=""):
    """Scatter/line Cp on both sides with 'Suction side' / 'Pressure side' legend entries."""
    suc, pre = split_sides(y, cp)
    for m, col, lab in ((suc, SUCTION, "Suction side"), (pre, PRESSURE, "Pressure side")):
        o = np.argsort(x[m])
        if marker:
            ax.plot(x[m], cp[m], ".", color=col, ms=ms, label=lab + label_suffix)
        else:
            ax.plot(x[m][o], cp[m][o], "-", color=col, lw=1.4, label=lab + label_suffix)


def style_cp_axes(ax):
    ax.invert_yaxis()  # the usual aerodynamics convention: negative Cp up, so suction looks like lift
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"$x/c$")
    ax.set_ylabel(r"Pressure coefficient, $C_p$")
