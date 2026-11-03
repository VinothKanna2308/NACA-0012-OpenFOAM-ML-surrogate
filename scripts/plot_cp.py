#!/usr/bin/env python3
"""Quick Cp(x) plot for a single run.

    python scripts/plot_cp.py runs/a10_medium [out.png]

I made this for the moment right after a run finishes, when I just want to see whether the
pressure distribution looks sane before I bother with the full make_plots.py set.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postprocess  # noqa: E402
import plotstyle  # noqa: E402

plotstyle.apply()

case = sys.argv[1]
x, y, cp = postprocess.surface_cp(case)
fig, ax = plt.subplots(figsize=(6.5, 4.8))
plotstyle.plot_cp_sides(ax, x, y, cp, marker=True, ms=4, label_suffix=" (OpenFOAM)")
plotstyle.style_cp_axes(ax)
ax.set_title(f"Surface pressure coefficient - {os.path.basename(os.path.normpath(case))}")
ax.legend(loc="lower right")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", "cp.png")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
