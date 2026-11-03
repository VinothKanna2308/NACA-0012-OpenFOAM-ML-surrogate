#!/usr/bin/env python3
"""Video of the angle-of-attack sweep (steady SA solutions, one frame group per angle).

    python scripts/make_video.py                  # -> results/videos/alpha_sweep.mp4
    python scripts/make_video.py --hold 1.5 --fps 20

I made this so someone can watch how the flow changes with angle instead of flipping through ten
PNGs. It is a nicer way to show where the suction peak grows and the wake thickens.

Be careful how you read it: the sweep only stores converged steady states, so this is NOT a time
history. I hold each angle for ``--hold`` seconds and I don't interpolate between angles, because
inventing in-between flow fields would show something the solver never computed. For a real time
history see ../NACA_0012_URANS (scripts/make_video.py there).
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_plots as mp  # noqa: E402
import postprocess  # noqa: E402

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})


def airfoil_mask(tri, x, y):
    # The mesh only has cells outside the airfoil, so the triangulation would otherwise fill the
    # inside with fake contours. I hide any triangle whose centre is inside the NACA 0012 thickness
    # (the 0.6 factor is 12 % thickness / 0.2) and give it a small margin so no sliver is left.
    xc, yc = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
    xx = np.clip(xc, 0, 1)
    yt = 0.6 * (0.2969 * np.sqrt(xx) - 0.1260 * xx - 0.3516 * xx**2 + 0.2843 * xx**3 - 0.1036 * xx**4)
    return (xc > 0) & (xc < 1) & (np.abs(yc) < yt + 2e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(mp.ROOT, "results", "videos", "alpha_sweep.mp4"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold", type=float, default=1.5, help="seconds per angle")
    a = ap.parse_args()

    runs = mp.sweep_runs()
    sw = mp.load_sweep()
    if not runs or sw is None:
        sys.exit("no sweep runs / results/sweep.csv found")

    data = []
    for al, d in runs:
        t = os.path.join(d, postprocess.latest_time(d))
        C = mp._internal_field(os.path.join(t, "C"), "vector")
        p = mp._internal_field(os.path.join(t, "p"), "scalar")
        U = mp._internal_field(os.path.join(t, "U"), "vector")
        x, y, cp = postprocess.surface_cp(d)
        # kinematic p with U = 1 and rho = 1, so Cp = p / 0.5 = 2p
        data.append(dict(alpha=al, C=C, cp_cell=2 * p, umag=np.hypot(U[:, 0], U[:, 1]), surf=(x, y, cp)))

    C = data[0]["C"]  # same mesh for every angle
    tri = mtri.Triangulation(C[:, 0], C[:, 1])
    tri.set_mask(airfoil_mask(tri, C[:, 0], C[:, 1]))
    # Contour levels and the Cp axis limit are fixed over all frames. If each frame scaled itself
    # the colours would change meaning between angles and the video would be misleading.
    lv_cp, lv_u = np.linspace(-6, 1.5, 61), np.linspace(0, 1.8, 46)
    cp_min = min(d["surf"][2].min() for d in data) - 0.5

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=16)
    for d in data:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2)
        ax_cp, ax_u, ax_s, ax_l = (fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2))
        for ax, f, lab, cm, lv in ((ax_cp, d["cp_cell"], r"$C_p$", "RdBu_r", lv_cp),
                                   (ax_u, d["umag"], r"$|U|/U_\infty$", "viridis", lv_u)):
            cs = ax.tricontourf(tri, np.clip(f, lv[0], lv[-1]), levels=lv, cmap=cm, extend="both")
            fig.colorbar(cs, ax=ax, label=lab, pad=0.02)
            ax.set_aspect("equal")
            ax.set(xlim=(-0.4, 1.6), ylim=(-0.5, 0.5), xlabel="x/c", ylabel="y/c", title=lab)
        x, y, cp = d["surf"]
        up = y >= 0
        ax_s.plot(x[up], cp[up], "b-", label="upper")
        ax_s.plot(x[~up], cp[~up], "r--", label="lower")
        ax_s.set_ylim(1.2, cp_min)
        ax_s.set(xlabel="x/c", ylabel=r"$C_p$", title="Surface pressure")
        ax_s.legend(loc="lower right")
        ax_l.plot(sw["alpha"], sw["Cl"], "-", color="0.6")
        ax_l.plot(d["alpha"], np.interp(d["alpha"], sw["alpha"], sw["Cl"]), "ro", ms=9)
        ax_l.set(xlabel=r"$\alpha$ [deg]", ylabel=r"$C_L$", title="Lift curve")
        fig.suptitle(rf"NACA 0012, Re = 6e6, SA   -   $\alpha$ = {d['alpha']:g}°   (steady solutions, not a time history)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        # render once, then write the same frame repeatedly for the hold time
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        for _ in range(int(round(a.hold * a.fps))):
            writer.append_data(frame)
    writer.close()
    print("wrote", os.path.relpath(a.out, mp.ROOT))


if __name__ == "__main__":
    main()
