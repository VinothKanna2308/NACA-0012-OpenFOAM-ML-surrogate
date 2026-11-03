#!/usr/bin/env python3
"""Export every result of the project as PNG images into results/figures/.

    python scripts/make_plots.py                 # everything that can be built from runs/ and results/
    python scripts/make_plots.py --only sweep cp # a subset
    python scripts/make_plots.py --no-fields     # skip the (slower) flow-field contours

Groups: sweep, cp, history, mesh, yplus, validation, surrogate, fields.  Each group is skipped
with a message when its input data is missing, so the script works on a partial set of runs.

Every figure in the README comes from this one script. I did that on purpose: if you change a
run or the plot style, one command rebuilds all of them and none of the pictures can go stale or
come from a hand-edited copy. I skip missing inputs instead of crashing so I can look at results
while the sweep is still running.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ml"))
import postprocess  # noqa: E402
import plotstyle  # noqa: E402

plotstyle.apply()
OUT = os.path.join(ROOT, "results", "figures")
LEVELS = ("coarse", "medium", "fine")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_sweep():
    path = os.path.join(ROOT, "results", "sweep.csv")
    if not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=",", names=True)
    o = np.argsort(d["alpha"])
    return {k: d[k][o] for k in d.dtype.names}


def sweep_runs():
    """[(alpha, case_dir)] for runs/sweep_a*_medium, sorted by alpha."""
    out = []
    for d in glob.glob(os.path.join(ROOT, "runs", "sweep_a*_*")):
        m = re.search(r"sweep_a(-?[\d.]+)_", os.path.basename(d))
        if m:
            out.append((float(m.group(1)), d))
    return sorted(out)


def reference_alpha10():
    path = os.path.join(ROOT, "reference", "sa_reference_alpha10.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {r["code"]: (float(r["CL_alpha10"]), float(r["CD_alpha10"])) for r in csv.DictReader(f)}


# ---------------------------------------------------------------- sweep: coefficients vs alpha
def plot_sweep():
    s = load_sweep()
    if s is None:
        return print("sweep: results/sweep.csv missing, skipped")
    a, cl, cd, cm = s["alpha"], s["Cl"], s["Cd"], s["Cm"]
    ref = reference_alpha10()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(a, cl, "o-", label="OpenFOAM (SA)")
    # thin-airfoil line as a sanity check: the linear part of my curve should follow its slope
    ax.plot(a, 2 * np.pi * np.radians(a), "k--", lw=1, label=r"Thin-airfoil theory, $2\pi\alpha$")
    if ref:
        ax.plot(10, ref["CFL3D"][0], "r*", ms=12, label="CFL3D SA (TMR)")
    ax.set(xlabel=r"$\alpha$ [deg]", ylabel=r"$C_L$", title="Lift curve")
    ax.legend()
    save(fig, "cl_vs_alpha.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(a, cd, "o-", label="OpenFOAM (SA)")
    if ref:
        ax.plot(10, ref["CFL3D"][1], "r*", ms=12, label="CFL3D SA (TMR)")
    ax.set(xlabel=r"$\alpha$ [deg]", ylabel=r"$C_D$", title="Drag vs angle of attack")
    ax.legend()
    save(fig, "cd_vs_alpha.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(a, cm, "o-")
    ax.set(xlabel=r"$\alpha$ [deg]", ylabel=r"$C_m$", title="Pitching moment")
    save(fig, "cm_vs_alpha.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(cd, cl, "o-", label="OpenFOAM (SA)")
    if ref:
        ax.plot(ref["CFL3D"][1], ref["CFL3D"][0], "r*", ms=12, label="CFL3D SA (TMR), 10°")
    ax.set(xlabel=r"$C_D$", ylabel=r"$C_L$", title="Drag polar")
    ax.legend()
    save(fig, "polar_cl_cd.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    m = a > 0  # L/D at zero or negative alpha is meaningless (or divides by ~0), so I only plot positive angles
    ax.plot(a[m], cl[m] / cd[m], "o-")
    ax.set(xlabel=r"$\alpha$ [deg]", ylabel=r"$C_L/C_D$", title="Aerodynamic efficiency")
    save(fig, "efficiency_vs_alpha.png")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (y, lab) in zip(axs.flat, [(cl, r"$C_L$"), (cd, r"$C_D$"), (cm, r"$C_m$"), (None, r"$C_L/C_D$")]):
        if y is None:
            ax.plot(a[m], cl[m] / cd[m], "o-")
        else:
            ax.plot(a, y, "o-")
        ax.set(xlabel=r"$\alpha$ [deg]", ylabel=lab)
    fig.suptitle(plotstyle.CASE_TITLE, fontweight="bold")
    save(fig, "sweep_summary.png")


# ---------------------------------------------------------------- Cp distributions
def _cp(case):
    try:
        return postprocess.surface_cp(case)
    except Exception as e:  # noqa: BLE001
        print(f"cp: {os.path.basename(case)} skipped ({e})")
        return None


def _cp_axes(ax, x, y, cp, label=None, color=None):
    """Overlay curve: solid = suction side, dashed = pressure side.

    On overlays colour already means mesh level or angle, so I use line style for the side.
    """
    suc, pre = plotstyle.split_sides(y, cp)
    for m, ls, lab in ((suc, "-", label), (pre, "--", None)):
        o = np.argsort(x[m])
        ax.plot(x[m][o], cp[m][o], ls, color=color, lw=1.3, label=lab)


def _side_key(ax):
    """Legend explaining the line styles of overlay plots."""
    ax.legend(handles=[Line2D([], [], color="0.2", ls="-", label="Suction side"),
                       Line2D([], [], color="0.2", ls="--", label="Pressure side")],
              loc="lower right", fontsize=8)


def plot_cp():
    # single plots for every run
    cases = sorted(glob.glob(os.path.join(ROOT, "runs", "*_*")))
    cps = {}
    for c in cases:
        r = _cp(c)
        if r is None:
            continue
        cps[os.path.basename(c)] = r
        x, y, cp = r
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plotstyle.plot_cp_sides(ax, x, y, cp, marker=True)
        plotstyle.style_cp_axes(ax)
        ax.set_title(f"Surface $C_p$ - {os.path.basename(c)}")
        ax.legend(loc="lower right")
        save(fig, f"cp_{os.path.basename(c)}.png")

    # mesh-level overlay at alpha = 10: if the three curves lie on top of each other, the surface
    # pressure doesn't depend on the mesh, which is a stronger check than Cl and Cd alone
    lv = {l: cps[f"a10_{l}"] for l in LEVELS if f"a10_{l}" in cps}
    if len(lv) > 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for (l, (x, y, cp)), col in zip(lv.items(), ("C0", "C1", "C2")):
            _cp_axes(ax, x, y, cp, label=l, color=col)
        plotstyle.style_cp_axes(ax)
        ax.set_title(r"$C_p$ mesh sensitivity, $\alpha$ = 10°")
        leg = ax.legend(title="Mesh level", loc="upper right")
        _side_key(ax)
        ax.add_artist(leg)
        save(fig, "cp_mesh_comparison.png")

    # alpha overlay
    sw = [(al, cps[os.path.basename(d)]) for al, d in sweep_runs() if os.path.basename(d) in cps]
    if sw:
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.get_cmap("viridis")
        for i, (al, (x, y, cp)) in enumerate(sw):
            _cp_axes(ax, x, y, cp, label=f"{al:g}°", color=cmap(i / max(1, len(sw) - 1)))
        plotstyle.style_cp_axes(ax)
        ax.set_title(r"$C_p$ vs angle of attack")
        leg = ax.legend(ncol=2, fontsize=8, title=r"Angle of attack $\alpha$", loc="upper right")
        _side_key(ax)
        ax.add_artist(leg)
        save(fig, "cp_alpha_overlay.png")

        # panel grid, one subplot per angle
        n = len(sw)
        nc = 4
        nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(3.6 * nc, 3 * nr), sharex=True, sharey=True, squeeze=False)
        for ax, (al, (x, y, cp)) in zip(axs.flat, sw):
            _cp_axes(ax, x, y, cp, color="C0")
            ax.set_title(rf"$\alpha$ = {al:g}°", fontsize=10)
        for ax in axs.flat[n:]:
            ax.axis("off")
        axs.flat[0].invert_yaxis()
        axs.flat[0].legend(handles=[Line2D([], [], color="C0", ls="-", label="Suction side"),
                                    Line2D([], [], color="C0", ls="--", label="Pressure side")],
                           loc="lower right", fontsize=8)
        fig.suptitle(r"Surface $C_p$ for each angle of attack", fontweight="bold")
        for ax in axs[-1]:
            ax.set_xlabel(r"$x/c$")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$C_p$")
        save(fig, "cp_alpha_panels.png")


# ---------------------------------------------------------------- convergence histories
def _residuals(case):
    # I read the residuals from the solver log because it needs no extra function object
    log = os.path.join(case, "log.simpleFoam")
    if not os.path.exists(log):
        return None
    res, it = {}, 0
    with open(log) as f:
        for line in f:
            if line.startswith("Time = "):
                it = int(float(line.split("=")[1]))
            m = re.search(r"Solving for (\w+), Initial residual = ([\d.eE+-]+)", line)
            if m:
                res.setdefault(m.group(1), []).append((it, float(m.group(2))))
    return {k: np.array(v) for k, v in res.items()} or None


def plot_history():
    for c in sorted(glob.glob(os.path.join(ROOT, "runs", "*_*"))):
        name = os.path.basename(c)
        try:
            d = postprocess.read_force_coeffs(c)
        except Exception as e:  # noqa: BLE001
            print(f"history: {name} skipped ({e})")
            continue
        res = _residuals(c)
        fig, axs = plt.subplots(1, 3 if res else 2, figsize=(5 * (3 if res else 2), 4))
        axs[0].plot(d["Time"], d["Cl"])
        axs[0].set(xlabel="iteration", ylabel=r"$C_L$", title="Lift history")
        axs[1].plot(d["Time"], d["Cd"], "C1")
        axs[1].set(xlabel="iteration", ylabel=r"$C_D$", title="Drag history")
        if res:
            for k, v in res.items():
                axs[2].semilogy(v[:, 0], v[:, 1], label=k, lw=1)
            axs[2].set(xlabel="iteration", ylabel="initial residual", title="Residuals")
            axs[2].legend(fontsize=8)
        fig.suptitle(name)
        save(fig, f"history_{name}.png")


# ---------------------------------------------------------------- mesh study
def plot_mesh():
    path = os.path.join(ROOT, "results", "mesh_study.json")
    if not os.path.exists(path):
        return print("mesh: results/mesh_study.json missing, skipped")
    ms = read_json(path)
    sys.path.insert(0, HERE)
    from mesh_study import N_CELLS
    runs, gci = ms["runs"], ms["gci"]
    n = np.array([N_CELLS[l] for l in LEVELS if l in runs], dtype=float)
    h = 1 / np.sqrt(n)  # characteristic cell size of a 2D mesh; h -> 0 is the "infinitely fine" limit
    ref = reference_alpha10()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, rk, lab in ((axs[0], "Cl", 0, r"$C_L$"), (axs[1], "Cd", 1, r"$C_D$")):
        v = np.array([runs[l][key] for l in LEVELS if l in runs])
        ax.plot(h, v, "o-", label="OpenFOAM")
        g = gci.get(key, {})
        if "f_ext" in g:
            ax.plot(0, g["f_ext"], "gs", label=f"Richardson ext. (GCI fine {100 * g['gci_fine']:.2f} %)")
        if ref:
            ax.axhline(ref["CFL3D"][rk], color="r", ls="--", lw=1, label="CFL3D SA")
        for hi, ni, vi in zip(h, n, v):
            ax.annotate(f"{int(ni):,}", (hi, vi), textcoords="offset points", xytext=(4, 6), fontsize=8)
        ax.set(xlabel=r"$h \propto N^{-1/2}$", ylabel=lab, title=f"Mesh convergence of {lab}")
        ax.set_xlim(left=-0.0005)
        ax.legend(fontsize=8)
    save(fig, "mesh_convergence.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ls = [l for l in LEVELS if l in runs and "yplus_max" in runs[l]]
    x = np.arange(len(ls))
    for i, (k, lab) in enumerate((("yplus_min", "min"), ("yplus_mean", "mean"), ("yplus_max", "max"))):
        ax.bar(x + (i - 1) * 0.26, [runs[l][k] for l in ls], 0.26, label=lab)
    ax.axhline(1, color="k", ls="--", lw=1)  # SA is meant to be run with y+ around 1 or below
    ax.set_xticks(x, ls)
    ax.set(ylabel=r"$y^+$", title=r"Wall $y^+$ on the airfoil, $\alpha$ = 10°")
    ax.legend()
    save(fig, "yplus_by_mesh.png")


def plot_yplus():
    for c in sorted(glob.glob(os.path.join(ROOT, "runs", "a10_*"))):
        try:
            t = os.path.join(c, postprocess.latest_time(c))
            yp = postprocess.read_patch_field(os.path.join(t, "yPlus"), "airfoil")
            cc = postprocess.read_patch_field(os.path.join(t, "C"), "airfoil")
        except Exception as e:  # noqa: BLE001
            print(f"yplus: {os.path.basename(c)} skipped ({e})")
            continue
        up = cc[:, 1] >= 0
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(cc[up, 0], yp[up], ".", color=plotstyle.SUCTION, ms=3, label="Suction side")
        ax.plot(cc[~up, 0], yp[~up], ".", color=plotstyle.PRESSURE, ms=3, label="Pressure side")
        ax.axhline(1, color="k", ls="--", lw=1, label=r"$y^+ = 1$")
        ax.set(xlabel=r"$x/c$", ylabel=r"$y^+$", title=rf"Wall $y^+$ along the airfoil - {os.path.basename(c)}")
        ax.legend()
        save(fig, f"yplus_{os.path.basename(c)}.png")


# ---------------------------------------------------------------- validation
def plot_validation():
    path = os.path.join(ROOT, "results", "validation.json")
    ref = reference_alpha10()
    if not os.path.exists(path) or not ref:
        return print("validation: input missing, skipped")
    v = read_json(path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, i, mine, lab in ((axs[0], 0, v["CL"], r"$C_L$"), (axs[1], 1, v["CD"], r"$C_D$")):
        names = list(ref)
        vals = [ref[k][i] for k in names]
        ax.bar(names, vals, color="0.7", label="TMR SA codes")
        ax.axhline(mine, color="C3", lw=2, label="OpenFOAM")
        lo, hi = min(vals + [mine]), max(vals + [mine])
        # I zoom in on the data range instead of starting at zero. The codes differ by about a
        # percent, so with a zero baseline every bar would look identical
        pad = (hi - lo) * 0.3 or 0.01
        ax.set_ylim(lo - pad, hi + pad)
        ax.set(ylabel=lab, title=f"{lab} at 10° vs other SA codes")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    fig.suptitle(f"dCL = {v['dCL_pct']:+.2f} %, dCD = {v['dCD_pct']:+.2f} % vs CFL3D")
    save(fig, "validation_vs_tmr.png")


# ---------------------------------------------------------------- surrogate
def plot_surrogate():
    path = os.path.join(ROOT, "results", "sweep.csv")
    if not os.path.exists(path):
        return print("surrogate: results/sweep.csv missing, skipped")
    # With only 10 CFD points a single train/test split is noisy, so I also show leave-one-out
    # errors, which I trust more as a measure of how good the surrogate really is
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    import train_surrogate as ts
    X, Y = ts.load(path)
    xs = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    mods = ts.models()
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for k, lab in enumerate((r"$C_L$", r"$C_D$")):
        ax, axr = axs[0, k], axs[1, k]
        ax.plot(X, Y[:, k], "ko", label="CFD")
        for (name, m), col in zip(mods.items(), ("C0", "C3")):
            m.fit(X, Y[:, k])
            ax.plot(xs, m.predict(xs), color=col, label=name)
            loo = cross_val_predict(m, X, Y[:, k], cv=LeaveOneOut())
            axr.plot(X, loo - Y[:, k], "o-", color=col, label=name)
        ax.set(xlabel=r"$\alpha$ [deg]", ylabel=lab, title=f"Surrogate fit of {lab}")
        ax.legend()
        axr.axhline(0, color="k", lw=1)
        axr.set(xlabel=r"$\alpha$ [deg]", ylabel="LOO prediction - CFD", title=f"Leave-one-out error of {lab}")
        axr.legend()
    save(fig, "surrogate_fit.png")

    mpath = os.path.join(ROOT, "results", "surrogate_metrics.json")
    if os.path.exists(mpath):
        m = read_json(mpath)
        names = [n for n in ("quadratic", "mlp") if n in m]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, ttl in ((axs[0], "MAE", "Mean absolute error"), (axs[1], "R2", r"$R^2$")):
            w = 0.2
            x = np.arange(2)
            for j, (n, split) in enumerate([(n, s) for n in names for s in ("test", "loo")]):
                ax.bar(x + (j - 1.5) * w, [m[n][t][f"{key}_{split}"] for t in ("Cl", "Cd")], w, label=f"{n} {split}")
            ax.set_xticks(x, ["Cl", "Cd"])
            ax.set_title(ttl)
            if key == "MAE":
                ax.set_yscale("log")  # Cl and Cd errors differ by orders of magnitude, so log scale
            ax.legend(fontsize=8)
        save(fig, "surrogate_metrics.png")


# ---------------------------------------------------------------- flow fields
def _internal_field(path, kind):
    # same idea as postprocess.py: read the ASCII field directly so there is no ParaView step
    with open(path) as f:
        text = f.read()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", text)
    n = int(m.group(2))
    end = text.index("\n)", m.end())
    raw = text[m.end():end].replace("(", " ").replace(")", " ").split()
    arr = np.array(raw, dtype=float)
    return arr.reshape(n, 3) if kind == "vector" else arr


def plot_fields(cases=None):
    import matplotlib.tri as mtri
    cases = cases or [os.path.join(ROOT, "runs", n) for n in ("a10_medium", "sweep_a0_medium", "sweep_a14_medium")]
    for c in cases:
        if not os.path.isdir(c):
            continue
        name = os.path.basename(c)
        try:
            t = os.path.join(c, postprocess.latest_time(c))
            C = _internal_field(os.path.join(t, "C"), "vector")
            p = _internal_field(os.path.join(t, "p"), "scalar")
            U = _internal_field(os.path.join(t, "U"), "vector")
        except Exception as e:  # noqa: BLE001
            print(f"fields: {name} skipped ({e})")
            continue
        x, y = C[:, 0], C[:, 1]
        cp = p / 0.5
        umag = np.hypot(U[:, 0], U[:, 1])
        box = np.ones_like(x, dtype=bool)
        tri = mtri.Triangulation(x, y)
        # The mesh has no cells inside the airfoil, so matplotlib would draw contours across it.
        # I hide triangles whose centre is inside the NACA 0012 outline (the same as in make_video.py)
        xc, yc = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
        xx = np.clip(xc, 0, 1)
        yt = 0.6 * (0.2969 * np.sqrt(xx) - 0.1260 * xx - 0.3516 * xx**2 + 0.2843 * xx**3 - 0.1036 * xx**4)
        tri.set_mask((xc > 0) & (xc < 1) & (np.abs(yc) < yt + 2e-3))
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, f, lab, cm, lv in (
            (axs[0], cp[box], r"$C_p$", "RdBu_r", np.linspace(-3.5, 1.0, 46)),
            (axs[1], umag[box] / 1.0, r"$|U|/U_\infty$", "viridis", np.linspace(0, 1.6, 41)),
        ):
            cs = ax.tricontourf(tri, np.clip(f, lv[0], lv[-1]), levels=lv, cmap=cm, extend="both")
            fig.colorbar(cs, ax=ax, label=lab)
            ax.set_aspect("equal")
            ax.set_xlim(-0.4, 1.6)
            ax.set_ylim(-0.5, 0.5)
            ax.set(xlabel="x/c", ylabel="y/c", title=f"{lab} - {name}")
        save(fig, f"field_{name}.png")


GROUPS = {
    "sweep": plot_sweep, "cp": plot_cp, "history": plot_history, "mesh": plot_mesh,
    "yplus": plot_yplus, "validation": plot_validation, "surrogate": plot_surrogate,
    "fields": plot_fields,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=GROUPS)
    ap.add_argument("--no-fields", action="store_true")
    a = ap.parse_args()
    for g in a.only or [g for g in GROUPS if not (a.no_fields and g == "fields")]:
        GROUPS[g]()
