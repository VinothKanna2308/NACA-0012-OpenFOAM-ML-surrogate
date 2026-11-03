#!/usr/bin/env python3
"""Mesh-convergence study (coarse/medium/fine) with Richardson extrapolation and GCI.

    python scripts/mesh_study.py --run            # runs 3 cases (needs OpenFOAM)
    python scripts/mesh_study.py                  # only evaluates existing runs/*/results.json

I wanted a number for how much the mesh affects the answer, not just "the fine mesh looks fine".
So I run three meshes and use Richardson extrapolation and the GCI, following Celik et al. (2008).
That is the standard procedure, so a reader can compare my numbers with other studies.

The refinement ratio is r = sqrt(N_fine / N_medium) (and the same for medium/coarse), because the
mesh is 2D and the cell count grows with the square of the linear refinement.
I use the safety factor Fs = 1.25, which is the value recommended when you have three grids.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# hard-coded on purpose: these are the counts of the meshes that mesh/make_mesh.py produces
N_CELLS = {"coarse": 20736, "medium": 46656, "fine": 104976}


def gci(f1, f2, f3, r21, r32, fs=1.25):
    """f1 = fine, f2 = medium, f3 = coarse.  Returns dict (Celik et al. 2008)."""
    e21, e32 = f2 - f1, f3 - f2
    # identical results on two meshes would make the log below blow up, so I bail out early
    if e21 == 0 or e32 == 0:
        return dict(p=float("nan"), f_ext=f1, gci_fine=0.0, monotonic=True)
    s = math.copysign(1.0, e32 / e21)
    p = 2.0
    for _ in range(200):  # fixed-point iteration
        q = math.log((r21**p - s) / (r32**p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-8:
            break
        p = 0.5 * (p + p_new)  # I average with the old value (under-relax) so the iteration doesn't oscillate
    f_ext = (r21**p * f1 - f2) / (r21**p - 1)
    ea = abs((f1 - f2) / f1)
    gci_fine = fs * ea / (r21**p - 1)
    # monotonic=False means the mesh convergence isn't clean, and I wouldn't trust the GCI then
    return dict(p=p, f_ext=f_ext, e_a=ea, gci_fine=gci_fine, monotonic=s > 0)


def evaluate(runs_dir="runs", alpha=10):
    res = {}
    for lv in ("coarse", "medium", "fine"):
        path = os.path.join(runs_dir, f"a{alpha:g}_{lv}", "results.json")
        with open(path) as f:
            res[lv] = json.load(f)
    r21 = math.sqrt(N_CELLS["fine"] / N_CELLS["medium"])
    r32 = math.sqrt(N_CELLS["medium"] / N_CELLS["coarse"])
    out = {k: gci(res["fine"][k], res["medium"][k], res["coarse"][k], r21, r32) for k in ("Cl", "Cd")}
    out["r21"], out["r32"] = r21, r32
    return res, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--n-iter", type=int, default=3000)
    a = ap.parse_args()
    if a.run:
        import run_case
        for lv in ("coarse", "medium", "fine"):
            run_case.run(os.path.join(a.runs_dir, f"a{a.alpha:g}_{lv}"), a.alpha, lv, a.n_iter)
    res, out = evaluate(a.runs_dir, a.alpha)
    print(json.dumps(out, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/mesh_study.json", "w") as f:
        json.dump(dict(runs=res, gci=out), f, indent=2)
