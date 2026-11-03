#!/usr/bin/env python3
"""Render + run one case (needs OpenFOAM sourced) and write results.json.

    python scripts/run_case.py runs/a10_medium --alpha 10 --level medium --n-iter 3000

This is the one entry point I use for a single run: mesh, case files, solver, post-processing in
one command. I did that so every run in this project goes through exactly the same steps, whether
it's one validation case or part of a sweep, and the numbers in results.json are comparable.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foamcase  # noqa: E402
import postprocess  # noqa: E402


def run(case_dir, alpha, level, n_iter, re_=6e6, n_avg=200, mesh_dir=None):
    # build the mesh only if it isn't there yet, since meshing is slower than I want to repeat
    if not os.path.exists(os.path.join(foamcase.ROOT, "mesh", f"naca0012_{level}.msh")) and not mesh_dir:
        subprocess.check_call([sys.executable, os.path.join(foamcase.ROOT, "mesh", "make_mesh.py"),
                               "--level", level])
    foamcase.render_case(case_dir, alpha, re_, level, n_iter, mesh_dir=mesh_dir)
    subprocess.check_call(["./Allrun"], cwd=case_dir)
    # postprocess.py reads the files directly and has no OpenFOAM libs, so I write the cell centres
    # out here to give it the x/y positions it needs for the Cp plots
    subprocess.check_call("postProcess -func writeCellCentres -latestTime > log.writeCellCentres 2>&1",
                          shell=True, cwd=case_dir)
    # I average the last n_avg iterations instead of taking the final one, because the force
    # coefficients can wiggle slightly even when the residuals look converged
    res = postprocess.summarise(case_dir, n_avg)
    res.update(alpha=alpha, level=level, Re=re_)
    with open(os.path.join(case_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--level", default="medium")
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--n-avg", type=int, default=200)
    a = ap.parse_args()
    print(json.dumps(run(a.case_dir, a.alpha, a.level, a.n_iter, n_avg=a.n_avg), indent=2))
