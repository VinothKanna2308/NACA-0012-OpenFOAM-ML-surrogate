#!/usr/bin/env python3
"""Run an AoA sweep (design-of-experiments for the surrogate) -> results/sweep.csv.

    python scripts/sweep.py --alphas -4 -2 0 2 4 6 8 10 12 14 --level medium

Needs OpenFOAM. This is where the training data for the ML surrogate comes from, so I picked
-4 to 14 degrees: it covers the linear lift region plus some of the approach to stall, and stays
below where steady RANS stops being believable. I use the medium mesh because the mesh study
showed it is already close to the fine result, at less than half the cost.
Each case starts from scratch (no restart from the previous angle), so every point is independent.
The time per case is only my estimate, tens of minutes on 4-8 cores.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_case  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--alphas", type=float, nargs="+", default=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
ap.add_argument("--level", default="medium")
ap.add_argument("--n-iter", type=int, default=3000)
ap.add_argument("--runs-dir", default="runs")
ap.add_argument("--out", default="results/sweep.csv")
a = ap.parse_args()

os.makedirs(os.path.dirname(a.out), exist_ok=True)
rows = []
for al in a.alphas:
    r = run_case.run(os.path.join(a.runs_dir, f"sweep_a{al:g}_{a.level}"), al, a.level, a.n_iter)
    rows.append(dict(alpha=al, Cl=r["Cl"], Cd=r["Cd"], Cm=r.get("Cm", float("nan"))))
    # I rewrite the csv after every angle, so if a run crashes late I keep what's already done
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "Cl", "Cd", "Cm"])
        w.writeheader()
        w.writerows(rows)
print("wrote", a.out)
