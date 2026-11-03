#!/usr/bin/env python3
"""Surrogate for (Cl, Cd) over angle of attack: MLP vs quadratic response surface.

The quadratic polynomial plays the role of the regression correlation used in the master thesis
(HTC polynomial over test conditions); the MLP is the 'ML' alternative.  Both are scored on
held-out points and with leave-one-out CV, and the inference speed-up over a CFD run is reported.

    python ml/train_surrogate.py --data results/sweep.csv
    python ml/train_surrogate.py --synthetic        # smoke test on made-up data (NOT CFD results)
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def synthetic(n=24, seed=0):
    """Thin-airfoil-like curves with a stall roll-off.  Only for testing the pipeline."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(-4, 14, n))
    cl = 0.11 * a - 0.0006 * np.clip(a - 8, 0, None) ** 2 * 10
    cd = 0.008 + 0.0001 * a**2
    return a.reshape(-1, 1), np.column_stack([cl, cd])


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["alpha"].reshape(-1, 1), np.column_stack([d["Cl"], d["Cd"]])


def models(seed=0):
    poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
    # The target is scaled too: Cd is ~0.01, and without scaling the network's default L2
    # penalty dominates the tiny targets and the fit collapses (negative R2).
    mlp = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(),
                                MLPRegressor(hidden_layer_sizes=(16, 16), activation="tanh",
                                             solver="lbfgs", max_iter=5000, random_state=seed)),
        transformer=StandardScaler())
    return {"quadratic": poly, "mlp": mlp}


def evaluate(X, Y, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(2, int(round(test_frac * len(X))))
    te, tr = idx[:n_test], idx[n_test:]
    out = {}
    for name, model in models(seed).items():
        m = {}
        for k, label in enumerate(("Cl", "Cd")):
            model.fit(X[tr], Y[tr, k])
            pred = model.predict(X[te])
            loo = cross_val_predict(model, X, Y[:, k], cv=LeaveOneOut())
            m[label] = dict(
                MAE_test=float(mean_absolute_error(Y[te, k], pred)),
                R2_test=float(r2_score(Y[te, k], pred)) if len(te) > 1 else float("nan"),
                MAE_loo=float(mean_absolute_error(Y[:, k], loo)),
                R2_loo=float(r2_score(Y[:, k], loo)),
            )
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict(X)
        m["predict_ms_per_call"] = (time.perf_counter() - t0) / 100 * 1e3
        out[name] = m
    out["n_points"], out["n_test"] = int(len(X)), int(n_test)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/sweep.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cfd-minutes", type=float, default=None,
                    help="wall-clock minutes of one CFD run, to report the speed-up")
    ap.add_argument("--out", default="results/surrogate_metrics.json")
    a = ap.parse_args()
    X, Y = synthetic() if a.synthetic else load(a.data)
    res = evaluate(X, Y)
    res["synthetic_data"] = bool(a.synthetic)
    if a.cfd_minutes:
        for name in ("quadratic", "mlp"):
            res[name]["speedup_vs_cfd"] = a.cfd_minutes * 60e3 / res[name]["predict_ms_per_call"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))