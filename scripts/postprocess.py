#!/usr/bin/env python3
"""Parse OpenFOAM results (pure Python, no OpenFOAM libs needed).

* forceCoeffs.dat  -> converged CL / CD / Cm (mean of the last N iterations)
* p / C / yPlus    -> Cp(x) on the airfoil surface and y+ statistics

I wrote my own parser, in plain Python, for the OpenFOAM output. That way anyone can re-do the
analysis on the result files without a working OpenFOAM install, or without ParaView, and nothing
is hidden in a GUI step. The price is that it only reads ASCII output, so keep writeFormat ascii.

Cell/patch-face centres come from ``postProcess -func writeCellCentres``
(run_case.py calls it); Cp = p / (0.5 U^2) because the solver is kinematic (rho = 1).
"""
import glob
import json
import os
import re
import sys

import numpy as np


def read_force_coeffs(case_dir):
    # I'm on ESI v2606, which writes coefficient.dat. I also accept the old forceCoeffs.dat name
    # so older OpenFOAM versions still work
    files = sorted(glob.glob(os.path.join(case_dir, "postProcessing", "forceCoeffs", "*", "coefficient.dat"))
                   + glob.glob(os.path.join(case_dir, "postProcessing", "forceCoeffs", "*", "forceCoeffs.dat")))
    if not files:
        raise FileNotFoundError("no forceCoeffs.dat found")
    cols, rows = None, []
    with open(files[-1]) as f:
        for line in f:
            if line.startswith("#"):
                names = line[1:].split()
                if names and names[0] == "Time":
                    cols = names
                continue
            if line.strip():
                rows.append([float(v) for v in line.split()])
    if cols is None:
        raise ValueError("no column header in forceCoeffs.dat")
    data = np.array(rows)
    out = {c: data[:, i] for i, c in enumerate(cols)}
    if "Cm" not in out and "CmPitch" in out:  # newer releases name the pitch moment CmPitch
        out["Cm"] = out["CmPitch"]
    return out


def converged_coeffs(case_dir, n_avg=200):
    d = read_force_coeffs(case_dir)
    n = min(n_avg, len(d["Time"]))
    # the std of the averaging window is saved too, as a quick check that the run really settled
    out = {k: float(np.mean(d[k][-n:])) for k in ("Cl", "Cd", "Cm") if k in d}
    out["Cl_std"] = float(np.std(d["Cl"][-n:]))
    out["Cd_std"] = float(np.std(d["Cd"][-n:]))
    out["iterations"] = int(d["Time"][-1])
    return out


def latest_time(case_dir):
    # steady runs write the iteration number as the time folder name, so the biggest number wins
    times = []
    for d in os.listdir(case_dir):
        try:
            times.append((float(d), d))
        except ValueError:
            pass
    return max(times)[1] if times else None


def read_patch_field(path, patch):
    """Return the patch values of an ASCII volScalarField / volVectorField as an array."""
    with open(path) as f:
        text = f.read()
    m = re.search(r"boundaryField\s*\{", text)
    body = text[m.end():]
    m = re.search(r"\b" + re.escape(patch) + r"\s*\{", body)
    blk = body[m.end(): body.index("}", m.end())]
    m = re.search(r"value\s+(uniform\s+([^;]+);|nonuniform\s+List<(\w+)>\s*(\d+)\s*\((.*?)\)\s*;)", blk, re.S)
    if m is None:
        # zeroGradient/fixedFluxPressure patches store no value: use the owner-cell value
        return _owner_cell_values(path, patch)
    if m.group(2) is not None:
        return np.array([float(v) for v in m.group(2).replace("(", " ").replace(")", " ").split()])
    kind, n, raw = m.group(3), int(m.group(4)), m.group(5)
    vals = np.array([float(v) for v in raw.replace("(", " ").replace(")", " ").split()])
    return vals.reshape(n, 3) if kind == "vector" else vals


def _read_list(path, kind):
    """Read the first top-level ASCII list (label or scalar) after the FoamFile header."""
    with open(path) as f:
        text = f.read()
    return _parse_list(text[text.index("}") + 1:], kind)


def _parse_list(text, kind):
    m = re.search(r"(\d+)\s*\(", text)
    n = int(m.group(1))
    end = text.index(")", m.end())
    arr = np.array(text[m.end():end].split(), dtype=int if kind == "label" else float)
    return arr[:n]


def _owner_cell_values(path, patch):
    case = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    poly = os.path.join(case, "constant", "polyMesh")
    with open(os.path.join(poly, "boundary")) as f:
        m = re.search(r"\b" + re.escape(patch) + r"\s*\{(.*?)\}", f.read(), re.S)
    start = int(re.search(r"startFace\s+(\d+)", m.group(1)).group(1))
    nf = int(re.search(r"nFaces\s+(\d+)", m.group(1)).group(1))
    owner = _read_list(os.path.join(poly, "owner"), "label")[start:start + nf]
    with open(path) as f:
        text = f.read()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*", text)
    if m is None:
        raise ValueError(f"no value for patch {patch} in {path}")
    return _parse_list(text[m.end():], "scalar")[owner]


def surface_cp(case_dir, U=1.0, patch="airfoil"):
    t = os.path.join(case_dir, latest_time(case_dir))
    p = read_patch_field(os.path.join(t, "p"), patch)
    c = read_patch_field(os.path.join(t, "C"), patch)
    cp = p / (0.5 * U * U)
    order = np.argsort(c[:, 0])
    return c[order, 0], c[order, 1], cp[order]


def yplus_stats(case_dir, patch="airfoil"):
    t = os.path.join(case_dir, latest_time(case_dir))
    yp = read_patch_field(os.path.join(t, "yPlus"), patch)
    return dict(yplus_max=float(yp.max()), yplus_mean=float(yp.mean()), yplus_min=float(yp.min()))


def summarise(case_dir, n_avg=200):
    out = converged_coeffs(case_dir, n_avg)
    # y+ is only a nice-to-have, so a failure there must not throw away Cl and Cd
    try:
        out.update(yplus_stats(case_dir))
    except Exception as e:  # noqa: BLE001
        out["yplus_error"] = str(e)
    return out


if __name__ == "__main__":
    print(json.dumps(summarise(sys.argv[1]), indent=2))
