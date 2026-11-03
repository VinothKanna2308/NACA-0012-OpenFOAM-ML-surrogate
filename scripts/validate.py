#!/usr/bin/env python3
"""Compare a run with the NASA TMR SA reference (CFL3D) and write results/validation.json.

    python scripts/validate.py runs/a10_medium/results.json

I compare against CFL3D because it is the TMR's own reference solution for Spalart-Allmaras, so
it's the fairest target for the same turbulence model. I also check whether my values fall inside
the spread of all the SA codes in the TMR. Matching one code could be luck, but sitting inside the
spread of seven says more. When I'm outside it, this script says so and doesn't hide it.
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def validate(results_json):
    with open(results_json) as f:
        r = json.load(f)
    with open(os.path.join(ROOT, "reference", "sa_reference_alpha10.csv")) as f:
        ref = {row["code"]: row for row in csv.DictReader(f)}
    cl_all = [float(v["CL_alpha10"]) for v in ref.values()]
    cd_all = [float(v["CD_alpha10"]) for v in ref.values()]
    cl_ref, cd_ref = float(ref["CFL3D"]["CL_alpha10"]), float(ref["CFL3D"]["CD_alpha10"])
    out = dict(
        CL=r["Cl"], CD=r["Cd"], CL_ref_CFL3D=cl_ref, CD_ref_CFL3D=cd_ref,
        dCL_pct=100 * (r["Cl"] - cl_ref) / cl_ref, dCD_pct=100 * (r["Cd"] - cd_ref) / cd_ref,
        CL_in_code_spread=min(cl_all) <= r["Cl"] <= max(cl_all),
        CD_in_code_spread=min(cd_all) <= r["Cd"] <= max(cd_all),
    )
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(validate(sys.argv[1]), indent=2))
