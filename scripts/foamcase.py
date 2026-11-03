#!/usr/bin/env python3
"""Render an OpenFOAM case directory from ``templates/`` (no OpenFOAM needed).

    python scripts/foamcase.py case --alpha 10 --re 6e6 --level medium

I generate every case from templates instead of copying and hand-editing a case folder. When I
sweep the angle of attack I only want alpha to change, and this way I can't forget to update one
of the velocity or force directions in some file. It also means you can check the setup without
having OpenFOAM installed.

The free-stream follows the NASA TMR NACA 0012 case so my numbers are comparable: chord c = 1,
fully turbulent Spalart-Allmaras, nuTilda_inf = 3 nu, hence nut_inf = 0.210438 nu.
I used the incompressible solver. TMR runs at Mach 0.15, which I judged low enough to compare
directly, but it is a real difference and I list it in the README under "Limitations".
"""

import argparse
import math
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "templates")

# template file -> destination inside the case
LAYOUT = {
    "U.tmpl": "0/U",
    "p.tmpl": "0/p",
    "nut.tmpl": "0/nut",
    "nuTilda.tmpl": "0/nuTilda",
    "transportProperties.tmpl": "constant/transportProperties",
    "turbulenceProperties.tmpl": "constant/turbulenceProperties",
    "controlDict.tmpl": "system/controlDict",
    "fvSchemes.tmpl": "system/fvSchemes",
    "fvSolution.tmpl": "system/fvSolution",
}

CV1 = 7.1


def nut_ratio(chi):
    """nut/nu for a given nuTilda/nu (Spalart-Allmaras fv1 function)."""
    return chi**3 / (chi**3 + CV1**3) * chi


def params(alpha, re, n_iter, dz, U=1.0, write_interval=None):
    a = math.radians(alpha)
    # I keep U = 1 and chord = 1 so Re = 1/nu. That also makes Cp = 2p, which keeps the maths simple
    nu = U * 1.0 / re
    return {
        "alpha": f"{alpha:g}",
        "Re": f"{re:g}",
        "U": f"{U:g}",
        "Ux": f"{U * math.cos(a):.10g}",
        "Uy": f"{U * math.sin(a):.10g}",
        "nu": f"{nu:.10g}",
        "nuTilda_inf": f"{3 * nu:.10g}",
        "nut_inf": f"{nut_ratio(3.0) * nu:.10g}",
        # I rotate the flow rather than the mesh, so the lift and drag directions have to rotate too
        "liftDir": f"{-math.sin(a):.10g} {math.cos(a):.10g} 0",
        "dragDir": f"{math.cos(a):.10g} {math.sin(a):.10g} 0",
        "Aref": f"{dz:g}",  # the mesh is one cell thick, so reference area = chord * dz = dz
        "n_iter": str(n_iter),
        "write_interval": str(write_interval or n_iter),
    }


def render_text(text, p):
    out = re.sub(r"@@(\w+)@@", lambda m: p[m.group(1)], text)
    # I want a loud failure if a placeholder is left over, not a case that silently runs wrong
    left = re.findall(r"@@\w+@@", out)
    if left:
        raise KeyError(f"unresolved placeholders: {left}")
    return out


def render_case(case_dir, alpha=10.0, re_=6e6, level="medium", n_iter=3000, dz=0.1,
                mesh_dir=None, write_interval=None):
    p = params(alpha, re_, n_iter, dz, write_interval=write_interval)
    for tmpl, dest in LAYOUT.items():
        with open(os.path.join(TEMPLATES, tmpl)) as f:
            text = render_text(f.read(), p)
        path = os.path.join(case_dir, dest)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    for name in ("Allrun", "Allclean"):
        src = os.path.join(TEMPLATES, name)
        if os.path.exists(src):
            dst = os.path.join(case_dir, name)
            shutil.copy(src, dst)
            os.chmod(dst, 0o755)
    # each case gets its own copy of fix_boundary.py so Allrun works when the folder is moved elsewhere
    shutil.copy(os.path.join(ROOT, "scripts", "fix_boundary.py"), os.path.join(case_dir, "fix_boundary.py"))
    open(os.path.join(case_dir, "case.foam"), "w").close()  # empty file so ParaView can open the case
    # small note so I can tell later what a run folder was made with
    with open(os.path.join(case_dir, "case_params.txt"), "w") as f:
        f.write(f"level {level}\nalpha {alpha:g}\nRe {re_:g}\n")
    mesh_dir = mesh_dir or os.path.join(ROOT, "mesh")
    msh = os.path.join(mesh_dir, f"naca0012_{level}.msh")
    if os.path.exists(msh):
        shutil.copy(msh, os.path.join(case_dir, "naca0012.msh"))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("case_dir")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--re", type=float, default=6e6)
    ap.add_argument("--level", default="medium", choices=["coarse", "medium", "fine"])
    ap.add_argument("--n-iter", type=int, default=3000)
    ap.add_argument("--dz", type=float, default=0.1)
    ap.add_argument("--mesh-dir", default=None)
    a = ap.parse_args()
    render_case(a.case_dir, a.alpha, a.re, a.level, a.n_iter, a.dz, a.mesh_dir)
    print(f"case rendered in {a.case_dir}")


if __name__ == "__main__":
    main()
