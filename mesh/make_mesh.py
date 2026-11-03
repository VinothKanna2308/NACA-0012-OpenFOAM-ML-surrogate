#!/usr/bin/env python3
"""Generate the NACA 0012 C-mesh as a Gmsh .msh file for OpenFOAM's gmshToFoam.

    python mesh/make_mesh.py --level medium
    python mesh/make_mesh.py --level all          # coarse, medium, fine
    python mesh/make_mesh.py --level medium --plot
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshgen  # noqa: E402


def make(level: str, out_dir: str, r_far: float, l_wake: float, plot: bool):
    params = meshgen.MeshParams(meshgen.LEVELS[level], r_far=r_far, l_wake=l_wake)
    grid = meshgen.build_grid(params)
    q = meshgen.quality(grid)
    path = os.path.join(out_dir, f"naca0012_{level}.msh")
    info = meshgen.write_msh(grid, path)
    summary = dict(level=level, file=path, **grid.info, **info, quality=q)
    with open(os.path.join(out_dir, f"naca0012_{level}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{level}] {grid.n_cells} cells  wall-normal growth q_BL={grid.info['q_bl']:.3f}  "
          f"max non-orth={q['max_nonortho']:.1f} deg  max aspect={q['max_aspect']:.0f}  "
          f"h_wall={q['h_wall_min']:.2e}  -> {path}")
    if plot:
        import plot_mesh
        plot_mesh.plot(grid, os.path.join(out_dir, f"naca0012_{level}.png"))
    return summary


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--level", default="medium", choices=["coarse", "medium", "fine", "all"])
    ap.add_argument("--out-dir", default=here)
    ap.add_argument("--r-far", type=float, default=100.0, help="far-field radius in chords")
    ap.add_argument("--l-wake", type=float, default=100.0, help="wake length in chords")
    ap.add_argument("--plot", action="store_true", help="also write a PNG of the mesh")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    levels = ["coarse", "medium", "fine"] if a.level == "all" else [a.level]
    for lv in levels:
        make(lv, a.out_dir, a.r_far, a.l_wake, a.plot)


if __name__ == "__main__":
    main()
