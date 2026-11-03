#!/usr/bin/env python3
"""Independent sanity check of a generated .msh file (no OpenFOAM needed).

Parses the MSH v2.2 file again and verifies what ``gmshToFoam`` + ``checkMesh``
will need:

* every hexahedron has positive volume (right-handed node order),
* every interior face is shared by exactly two cells,
* every boundary face (shared by one cell) is covered by exactly one patch quad,
* patch quads do not touch interior faces,
* a few geometric numbers (non-orthogonality, aspect ratio) from the 3D data.

Usage:  python mesh/check_mesh.py mesh/naca0012_medium.msh
"""

import sys
from collections import defaultdict

import numpy as np

# faces of an OpenFOAM/Gmsh hexahedron (node indices 0..7)
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5))


def read_msh(path):
    with open(path) as f:
        lines = f.read().split("\n")
    it = iter(lines)
    names, nodes, quads, hexes = {}, {}, [], []
    for line in it:
        if line == "$PhysicalNames":
            for _ in range(int(next(it))):
                dim, tag, name = next(it).split(None, 2)
                names[(int(dim), int(tag))] = name.strip('"')
        elif line == "$Nodes":
            n = int(next(it))
            xyz = np.zeros((n + 1, 3))
            for _ in range(n):
                p = next(it).split()
                xyz[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
            nodes = xyz
        elif line == "$Elements":
            for _ in range(int(next(it))):
                p = next(it).split()
                etype, ntags = int(p[1]), int(p[2])
                phys = int(p[3])
                conn = tuple(int(v) for v in p[3 + ntags :])
                if etype == 3:
                    quads.append((phys, conn))
                elif etype == 5:
                    hexes.append(conn)
    return names, nodes, quads, hexes


def check(path, verbose=True):
    names, xyz, quads, hexes = read_msh(path)
    problems = []

    # --- volumes -----------------------------------------------------------
    h = np.array(hexes)
    p = xyz[h]  # (nc, 8, 3)
    # volume via triple products of the three edge vectors at node 0
    e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 3] - p[:, 0], p[:, 4] - p[:, 0]
    vol = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
    if vol.min() <= 0:
        problems.append(f"{int((vol <= 0).sum())} hexahedra with non-positive volume")

    # --- face topology -----------------------------------------------------
    face_cells = defaultdict(list)
    for c, conn in enumerate(hexes):
        for f in HEX_FACES:
            face_cells[frozenset(conn[k] for k in f)].append(c)
    n_int = sum(1 for v in face_cells.values() if len(v) == 2)
    boundary = {k for k, v in face_cells.items() if len(v) == 1}
    bad = [k for k, v in face_cells.items() if len(v) > 2]
    if bad:
        problems.append(f"{len(bad)} faces shared by more than two cells")

    patch_faces = defaultdict(int)
    covered = set()
    for phys, conn in quads:
        key = frozenset(conn)
        if key not in face_cells:
            problems.append("a patch quad does not match any cell face")
            break
        if len(face_cells[key]) != 1:
            problems.append("a patch quad lies on an interior face")
            break
        if key in covered:
            problems.append("a boundary face is covered by two patch quads")
            break
        covered.add(key)
        patch_faces[names[(2, phys)]] += 1
    missing = boundary - covered
    if missing:
        problems.append(f"{len(missing)} boundary faces belong to no patch (would become defaultFaces)")

    # --- geometry from the 3D data ----------------------------------------
    cc = p.mean(axis=1)
    nonortho = []
    for key, cells in face_cells.items():
        if len(cells) != 2:
            continue
        a, b = cells
        fp = xyz[list(key)]
        # face normal from the first three non-collinear points of the quad
        # (the 4 nodes are coplanar for our extruded cells)
        c0 = fp.mean(axis=0)
        n = np.cross(fp[1] - fp[0], fp[2] - fp[0])
        if np.linalg.norm(n) < 1e-30:
            n = np.cross(fp[1] - fp[0], fp[3] - fp[0])
        d = cc[b] - cc[a]
        cosang = abs(np.dot(n, d)) / (np.linalg.norm(n) * np.linalg.norm(d))
        nonortho.append(np.degrees(np.arccos(min(1.0, cosang))))
    nonortho = np.array(nonortho)

    out = dict(
        n_cells=len(hexes), n_internal_faces=n_int, n_boundary_faces=len(boundary),
        patches=dict(patch_faces), min_volume=float(vol.min()),
        max_nonortho=float(nonortho.max()), mean_nonortho=float(nonortho.mean()),
        problems=problems,
    )
    if verbose:
        print(f"{path}")
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    ok = True
    for path in sys.argv[1:]:
        res = check(path)
        ok &= not res["problems"]
    sys.exit(0 if ok else 1)
