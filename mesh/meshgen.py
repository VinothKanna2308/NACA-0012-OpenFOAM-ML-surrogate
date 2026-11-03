"""Structured C-mesh generator for a sharp-trailing-edge NACA 0012 airfoil.

Output: Gmsh MSH v2.2 (ASCII) with one-cell-thick hexahedra, readable by
OpenFOAM's ``gmshToFoam``.  Pure numpy, no Gmsh installation needed.

Topology (one structured block, i = streamwise, j = wall-normal)::

    i = 0 .............. nw ........... nw+ns ........... nw+2ns ............ ni-1
        lower wake       lower surface     upper surface        upper wake
        (outlet -> TE)   (TE -> LE)        (LE -> TE)           (TE -> outlet)

The wake cut (j = 0 for the two wake segments) is closed by *sharing nodes*,
so cells above and below it are connected through interior faces.  The
trailing edge is sharp; its upper and lower nodes are merged as well.

Wall-normal lines are cubic Hermite curves that leave the wall along the local
surface normal and run straight into the far-field boundary.  Spacing along
them is geometric (first cell ``h1``) up to a boundary-layer thickness ``t_bl``
and then geometric again with a ratio solved per line to reach the far field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- NACA 0012, sharp TE, coefficients from the NASA Turbulence Modeling
# Resource (NACA 0012 validation case).  The five coefficients sum to zero, so
# the polynomial closes exactly at x/c = 1 (verified in tests/test_meshgen.py).
_T = 0.594689181
_A = (0.298222773, -0.127125232, -0.357907906, 0.291984971, -0.105174606)


def half_thickness(x):
    """Half thickness y(x) of the sharp-TE NACA 0012 for x/c in [0, 1]."""
    x = np.asarray(x, dtype=float)
    return _T * (
        _A[0] * np.sqrt(x) + _A[1] * x + _A[2] * x**2 + _A[3] * x**3 + _A[4] * x**4
    )


@dataclass(frozen=True)
class Level:
    name: str
    ns: int  # cells along each airfoil surface
    nw: int  # cells along each wake half
    nj: int  # cells in the wall-normal direction
    h1: float  # first cell height [chord lengths]


# Cell counts grow by a factor 1.5 per direction between levels (2D cell ratio
# 2.25), first-cell height shrinks by 1.5, so the three meshes are
# approximately geometrically similar (needed for the GCI study).
LEVELS = {
    "coarse": Level("coarse", ns=96, nw=48, nj=72, h1=9.0e-6),
    "medium": Level("medium", ns=144, nw=72, nj=108, h1=6.0e-6),
    "fine": Level("fine", ns=216, nw=108, nj=162, h1=4.0e-6),
}


@dataclass
class MeshParams:
    level: Level
    r_far: float = 100.0  # far-field radius [chords], centre at the TE
    l_wake: float = 100.0  # wake length behind the TE [chords]
    dz: float = 0.1  # extrusion depth (one cell) [chords]
    t_bl: float = 0.3  # thickness of the geometric boundary-layer segment
    frac_bl: float = 0.6  # fraction of wall-normal cells in that segment
    clust: float = 0.9  # surface clustering strength at LE/TE (0 = uniform)
    wake_aspect: float = 100.0  # max streamwise/wall-normal aspect ratio of wake cells
    beta: float = 0.3  # start-tangent length of the radial Hermite lines, in units of the line length


@dataclass
class Grid:
    X: np.ndarray  # (ni, nj+1)
    Y: np.ndarray
    params: MeshParams
    info: dict = field(default_factory=dict)

    @property
    def ni(self):
        return self.X.shape[0]

    @property
    def nj(self):
        return self.X.shape[1] - 1

    @property
    def n_cells(self):
        return (self.ni - 1) * self.nj


# ----------------------------------------------------------------------------
# 1D helpers
# ----------------------------------------------------------------------------


def geometric_ratio(first: float, total: float, n: int) -> float:
    """Ratio q with first*(q^n - 1)/(q - 1) = total (bisection)."""
    if abs(total - first * n) < 1e-14 * total:
        return 1.0
    lo, hi = (1.0 + 1e-12, 4.0) if total > first * n else (0.05, 1.0 - 1e-12)

    def f(q):
        return first * (q**n - 1.0) / (q - 1.0) - total

    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (f(lo) > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def surface_nodes(ns: int, clust: float):
    """Upper-surface nodes from LE (p=0) to TE (p=ns).

    Returns x, y, nx, ny, u where u is the arc-length fraction from the LE.
    """
    xd = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 40001)))  # dense, LE->TE
    yd = half_thickness(xd)
    ds = np.hypot(np.diff(xd), np.diff(yd))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    xi = np.linspace(0.0, 1.0, ns + 1)
    g = xi - clust * np.sin(2.0 * np.pi * xi) / (2.0 * np.pi)  # clustered, monotone
    st = g * s[-1]
    x = np.interp(st, s, xd)
    y = np.interp(st, s, yd)
    # tangent from the dense curve (central differences), then interpolate
    tx = np.gradient(xd, s)
    ty = np.gradient(yd, s)
    tx_n = np.interp(st, s, tx)
    ty_n = np.interp(st, s, ty)
    norm = np.hypot(tx_n, ty_n)
    tx_n, ty_n = tx_n / norm, ty_n / norm
    nx, ny = -ty_n, tx_n  # outward normal of the upper surface
    # Near the sharp TE the true surface normals tilt downstream and would
    # cross the vertical radial line that leaves the TE.  Blend them towards
    # vertical over the last 20 % of chord (a few degrees of non-orthogonality).
    tb = np.clip((x - 0.8) / 0.2, 0.0, 1.0)
    w = tb * tb * (3.0 - 2.0 * tb)
    nx, ny = (1.0 - w) * nx, (1.0 - w) * ny + w
    nrm = np.hypot(nx, ny)
    nx, ny = nx / nrm, ny / nrm
    nx[0], ny[0] = -1.0, 0.0  # leading edge
    nx[-1], ny[-1] = 0.0, 1.0  # trailing edge: vertical radial line
    x[0], y[0] = 0.0, 0.0
    x[-1], y[-1] = 1.0, 0.0
    return x, y, nx, ny, g, s[-1]


def radial_distances(nj: int, h1: float, t_bl: float, frac_bl: float):
    """Boundary-layer segment distances d_0..d_nA (absolute, chord units)."""
    n_a = max(2, int(round(frac_bl * nj)))
    q = geometric_ratio(h1, t_bl, n_a)
    j = np.arange(n_a + 1)
    d = h1 * (q**j - 1.0) / (q - 1.0) if q != 1.0 else h1 * j
    d[-1] = t_bl
    return n_a, q, d


def hermite(x0, t0, xf, t1, t):
    """Cubic Hermite curve points for parameter array t (shape (m,))."""
    t = t[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * x0 + h10 * t0 + h01 * xf + h11 * t1


# ----------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------


def build_grid(params: MeshParams) -> Grid:
    lv = params.level
    ns, nw, nj = lv.ns, lv.nw, lv.nj
    R, Lw = params.r_far, params.l_wake
    ni = 2 * nw + 2 * ns + 1

    xs, ys, nxs, nys, u, S = surface_nodes(ns, params.clust)
    # wake: geometric growth starting from the TE surface cell length
    ds_te = math.hypot(xs[-1] - xs[-2], ys[-1] - ys[-2])
    q_w = geometric_ratio(ds_te, Lw, nw)
    k = np.arange(nw + 1)
    dw = ds_te * (q_w**k - 1.0) / (q_w - 1.0)
    dw[-1] = Lw

    n_a, q_a, d_a = radial_distances(nj, lv.h1, params.t_bl, params.frac_bl)

    # Wake lines: the streamwise cell length grows to ~20 chords, so a wall-sized
    # first cell would give aspect ratios of 1e6.  Cap the wake aspect ratio by
    # growing the first-cell height (and the boundary-layer segment) with it.
    seg_w = np.diff(dw)
    dw_len = np.maximum(np.concatenate([seg_w[:1], seg_w]), np.concatenate([seg_w, seg_w[-1:]]))

    def wake_dists(kk):
        # additive, slowly varying shift: neighbouring (narrow) columns then
        # differ by a small fraction of their width, unlike a rescaled profile
        extra = max(0.0, dw_len[kk] / params.wake_aspect - lv.h1)
        return d_a + extra * np.arange(n_a + 1)

    # Far-field x spacing of the wake lines: start with the spacing of the
    # neighbouring surface lines (R * dphi) so the lines fan out instead of
    # being parallel slits next to the trailing edge.
    ff = max(ds_te, R * 0.5 * math.pi * (u[-1] - u[-2]))
    q_f = geometric_ratio(ff, Lw, nw)
    dwf = ff * (q_f**k - 1.0) / (q_f - 1.0)
    dwf[-1] = Lw

    # base points X0, wall normals n and far-field points Xf for every i
    X0 = np.zeros((ni, 2))
    N0 = np.zeros((ni, 2))
    XF = np.zeros((ni, 2))
    # lower wake: i = 0..nw ; k_idx = nw - i (0 at TE)
    for i in range(0, nw + 1):
        kk = nw - i
        X0[i] = (1.0 + dw[kk], 0.0)
        N0[i] = (0.0, -1.0)
        XF[i] = (1.0 + dwf[kk], -R)
    # lower surface: i = nw..nw+ns, m = i-nw (0 = TE), p = ns - m
    for i in range(nw, nw + ns + 1):
        p = ns - (i - nw)
        X0[i] = (xs[p], -ys[p])
        N0[i] = (nxs[p], -nys[p])
        phi = math.pi + 0.5 * math.pi * u[p]
        XF[i] = (1.0 + R * math.cos(phi), R * math.sin(phi))
    # upper surface: i = nw+ns..nw+2ns, p = i-(nw+ns)
    for i in range(nw + ns, nw + 2 * ns + 1):
        p = i - (nw + ns)
        X0[i] = (xs[p], ys[p])
        N0[i] = (nxs[p], nys[p])
        phi = math.pi - 0.5 * math.pi * u[p]
        XF[i] = (1.0 + R * math.cos(phi), R * math.sin(phi))
    # upper wake: i = nw+2ns .. ni-1, kk = i - (nw+2ns)
    for i in range(nw + 2 * ns, ni):
        kk = i - (nw + 2 * ns)
        X0[i] = (1.0 + dw[kk], 0.0)
        N0[i] = (0.0, 1.0)
        XF[i] = (1.0 + dwf[kk], R)

    X = np.zeros((ni, nj + 1))
    Y = np.zeros((ni, nj + 1))
    n_b = nj - n_a
    q_b_all = np.zeros(ni)
    # dense, wall-clustered Hermite parameter used to measure arc length
    t_dense = np.concatenate([[0.0], np.geomspace(1e-11, 1.0, 24000)])
    for i in range(ni):
        if i <= nw:
            d_a_i = wake_dists(nw - i)
        elif i >= nw + 2 * ns:
            d_a_i = wake_dists(i - (nw + 2 * ns))
        else:
            d_a_i = d_a
        D = XF[i] - X0[i]
        L = float(np.hypot(*D))
        T0 = params.beta * L * N0[i]
        dense = hermite(X0[i], T0, XF[i], D, t_dense)
        seg = np.hypot(np.diff(dense[:, 0]), np.diff(dense[:, 1]))
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = arc[-1]
        # Every line (surface, TE column, wake) uses the same recipe:
        # geometric boundary-layer segment with absolute distances d_a, then a
        # geometric outer segment whose ratio is solved to end on the far
        # field.  Identical distributions on neighbouring lines keep the
        # constant-j grid lines smooth (low non-orthogonality) everywhere.
        delta = d_a_i[-1] - d_a_i[-2]
        span = total - d_a_i[-1]
        lo, hi = 1.0 + 1e-12, 6.0
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            val = delta * mid * (mid**n_b - 1.0) / (mid - 1.0) - span
            if val > 0:
                hi = mid
            else:
                lo = mid
        q_b = 0.5 * (lo + hi)
        m = np.arange(1, n_b + 1)
        d_b = d_a_i[-1] + delta * q_b * (q_b**m - 1.0) / (q_b - 1.0)
        d_b[-1] = total
        s_nodes = np.concatenate([d_a_i, d_b])
        t_nodes = np.interp(s_nodes, arc, t_dense)
        pts = hermite(X0[i], T0, XF[i], D, t_nodes)
        X[i], Y[i] = pts[:, 0], pts[:, 1]
        X[i, 0], Y[i, 0] = X0[i]
        X[i, nj], Y[i, nj] = XF[i]
        q_b_all[i] = q_b

    info = dict(
        ni=ni, nj=nj, n_cells=(ni - 1) * nj, q_bl=q_a, q_wake=q_w,
        q_outer_max=float(q_b_all.max()), n_bl=n_a, ds_te=ds_te, arc_upper=S,
    )
    return Grid(X, Y, params, info)


# ----------------------------------------------------------------------------
# quality metrics (2D, per cell / per face)
# ----------------------------------------------------------------------------


def quality(grid: Grid) -> dict:
    X, Y = grid.X, grid.Y
    x00, x10, x11, x01 = X[:-1, :-1], X[1:, :-1], X[1:, 1:], X[:-1, 1:]
    y00, y10, y11, y01 = Y[:-1, :-1], Y[1:, :-1], Y[1:, 1:], Y[:-1, 1:]
    # signed area (shoelace), must be positive for the (i,j,k) right-handed order
    area = 0.5 * (
        (x00 * y10 - x10 * y00) + (x10 * y11 - x11 * y10)
        + (x11 * y01 - x01 * y11) + (x01 * y00 - x00 * y01)
    )
    xc = 0.25 * (x00 + x10 + x11 + x01)
    yc = 0.25 * (y00 + y10 + y11 + y01)

    # internal angles
    def ang(ax, ay, bx, by, cx, cy):  # angle at b between a and c (degrees)
        v1x, v1y, v2x, v2y = ax - bx, ay - by, cx - bx, cy - by
        c = (v1x * v2x + v1y * v2y) / (np.hypot(v1x, v1y) * np.hypot(v2x, v2y))
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    angs = np.stack([
        ang(x01, y01, x00, y00, x10, y10), ang(x00, y00, x10, y10, x11, y11),
        ang(x10, y10, x11, y11, x01, y01), ang(x11, y11, x01, y01, x00, y00),
    ])
    # edge lengths -> aspect ratio
    li = 0.5 * (np.hypot(x10 - x00, y10 - y00) + np.hypot(x11 - x01, y11 - y01))
    lj = 0.5 * (np.hypot(x01 - x00, y01 - y00) + np.hypot(x11 - x10, y11 - y10))
    aspect = np.maximum(li / lj, lj / li)

    # non-orthogonality: angle between the face normal and the cell-centre line
    def nonortho(dx, dy, ex, ey):  # centre vector (dx,dy), face edge (ex,ey)
        nx, ny = ey, -ex
        c = np.abs(dx * nx + dy * ny) / (np.hypot(dx, dy) * np.hypot(nx, ny))
        return np.degrees(np.arccos(np.clip(c, 0, 1)))

    # faces between cell (i,j) and (i+1,j): the shared edge is x[i+1, j:j+2]
    no_i = nonortho(
        xc[1:, :] - xc[:-1, :], yc[1:, :] - yc[:-1, :],
        X[1:-1, 1:] - X[1:-1, :-1], Y[1:-1, 1:] - Y[1:-1, :-1],
    )
    no_j = nonortho(
        xc[:, 1:] - xc[:, :-1], yc[:, 1:] - yc[:, :-1],
        X[1:, 1:-1] - X[:-1, 1:-1], Y[1:, 1:-1] - Y[:-1, 1:-1],
    )
    # wall-normal growth ratio of neighbouring cells
    hj = np.hypot(X[:, 1:] - X[:, :-1], Y[:, 1:] - Y[:, :-1])
    growth = hj[:, 1:] / hj[:, :-1]
    # first cell height at the wall (airfoil faces only)
    nw, ns = grid.params.level.nw, grid.params.level.ns
    h_wall = hj[nw : nw + 2 * ns, 0]

    return dict(
        n_cells=int(area.size),
        min_area=float(area.min()),
        min_angle=float(angs.min()),
        max_angle=float(angs.max()),
        max_aspect=float(aspect.max()),
        max_nonortho=float(max(no_i.max(), no_j.max())),
        mean_nonortho=float(np.mean(np.concatenate([no_i.ravel(), no_j.ravel()]))),
        max_growth=float(growth.max()),
        min_growth=float(growth.min()),
        h_wall_min=float(h_wall.min()),
        h_wall_max=float(h_wall.max()),
    )


def cell_centres(grid: Grid):
    """Arithmetic cell centres, ordered like the cells in the MSH file."""
    X, Y = grid.X, grid.Y
    xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
    yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
    return xc, yc  # shape (ni-1, nj); cell id = i*nj + j


# ----------------------------------------------------------------------------
# MSH writer
# ----------------------------------------------------------------------------

PATCH_TAGS = {"airfoil": 1, "farfield": 2, "outlet": 3, "frontAndBack": 4}
VOLUME_TAG = 5
VOLUME_NAME = "fluid"


def node_ids(grid: Grid):
    """Unique 2D node ids with the wake cut and the trailing edge merged."""
    ni, nj1 = grid.X.shape
    nw = grid.params.level.nw
    ids = -np.ones((ni, nj1), dtype=np.int64)
    n = 0
    for i in range(ni):
        for j in range(nj1):
            if j == 0 and i < nw + 1:
                continue  # lower wake line + TE: filled from the upper side
            ids[i, j] = n
            n += 1
    for i in range(0, nw + 1):  # merge lower wake / TE with their upper partner
        ids[i, 0] = ids[ni - 1 - i, 0]
    return ids, n


def write_msh(grid: Grid, path: str) -> dict:
    ids, n2d = node_ids(grid)
    ni, nj1 = grid.X.shape
    nj = nj1 - 1
    lv = grid.params.level
    nw, ns = lv.nw, lv.ns
    dz = grid.params.dz
    z0, z1 = -0.5 * dz, 0.5 * dz

    # unique node coordinates
    xu = np.zeros(n2d)
    yu = np.zeros(n2d)
    for i in range(ni):
        for j in range(nj1):
            xu[ids[i, j]] = grid.X[i, j]
            yu[ids[i, j]] = grid.Y[i, j]

    def nid(i, j, k):  # k = 0 -> z0, 1 -> z1 (1-based ids for the MSH file)
        return int(ids[i, j]) + 1 + k * n2d

    n_hex = (ni - 1) * nj
    patch_quads = {name: [] for name in PATCH_TAGS}
    for i in range(nw, nw + 2 * ns):
        patch_quads["airfoil"].append((nid(i, 0, 0), nid(i + 1, 0, 0), nid(i + 1, 0, 1), nid(i, 0, 1)))
    for i in range(ni - 1):
        patch_quads["farfield"].append((nid(i, nj, 0), nid(i + 1, nj, 0), nid(i + 1, nj, 1), nid(i, nj, 1)))
    for j in range(nj):
        patch_quads["outlet"].append((nid(0, j, 0), nid(0, j + 1, 0), nid(0, j + 1, 1), nid(0, j, 1)))
        patch_quads["outlet"].append((nid(ni - 1, j, 0), nid(ni - 1, j + 1, 0), nid(ni - 1, j + 1, 1), nid(ni - 1, j, 1)))
    for i in range(ni - 1):
        for j in range(nj):
            for k in (0, 1):
                patch_quads["frontAndBack"].append(
                    (nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k)))

    n_quads = sum(len(v) for v in patch_quads.values())
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n%d\n" % (len(PATCH_TAGS) + 1))
        for name, tag in PATCH_TAGS.items():
            f.write('2 %d "%s"\n' % (tag, name))
        f.write('3 %d "%s"\n$EndPhysicalNames\n' % (VOLUME_TAG, VOLUME_NAME))
        f.write("$Nodes\n%d\n" % (2 * n2d))
        for k, z in enumerate((z0, z1)):
            for n in range(n2d):
                f.write("%d %.15g %.15g %.15g\n" % (n + 1 + k * n2d, xu[n], yu[n], z))
        f.write("$EndNodes\n")
        f.write("$Elements\n%d\n" % (n_quads + n_hex))
        eid = 1
        for name, quads in patch_quads.items():
            tag = PATCH_TAGS[name]
            for q in quads:
                f.write("%d 3 2 %d %d %d %d %d %d\n" % (eid, tag, tag, *q))
                eid += 1
        for i in range(ni - 1):
            for j in range(nj):
                f.write("%d 5 2 %d %d %d %d %d %d %d %d %d %d\n" % (
                    eid, VOLUME_TAG, VOLUME_TAG,
                    nid(i, j, 0), nid(i + 1, j, 0), nid(i + 1, j + 1, 0), nid(i, j + 1, 0),
                    nid(i, j, 1), nid(i + 1, j, 1), nid(i + 1, j + 1, 1), nid(i, j + 1, 1)))
                eid += 1
        f.write("$EndElements\n")
    return dict(n_nodes=2 * n2d, n_hex=n_hex, n_quads=n_quads,
                patch_faces={k: len(v) for k, v in patch_quads.items()})
