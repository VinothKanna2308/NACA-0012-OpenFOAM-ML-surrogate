import json
import math
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("mesh", "scripts", "ml"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import check_mesh  # noqa: E402
import foamcase  # noqa: E402
import meshgen  # noqa: E402
import mesh_study  # noqa: E402
import postprocess  # noqa: E402
import train_surrogate  # noqa: E402


@pytest.fixture(scope="module")
def coarse_msh(tmp_path_factory):
    d = tmp_path_factory.mktemp("msh")
    grid = meshgen.build_grid(meshgen.MeshParams(meshgen.LEVELS["coarse"]))
    path = str(d / "c.msh")
    meshgen.write_msh(grid, path)
    return grid, path


def test_mesh_counts_and_quality(coarse_msh):
    grid, _ = coarse_msh
    q = meshgen.quality(grid)
    assert grid.n_cells == 20736
    assert q["max_nonortho"] < 35
    assert q["h_wall_min"] < 1e-5


def test_mesh_topology(coarse_msh):
    _, path = coarse_msh
    res = check_mesh.check(path, verbose=False)
    assert res["problems"] == []
    assert res["min_volume"] > 0
    assert set(res["patches"]) == {"airfoil", "farfield", "outlet", "frontAndBack"}


def test_airfoil_is_sharp_naca0012():
    assert meshgen.half_thickness(0.0) == pytest.approx(0.0, abs=1e-12)
    assert meshgen.half_thickness(1.0) == pytest.approx(0.0, abs=1e-6)  # sharp TE
    assert meshgen.half_thickness(0.3) == pytest.approx(0.06, abs=1e-3)  # 12 % thick


def test_case_rendering(tmp_path):
    p = foamcase.render_case(str(tmp_path), alpha=10, level="coarse")
    for f in foamcase.LAYOUT.values():
        text = (tmp_path / f).read_text()
        assert "@@" not in text
        assert text.count("{") == text.count("}")
    assert float(p["nu"]) == pytest.approx(1 / 6e6)
    assert float(p["nut_inf"]) / float(p["nu"]) == pytest.approx(0.210438, rel=1e-4)
    assert "10" in (tmp_path / "case_params.txt").read_text()
    assert os.access(tmp_path / "Allrun", os.X_OK)


def test_lift_drag_directions_orthogonal():
    p = foamcase.params(10, 6e6, 100, 0.1)
    l = np.array(p["liftDir"].split(), float)
    d = np.array(p["dragDir"].split(), float)
    assert abs(l @ d) < 1e-12


def test_gci_recovers_second_order():
    r = 1.5
    f = [1.0 + 0.1 * h**2 for h in (1.0, r, r * r)]
    out = mesh_study.gci(*f, r, r)
    assert out["p"] == pytest.approx(2.0, abs=1e-4)
    assert out["f_ext"] == pytest.approx(1.0, abs=1e-6)


def test_force_coeff_parser(tmp_path):
    d = tmp_path / "postProcessing" / "forceCoeffs" / "0"
    d.mkdir(parents=True)
    lines = ["# Force coefficients", "# Time Cm Cd Cl Cl(f) Cl(r)"]
    lines += [f"{i} 0.01 0.0123 1.09 0 0" for i in range(1, 301)]
    (d / "forceCoeffs.dat").write_text("\n".join(lines))
    r = postprocess.converged_coeffs(str(tmp_path), n_avg=100)
    assert r["Cl"] == pytest.approx(1.09)
    assert r["Cd"] == pytest.approx(0.0123)
    assert r["iterations"] == 300


def test_patch_field_parser(tmp_path):
    f = tmp_path / "p"
    f.write_text("""boundaryField
{
    farfield { type freestreamPressure; value uniform 0; }
    airfoil
    {
        type zeroGradient;
        value nonuniform List<scalar> 3(0.5 -1.0 2.0);
    }
}""")
    v = postprocess.read_patch_field(str(f), "airfoil")
    assert list(v) == [0.5, -1.0, 2.0]


def test_surrogate_pipeline_runs():
    X, Y = train_surrogate.synthetic()
    res = train_surrogate.evaluate(X, Y)
    assert res["quadratic"]["Cl"]["R2_loo"] > 0.9
    assert res["mlp"]["Cl"]["R2_loo"] > 0.9
