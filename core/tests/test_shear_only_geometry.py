import math
import numpy as np
import pytest
from lls_core.shear_only_geometry import (
    deskew_trig, pixel_step, shear_only_output_shape, level_angle,
    shear_only_forward_affine, shear_only_subblock_offset,
    shear_only_display_affine_zyx,
)


@pytest.mark.parametrize("bad", ["Z", "y", "", None])
def test_bad_skew_raises(bad):
    # The shared builder is the choke point for the display affine and the
    # sub-block offset; both must reject an unknown skew (guard preserved).
    with pytest.raises(ValueError):
        shear_only_forward_affine(45.0, 2.0, 1.04, 1.04, bad)
    with pytest.raises(ValueError):
        shear_only_subblock_offset(0, 0, 0, 45.0, 2.0, 1.04, 1.04, bad)
    with pytest.raises(ValueError):
        shear_only_display_affine_zyx((10, 8, 6), 45.0, 2.0, 1.04, 1.04, bad)

def test_trig_and_step():
    t, s, c = deskew_trig(30.0)
    assert math.isclose(t, math.tan(math.radians(30.0)))
    assert math.isclose(s, math.sin(math.radians(30.0)))
    assert math.isclose(c, math.cos(math.radians(30.0)))
    assert math.isclose(pixel_step(2.0, 1.04), 2.0 / 1.04)

# dy != dx in the anisotropic cases so the per-skew lateral selection is pinned:
# Y-skew must use step=dz/dy and X-skew step=dz/dx. With dy != dx the two skews'
# nz differ, and an accidental dy/dx swap in one skew would break these formulas.
@pytest.mark.parametrize("dy,dx", [(1.04, 1.04), (1.04, 0.9)])
def test_shear_only_shape_y_skew_formula(dy, dx):
    # Y-skew scan-driven formulas (oracle-validated), step = dz/dy:
    #   nz = ceil((n_scan-1)*sin*step) + 1
    #   ny = ceil((n_scan-1)*cos*step + (n_y-1)) + 1
    #   nx = n_x
    raw = (24, 70, 80)  # (scan/Z, Y, X)
    ang, dz = 45.0, 2.0
    _, s, c = deskew_trig(ang)
    step = pixel_step(dz, dy)   # Y-skew lateral is dy
    nz, ny, nx = shear_only_output_shape(raw, ang, dz, dy, dx, "Y")
    assert nz == int(math.ceil((raw[0] - 1) * s * step)) + 1
    assert ny == int(math.ceil((raw[0] - 1) * c * step + (raw[1] - 1))) + 1
    assert nx == raw[2]

@pytest.mark.parametrize("dy,dx", [(1.04, 1.04), (1.04, 0.9)])
def test_shear_only_shape_x_skew_transposes_roles(dy, dx):
    # X-skew scan-driven formulas (oracle-validated), step = dz/dx:
    #   nz = ceil((n_scan-1)*sin*step) + 1
    #   ny = n_y
    #   nx = ceil((n_scan-1)*cos*step + (n_x-1)) + 1
    raw = (24, 70, 80)
    ang, dz = 45.0, 2.0
    _, s, c = deskew_trig(ang)
    step = pixel_step(dz, dx)   # X-skew lateral is dx (NOT dy)
    nz, ny, nx = shear_only_output_shape(raw, ang, dz, dy, dx, "X")
    assert nz == int(math.ceil((raw[0] - 1) * s * step)) + 1   # scan drives axial
    assert ny == raw[1]                                          # raw Y passes through
    assert nx == int(math.ceil((raw[0] - 1) * c * step + (raw[2] - 1))) + 1


def test_shear_only_shape_anisotropic_axial_differs_per_skew():
    # Direct cross-check that the two skews select different lateral pixels:
    # with dy != dx, Y-skew (step=dz/dy) and X-skew (step=dz/dx) must yield
    # DIFFERENT axial extents. If both used the same lateral pixel these agree,
    # so this pins the per-skew selection at the shape level.
    raw = (24, 70, 80)
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 0.9
    nz_y = shear_only_output_shape(raw, ang, dz, dy, dx, "Y")[0]
    nz_x = shear_only_output_shape(raw, ang, dz, dy, dx, "X")[0]
    assert nz_y != nz_x

def test_level_angle_equals_deskew_angle():
    # Identity for both skew branches, at several angles.
    for skew in ("Y", "X"):
        for ang in (30.0, 45.0, 57.2):
            assert level_angle(ang, skew) == ang
