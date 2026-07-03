"""Shear-only (coverslip-frame) fast MIP tests (Task 8).

Verifies that deskew_mip(..., frame="shear_only") produces a 2-D projection
that matches shear_only_deskew(...).max(axis=0) for both skew directions,
and that the default objective behaviour is byte-identical to the pre-existing
call signature (back-compat gate).
"""
import numpy as np
import pytest
import pyclesperanto_prototype as cle
from lls_core.mip import deskew_mip
from lls_core.shear_only_deskew import shear_only_deskew


@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_mip_matches_full_shear_only_deskew(skew):
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    raw[14:18, 40:65, 30:70] = 220
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    full = np.asarray(cle.pull(shear_only_deskew(raw, ang, dz, dy, dx, skew=skew)))
    gt = full.max(axis=0)  # collapse coverslip-normal axis
    mip = deskew_mip(raw, ang, dz, dy, dx, skew=skew, frame="shear_only",
                     target_shape=gt.shape)
    assert mip.shape == gt.shape
    denom = max(gt.max(), 1.0)
    assert np.abs(gt - mip).mean() / denom < 0.03
    assert np.corrcoef(gt.ravel(), mip.ravel())[0, 1] > 0.97


def test_objective_mip_unchanged():
    # frame defaults to objective and equals the pre-existing behaviour
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    a = deskew_mip(raw, 45, 2.0, 1.04, 1.04, skew="Y")
    b = deskew_mip(raw, 45, 2.0, 1.04, 1.04, skew="Y", frame="objective")
    np.testing.assert_array_equal(a, b)
