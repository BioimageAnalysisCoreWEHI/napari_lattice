"""Regression test for crop_volume_deskew ROI placement (objective / stock deskew).

ROI crop must match the same region sliced from the full deskewed volume at off-centre
positions along the shear axis. 
This catches 

(a) the off-centre trim bug -- invisible on
narrow FOV, growing with the lateral field width and largest off-centre -- and 

(b) the X-skew skew_dir threading fix (skew_dir was previously defaulted to Y inside _process_crop,
so an X-skew crop landed at the wrong X). One parametrized test covers both skews; the centre
position is omitted because low/high are strictly more discriminating for the trim bug.
"""
import numpy as np
import pytest
import pyclesperanto as cle

from lls_core.llsz_core import crop_volume_deskew
from lls_core import DeskewDirection
from tests.utils import requires_real_gpu


def _raw_wide(skew):
    # Uniform background + off-centre markers, wide along the shear axis so every ROI
    # position contains discriminating structure (guarded below via gt.std()).
    if skew == "Y":
        r = np.full((150, 400, 120), 30.0, np.float32)   # wide Y (shear axis)
        for z, a in [(30, 80), (75, 200), (120, 330)]:
            r[z - 3:z + 3, a - 6:a + 6, 55:65] = 800.0
    else:
        r = np.full((150, 120, 400), 30.0, np.float32)   # wide X (shear axis)
        for z, a in [(30, 80), (75, 200), (120, 330)]:
            r[z - 3:z + 3, 55:65, a - 6:a + 6] = 800.0
    return r


@requires_real_gpu
@pytest.mark.parametrize("skew", ["Y", "X"])
@pytest.mark.parametrize("angle", [30.0, 45.0])
@pytest.mark.parametrize("pos", ["low", "high"])   # off-centre: where the trim bug is largest
def test_crop_placement_objective(skew, angle, pos):
    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X
    deskew = cle.deskew_y if skew == "Y" else cle.deskew_x
    raw = _raw_wide(skew)
    full = np.asarray(cle.pull(deskew(
        raw, angle=angle, voxel_size_x=1, voxel_size_y=1, voxel_size_z=1)))

    ax = 1 if skew == "Y" else 2          # sheared (wide) output axis
    other = 2 if skew == "Y" else 1
    H = 60
    frac = {"low": 0.2, "high": 0.8}[pos]
    a0 = int(np.clip(frac * full.shape[ax] - H / 2, 0, full.shape[ax] - H)); a1 = a0 + H
    b0 = full.shape[other] // 2 - H // 2; b1 = b0 + H

    # ROI corners are [y, x]: moving window on the shear axis, fixed on the other axis.
    if skew == "Y":
        roi = [[a0, b0], [a0, b1], [a1, b1], [a1, b0]]
        gt = full[:, a0:a1, b0:b1]
    else:
        roi = [[b0, a0], [b0, a1], [b1, a1], [b1, a0]]
        gt = full[:, b0:b1, a0:a1]

    assert gt.std() > 1.0                 # ROI must contain discriminating content, else vacuous

    crop = np.asarray(crop_volume_deskew(
        original_volume=raw, roi_shape=[roi], angle_in_degrees=angle,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1,
        z_start=0, z_end=full.shape[0], skew_dir=skew_dir)).astype(np.float32)

    qm = min(crop.shape[0], gt.shape[0])          # deskew Z-depth can differ by 1
    assert crop.shape[1:] == gt.shape[1:], f"shape {crop.shape[1:]} != {gt.shape[1:]} (skew={skew})"
    assert np.allclose(crop[:qm], gt[:qm], atol=1e-2), f"crop data mismatch (skew={skew}, {pos}, {angle})"

    # Placement: the best-matching window offset along the shear axis must be a0.
    w = crop.shape[ax]
    def window(k):
        return full[:qm, k:k + w, b0:b1] if skew == "Y" else full[:qm, b0:b1, k:k + w]
    errs = [float(np.mean(np.abs(window(k) - crop[:qm]))) for k in range(full.shape[ax] - w + 1)]
    assert abs(int(np.argmin(errs)) - a0) <= 1, (
        f"crop placed at wrong {'YX'[ax - 1]} (argmin={np.argmin(errs)}, expected={a0}, "
        f"skew={skew}, pos={pos}, angle={angle})")
