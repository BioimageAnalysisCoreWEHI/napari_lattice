"""Regression test for crop_volume_deskew ROI placement.

Verifies that ROI crops match the same region sliced from the full deskewed
volume at low, centre, and high positions, catching the off-centre trim bug.

The error is invisible on narrow-FOV data (e.g.
the 5x5x5 volume in ``test_crop_deskew.py``, which is firmly in the near-zero
regime) but grows with the lateral field width.
"""
import numpy as np
import pytest
import pyclesperanto_prototype as cle

from lls_core.llsz_core import crop_volume_deskew
from lls_core import DeskewDirection


def _raw():
    # Uniform background + a few off-centre markers. The deskew shear spreads each
    # marker across the skew axis, so every ROI position contains discriminating
    # structure (guarded below via gt.std()).
    r = np.full((150, 400, 120), 30.0, np.float32)
    for z, y in [(30, 80), (75, 200), (120, 330)]:
        r[z - 3:z + 3, y - 6:y + 6, 55:65] = 800.0
    return r


@pytest.mark.parametrize("angle", [30.0, 45.0])
@pytest.mark.parametrize("pos", ["low", "centre", "high"])  # low/high are far from centre
def test_crop_placement(angle, pos):
    raw = _raw()
    full = np.asarray(cle.pull(cle.deskew_y(
        raw, angle_in_degrees=angle,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1)))

    H = 60
    frac = {"low": 0.2, "centre": 0.5, "high": 0.8}[pos]
    y0 = int(np.clip(frac * full.shape[1] - H / 2, 0, full.shape[1] - H))
    y1 = y0 + H
    x0 = full.shape[2] // 2 - H // 2
    x1 = x0 + H
    roi = [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]
    gt = full[:, y0:y1, x0:x1]

    # Guard: the ROI must contain varying content, otherwise the comparison below
    # would be vacuous (a mis-placed crop could match a uniform background).
    assert gt.std() > 1.0

    crop = np.asarray(crop_volume_deskew(
        original_volume=raw, roi_shape=[roi], angle_in_degrees=angle,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1,
        z_start=0, z_end=full.shape[0], skew_dir=DeskewDirection.Y)).astype(np.float32)

    qm = min(crop.shape[0], gt.shape[0])           # deskew Z-depth can differ by 1
    assert crop.shape[1:] == gt.shape[1:]          # right size (catches empty/short crops)
    assert np.allclose(crop[:qm], gt[:qm], atol=1e-2)   # right place + right data

    # Stronger placement check: the crop must originate from rows [y0:y1] of the
    # full deskew, not elsewhere. This is what fails badly for off-centre ROIs.
    h = crop.shape[1]
    errs = [float(np.mean(np.abs(full[:qm, k:k + h, x0:x1] - crop[:qm])))
            for k in range(full.shape[1] - h + 1)]
    assert abs(int(np.argmin(errs)) - y0) <= 1


def _raw_xskew():
    # For X-skew objective: markers spread along the X-shear axis.
    # Volume is (scan=150, Y=120, X=400) so X is wide enough to test off-centre placement.
    r = np.full((150, 120, 400), 30.0, np.float32)
    for z, x in [(30, 80), (75, 200), (120, 330)]:
        r[z - 3:z + 3, 55:65, x - 6:x + 6] = 800.0
    return r


@pytest.mark.parametrize("angle", [30.0, 45.0])
@pytest.mark.parametrize("pos", ["low", "centre", "high"])
def test_crop_placement_x_skew_objective(angle, pos):
    """Regression: X-skew OBJECTIVE crop (coverslip_rotation=True, stock deskew) must land at
    the correct X position in the deskewed volume.  `skew_dir` was previously
    defaulted to Y inside _process_crop; it is now threaded correctly
    (`skew_dir=self.skew_dir`).  This test locks in that fix for the objective path.

    Mirrors test_crop_placement but along the X-shear axis using cle.deskew_x
    as ground truth.
    """
    raw = _raw_xskew()
    full = np.asarray(cle.pull(cle.deskew_x(
        raw, angle_in_degrees=angle,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1)))

    H = 60
    frac = {"low": 0.2, "centre": 0.5, "high": 0.8}[pos]
    x0 = int(np.clip(frac * full.shape[2] - H / 2, 0, full.shape[2] - H))
    x1 = x0 + H
    y0 = full.shape[1] // 2 - H // 2
    y1 = y0 + H
    roi = [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]
    gt = full[:, y0:y1, x0:x1]

    # Guard: ROI must contain discriminating content.
    assert gt.std() > 1.0

    crop = np.asarray(crop_volume_deskew(
        original_volume=raw, roi_shape=[roi], angle_in_degrees=angle,
        voxel_size_x=1, voxel_size_y=1, voxel_size_z=1,
        z_start=0, z_end=full.shape[0],
        skew_dir=DeskewDirection.X)).astype(np.float32)

    qm = min(crop.shape[0], gt.shape[0])
    assert crop.shape[1:] == gt.shape[1:], (
        f"Shape mismatch: crop={crop.shape[1:]}, gt={gt.shape[1:]} (angle={angle}, pos={pos})")
    assert np.allclose(crop[:qm], gt[:qm], atol=1e-2), (
        f"X-skew objective crop data mismatch (angle={angle}, pos={pos})")

    # Placement check: best-matching column offset must be x0.
    w = crop.shape[2]
    errs = [float(np.mean(np.abs(full[:qm, y0:y1, k:k + w] - crop[:qm])))
            for k in range(full.shape[2] - w + 1)]
    assert abs(int(np.argmin(errs)) - x0) <= 1, (
        f"X-skew crop placed at wrong X (argmin={np.argmin(errs)}, expected={x0}, "
        f"angle={angle}, pos={pos})")
