"""
Tests for the single-pass deskewed MIP (lls_core.mip).

These verify the push/scatter MIP kernel against the ground-truth
"full deskew then max-project" using pyclesperanto, for both skew directions.
The push kernel is approximate (max-dilation splat), so we assert high
correlation and small mean error rather than exact equality.
"""
from __future__ import annotations

import numpy as np
import pytest
import pyclesperanto_prototype as cle

from lls_core.mip import deskew_mip, deskew_mip_from_lattice


def _ground_truth_mip(raw, func, theta, dz, dy, dx):
    full = np.asarray(cle.pull(func(
        raw, angle_in_degrees=theta, voxel_size_x=dx, voxel_size_y=dy, voxel_size_z=dz
    )))
    return full.max(axis=0)


@pytest.mark.parametrize("skew,func", [("Y", cle.deskew_y), ("X", cle.deskew_x)])
def test_deskew_mip_matches_full_deskew(skew, func):
    theta, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    raw[14:18, 40:65, 30:70] = 220

    gt = _ground_truth_mip(raw, func, theta, dz, dy, dx)
    mip = deskew_mip(raw, theta, dz, dy, dx, skew=skew)

    # Allow a 1px size difference from grid rounding; compare the overlap.
    h = min(gt.shape[0], mip.shape[0])
    w = min(gt.shape[1], mip.shape[1])
    a, b = gt[:h, :w], mip[:h, :w]
    denom = max(a.max(), 1.0)
    assert abs(gt.shape[0] - mip.shape[0]) <= 1
    assert abs(gt.shape[1] - mip.shape[1]) <= 1
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98
    assert np.abs(a - b).mean() / denom < 0.02


def test_deskew_mip_preserves_dtype():
    raw = np.zeros((10, 20, 20), dtype=np.uint16)
    raw[2:5, 5:15, 5:15] = 300
    mip = deskew_mip(raw, 30.0, 2.0, 1.0, 1.0, skew="Y")
    assert mip.dtype == np.uint16
    assert mip.max() == 300  # max preserved exactly for integer max-accumulation


def test_deskew_mip_is_2d_and_collapses_axial():
    raw = np.zeros((10, 20, 20), dtype=np.float32)
    raw[3:6, 8:12, 8:12] = 50
    mip = deskew_mip(raw, 30.0, 2.0, 1.0, 1.0, skew="Y")
    assert mip.ndim == 2


def test_deskew_mip_rejects_non_3d():
    with pytest.raises(ValueError):
        deskew_mip(np.zeros((4, 4)), 30.0, 1.0, 1.0, 1.0)


def test_deskew_mip_rejects_unknown_skew():
    with pytest.raises(ValueError):
        deskew_mip(np.zeros((4, 4, 4)), 30.0, 1.0, 1.0, 1.0, skew="Z")


@pytest.mark.parametrize("scan_scale_dz", [2.0, 5.0, 10.0])
def test_deskew_mip_no_striping_at_large_scan_scale(scan_scale_dz):
    # The bounded pull (gather) must be free of striping regardless of the
    # scan-step-to-pixel ratio dz/dy. A thin feature spanning all scan planes
    # projects to a continuous line with no interior holes (a scatter/push
    # leaves holes here once dz/dy is large).
    raw = np.zeros((30, 40, 20), dtype=np.float32)
    raw[:, 18:20, 8:12] = 100.0
    mip = deskew_mip(raw, 45.0, scan_scale_dz, 1.0, 1.04, skew="Y", interpolation="nearest")
    col = mip[:, 9]
    ys = np.where(col > 50)[0]
    assert ys.size > 0
    interior = col[ys.min():ys.max() + 1]
    assert (interior <= 50).sum() == 0  # no holes


def test_deskew_mip_linear_interpolation_matches_cle():
    theta, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    gt = _ground_truth_mip(raw, cle.deskew_y, theta, dz, dy, dx)
    mip = deskew_mip(raw, theta, dz, dy, dx, skew="Y", interpolation="linear", target_shape=gt.shape)
    assert mip.shape == gt.shape
    assert np.corrcoef(gt.ravel(), mip.ravel())[0, 1] > 0.98


def test_deskew_mip_rejects_bad_interpolation():
    with pytest.raises(ValueError):
        deskew_mip(np.zeros((4, 4, 4)), 30.0, 1.0, 1.0, 1.0, interpolation="cubic")


def test_deskew_mip_no_holes_in_feature():
    # A solid block must project to a solid (gap-free) region: the max-dilation
    # splat is specifically there to prevent striping/holes.
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[:, 20:40, 20:40] = 100.0  # spans all scan planes
    mip = deskew_mip(raw, 45.0, 2.0, 1.04, 1.04, skew="Y")
    # Within the bright region's bounding box there should be no zero holes
    ys, xs = np.where(mip > 50)
    assert ys.size > 0
    sub = mip[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # Allow a thin border of partial pixels; require the interior to be hole-free
    interior = sub[1:-1, 1:-1]
    if interior.size:
        assert (interior > 50).mean() > 0.95


def test_deskew_mip_from_lattice():
    import tempfile
    from xarray import DataArray
    from lls_core.models.lattice_data import LatticeData

    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(2.0, 1.04, 1.04),
            angle=45,
            save_name="t",
            save_dir=tmpdir,
        )
        mip = deskew_mip_from_lattice(lattice)
        # The MIP grid must be pinned exactly to the lattice's deskewed shape
        # (collapsing the deskewed Z axis), so ROIs land on the crop grid.
        zd, yd, xd = lattice.derived.deskew_vol_shape
        assert mip.shape == (yd, xd)

    gt = _ground_truth_mip(raw, cle.deskew_y, 45.0, 2.0, 1.04, 1.04)
    h = min(gt.shape[0], mip.shape[0]); w = min(gt.shape[1], mip.shape[1])
    assert np.corrcoef(gt[:h, :w].ravel(), mip[:h, :w].ravel())[0, 1] > 0.98


def test_save_mip_writes_2d_via_existing_writer():
    # save_mip routes through the normal writer machinery as a singleton-Z slice,
    # producing a 2D MIP whose lateral shape equals the deskewed shape.
    import tempfile
    import tifffile
    from pathlib import Path
    from xarray import DataArray
    from lls_core.models.lattice_data import LatticeData

    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(2.0, 1.04, 1.04),
            angle=45,
            save_name="mip_test",
            save_dir=tmpdir,
            save_type="tiff",
            save_mip=True,
        )
        lattice.save()
        files = list(Path(tmpdir).glob("*.tif"))
        assert len(files) == 1, files
        img = np.squeeze(tifffile.imread(str(files[0])))
        zd, yd, xd = lattice.derived.deskew_vol_shape
        assert img.shape == (yd, xd)
        assert img.max() > 0  # the bright feature survived the projection


def test_save_mip_ignores_crop():
    # MIP is whole-FOV: enabling crop must not change the MIP output shape.
    import tempfile
    import tifffile
    from pathlib import Path
    from xarray import DataArray
    from lls_core.models.crop import CropParams
    from lls_core.models.lattice_data import LatticeData

    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(2.0, 1.04, 1.04),
            angle=45,
            save_name="mip_crop",
            save_dir=tmpdir,
            save_type="tiff",
            save_mip=True,
            crop=CropParams(roi_list=[[[0, 0], [0, 20], [20, 20], [20, 0]]], z_range=(0, 10)),
        )
        lattice.save()
        img = np.squeeze(tifffile.imread(str(next(Path(tmpdir).glob("*.tif")))))
        zd, yd, xd = lattice.derived.deskew_vol_shape
        assert img.shape == (yd, xd)  # whole-FOV, not the cropped ROI


@pytest.mark.parametrize("skew,func", [("Y", cle.deskew_y), ("X", cle.deskew_x)])
def test_deskew_mip_target_shape_pins_grid_to_cle(skew, func):
    theta, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    raw[14:18, 40:65, 30:70] = 220

    gt = _ground_truth_mip(raw, func, theta, dz, dy, dx)
    mip = deskew_mip(raw, theta, dz, dy, dx, skew=skew, target_shape=gt.shape)

    # Exact dimensional match to the reference deskew, and aligned content
    assert mip.shape == gt.shape
    denom = max(gt.max(), 1.0)
    assert np.abs(gt - mip).mean() / denom < 0.02

    # Bright-feature centroid must coincide within a sub-pixel
    def centroid(m):
        ys, xs = np.where(m > 0.5 * m.max())
        return np.array([ys.mean(), xs.mean()])
    assert np.allclose(centroid(gt), centroid(mip), atol=1.0)
