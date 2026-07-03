import math as _math
import tempfile

import numpy as np
import pytest
import pyclesperanto_prototype as cle
from xarray import DataArray

from lls_core import DeskewDirection
from lls_core.models.crop import CropParams
from lls_core.models.deskew import DeskewParams
from lls_core.models.lattice_data import LatticeData
from lls_core.shear_only_geometry import (
    deskew_trig, pixel_step, shear_only_output_shape, shear_only_display_affine_zyx,
)
from tests.reference_deskew import numba_deskew, two_pass_shear_only, tilt_residual_slope, _tight_crop


def _feature_volume():
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    raw[14:18, 40:65, 30:70] = 220
    return raw


def test_numba_objective_matches_cle_deskew_y():
    raw = _feature_volume()
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    gt = np.asarray(cle.pull(cle.deskew_y(
        raw, angle_in_degrees=ang, voxel_size_x=dx, voxel_size_y=dy, voxel_size_z=dz)))
    nb = numba_deskew(raw, ang, dz, dy, dx, skew="Y", frame="objective",
                      out_shape_zyx=gt.shape)
    assert nb.shape == gt.shape
    # Compare per-Z slice max-projection to avoid tiny per-voxel edge diffs
    denom = max(gt.max(), 1.0)
    assert np.abs(nb - gt).mean() / denom < 0.02
    assert np.corrcoef(nb.ravel(), gt.ravel())[0, 1] > 0.98


def _centroid(vol):
    idx = np.argwhere(vol > 0.5 * vol.max())
    return idx.mean(0)


def test_numba_shear_only_matches_two_pass_ground_truth():
    raw = _feature_volume()
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    gt = two_pass_shear_only(raw, ang, dz, dy, dx, skew="Y")
    shape = shear_only_output_shape(raw.shape, ang, dz, dy, dx, "Y")
    nb_full = numba_deskew(raw, ang, dz, dy, dx, skew="Y", frame="shear_only",
                           out_shape_zyx=shape)
    # Tight-crop the numba full-box output (vol > 0) before computing its
    # normalised centroid, so we compare content-to-content with the already-
    # cropped oracle (two_pass_shear_only tight-crops internally).
    nb = _tight_crop(nb_full)
    # Content alignment: the bright features must sit at the same relative
    # position (normalised centroids) despite different bounding boxes.
    cn = _centroid(nb) / np.array(nb.shape)
    cg = _centroid(gt) / np.array(gt.shape)
    np.testing.assert_allclose(cn, cg, atol=0.06)


def _sheared_axis_view(vol, skew):
    """tilt_residual_slope only measures drift along output-Y. For X-skew the
    tilt lives along X, so swap Y<->X to put the sheared axis under the metric.
    Without this the X-skew level test is vacuous (slope is 0 for any result)."""
    return np.swapaxes(vol, 1, 2) if skew == "X" else vol


@pytest.mark.parametrize("skew", ["Y", "X"])
def test_numba_shear_only_both_skews_level(skew):
    # Off-centre COMPACT blob, narrow in the sheared lateral axis and spanning
    # only mid scan planes, with genuine tilt-detection power: if the shear is
    # NOT removed (objective frame) the structure's centroid drifts monotonically
    # along the sheared axis with output-Z (slope ~0.15), whereas correct
    # shear-only leveling gives slope ~0.
    # NOTE: "level" here assumes OPM acquisition geometry (coverslip_rotation=False);
    # for Zeiss LLS the objective/True path is the coverslip-level one instead.
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    if skew == "Y":
        raw[8:22, 18:24, 28:34] = 300.0   # narrow in raw-Y (sheared), off-centre
    else:
        raw[8:22, 28:34, 18:24] = 300.0   # narrow in raw-X (sheared), off-centre
    shape = shear_only_output_shape(raw.shape, ang, dz, dy, dx, skew)

    level = numba_deskew(raw, ang, dz, dy, dx, skew=skew, frame="shear_only",
                         out_shape_zyx=shape)
    slope = abs(tilt_residual_slope(_sheared_axis_view(level, skew)))
    assert slope < 0.06, f"shear-only result is not level (skew={skew}): slope={slope:.4f}"

    # Guard against a vacuous pass: on the SAME content the un-leveled objective
    # frame must show a clearly larger residual, i.e. the metric can detect tilt
    # for this content/skew (measured ~0.12; require it to exceed the 0.06 gate).
    tilted = numba_deskew(raw, ang, dz, dy, dx, skew=skew, frame="objective",
                          out_shape_zyx=shape)
    tilted_slope = abs(tilt_residual_slope(_sheared_axis_view(tilted, skew)))
    assert tilted_slope > 0.06, (
        f"tilt metric has no power for this content (skew={skew}): "
        f"objective-frame slope={tilted_slope:.4f} did not exceed the 0.06 gate")


@pytest.mark.parametrize("skew", ["Y", "X"])
@pytest.mark.parametrize("dy,dx", [
    (1.04, 1.04),        # isotropic
    (1.04, 0.9),         # anisotropic: pins the per-skew lateral selection
])                        #   (Y uses step=dz/dy, X uses step=dz/dx) end-to-end
def test_opencl_shear_only_matches_numba(skew, dy, dx):
    """OpenCL single-pass kernel must match the frozen numba shear-only map.

    The anisotropic (dy != dx) case pins the kernel's lateral-axis selection:
    a Y/X dy/dx swap would change nz (via step) and break parity for one skew.
    """
    from lls_core.shear_only_deskew import shear_only_deskew

    raw = _feature_volume()
    ang, dz = 45.0, 2.0
    shape = shear_only_output_shape(raw.shape, ang, dz, dy, dx, skew)
    nb = numba_deskew(raw, ang, dz, dy, dx, skew=skew, frame="shear_only",
                      out_shape_zyx=shape)
    gpu = np.asarray(cle.pull(shear_only_deskew(raw, ang, dz, dy, dx, skew=skew)))
    assert gpu.shape == nb.shape == shape, (
        f"Shape mismatch: gpu={gpu.shape}, nb={nb.shape}, expected={shape}"
    )
    denom = max(float(nb.max()), 1.0)
    mean_err = float(np.abs(gpu - nb).mean()) / denom
    corr = float(np.corrcoef(gpu.ravel(), nb.ravel())[0, 1])
    assert mean_err < 0.02, f"mean_err={mean_err:.4f} >= 0.02 (skew={skew})"
    assert corr > 0.98, f"corr={corr:.4f} <= 0.98 (skew={skew})"


def test_deskew_params_shear_only_shape_and_default():
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    da = DataArray(raw, dims=["Z", "Y", "X"])
    default = DeskewParams(input_image=da, physical_pixel_sizes=(2.0, 1.04, 1.04), angle=45)
    cover = DeskewParams(input_image=da, physical_pixel_sizes=(2.0, 1.04, 1.04),
                         angle=45, coverslip_rotation=False)
    # default is stock deskew (Zeiss LLS) and coverslip_rotation defaults True
    assert default.coverslip_rotation is True
    # shear-only derived shape equals shear_only_output_shape
    assert tuple(cover.derived.deskew_vol_shape) == shear_only_output_shape(
        (24, 70, 80), 45, 2.0, 1.04, 1.04, "Y")
    # the two shapes differ (shear-only is a different box)
    assert tuple(cover.derived.deskew_vol_shape) != tuple(default.derived.deskew_vol_shape)


def test_non_crop_shear_only_is_level():
    """With coverslip_rotation=False (OPM/SOPi geometry), _process_non_crop must return a ZYX
    array whose shape equals shear_only_output_shape and that is coverslip-level (tilt
    residual slope ~0).  NOTE: "level" here assumes OPM acquisition geometry; for Zeiss LLS
    it is coverslip_rotation=True (the objective path) that is the coverslip-level one."""
    # Full-scan slab: strong objective tilt (~0.23 through the pipeline), so the gate below
    # is non-vacuous — the un-leveled result would clearly fail it (see guard).
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[:, 24:36, 20:40] = 200.0

    def _pipeline(coverslip_rotation: bool):
        with tempfile.TemporaryDirectory() as d:
            lat = LatticeData(input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                              physical_pixel_sizes=(2.0, 1.04, 1.04), angle=45,
                              coverslip_rotation=coverslip_rotation, save_name="t", save_dir=d)
            return np.asarray(next(iter(lat.process().slices)).data)

    out = _pipeline(False)
    assert abs(tilt_residual_slope(out)) < 0.06
    assert out.shape == shear_only_output_shape((30, 60, 60), 45, 2.0, 1.04, 1.04, "Y")

    # Anti-vacuous guard: the un-leveled objective result (coverslip_rotation=True) must
    # clearly exceed the gate, proving the metric has tilt-detection power for this content.
    tilted = _pipeline(True)
    assert abs(tilt_residual_slope(tilted)) > 0.06


def _shear_only_forward(p, yr, xr, angle_deg, dz, dy, dx, skew):
    """Frozen forward map raw (scan p, raw_y yr, raw_x xr) -> shear-only/coverslip (zc, yc, xc)."""
    _, sin_t, cos_t = deskew_trig(angle_deg)
    if skew == "Y":
        step = pixel_step(dz, dy)
        return sin_t * step * p, cos_t * step * p + yr, xr
    else:  # X-skew: X is the sheared axis, Y passes through
        step = pixel_step(dz, dx)
        return sin_t * step * p, yr, cos_t * step * p + xr


@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_crop_roundtrip_places_feature(skew):
    """A bright blob at a KNOWN raw location maps (via the frozen forward map) to a
    known shear-only/coverslip location. An ROI drawn AROUND that location must crop
    to a volume that CONTAINS the blob near the output centre; a CONTROL ROI far
    away must NOT capture it. This proves the crop localizes to the ROI, not just
    that "something survived"."""
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[10:14, 28:34, 26:32] = 500.0
    # blob raw centroid (voxel-centre convention)
    p_c, yr_c, xr_c = 11.5, 30.5, 28.5
    zc, yc, xc = _shear_only_forward(p_c, yr_c, xr_c, ang, dz, dy, dx, skew)

    nz, ny, nx = shear_only_output_shape((30, 60, 60), ang, dz, dy, dx, skew)
    hw = 12

    def make_roi(cy, cx):
        return [[[cy - hw, cx - hw], [cy - hw, cx + hw],
                 [cy + hw, cx + hw], [cy + hw, cx - hw]]]

    z0 = int(max(round(zc - hw), 0))
    z1 = int(min(round(zc + hw), nz))
    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X

    def run(roi):
        with tempfile.TemporaryDirectory() as d:
            lat = LatticeData(input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                              physical_pixel_sizes=(dz, dy, dx), angle=ang,
                              skew=skew_dir, coverslip_rotation=False,
                              crop=CropParams(roi_list=roi, z_range=(z0, z1)),
                              save_name="t", save_dir=d)
            return np.asarray(next(iter(lat.process().slices)).data)

    # (a) ROI centred on the blob's coverslip location: blob captured near centre
    on = run(make_roi(yc, xc))
    assert on.size > 0 and on.max() > 0, "on-target crop is empty"
    idx = np.argwhere(on > 0.5 * on.max())
    centroid = idx.mean(0)
    centre = (np.array(on.shape) - 1) / 2.0
    dz_off, dy_off, dx_off = np.abs(centroid - centre)
    assert dz_off <= 3 and dy_off <= 3 and dx_off <= 3, (
        f"blob not centred (skew={skew}): centroid={centroid}, centre={centre}")

    # (b) CONTROL ROI far from the blob (along the sheared axis): blob NOT captured
    if skew == "Y":
        ctrl = make_roi(yc + 35, xc)
    else:
        ctrl = make_roi(yc, xc + 35)
    off = run(ctrl)
    assert off.max() < 0.01 * on.max(), (
        f"control crop captured the blob (skew={skew}): off.max={off.max()}, on.max={on.max()}")


@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_crop_edge_roi_aligned(skew):
    """Regression: ROI touching the shear-only frame Z=0 edge (z0=0).

    When the raw sub-block halo pushes scan_start > 0, shear_only_subblock_offset
    returns off_zc > 0. This means z0 - off_zc < 0 — an edge ROI situation where
    the old _trim() silently clamped to 0, slicing from the sub-block's own origin
    and returning misaligned data.

    Fix: zero-pad the leading edge so the returned volume aligns to the ROI origin.
    Assertion: blob at a KNOWN shear-only/coverslip location maps to the correct position in
    a crop whose ROI starts at z0=0 (coverslip frame bottom edge).
    """
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    # Place a bright blob at raw scan plane 0–3 so it maps to coverslip Z near 0.
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[0:4, 28:34, 26:32] = 500.0
    # Blob raw centroid (first few scan planes)
    p_c, yr_c, xr_c = 1.5, 30.5, 28.5
    zc, yc, xc = _shear_only_forward(p_c, yr_c, xr_c, ang, dz, dy, dx, skew)

    nz, ny, nx = shear_only_output_shape((30, 60, 60), ang, dz, dy, dx, skew)
    hw = 8

    # ROI starting at z0=0 (coverslip edge): this forces the leading-edge trim path.
    # (yc, xc) already differ by skew via _shear_only_forward, so one ROI serves both.
    z0_roi = 0
    z1_roi = int(min(round(zc + hw), nz))
    roi = [[[yc - hw, xc - hw], [yc - hw, xc + hw],
            [yc + hw, xc + hw], [yc + hw, xc - hw]]]

    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X
    with tempfile.TemporaryDirectory() as d:
        lat = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(dz, dy, dx), angle=ang,
            skew=skew_dir, coverslip_rotation=False,
            crop=CropParams(roi_list=roi, z_range=(z0_roi, z1_roi)),
            save_name="t", save_dir=d,
        )
        out = np.asarray(next(iter(lat.process().slices)).data)

    # The blob must appear in the output (not silently zeroed out by a wrong slice).
    assert out.size > 0, "edge-ROI crop returned empty volume"
    assert out.max() > 0, (
        f"edge-ROI crop is all zeros (skew={skew}): blob at shear-only zc={zc:.2f} "
        f"not found in z_range=[{z0_roi},{z1_roi}] crop of shape {out.shape}")

    # The blob should be near the BOTTOM of the z-extent (low z-index) since z0_roi=0.
    idx = np.argwhere(out > 0.5 * out.max())
    blob_z = idx[:, 0].mean()
    # Blob coverslip z is zc; crop starts at z0_roi=0; so blob_z ≈ zc - z0_roi = zc.
    # Allow up to 3 px tolerance.
    expected_z = zc - z0_roi
    assert abs(blob_z - expected_z) <= 3, (
        f"edge-ROI blob at wrong z (skew={skew}): got blob_z={blob_z:.1f}, "
        f"expected≈{expected_z:.1f} (zc={zc:.2f}, z0_roi={z0_roi})")


# ---------------------------------------------------------------------------
# DeskewParams.new_dz reports correct Z voxel spacing
# ---------------------------------------------------------------------------
# Shear-only new_dz is the sheared lateral pixel (dy for Y, dx for X); objective
# stays sin(angle)*dz. dy != dx so a Y/X lateral swap would flip 1.04 <-> 0.9.
@pytest.mark.parametrize("skew,coverslip_rotation,expected_dz", [
    ("Y", False, 1.04),                                # shear-only Y -> dy
    ("X", False, 0.9),                                 # shear-only X -> dx
    ("Y", True, _math.sin(_math.radians(45.0)) * 2.0),  # objective -> sin*dz
])
def test_new_dz(skew, coverslip_rotation, expected_dz):
    da = DataArray(np.zeros((5, 10, 12), dtype=np.float32), dims=["Z", "Y", "X"])
    params = DeskewParams(
        input_image=da,
        physical_pixel_sizes=(2.0, 1.04, 0.9),
        angle=45,
        skew=skew,
        coverslip_rotation=coverslip_rotation,
    )
    assert params.new_dz == pytest.approx(expected_dz), (
        f"skew={skew}, coverslip_rotation={coverslip_rotation}: "
        f"expected new_dz={expected_dz}, got {params.new_dz}")


# ---------------------------------------------------------------------------
# Quick Deskew display affine: shear_only_display_affine_zyx tests
# ---------------------------------------------------------------------------

def _ss_cc_display(angle_deg, dz, d_lateral):
    angle_rad = _math.radians(angle_deg)
    step = pixel_step(dz, d_lateral)
    return _math.sin(angle_rad) * step, _math.cos(angle_rad) * step


@pytest.mark.parametrize("z,y,x", [(0, 0, 0), (1, 0, 0), (3, 2, 4), (0, 5, 7)])
def test_display_affine_y_skew(z, y, x):
    """For Y-skew: M @ [z,y,x,1] == [ss*z, cc*z + y, x, 1]."""
    angle_deg = 30.0
    dz, dy, dx = 0.3, 0.15, 0.15
    raw_shape_zyx = (10, 8, 6)

    M = shear_only_display_affine_zyx(raw_shape_zyx, angle_deg, dz, dy, dx, skew="Y")
    ss, cc = _ss_cc_display(angle_deg, dz, dy)

    result = M @ np.array([z, y, x, 1.0])
    expected = np.array([ss * z, cc * z + y, x, 1.0])
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("z,y,x", [(0, 0, 0), (1, 0, 0), (3, 2, 4), (0, 5, 7)])
def test_display_affine_x_skew(z, y, x):
    """For X-skew: M @ [z,y,x,1] == [ss*z, y, cc*z + x, 1]."""
    angle_deg = 45.0
    dz, dy, dx = 0.3, 0.15, 0.15
    raw_shape_zyx = (10, 8, 6)

    M = shear_only_display_affine_zyx(raw_shape_zyx, angle_deg, dz, dy, dx, skew="X")
    ss, cc = _ss_cc_display(angle_deg, dz, dx)

    result = M @ np.array([z, y, x, 1.0])
    expected = np.array([ss * z, y, cc * z + x, 1.0])
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("skew", ["Y", "X"])
@pytest.mark.parametrize("z,y,x", [(0, 0, 0), (1, 2, 3), (4, 0, 5)])
def test_display_affine_invert_scan_fold(skew, z, y, x):
    """With invert=True, mapping [z,y,x,1] equals invert=False applied to [nz-1-z, y, x, 1]."""
    angle_deg = 30.0
    dz, dy, dx = 0.3, 0.15, 0.15
    nz = 10
    raw_shape_zyx = (nz, 8, 6)

    M_normal = shear_only_display_affine_zyx(
        raw_shape_zyx, angle_deg, dz, dy, dx, skew=skew, invert_scan_direction=False
    )
    M_inv = shear_only_display_affine_zyx(
        raw_shape_zyx, angle_deg, dz, dy, dx, skew=skew, invert_scan_direction=True
    )

    result = M_inv @ np.array([z, y, x, 1.0])
    reference = M_normal @ np.array([nz - 1 - z, y, x, 1.0])
    np.testing.assert_allclose(result, reference, atol=1e-12)


def test_objective_display_transform_differs_under_coverslip_flag():
    """The coverslip flag must actually reach the display transform: a
    coverslip_rotation=False model must yield a DIFFERENT deskew_affine_transform_zyx
    (the shear-only one) than the coverslip_rotation=True (objective) model.

    The objective path's byte-identity to stock deskew is gated separately by
    test_non_crop_default_byte_identical (test_shear_only_backcompat.py).
    """
    raw = DataArray(np.zeros((5, 5, 5), dtype=np.float32), dims=["Z", "Y", "X"])

    obj = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), coverslip_rotation=True)
    cover = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), coverslip_rotation=False)

    m_obj = np.asarray(obj.derived.deskew_affine_transform_zyx)
    m_cover = np.asarray(cover.derived.deskew_affine_transform_zyx)

    assert not np.allclose(m_obj, m_cover), (
        "coverslip_rotation=False must produce a different display affine than True"
    )
