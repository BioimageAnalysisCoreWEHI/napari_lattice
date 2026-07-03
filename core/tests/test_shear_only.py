"""Consolidated shear-only (coverslip-frame / OPM) deskew test suite.

Merges the former test_shear_only_{geometry,deskew,mip,backcompat}.py. Sections:
  - GEOMETRY            pure-math unit tests (no GPU)
  - KERNEL PARITY       numba reference vs cle / OpenCL kernel (GPU)
  - MODEL               DeskewParams shape/new_dz/display-affine (no GPU)
  - PIPELINE (non-crop) shear-only leveling through LatticeData (GPU)
  - PIPELINE (crop)     crop ROI mapping, edge/zero-pad, decon, invert-scan (GPU)
  - MIP                 fast shear-only MIP (numba)
  - BACKCOMPAT          coverslip_rotation=True byte-identical to stock deskew (GPU)

Metrics/reference kernels live in tests/reference_deskew.py.
"""
import math
import tempfile

import numpy as np
import pytest
import pyclesperanto_prototype as cle
from xarray import DataArray

from lls_core import DeskewDirection
from lls_core.models.crop import CropParams
from lls_core.models.deskew import DeskewParams
from lls_core.models.lattice_data import LatticeData
from lls_core.mip import deskew_mip
from lls_core.shear_only_deskew import shear_only_deskew
from lls_core.shear_only_geometry import (
    deskew_trig, pixel_step, shear_only_output_shape, level_angle,
    shear_only_forward_affine, shear_only_subblock_offset,
    shear_only_display_affine_zyx,
)
from tests.reference_deskew import numba_deskew, two_pass_shear_only, _tight_crop, yz_slab_angle


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _feature_volume():
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    raw[14:18, 40:65, 30:70] = 220
    return raw


def _coverslip_normal_post_raw(ang, dz, dy, dx, skew, c0=50):
    """Raw volume whose bright feature is a coverslip-NORMAL post: a diagonal streak in
    raw along the sheared axis (raw lateral decreases with scan plane by the deskew
    angle) that maps to a VERTICAL line in the leveled coverslip frame. Y-skew streaks
    in raw-Y, X-skew in raw-X. See shear_only_inverse_map."""
    d_lat = dy if skew == "Y" else dx
    ss = math.sin(math.radians(ang)) * (dz / d_lat)      # zc per scan plane
    tan = math.tan(math.radians(ang))
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    for p in range(0, 26):
        r = int(round(c0 - ss * p / tan))                # raw lateral = c0 - zc/tan, zc = ss*p
        if 0 <= r < 60:
            if skew == "Y":
                raw[p, max(r - 2, 0):r + 3, 26:34] = 200.0
            else:
                raw[p, 26:34, max(r - 2, 0):r + 3] = 200.0
    return raw


def _shear_only_forward(p, yr, xr, angle_deg, dz, dy, dx, skew):
    """Frozen forward map raw (scan p, raw_y yr, raw_x xr) -> shear-only/coverslip (zc, yc, xc)."""
    _, sin_t, cos_t = deskew_trig(angle_deg)
    if skew == "Y":
        step = pixel_step(dz, dy)
        return sin_t * step * p, cos_t * step * p + yr, xr
    else:  # X-skew: X is the sheared axis, Y passes through
        step = pixel_step(dz, dx)
        return sin_t * step * p, yr, cos_t * step * p + xr


def _ss_cc_display(angle_deg, dz, d_lateral):
    angle_rad = math.radians(angle_deg)
    step = pixel_step(dz, d_lateral)
    return math.sin(angle_rad) * step, math.cos(angle_rad) * step


# ===========================================================================
# GEOMETRY  (pure-math, no GPU)
# ===========================================================================
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


def test_shear_only_shape_anisotropic_axial_differs_per_skew():
    # With dy != dx, Y-skew (step=dz/dy) and X-skew (step=dz/dx) must yield DIFFERENT
    # axial extents. If both used the same lateral pixel these agree, so this pins the
    # per-skew lateral selection at the shape level (non-tautological cross-check). The
    # exact shape values are validated end-to-end by the two-pass oracle + OpenCL parity.
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


# ===========================================================================
# KERNEL PARITY  (numba reference vs cle / OpenCL, GPU)
# ===========================================================================
def test_numba_objective_matches_cle_deskew_y():
    raw = _feature_volume()
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    gt = np.asarray(cle.pull(cle.deskew_y(
        raw, angle_in_degrees=ang, voxel_size_x=dx, voxel_size_y=dy, voxel_size_z=dz)))
    nb = numba_deskew(raw, ang, dz, dy, dx, skew="Y", frame="objective",
                      out_shape_zyx=gt.shape)
    assert nb.shape == gt.shape
    denom = max(gt.max(), 1.0)
    assert np.abs(nb - gt).mean() / denom < 0.02
    assert np.corrcoef(nb.ravel(), gt.ravel())[0, 1] > 0.98


def test_numba_shear_only_matches_two_pass_ground_truth():
    """Independent-oracle check of the frozen shear-only map. The single-pass numba
    gather must match the two-pass ground truth (cle.deskew_y THEN cle.rotate-to-level,
    a completely different code path). Compares the tight-cropped CONTENT bounding box
    -- which catches shear-magnitude / step / scale errors that a normalised centroid
    silently passes -- AND the voxel correlation of the origin-aligned volumes, which
    catches structure errors. (Y-skew only: the two-pass cle.rotate for X-skew rotates
    about a different axis and yields a transposed box that is not directly comparable.)"""
    raw = _feature_volume()
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    gt = two_pass_shear_only(raw, ang, dz, dy, dx, skew="Y")          # independent oracle
    shape = shear_only_output_shape(raw.shape, ang, dz, dy, dx, "Y")
    nb = _tight_crop(numba_deskew(raw, ang, dz, dy, dx, skew="Y",
                                  frame="shear_only", out_shape_zyx=shape))

    # (1) Content bounding box must match within a couple of voxels. A wrong shear
    #     magnitude / step scales the box and is caught here; a centroid alone is not.
    assert all(abs(int(a) - int(b)) <= 2 for a, b in zip(nb.shape, gt.shape)), (
        f"shear-only content box {nb.shape} != two-pass ground truth {gt.shape}")

    # (2) Voxel correlation of the origin-aligned (padded to common shape) volumes.
    common = np.maximum(nb.shape, gt.shape)
    pad = lambda v: np.pad(v, [(0, int(common[i] - v.shape[i])) for i in range(3)])
    corr = float(np.corrcoef(pad(nb).ravel(), pad(gt).ravel())[0, 1])
    assert corr > 0.9, f"shear-only vs two-pass correlation too low: {corr:.3f}"


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


# ===========================================================================
# MODEL  (DeskewParams: shape / new_dz / display affine, no GPU)
# ===========================================================================
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


# Shear-only new_dz is the sheared lateral pixel (dy for Y, dx for X); objective
# stays sin(angle)*dz. dy != dx so a Y/X lateral swap would flip 1.04 <-> 0.9.
@pytest.mark.parametrize("skew,coverslip_rotation,expected_dz", [
    ("Y", False, 1.04),                                # shear-only Y -> dy
    ("X", False, 0.9),                                 # shear-only X -> dx
    ("Y", True, math.sin(math.radians(45.0)) * 2.0),   # objective -> sin*dz
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


@pytest.mark.parametrize("z,y,x", [(0, 0, 0), (1, 0, 0), (3, 2, 4), (0, 5, 7)])
def test_display_affine_y_skew(z, y, x):
    """For Y-skew: M @ [z,y,x,1] == [ss*z, cc*z + y, x, 1]."""
    angle_deg = 30.0
    dz, dy, dx = 0.3, 0.15, 0.15
    M = shear_only_display_affine_zyx((10, 8, 6), angle_deg, dz, dy, dx, skew="Y")
    ss, cc = _ss_cc_display(angle_deg, dz, dy)
    result = M @ np.array([z, y, x, 1.0])
    expected = np.array([ss * z, cc * z + y, x, 1.0])
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("z,y,x", [(0, 0, 0), (1, 0, 0), (3, 2, 4), (0, 5, 7)])
def test_display_affine_x_skew(z, y, x):
    """For X-skew: M @ [z,y,x,1] == [ss*z, y, cc*z + x, 1]."""
    angle_deg = 45.0
    dz, dy, dx = 0.3, 0.15, 0.15
    M = shear_only_display_affine_zyx((10, 8, 6), angle_deg, dz, dy, dx, skew="X")
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
    M_normal = shear_only_display_affine_zyx(
        (nz, 8, 6), angle_deg, dz, dy, dx, skew=skew, invert_scan_direction=False)
    M_inv = shear_only_display_affine_zyx(
        (nz, 8, 6), angle_deg, dz, dy, dx, skew=skew, invert_scan_direction=True)
    result = M_inv @ np.array([z, y, x, 1.0])
    reference = M_normal @ np.array([nz - 1 - z, y, x, 1.0])
    np.testing.assert_allclose(result, reference, atol=1e-12)


def test_objective_display_transform_differs_under_coverslip_flag():
    """The coverslip flag must actually reach the display transform: a
    coverslip_rotation=False model must yield a DIFFERENT deskew_affine_transform_zyx
    (the shear-only one) than the coverslip_rotation=True (objective) model. The
    objective path's byte-identity to stock deskew is gated separately by
    test_non_crop_default_byte_identical."""
    raw = DataArray(np.zeros((5, 5, 5), dtype=np.float32), dims=["Z", "Y", "X"])
    obj = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), coverslip_rotation=True)
    cover = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), coverslip_rotation=False)
    m_obj = np.asarray(obj.derived.deskew_affine_transform_zyx)
    m_cover = np.asarray(cover.derived.deskew_affine_transform_zyx)
    assert not np.allclose(m_obj, m_cover), (
        "coverslip_rotation=False must produce a different display affine than True")


# ===========================================================================
# PIPELINE (non-crop)  — shear-only leveling through LatticeData (GPU)
# ===========================================================================
@pytest.mark.parametrize("skew,ang,dy,dx", [
    ("Y", 45.0, 1.04, 1.04), ("X", 45.0, 1.04, 1.04),   # baseline, both skews
    ("Y", 30.0, 1.04, 1.04), ("X", 30.0, 1.04, 1.04),   # angle robustness, both skews
    ("Y", 45.0, 1.04, 0.9),  ("X", 45.0, 1.04, 0.9),    # anisotropic (dy/dx wiring), both skews
])
def test_shear_only_post_is_upright(skew, ang, dy, dx):
    """Flatness check via a coverslip-normal post. Through the real LatticeData pipeline,
    coverslip_rotation=False must (a) return the shear_only_output_shape box and (b) leave
    the post UPRIGHT (~90 deg in the sheared-axis projection); the un-leveled objective
    frame (coverslip_rotation=True) leaves it tilted off vertical by ~the deskew angle.

    The gates are angle-robust: the leveled post reads ~90 deg at every angle (absolute
    ``> 80`` gate), and it must be clearly more upright than the objective result
    (relative ``level - objective > 15`` gate). A fixed objective threshold is NOT used
    because the objective angle scales with the deskew angle (~90 - deskew_angle) and
    collapses toward any fixed gate at shallow angles.

    Assumes OPM acquisition geometry; for Zeiss LLS it is coverslip_rotation=True that is
    the coverslip-level one."""
    dz = 2.0
    raw = _coverslip_normal_post_raw(ang, dz, dy, dx, skew)
    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X

    def _pipeline(coverslip_rotation: bool):
        with tempfile.TemporaryDirectory() as d:
            lat = LatticeData(input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                              physical_pixel_sizes=(dz, dy, dx), angle=ang, skew=skew_dir,
                              coverslip_rotation=coverslip_rotation, save_name="t", save_dir=d)
            return np.asarray(next(iter(lat.process().slices)).data)

    level = _pipeline(False)
    assert level.shape == shear_only_output_shape((30, 60, 60), ang, dz, dy, dx, skew)

    level_deg = yz_slab_angle(level, skew)
    tilted_deg = yz_slab_angle(_pipeline(True), skew)
    assert level_deg > 80, (
        f"leveled post not upright (skew={skew}, ang={ang}): {level_deg:.1f} deg (expected ~90)")
    assert level_deg - tilted_deg > 15, (
        f"leveled post not clearly more upright than objective (skew={skew}, ang={ang}): "
        f"level={level_deg:.1f} objective={tilted_deg:.1f} sep={level_deg - tilted_deg:.1f}")


# ===========================================================================
# PIPELINE (crop)  — ROI mapping, edge zero-pad, deconvolution, invert-scan (GPU)
# ===========================================================================
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
    p_c, yr_c, xr_c = 11.5, 30.5, 28.5   # blob raw centroid (voxel-centre convention)
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
    centroid = np.argwhere(on > 0.5 * on.max()).mean(0)
    centre = (np.array(on.shape) - 1) / 2.0
    dz_off, dy_off, dx_off = np.abs(centroid - centre)
    assert dz_off <= 3 and dy_off <= 3 and dx_off <= 3, (
        f"blob not centred (skew={skew}): centroid={centroid}, centre={centre}")

    # (b) CONTROL ROI far from the blob (along the sheared axis): blob NOT captured
    ctrl = make_roi(yc + 35, xc) if skew == "Y" else make_roi(yc, xc + 35)
    off = run(ctrl)
    assert off.max() < 0.01 * on.max(), (
        f"control crop captured the blob (skew={skew}): off.max={off.max()}, on.max={on.max()}")


@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_crop_edge_roi_aligned(skew):
    """Regression: ROI touching the shear-only frame Z=0 edge (z0=0).

    When the raw sub-block halo pushes scan_start > 0, shear_only_subblock_offset returns
    off_zc > 0, so z0 - off_zc < 0 -- an edge ROI where the old _trim() silently clamped
    to 0, slicing from the sub-block's own origin and returning misaligned data. Fix:
    zero-pad the leading edge so the returned volume aligns to the ROI origin."""
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    # Place a bright blob at raw scan plane 0-3 so it maps to coverslip Z near 0.
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[0:4, 28:34, 26:32] = 500.0
    p_c, yr_c, xr_c = 1.5, 30.5, 28.5
    zc, yc, xc = _shear_only_forward(p_c, yr_c, xr_c, ang, dz, dy, dx, skew)

    nz, ny, nx = shear_only_output_shape((30, 60, 60), ang, dz, dy, dx, skew)
    hw = 8
    z0_roi = 0                                   # ROI at the coverslip edge forces leading-pad path
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

    assert out.size > 0, "edge-ROI crop returned empty volume"
    assert out.max() > 0, (
        f"edge-ROI crop is all zeros (skew={skew}): blob at shear-only zc={zc:.2f} "
        f"not found in z_range=[{z0_roi},{z1_roi}] crop of shape {out.shape}")

    # Blob should be near the BOTTOM of the z-extent (low z-index) since z0_roi=0.
    blob_z = np.argwhere(out > 0.5 * out.max())[:, 0].mean()
    expected_z = zc - z0_roi
    assert abs(blob_z - expected_z) <= 3, (
        f"edge-ROI blob at wrong z (skew={skew}): got blob_z={blob_z:.1f}, "
        f"expected~{expected_z:.1f} (zc={zc:.2f}, z0_roi={z0_roi})")


def test_shear_only_crop_deconvolution_cpu():
    """Covers the CPU deconvolution branch of _crop_volume_deskew_shear_only.

    Deconvolution runs on the raw sub-block BEFORE shear_only_deskew + sub-block-offset
    trim, so an ordering bug or a wrong trim applied to the deconvolved volume would show
    up as a shape/placement change versus the no-decon crop. Kept tiny (small volume,
    small ROI, num_iter=1) because skimage Richardson-Lucy is slow."""
    from lls_core.llsz_core import crop_volume_deskew
    from lls_core.deconvolution import DeconvolutionChoice

    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((20, 40, 40), dtype=np.float32)
    raw[6:10, 18:24, 16:22] = 500.0
    zc, yc, xc = _shear_only_forward(7.5, 20.5, 18.5, ang, dz, dy, dx, "Y")
    nz, _, _ = shear_only_output_shape((20, 40, 40), ang, dz, dy, dx, "Y")
    hw = 6
    z0, z1 = int(max(round(zc - hw), 0)), int(min(round(zc + hw), nz))
    roi = [[[yc - hw, xc - hw], [yc - hw, xc + hw], [yc + hw, xc + hw], [yc + hw, xc - hw]]]
    psf = np.zeros((3, 3, 3), np.float32); psf[1, 1, 1] = 1.0; psf += 0.02; psf /= psf.sum()

    common = dict(original_volume=raw, roi_shape=roi, angle_in_degrees=ang,
                  voxel_size_x=dx, voxel_size_y=dy, voxel_size_z=dz, z_start=z0, z_end=z1,
                  skew_dir=DeskewDirection.Y, coverslip_rotation=False)
    plain = np.asarray(crop_volume_deskew(**common)).astype(np.float32)
    decon = np.asarray(crop_volume_deskew(
        deconvolution=True, decon_processing=DeconvolutionChoice.cpu, psf=psf, num_iter=1,
        **common)).astype(np.float32)

    assert decon.shape == plain.shape, f"decon crop reshaped: {decon.shape} != {plain.shape}"
    assert decon.max() > 0, "decon crop is empty"
    # Deconvolution must not move the feature: same >0.5*max centroid within 1 voxel.
    c_plain = np.argwhere(plain > 0.5 * plain.max()).mean(0)
    c_decon = np.argwhere(decon > 0.5 * decon.max()).mean(0)
    assert np.all(np.abs(c_plain - c_decon) <= 1.0), (
        f"deconvolution shifted the feature: plain={c_plain}, decon={c_decon}")


@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_invert_scan_direction_matches_flipped_input(skew):
    """coverslip_rotation=False with invert_scan_direction=True must equal deskewing the
    scan-flipped raw with invert_scan_direction=False, through the real pipeline. The scan
    flip is applied before the shear-only deskew, so this pins that the invert branch
    composes correctly with the shear-only frame (a double-flip or wrong-axis interaction
    would break the byte-equality)."""
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[4:9, 20:30, 25:35] = 300.0   # asymmetric along the scan axis so a flip is detectable
    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X

    def proc(vol, invert):
        with tempfile.TemporaryDirectory() as d:
            lat = LatticeData(input_image=DataArray(vol, dims=["Z", "Y", "X"]),
                              physical_pixel_sizes=(dz, dy, dx), angle=ang, skew=skew_dir,
                              coverslip_rotation=False, invert_scan_direction=invert,
                              save_name="t", save_dir=d)
            return np.asarray(next(iter(lat.process().slices)).data)

    out_inv = proc(raw, invert=True)
    out_flip = proc(raw[::-1].copy(), invert=False)   # flip scan axis, no invert
    assert out_inv.shape == out_flip.shape
    np.testing.assert_array_equal(out_inv, out_flip)


# ===========================================================================
# MIP  (fast shear-only maximum-intensity projection)
# ===========================================================================
@pytest.mark.parametrize("dy,dx", [(1.04, 1.04), (1.04, 0.9)])   # isotropic + anisotropic
@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_mip_matches_full_shear_only_deskew(skew, dy, dx):
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    raw[14:18, 40:65, 30:70] = 220
    ang, dz = 45.0, 2.0
    full = np.asarray(cle.pull(shear_only_deskew(raw, ang, dz, dy, dx, skew=skew)))
    gt = full.max(axis=0)  # collapse coverslip-normal axis
    mip = deskew_mip(raw, ang, dz, dy, dx, skew=skew, frame="shear_only", target_shape=gt.shape)
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


# ===========================================================================
# BACKCOMPAT  (coverslip_rotation=True byte-identical to stock deskew, GPU)
# ===========================================================================
@pytest.mark.parametrize("skew,skew_dir,ref_func", [
    ("Y", DeskewDirection.Y, cle.deskew_y),
    ("X", DeskewDirection.X, cle.deskew_x),
])
def test_non_crop_default_byte_identical(skew, skew_dir, ref_func):
    """With coverslip_rotation=True (the default), the non-crop pipeline must be
    byte-identical to stock pyclesperanto deskew. Anisotropic voxels (dy != dx) keep the
    X path meaningfully distinct."""
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    with tempfile.TemporaryDirectory() as d:
        lat = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(2.0, 1.04, 0.9), angle=45,
            skew=skew_dir, coverslip_rotation=True,   # True = stock deskew (Zeiss LLS default)
            save_name="t", save_dir=d)
        out = np.asarray(next(iter(lat.process().slices)).data)
    ref = np.asarray(cle.pull_zyx(ref_func(
        raw, angle_in_degrees=45, voxel_size_x=0.9, voxel_size_y=1.04, voxel_size_z=2.0)))
    np.testing.assert_array_equal(out, ref)
