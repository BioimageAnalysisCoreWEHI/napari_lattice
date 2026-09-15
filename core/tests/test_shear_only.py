"""Shear-only (coverslip-frame / OPM) deskew test suite.

"Shear-only" deskew levels the raw volume against the coverslip WITHOUT the extra
rotation that stock deskew adds. This suite is large on purpose: it checks that the
shear-only geometry is correct for both the full 3D volume and the MIP, and that
turning the feature off keeps the old behaviour exactly.

Sections:
    - GEOMETRY            pure-math unit tests (no GPU)
    - KERNEL PARITY       numba reference vs cle / OpenCL kernel (GPU)
    - MODEL               DeskewParams shape/new_dz/display-affine (no GPU)
    - PIPELINE (non-crop) shear-only leveling through LatticeData (GPU)
    - PIPELINE (crop)     crop ROI mapping, edge/zero-pad, decon, invert-scan (GPU)
    - MIP                 fast shear-only MIP (numba)
    - BACKCOMPAT          coverslip_rotation=True byte-identical to stock deskew (GPU)

What actually proves the geometry is correct:
  A "true oracle" is an independent source of truth that can catch a wrong geometry
  map. A "fidelity check" only proves two implementations agree with each other -- if
  they share the same map, both can be wrong the same way and still pass. This suite is
  careful about which check is which:

  * The shear-only map is a single frozen formula. Its one true oracle is
    test_numba_shear_only_matches_two_pass_ground_truth, which compares it against a
    completely separate code path (cle.deskew_y, then cle.rotate-to-level).
    That oracle only works for Y-skew: the X-skew two-pass rotates about a different
    axis, so its content can't be compared directly (the bounding box happens to match
    at 45 deg, but the content correlation collapses, and the box drifts at shallow angles).
  * X-skew geometry is proved instead by test_shear_only_post_is_upright (both skews;
    two angles for Y, 45 deg for X), the MIP-vs-full-deskew parity, and the OpenCL<->numba parity.
  * OpenCL<->numba and MIP<->full-deskew both reuse the frozen map, so they are fidelity
    checks, not independent oracles.

Metrics and reference kernels live in tests/reference_deskew.py.
"""
import math
import tempfile

import numpy as np
import pytest
import pyclesperanto as cle
from xarray import DataArray

from lls_core import DeskewDirection
from lls_core.models.crop import CropParams
from lls_core.models.deskew import DeskewParams
from lls_core.models.lattice_data import LatticeData
from lls_core.mip import deskew_mip
from lls_core.shear_only_deskew import shear_only_deskew
from lls_core.shear_only_geometry import (
    deskew_trig, pixel_step, shear_only_output_shape,
    shear_only_forward_affine, shear_only_subblock_offset,
    shear_only_display_affine_zyx,
)
from tests.reference_deskew import numba_deskew, two_pass_shear_only, _tight_crop, yz_slab_angle
from tests.utils import requires_real_gpu


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _feature_volume():
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    raw[14:18, 40:65, 30:70] = 220
    return raw


def _coverslip_normal_post_raw(ang, dz, dy, dx, skew, c0=50):
    """Build a raw volume whose bright feature is a "coverslip-normal post" -- a feature
    that should stand vertical once the volume is leveled. In raw coordinates it is a
    diagonal streak along the sheared axis (its raw lateral position shifts with the scan
    plane by the deskew angle); a correct shear-only map turns that streak into a VERTICAL
    line in the leveled coverslip frame. Y-skew streaks along raw-Y, X-skew along raw-X.
    See shear_only_inverse_map."""
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
@pytest.mark.parametrize("bad", ["Z", None])
def test_bad_skew_raises(bad):
    # Forward affine, sub-block offset, and display affine must all reject unknown skew.
    with pytest.raises(ValueError):
        shear_only_forward_affine(45.0, 2.0, 1.04, 1.04, bad)
    with pytest.raises(ValueError):
        shear_only_subblock_offset(0, 0, 0, 45.0, 2.0, 1.04, 1.04, bad)
    with pytest.raises(ValueError):
        shear_only_display_affine_zyx((10, 8, 6), 45.0, 2.0, 1.04, 1.04, bad)


# ===========================================================================
# KERNEL PARITY  (numba reference vs cle / OpenCL, GPU)
# ===========================================================================
@requires_real_gpu
def test_numba_objective_matches_cle_deskew_y():
    # Numba objective output should match trusted cle.deskew_y.
    raw = _feature_volume()
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    gt = np.asarray(cle.pull(cle.deskew_y(
        raw, angle=ang, voxel_size_x=dx, voxel_size_y=dy, voxel_size_z=dz)))
    nb = numba_deskew(raw, ang, dz, dy, dx, skew="Y", frame="objective", out_shape_zyx=gt.shape)
    assert nb.shape == gt.shape
    denom = max(gt.max(), 1.0)
    assert np.abs(nb - gt).mean() / denom < 0.02
    assert np.corrcoef(nb.ravel(), gt.ravel())[0, 1] > 0.98


@requires_real_gpu
@pytest.mark.parametrize("ang", [45.0, 30.0])
def test_numba_shear_only_matches_two_pass_ground_truth(ang):
    """The one true oracle for the frozen shear-only map: compare the single-pass numba
    gather against a two-pass ground truth (cle.deskew_y THEN cle.rotate-to-level -- a
    completely different code path). Two comparisons are made:
        1. the tight-cropped CONTENT bounding box -- catches shear-magnitude / step / scale
            errors that a normalised centroid would silently pass, and
        2. the voxel correlation of the origin-aligned volumes -- catches structure errors.

    Run at two angles so shallow-angle geometry is checked, not just 45 deg. Y-skew only:
    the X-skew two-pass rotates about a different axis, so its content can't be compared
    directly (see module docstring); X is pinned by test_shear_only_post_is_upright."""
    raw = _feature_volume()
    dz, dy, dx = 2.0, 1.04, 1.04
    gt = two_pass_shear_only(raw, ang, dz, dy, dx, skew="Y")
    shape = shear_only_output_shape(raw.shape, ang, dz, dy, dx, "Y")
    nb = _tight_crop(numba_deskew(raw, ang, dz, dy, dx, skew="Y",
                                    frame="shear_only", out_shape_zyx=shape))

    # (1) Content box within 3 vox. 45 deg matches the two-pass exactly; at 30 deg the
    #     two-pass double-resample + rotate erodes a few edge voxels (measured <=3).
    assert all(abs(int(a) - int(b)) <= 3 for a, b in zip(nb.shape, gt.shape)), (
        f"shear-only content box {nb.shape} != two-pass ground truth {gt.shape} (ang={ang})")

    # (2) Voxel correlation of the origin-aligned (padded to common shape) volumes.
    common = np.maximum(nb.shape, gt.shape)
    pad = lambda v: np.pad(v, [(0, int(common[i] - v.shape[i])) for i in range(3)])
    corr = float(np.corrcoef(pad(nb).ravel(), pad(gt).ravel())[0, 1])
    assert corr > 0.90, f"shear-only vs two-pass correlation too low: {corr:.3f} (ang={ang})"


@pytest.mark.parametrize("skew", ["Y", "X"])
@pytest.mark.parametrize("dy,dx", [
    (1.04, 1.04),        # isotropic
    (1.04, 0.9),         # anisotropic: pins the kernel's per-skew lateral-axis selection
])
def test_opencl_shear_only_matches_numba(skew, dy, dx):
    """The OpenCL single-pass kernel must match the frozen numba shear-only map (both skews).

    The anisotropic case (dy != dx) checks that the kernel picks the right lateral axis:
    swapping dy/dx between Y and X would change nz (through `step`) and break parity for one
    of the skews. In practice the two are bit-exact (max|err| ~ 0), because every raw read
    stays inside the in-bounds guard."""
    raw = _feature_volume()
    ang, dz = 45.0, 2.0
    shape = shear_only_output_shape(raw.shape, ang, dz, dy, dx, skew)
    nb = numba_deskew(raw, ang, dz, dy, dx, skew=skew, frame="shear_only", out_shape_zyx=shape)
    gpu = np.asarray(cle.pull(shear_only_deskew(raw, ang, dz, dy, dx, skew=skew)))
    assert gpu.shape == nb.shape == shape, f"Shape mismatch: gpu={gpu.shape}, nb={nb.shape}, expected={shape}"
    denom = max(float(nb.max()), 1.0)
    assert float(np.abs(gpu - nb).mean()) / denom < 0.02, f"mean_err too high (skew={skew})"
    assert float(np.corrcoef(gpu.ravel(), nb.ravel())[0, 1]) > 0.98, f"corr too low (skew={skew})"


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
    # shear-only derived shape equals shear_only_output_shape, and differs from the objective box
    assert tuple(cover.derived.deskew_vol_shape) == shear_only_output_shape(
        (24, 70, 80), 45, 2.0, 1.04, 1.04, "Y")
    assert tuple(cover.derived.deskew_vol_shape) != tuple(default.derived.deskew_vol_shape)


# Shear-only new_dz is the sheared lateral pixel (dy for Y, dx for X); objective stays
# sin(angle)*dz. dy != dx so a Y/X lateral swap would flip 1.04 <-> 0.9.
@pytest.mark.parametrize("skew,coverslip_rotation,expected_dz", [
    ("Y", False, 1.04),                                # shear-only Y -> dy
    ("X", False, 0.9),                                 # shear-only X -> dx
    ("Y", True, math.sin(math.radians(45.0)) * 2.0),   # objective -> sin*dz
])
def test_new_dz(skew, coverslip_rotation, expected_dz):
    da = DataArray(np.zeros((5, 10, 12), dtype=np.float32), dims=["Z", "Y", "X"])
    params = DeskewParams(input_image=da, physical_pixel_sizes=(2.0, 1.04, 0.9),
                            angle=45, skew=skew, coverslip_rotation=coverslip_rotation)
    assert params.new_dz == pytest.approx(expected_dz)


@pytest.mark.parametrize("skew", ["Y", "X"])
@pytest.mark.parametrize("z,y,x", [(1, 0, 0), (3, 2, 4)])
def test_display_affine_forward(skew, z, y, x):
    """M @ [z,y,x,1] equals the closed-form shear-only forward map:
    Y -> [ss*z, cc*z + y, x];  X -> [ss*z, y, cc*z + x]."""
    angle_deg, dz, dy, dx = 30.0, 0.3, 0.15, 0.15
    M = shear_only_display_affine_zyx((10, 8, 6), angle_deg, dz, dy, dx, skew=skew)
    ss, cc = _ss_cc_display(angle_deg, dz, dy if skew == "Y" else dx)
    result = M @ np.array([z, y, x, 1.0])
    expected = [ss * z, cc * z + y, x, 1.0] if skew == "Y" else [ss * z, y, cc * z + x, 1.0]
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("skew", ["Y", "X"])
@pytest.mark.parametrize("z,y,x", [(1, 2, 3), (4, 0, 5)])
def test_display_affine_invert_scan_fold(skew, z, y, x):
    """With invert=True, mapping [z,y,x,1] equals invert=False applied to [nz-1-z, y, x, 1]."""
    angle_deg, dz, dy, dx, nz = 30.0, 0.3, 0.15, 0.15, 10
    M_normal = shear_only_display_affine_zyx((nz, 8, 6), angle_deg, dz, dy, dx, skew=skew,
                                                invert_scan_direction=False)
    M_inv = shear_only_display_affine_zyx((nz, 8, 6), angle_deg, dz, dy, dx, skew=skew,
                                            invert_scan_direction=True)
    np.testing.assert_allclose(M_inv @ np.array([z, y, x, 1.0]),
                                M_normal @ np.array([nz - 1 - z, y, x, 1.0]), atol=1e-12)


def test_objective_display_transform_differs_under_coverslip_flag():
    """The coverslip flag must reach the display transform: coverslip_rotation=False must
    yield a DIFFERENT deskew_affine_transform_zyx than coverslip_rotation=True."""
    raw = DataArray(np.zeros((5, 5, 5), dtype=np.float32), dims=["Z", "Y", "X"])
    obj = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), coverslip_rotation=True)
    cover = DeskewParams(input_image=raw, physical_pixel_sizes=(1, 1, 1), coverslip_rotation=False)
    assert not np.allclose(np.asarray(obj.derived.deskew_affine_transform_zyx),
                            np.asarray(cover.derived.deskew_affine_transform_zyx))


# ===========================================================================
# PIPELINE (non-crop)  — shear-only leveling through LatticeData (GPU)
# ===========================================================================
@pytest.mark.parametrize("skew,ang,dy,dx", [
    ("Y", 45.0, 1.04, 1.04), ("X", 45.0, 1.04, 1.04),   # both skews, baseline
    ("Y", 30.0, 1.04, 1.04),                             # shallow-angle robustness
    ("X", 45.0, 1.04, 0.9),                              # X-skew anisotropic dy/dx wiring
])
def test_shear_only_post_is_upright(skew, ang, dy, dx):
    """A coverslip-normal post should stand vertical after leveling. Through the real
    LatticeData pipeline, coverslip_rotation=False must (a) return the shear_only_output_shape
    box and (b) leave the post upright (~90 deg in the sheared-axis projection); the objective
    frame instead leaves it tilted by ~the deskew angle. This is the main independent geometric
    check for X-skew (the two-pass oracle can't compare X): a wrong tan would tilt the post.
    Thresholds hold at any angle (upright > 80, and level - objective > 15)."""
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
    assert level_deg > 80, f"leveled post not upright (skew={skew}, ang={ang}): {level_deg:.1f}"
    assert level_deg - tilted_deg > 15, (
        f"leveled post not clearly more upright than objective (skew={skew}, ang={ang}): "
        f"level={level_deg:.1f} objective={tilted_deg:.1f}")


# ===========================================================================
# PIPELINE (crop)  — ROI mapping, edge zero-pad, deconvolution, invert-scan (GPU)
# ===========================================================================
@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_crop_roundtrip_places_feature(skew):
    """A bright blob at a KNOWN raw location maps (via the frozen forward map) to a known
    coverslip location. An ROI AROUND it must crop a volume that CONTAINS the blob near the
    output centre; a CONTROL ROI far away must NOT capture it (proves localization)."""
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[10:14, 28:34, 26:32] = 500.0
    zc, yc, xc = _shear_only_forward(11.5, 30.5, 28.5, ang, dz, dy, dx, skew)

    nz, ny, nx = shear_only_output_shape((30, 60, 60), ang, dz, dy, dx, skew)
    hw = 12

    def make_roi(cy, cx):
        return [[[cy - hw, cx - hw], [cy - hw, cx + hw], [cy + hw, cx + hw], [cy + hw, cx - hw]]]

    z0 = int(max(round(zc - hw), 0))
    z1 = int(min(round(zc + hw), nz))
    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X

    def run(roi):
        with tempfile.TemporaryDirectory() as d:
            lat = LatticeData(input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                                physical_pixel_sizes=(dz, dy, dx), angle=ang, skew=skew_dir,
                                coverslip_rotation=False, crop=CropParams(roi_list=roi, z_range=(z0, z1)),
                                save_name="t", save_dir=d)
            return np.asarray(next(iter(lat.process().slices)).data)

    # (a) ROI centred on the blob's coverslip location: blob captured near centre
    on = run(make_roi(yc, xc))
    assert on.size > 0 and on.max() > 0, "on-target crop is empty"
    centroid = np.argwhere(on > 0.5 * on.max()).mean(0)
    centre = (np.array(on.shape) - 1) / 2.0
    assert np.all(np.abs(centroid - centre) <= 3), f"blob not centred (skew={skew})"

    # (b) CONTROL ROI far from the blob (along the sheared axis): blob NOT captured
    ctrl = make_roi(yc + 35, xc) if skew == "Y" else make_roi(yc, xc + 35)
    off = run(ctrl)
    assert off.max() < 0.01 * on.max(), f"control crop captured the blob (skew={skew})"


@pytest.mark.parametrize("skew", ["Y", "X"])
def test_shear_only_crop_edge_roi_aligned(skew):
    """Regression: an ROI whose coverslip Z-range starts at the frame's Z=0 edge (z0_roi=0).
    The ROI origin then maps to raw scan plane 0, so the sub-block offset off_zc is 0 and the
    deskewed output must stay aligned to the ROI origin -- the blob must land at the low-z
    (bottom) edge, not be shifted inward or dropped. (_trim zero-pads the leading edge whenever
    a sub-block origin would map before the ROI origin, keeping this alignment.)"""
    ang, dz, dy, dx = 45.0, 2.0, 1.04, 1.04
    raw = np.zeros((30, 60, 60), dtype=np.float32)
    raw[0:4, 28:34, 26:32] = 500.0
    zc, yc, xc = _shear_only_forward(1.5, 30.5, 28.5, ang, dz, dy, dx, skew)

    nz, ny, nx = shear_only_output_shape((30, 60, 60), ang, dz, dy, dx, skew)
    hw = 8
    z0_roi = 0                                   # ROI at the coverslip edge forces leading-pad path
    z1_roi = int(min(round(zc + hw), nz))
    roi = [[[yc - hw, xc - hw], [yc - hw, xc + hw], [yc + hw, xc + hw], [yc + hw, xc - hw]]]

    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X
    with tempfile.TemporaryDirectory() as d:
        lat = LatticeData(input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                            physical_pixel_sizes=(dz, dy, dx), angle=ang, skew=skew_dir,
                            coverslip_rotation=False, crop=CropParams(roi_list=roi, z_range=(z0_roi, z1_roi)),
                            save_name="t", save_dir=d)
        out = np.asarray(next(iter(lat.process().slices)).data)

    assert out.size > 0 and out.max() > 0, f"edge-ROI crop is empty (skew={skew})"
    # Blob should be near the BOTTOM of the z-extent (low z-index) since z0_roi=0.
    blob_z = np.argwhere(out > 0.5 * out.max())[:, 0].mean()
    assert abs(blob_z - (zc - z0_roi)) <= 3, f"edge-ROI blob at wrong z (skew={skew}): {blob_z:.1f} vs {zc:.2f}"


def test_shear_only_crop_deconvolution_cpu():
    """Covers the CPU deconvolution branch of the shear-only crop path. Decon runs on the raw
    sub-block BEFORE shear_only_deskew + trim, so a wrong order or trim would move the feature
    or reshape the crop. Kept tiny (num_iter=1) because Richardson-Lucy is slow."""
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

    assert decon.shape == plain.shape and decon.max() > 0, "decon crop reshaped or empty"
    # Deconvolution must not move the feature.
    c_plain = np.argwhere(plain > 0.5 * plain.max()).mean(0)
    c_decon = np.argwhere(decon > 0.5 * decon.max()).mean(0)
    assert np.all(np.abs(c_plain - c_decon) <= 1.0), f"decon shifted feature: {c_plain} vs {c_decon}"


@pytest.mark.parametrize("skew", ["Y", "X"])
@pytest.mark.parametrize("use_crop", [False, True])
def test_shear_only_invert_scan_matches_flipped_input(skew, use_crop):
    """coverslip_rotation=False with invert_scan_direction=True must give byte-for-byte the same
    result as deskewing the scan-flipped raw with invert=False -- through the real pipeline, for
    BOTH the non-crop and crop paths. The scan flip happens upstream (apply_scan_flip) before the
    shear-only deskew, so this checks the invert branch composes with the shear-only frame and (for
    crop) with the ROI->raw inverse map. A double flip or wrong-axis interaction would break it.
    (No separate MIP invert path to test: the MIP uses the same pre-flipped volume.)"""
    ang, dz, dy, dx, Nz = 45.0, 2.0, 1.04, 1.04, 30
    raw = np.zeros((Nz, 60, 60), dtype=np.float32)
    raw[4:9, 20:30, 25:35] = 300.0   # asymmetric along scan so a flip is detectable
    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X

    crop = None
    if use_crop:
        # ROI at the flip-effective coverslip location so the crop actually captures the feature.
        zc, yc, xc = _shear_only_forward((Nz - 1) - 6.5, 24.5, 29.5, ang, dz, dy, dx, skew)
        hw = 10
        roi = [[[yc - hw, xc - hw], [yc - hw, xc + hw], [yc + hw, xc + hw], [yc + hw, xc - hw]]]
        z0, z1 = int(max(round(zc - hw), 0)), int(round(zc + hw))
        crop = CropParams(roi_list=roi, z_range=(z0, z1))

    def proc(vol, invert):
        with tempfile.TemporaryDirectory() as d:
            lat = LatticeData(input_image=DataArray(vol, dims=["Z", "Y", "X"]),
                                physical_pixel_sizes=(dz, dy, dx), angle=ang, skew=skew_dir,
                                coverslip_rotation=False, invert_scan_direction=invert,
                                crop=crop, save_name="t", save_dir=d)
            return np.asarray(next(iter(lat.process().slices)).data)

    out_inv = proc(raw, invert=True)
    out_flip = proc(raw[::-1].copy(), invert=False)   # flip scan axis, no invert
    assert out_inv.shape == out_flip.shape
    np.testing.assert_array_equal(out_inv, out_flip)
    if use_crop:
        assert out_inv.max() > 0, "crop+invert test is vacuous (feature not captured)"


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
    np.testing.assert_array_equal(
        deskew_mip(raw, 45, 2.0, 1.04, 1.04, skew="Y"),
        deskew_mip(raw, 45, 2.0, 1.04, 1.04, skew="Y", frame="objective"))


# ===========================================================================
# BACKCOMPAT  (coverslip_rotation=True byte-identical to stock deskew, GPU)
# ===========================================================================
@pytest.mark.parametrize("skew_dir,ref_func", [
    (DeskewDirection.Y, cle.deskew_y),
    (DeskewDirection.X, cle.deskew_x),
])
def test_non_crop_default_byte_identical(skew_dir, ref_func):
    """With coverslip_rotation=True (the default), the non-crop pipeline must be byte-identical
    to stock pyclesperanto deskew. Anisotropic voxels (dy != dx) keep the X path distinct."""
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    with tempfile.TemporaryDirectory() as d:
        lat = LatticeData(input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                            physical_pixel_sizes=(2.0, 1.04, 0.9), angle=45,
                            skew=skew_dir, coverslip_rotation=True, save_name="t", save_dir=d)
        out = np.asarray(next(iter(lat.process().slices)).data)
    ref = np.asarray(cle.pull(ref_func(
        raw, angle=45, voxel_size_x=0.9, voxel_size_y=1.04, voxel_size_z=2.0)))
    np.testing.assert_array_equal(out, ref)
