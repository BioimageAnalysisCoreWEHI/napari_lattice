"""Test-only numba deskew reference, two-pass coverslip ground truth, and a
tilt-residual metric. Not shipped. The objective kernel is a line-by-line port
of pyclesperanto's affine_transform_deskew_y_3d_x.cl, so it matches cle.deskew_y."""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from lls_core.shear_only_geometry import deskew_trig, pixel_step, level_angle


@njit(parallel=True, cache=True)
def _deskew_y_objective(raw, nz, ny, nx, step, tan_t, sin_t, cos_t):
    Nz, Ny, Nx = raw.shape
    out = np.zeros((nz, ny, nx), dtype=np.float32)
    for z in prange(nz):
        for y in range(ny):
            virtual_plane = y - z / tan_t
            plane_before = int(math.floor(virtual_plane / step))
            plane_after = plane_before + 1
            if plane_before < 0 or plane_after >= Nz:
                continue
            l_before = virtual_plane - plane_before * step
            l_after = step - l_before
            za = z / sin_t
            vpb = za + l_before * cos_t
            vpa = za - l_after * cos_t
            pos_before = int(math.floor(vpb))
            pos_after = int(math.floor(vpa))
            if pos_before < 0 or pos_after < 0 or pos_before >= Ny - 1 or pos_after >= Ny - 1:
                continue
            dzb = vpb - pos_before
            dza = vpa - pos_after
            zout = nz - 1 - z  # objective-frame write flip (matches the .cl)
            for x in range(nx):
                pix = (l_before * dza * raw[plane_after, pos_after + 1, x]
                       + l_before * (1.0 - dza) * raw[plane_after, pos_after, x]
                       + l_after * dzb * raw[plane_before, pos_before + 1, x]
                       + l_after * (1.0 - dzb) * raw[plane_before, pos_before, x]) / step
                out[zout, y, x] = pix
    return out


@njit(parallel=True, cache=True)
def _deskew_y_shear_only(raw, nz, ny, nx, step, tan_t, sin_t, cos_t):
    # Inverse of the frozen shear-only forward map; see shear_only_geometry
    # (bilinear gather over the two nearest scan planes / raw-Y rows, z0=y0=0).
    Nz, Ny, Nx = raw.shape
    out = np.zeros((nz, ny, nx), dtype=np.float32)
    ss = sin_t * step  # spacing of zc per scan-plane index: zc = ss * p
    for zc in prange(nz):
        scan = zc / ss
        plane = int(math.floor(scan))
        if plane < 0 or plane + 1 >= Nz:
            continue
        fp = scan - plane
        for yc in range(ny):
            raw_y = yc - zc / tan_t
            pos = int(math.floor(raw_y))
            if pos < 0 or pos + 1 >= Ny:
                continue
            fy = raw_y - pos
            for x in range(nx):
                v = ((1.0 - fp) * (1.0 - fy) * raw[plane, pos, x]
                     + (1.0 - fp) * fy * raw[plane, pos + 1, x]
                     + fp * (1.0 - fy) * raw[plane + 1, pos, x]
                     + fp * fy * raw[plane + 1, pos + 1, x])
                out[zc, yc, x] = v
    return out


def _as_zyx_for_x_skew(raw):
    # X-skew reuses the Y kernels by swapping X<->Y axes in and out.
    return np.ascontiguousarray(np.swapaxes(raw, 1, 2))


def _tight_crop(vol):
    """Trim all-zero margins so numba output can be compared content-to-content with the (already-cropped) oracle."""
    idx = np.argwhere(vol > 0)
    if idx.size == 0:
        return vol
    lo = idx.min(0)
    hi = idx.max(0) + 1
    return np.ascontiguousarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])


def numba_deskew(raw_zyx, angle_deg, dz, dy, dx, skew="Y", frame="objective", out_shape_zyx=None):
    tan_t, sin_t, cos_t = deskew_trig(angle_deg)
    raw = np.ascontiguousarray(raw_zyx.astype(np.float32))
    if skew == "X":
        raw = _as_zyx_for_x_skew(raw)
        d_lat = dx
    else:
        d_lat = dy
    step = pixel_step(dz, d_lat)
    if out_shape_zyx is None:
        raise ValueError("out_shape_zyx is required")
    nz, ny, nx = (out_shape_zyx if skew == "Y"
                  else (out_shape_zyx[0], out_shape_zyx[2], out_shape_zyx[1]))
    kernel = _deskew_y_objective if frame == "objective" else _deskew_y_shear_only
    out = kernel(raw, int(nz), int(ny), int(nx), step, tan_t, sin_t, cos_t)
    if skew == "X":
        out = np.ascontiguousarray(np.swapaxes(out, 1, 2))
    # coverslip path: return the FULL fixed-shape box (no tight-crop here).
    # Callers that need content-to-content comparison should call _tight_crop
    # themselves (e.g. test_numba_coverslip_matches_two_pass_ground_truth).
    # two_pass_shear_only's own crop is unchanged (it is the oracle).
    return out


def two_pass_shear_only(raw_zyx, angle_deg, dz, dy, dx, skew="Y"):
    """Ground truth: the current manual method (deskew then rotate-to-level)."""
    import pyclesperanto_prototype as cle
    raw = raw_zyx.astype(np.float32)
    if skew == "Y":
        desk = cle.deskew_y(raw, angle_in_degrees=angle_deg,
                            voxel_size_x=dx, voxel_size_y=dy, voxel_size_z=dz)
        leveled = cle.rotate(desk,
                             angle_around_x_in_degrees=level_angle(angle_deg, "Y"),
                             rotate_around_center=True, auto_size=True)
    else:
        desk = cle.deskew_x(raw, angle_in_degrees=angle_deg,
                            voxel_size_x=dx, voxel_size_y=dy, voxel_size_z=dz)
        leveled = cle.rotate(desk,
                             angle_around_y_in_degrees=level_angle(angle_deg, "X"),
                             rotate_around_center=True, auto_size=True)
    vol = np.asarray(cle.pull(leveled))
    nz = np.argwhere(vol > 0)
    if nz.size:
        lo = nz.min(0); hi = nz.max(0) + 1
        vol = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return vol


def yz_slab_angle(vol, skew="Y", thr_frac=0.5):
    """Angle (degrees) of the bright slab in the sheared-axis projection of a ZYX volume.

    Simple, visual flatness check: threshold to the bright slab, project along the
    passthrough lateral axis, and take the principal-axis angle in the (Z, sheared)
    plane. 90 == vertical (upright, along output-Z); 0 == horizontal. For a
    coverslip-NORMAL post a correctly leveled (shear-only) deskew gives ~90; the
    un-leveled objective frame leaves it tilted off vertical by ~the deskew angle.
    Y-skew projects along X (-> Z, Y); X-skew projects along Y (-> Z, X). Assumes OPM
    acquisition geometry."""
    arr = np.asarray(vol, dtype=np.float64)
    proj = arr.max(axis=2) if skew == "Y" else arr.max(axis=1)   # -> (Z, sheared-lateral)
    if proj.max() <= 0:
        return 0.0
    zz, ll = np.nonzero(proj > thr_frac * proj.max())
    if zz.size < 2:
        return 0.0
    pts = np.stack([zz, ll]).astype(float)
    pts -= pts.mean(1, keepdims=True)
    eigvals, eigvecs = np.linalg.eigh(np.cov(pts))
    dzc, dll = eigvecs[:, int(np.argmax(eigvals))]       # dominant principal direction
    return float(np.degrees(np.arctan2(abs(dzc), abs(dll))))
