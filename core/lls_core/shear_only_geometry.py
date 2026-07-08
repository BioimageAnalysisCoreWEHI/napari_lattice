"""
Shared shear-only (coverslip-frame) deskew geometry.

As we have both OpenCL and numba based implementations for deskewing, 
we need a single source of truth for the shear-only output shape and coordinate map,
which is then consumed by the 
- OpenCL deskew wrapper
- the numba MIP, and 
- the crop ROI
mappings so they cannot drift. Pure numpy/math; We do not include any pyclesperanto or numba imports.

Geometry follows the qi2lab orthogonal-deskew convention. The coverslip axial (Z)
extent is SCAN-driven -- `(n_scan-1)*sin(theta)*step + 1` (step = dz/d_lateral) -- NOT
raw-lateral-driven: verified against the two-pass oracle (moving a feature along raw-Y at a
fixed scan plane leaves its coverslip height unchanged). The sheared lateral extent is
`(n_scan-1)*cos(theta)*step + (n_lat-1) + 1`.

"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def deskew_trig(angle_deg: float) -> Tuple[float, float, float]:
    r = math.radians(angle_deg)
    return math.tan(r), math.sin(r), math.cos(r)


def pixel_step(voxel_size_z: float, voxel_size_lateral: float) -> float:
    """Scan step expressed in lateral pixels (dz / d_lateral)."""
    return voxel_size_z / voxel_size_lateral


def shear_only_forward_affine(
    angle_deg: float, dz: float, dy: float, dx: float, skew: str,
) -> np.ndarray:
    """Forward shear-only map as a 4x4 affine M (ZYX homogeneous).

    Maps a raw voxel index [z=scan plane, y, x, 1] to its coverslip-frame
    position [zc, yc, xc, 1]:

        Y-skew:  zc = ss*z ; yc = cc*z + y ; xc = x
        X-skew:  zc = ss*z ; yc = y        ; xc = cc*z + x

    with step = dz/dy (Y-skew) or dz/dx (X-skew), ss = sin*step, cc = cos*step.
    Shared builder for the Python-side shear-only geometry (display affine and
    sub-block offset); the numba/OpenCL kernels carry the same map as scalars.
    """
    if skew not in ("Y", "X"):
        raise ValueError(f"Unknown skew {skew!r}; expected 'Y' or 'X'")
    _, sin_t, cos_t = deskew_trig(angle_deg)
    step = pixel_step(dz, dy) if skew == "Y" else pixel_step(dz, dx)
    ss = sin_t * step
    cc = cos_t * step
    M = np.eye(4)
    M[0, 0] = ss
    M[1 if skew == "Y" else 2, 0] = cc
    return M


def shear_only_output_shape(
    raw_shape_zyx: Tuple[int, int, int],
    angle_deg: float,
    dz: float,
    dy: float,
    dx: float,
    skew: str,
) -> Tuple[int, int, int]:
    """Shear-only (coverslip-frame) output shape (nz, ny, nx) in ZYX.

    Y-skew: coverslip height (Z) is driven by the SCAN axis (raw axis 0) and
    lateral Y is sheared by scan displacement plus the raw-Y extent.  X passes
    through.  X-skew: roles of n_y and n_x are swapped.

    Scan-driven formulas (oracle-validated):
      Y-skew:
        nz = ceil((n_scan-1)*sin*step) + 1
        ny = ceil((n_scan-1)*cos*step + (n_y-1)) + 1
        nx = n_x
      X-skew:
        nz = ceil((n_scan-1)*sin*step) + 1
        ny = n_y
        nx = ceil((n_scan-1)*cos*step + (n_x-1)) + 1
    where step = dz/dy (Y-skew) or dz/dx (X-skew).
    """
    n_scan, n_y, n_x = raw_shape_zyx
    _, sin_t, cos_t = deskew_trig(angle_deg)
    if skew == "Y":
        step = pixel_step(dz, dy)
        nz = int(math.ceil((n_scan - 1) * sin_t * step)) + 1
        ny = int(math.ceil((n_scan - 1) * cos_t * step + (n_y - 1))) + 1
        nx = n_x
    elif skew == "X":
        step = pixel_step(dz, dx)
        nz = int(math.ceil((n_scan - 1) * sin_t * step)) + 1
        ny = n_y
        nx = int(math.ceil((n_scan - 1) * cos_t * step + (n_x - 1))) + 1
    else:
        raise ValueError(f"Unknown skew {skew!r}; expected 'Y' or 'X'")
    return nz, ny, nx


def shear_only_inverse_map(
    zc: float, yc: float, xc: float,
    angle_deg: float, dz: float, dy: float, dx: float, skew: str,
) -> Tuple[float, float, float]:
    """Frozen shear-only inverse map: coverslip (level) (zc, yc, xc) -> raw (scan p, raw_y, raw_x).

    Exact inverse of the forward map used by the deskew kernel:

      Y-skew forward: zc = sin*step*p ; yc = cos*step*p + raw_y ; xc = raw_x
      X-skew forward: zc = sin*step*p ; xc = cos*step*p + raw_x ; yc = raw_y

    with step = dz/dy (Y-skew) or dz/dx (X-skew). Used to map a crop ROI drawn
    in the coverslip frame back to the raw sub-volume that must be extracted.
    Must NOT be confused with the objective `get_inverse_affine_transform`.
    """
    tan_t, sin_t, _ = deskew_trig(angle_deg)
    if skew == "Y":
        step = pixel_step(dz, dy)
        ss = sin_t * step
        p = zc / ss
        raw_y = yc - zc / tan_t
        raw_x = xc
    elif skew == "X":
        step = pixel_step(dz, dx)
        ss = sin_t * step
        p = zc / ss
        raw_x = xc - zc / tan_t
        raw_y = yc
    else:
        raise ValueError(f"Unknown skew {skew!r}; expected 'Y' or 'X'")
    return p, raw_y, raw_x


def shear_only_subblock_offset(
    scan_start: float, rawy_start: float, rawx_start: float,
    angle_deg: float, dz: float, dy: float, dx: float, skew: str,
) -> Tuple[float, float, float]:
    """Shear-only (coverslip-frame) coordinate of a raw sub-block's origin.

    When a raw sub-block starting at raw index (scan_start, rawy_start, rawx_start)
    is deskewed on its own, its shear-only output is a pure translation of the
    corresponding region in the global coverslip frame. This returns that offset
    (off_zc, off_yc, off_xc) so a sub-block crop can be trimmed to a global ROI:

        global_coverslip = subblock_coverslip + (off_zc, off_yc, off_xc)

    Derived from the forward map evaluated at the sub-block origin (raw_y/raw_x = 0
    within the sub-block, scan plane 0 == global scan_start).
    """
    M = shear_only_forward_affine(angle_deg, dz, dy, dx, skew)
    off_zc, off_yc, off_xc, _ = M @ np.array(
        [scan_start, rawy_start, rawx_start, 1.0], dtype=float
    )
    return off_zc, off_yc, off_xc


def level_angle(angle_deg: float, skew: str) -> float:
    """Rotation that levels the stock deskew to the coverslip frame.

    Equals the deskew angle. For Y-skew this is a rotation about X;
    for X-skew, about Y. Sign is positive (validated against the two-pass oracle).
    """
    if skew not in ("Y", "X"):
        raise ValueError(f"Unknown skew {skew!r}; expected 'Y' or 'X'")
    return float(angle_deg)


def shear_only_display_affine_zyx(
    raw_shape_zyx: Tuple[int, int, int],
    angle_deg: float,
    dz: float,
    dy: float,
    dx: float,
    skew: str,
    invert_scan_direction: bool = False,
) -> np.ndarray:
    """Display affine (ZYX, 4x4) for the shear-only / OPM orientation.

    Returns a 4x4 matrix M such that output = M @ [z, y, x, 1]^T maps a raw
    voxel index (z=scan plane, y, x) to its position in the shear-only
    (coverslip_rotation=False) display frame.

    Forward map (Y-skew):
        zc = ss * z
        yc = cc * z + y
        xc = x

    Forward map (X-skew):
        zc = ss * z
        yc = y
        xc = cc * z + x

    where step = dz/dy (Y-skew) or dz/dx (X-skew),
    ss = sin(angle_rad) * step,
    cc = cos(angle_rad) * step.

    If invert_scan_direction=True the scan (Z) axis is flipped before applying
    the shear (z -> nz-1-z), so M_inv = M @ flip_zyx.
    """
    M = shear_only_forward_affine(angle_deg, dz, dy, dx, skew)

    if invert_scan_direction:
        nz = raw_shape_zyx[0]
        flip_zyx = np.eye(4)
        flip_zyx[0, 0] = -1.0
        flip_zyx[0, 3] = nz - 1
        return M @ flip_zyx

    return M
