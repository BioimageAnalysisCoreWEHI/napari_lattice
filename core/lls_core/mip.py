"""
Single-pass, memory-light deskewed MIP (no intermediate deskewed volume).
The main advantage here is that we create a MIP without deskewing the whole volume. 


Supports objective-frame and shear-only (coverslip) projections; see
``shear_only_geometry`` for the coordinate maps.  CPU kernel via numba.

Objective-frame is default deskewing with coverslip rotation
Shear-only: Deskew without coverslip rotation
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is expected, but degrade gracefully
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap

    prange = range  # type: ignore

# Warn at most once if we fall back to the un-jitted (pure-Python) kernel, which
# is orders of magnitude slower and effectively unusable on large volumes.
_warned_no_numba = False


def _warn_if_no_numba() -> None:
    global _warned_no_numba
    if not _HAVE_NUMBA and not _warned_no_numba:
        logger.warning(
            "numba is not installed; the deskewed MIP is running as a pure-Python "
            "loop, which is orders of magnitude slower and impractical for large "
            "volumes. Install numba (a declared dependency) for the JIT-compiled kernel."
        )
        _warned_no_numba = True


if TYPE_CHECKING:
    from lls_core import DeskewDirection
    from lls_core.models.deskew import DeskewParams
    from lls_core.types import ArrayLike


def _mip_output_size(n_scan: int, n_lat: int, scan_scale: float, cos_theta: float) -> int:
    """Length of the sheared output axis (the axis the shear shifts along)."""
    return int(math.ceil((n_scan - 1) * scan_scale + (n_lat - 1) * cos_theta)) + 1


@njit(parallel=True, cache=True)
def _pull_mip_y(data, scan_scale, cos_theta, out_rows, out_cols, linear):
    """Y-skew top-down MIP via bounded pull. data is (scan, raw_y, x); collapse
    the deskewed axial axis. Output is (out_rows, out_cols) = (sheared Y, X).
    Parallelise over output rows (yd)."""
    n_scan, n_y, n_x = data.shape
    out = np.zeros((out_rows, out_cols), dtype=data.dtype)
    cols = out_cols if out_cols < n_x else n_x
    for yd in prange(out_rows):
        # Bounded raw_y (j) range: the scan index n=(yd-cos*j)/scan_scale must
        # land in [0, n_scan). Solve for j and clip to [0, n_y).
        jlo = int(math.floor((yd - n_scan * scan_scale) / cos_theta))
        jhi = int(math.ceil(yd / cos_theta)) + 1
        if jlo < 0:
            jlo = 0
        if jhi > n_y:
            jhi = n_y
        for i in range(cols):
            m = out[yd, i]
            for j in range(jlo, jhi):
                nf = (yd - cos_theta * j) / scan_scale
                if linear:
                    # For integer `data`, the float blend is stored back into the
                    # integer-dtype output, so it truncates toward zero. That is
                    # acceptable for a MIP; use 'nearest' for an exact integer max.
                    n0 = int(math.floor(nf))
                    if n0 >= 0 and n0 + 1 < n_scan:
                        f = nf - n0
                        v = (1.0 - f) * data[n0, j, i] + f * data[n0 + 1, j, i]
                        if v > m:
                            m = v
                    elif 0 <= n0 < n_scan:
                        v = data[n0, j, i]
                        if v > m:
                            m = v
                else:
                    nn = int(nf + 0.5)
                    if 0 <= nn < n_scan:
                        v = data[nn, j, i]
                        if v > m:
                            m = v
            out[yd, i] = m
    return out


@njit(parallel=True, cache=True)
def _pull_mip_x(data, scan_scale, cos_theta, out_rows, out_cols, linear):
    """X-skew top-down MIP via bounded pull. data is (scan, y, raw_x); collapse
    the deskewed axial axis. Output is (out_rows, out_cols) = (Y, sheared X).
    Parallelise over output rows (yd = raw_y)."""
    n_scan, n_y, n_x = data.shape
    out = np.zeros((out_rows, out_cols), dtype=data.dtype)
    rows = out_rows if out_rows < n_y else n_y
    for yd in prange(rows):
        j = yd  # raw_y maps straight through for X-skew
        for xd in range(out_cols):
            ilo = int(math.floor((xd - n_scan * scan_scale) / cos_theta))
            ihi = int(math.ceil(xd / cos_theta)) + 1
            if ilo < 0:
                ilo = 0
            if ihi > n_x:
                ihi = n_x
            m = out[yd, xd]
            for i in range(ilo, ihi):
                nf = (xd - cos_theta * i) / scan_scale
                if linear:
                    n0 = int(math.floor(nf))
                    if n0 >= 0 and n0 + 1 < n_scan:
                        f = nf - n0
                        v = (1.0 - f) * data[n0, j, i] + f * data[n0 + 1, j, i]
                        if v > m:
                            m = v
                    elif 0 <= n0 < n_scan:
                        v = data[n0, j, i]
                        if v > m:
                            m = v
                else:
                    nn = int(nf + 0.5)
                    if 0 <= nn < n_scan:
                        v = data[nn, j, i]
                        if v > m:
                            m = v
            out[yd, xd] = m
    return out


@njit(parallel=True, cache=True)
def _pull_mip_y_shear_only(data, step, tan_t, sin_t, out_rows, out_cols, linear):
    """Y-skew shear-only (coverslip-frame) MIP via bounded pull.

    data is (scan, raw_y, x).  Collapses the coverslip-normal axis (zc) for
    each output pixel (yc=row, xc=col).  Frozen inverse map:
        plane  = zc / (sin_t * step)
        raw_y  = yc - zc / tan_t
        x      = xc (passthrough)
    Bilinear over (plane, raw_y).  Parallelise over output rows (yc).
    """
    n_scan, n_y, n_x = data.shape
    out = np.zeros((out_rows, out_cols), dtype=data.dtype)
    ss = sin_t * step  # zc spacing per scan-plane index: zc = ss * p_float
    nzc_max = int(math.ceil((n_scan - 1) * ss)) + 1  # max coverslip-Z extent
    cols = out_cols if out_cols < n_x else n_x
    for yc in prange(out_rows):
        for xc in range(cols):
            m = out[yc, xc]
            for zc in range(nzc_max):
                scan = zc / ss
                plane = int(math.floor(scan))
                if plane < 0 or plane + 1 >= n_scan:
                    continue
                fp = scan - plane
                raw_y = yc - zc / tan_t
                pos = int(math.floor(raw_y))
                if pos < 0 or pos + 1 >= n_y:
                    continue
                fy = raw_y - pos
                if linear:
                    v = ((1.0 - fp) * (1.0 - fy) * data[plane, pos, xc]
                         + (1.0 - fp) * fy * data[plane, pos + 1, xc]
                         + fp * (1.0 - fy) * data[plane + 1, pos, xc]
                         + fp * fy * data[plane + 1, pos + 1, xc])
                else:
                    # nearest in both dims
                    p0 = plane + 1 if fp >= 0.5 else plane
                    y0 = pos + 1 if fy >= 0.5 else pos
                    v = data[p0, y0, xc]
                if v > m:
                    m = v
            out[yc, xc] = m
    return out


@njit(parallel=True, cache=True)
def _pull_mip_x_shear_only(data, step, tan_t, sin_t, out_rows, out_cols, linear):
    """X-skew shear-only (coverslip-frame) MIP via bounded pull.

    data is (scan, y, raw_x).  Collapses the coverslip-normal axis (zc) for
    each output pixel (yc=row, xc=col).  Frozen inverse map:
        plane  = zc / (sin_t * step)
        raw_x  = xc - zc / tan_t
        y      = yc (passthrough)
    Bilinear over (plane, raw_x).  Parallelise over output rows (yc).
    """
    n_scan, n_y, n_x = data.shape
    out = np.zeros((out_rows, out_cols), dtype=data.dtype)
    ss = sin_t * step  # zc spacing per scan-plane index
    nzc_max = int(math.ceil((n_scan - 1) * ss)) + 1
    rows = out_rows if out_rows < n_y else n_y
    for yc in prange(rows):
        for xc in range(out_cols):
            m = out[yc, xc]
            for zc in range(nzc_max):
                scan = zc / ss
                plane = int(math.floor(scan))
                if plane < 0 or plane + 1 >= n_scan:
                    continue
                fp = scan - plane
                raw_x = xc - zc / tan_t
                pos = int(math.floor(raw_x))
                if pos < 0 or pos + 1 >= n_x:
                    continue
                fx = raw_x - pos
                if linear:
                    v = ((1.0 - fp) * (1.0 - fx) * data[plane, yc, pos]
                         + (1.0 - fp) * fx * data[plane, yc, pos + 1]
                         + fp * (1.0 - fx) * data[plane + 1, yc, pos]
                         + fp * fx * data[plane + 1, yc, pos + 1])
                else:
                    p0 = plane + 1 if fp >= 0.5 else plane
                    x0 = pos + 1 if fx >= 0.5 else pos
                    v = data[p0, yc, x0]
                if v > m:
                    m = v
            out[yc, xc] = m
    return out


def deskew_mip(
    data: "ArrayLike",
    angle_in_degrees: float = 30.0,
    voxel_size_z: float = 1.0,
    voxel_size_y: float = 1.0,
    voxel_size_x: float = 1.0,
    skew: "DeskewDirection | str" = "Y",
    interpolation: str = "nearest",
    target_shape: "Tuple[int, int] | None" = None,
    frame: str = "objective",
) -> np.ndarray:
    """
    Compute a deskewed maximum-intensity projection of a 3D raw stack without
    materialising the deskewed volume.

    Uses a bounded pull (gather) kernel, free of scatter artifacts (no
    gaps/striping, no edge dilation) regardless of the scan-step-to-pixel ratio.

    Args:
        data: 3D raw array (scan/Z, Y, X). Converted to a contiguous numpy array
            for the kernel (so it must fit in host RAM; for larger-than-RAM
            inputs, stream planes externally).
        angle_in_degrees: deskew angle relative to the coverslip.
        voxel_size_z: scan step between planes (along the scan direction).
        voxel_size_y, voxel_size_x: lateral pixel sizes.
        skew: deskew direction, "Y" or "X" (or a DeskewDirection).
        interpolation: "nearest" (default; fast, blocky) or "linear" (smoother,
            blends adjacent scan planes).
        target_shape: optional exact 2D output shape ``(rows, cols)`` to align
            the MIP grid with a reference deskew.  When omitted, the size is
            derived analytically.
        frame: "objective" (default) collapses the objective-frame axial axis;
            "shear_only" collapses the coverslip-normal axis (see shear_only_geometry).

    Returns:
        2D MIP, dtype matching the input.
    """
    _warn_if_no_numba()
    if frame not in ("objective", "shear_only"):
        raise ValueError(f"frame must be 'objective' or 'shear_only', got {frame!r}")
    skew_name = skew.name if hasattr(skew, "name") else str(skew)
    if interpolation not in ("nearest", "linear"):
        raise ValueError(f"interpolation must be 'nearest' or 'linear', got {interpolation!r}")
    linear = interpolation == "linear"

    arr = np.ascontiguousarray(data)
    if arr.ndim != 3:
        raise ValueError(f"deskew_mip expects a 3D (Z, Y, X) array, got {arr.ndim}D")

    n_scan, n_y, n_x = arr.shape

    if frame == "shear_only":
        # Shear-only (coverslip-frame) path: collapse zc, output is (yc, xc).
        from lls_core.shear_only_geometry import shear_only_output_shape, deskew_trig, pixel_step
        tan_t, sin_t, _ = deskew_trig(angle_in_degrees)
        raw_shape = (n_scan, n_y, n_x)
        if skew_name == "Y":
            step = pixel_step(voxel_size_z, voxel_size_y)
            if target_shape is not None:
                out_rows, out_cols = int(target_shape[0]), int(target_shape[1])
            else:
                _, out_rows, out_cols = shear_only_output_shape(
                    raw_shape, angle_in_degrees, voxel_size_z, voxel_size_y, voxel_size_x, "Y"
                )
            return _pull_mip_y_shear_only(arr, step, tan_t, sin_t, out_rows, out_cols, linear)
        elif skew_name == "X":
            step = pixel_step(voxel_size_z, voxel_size_x)
            if target_shape is not None:
                out_rows, out_cols = int(target_shape[0]), int(target_shape[1])
            else:
                _, out_rows, out_cols = shear_only_output_shape(
                    raw_shape, angle_in_degrees, voxel_size_z, voxel_size_y, voxel_size_x, "X"
                )
            return _pull_mip_x_shear_only(arr, step, tan_t, sin_t, out_rows, out_cols, linear)
        else:
            raise ValueError(f"Unknown skew direction {skew_name!r}; expected 'Y' or 'X'")

    # Objective-frame path (default; pre-existing behaviour).
    cos_theta = math.cos(math.radians(angle_in_degrees))
    if skew_name == "Y":
        scan_scale = voxel_size_z / voxel_size_y
        if target_shape is not None:
            out_rows, out_cols = int(target_shape[0]), int(target_shape[1])
        else:
            out_rows = _mip_output_size(n_scan, n_y, scan_scale, cos_theta)
            out_cols = n_x
        return _pull_mip_y(arr, scan_scale, cos_theta, out_rows, out_cols, linear)
    elif skew_name == "X":
        scan_scale = voxel_size_z / voxel_size_x
        if target_shape is not None:
            out_rows, out_cols = int(target_shape[0]), int(target_shape[1])
        else:
            out_rows = n_y
            out_cols = _mip_output_size(n_scan, n_x, scan_scale, cos_theta)
        return _pull_mip_x(arr, scan_scale, cos_theta, out_rows, out_cols, linear)
    else:
        raise ValueError(f"Unknown skew direction {skew_name!r}; expected 'Y' or 'X'")


def deskew_mip_from_lattice(
    deskew: "DeskewParams",
    time: int = 0,
    channel: int = 0,
    interpolation: str = "nearest",
) -> np.ndarray:
    """
    Convenience wrapper: compute the deskewed MIP for one (time, channel) of a
    DeskewParams/LatticeData, honouring its angle, pixel sizes, skew direction
    and `invert_scan_direction` (the scan flip is applied by `get_3d_slice`).

    The output grid is pinned to the lattice's deskewed shape
    (`derived.deskew_vol_shape`), so the MIP aligns exactly with a full
    deskew -- ROIs drawn on it are in the same coordinate frame the crop
    pipeline expects. (No pyclesperanto call is added here; the shape is read
    from the already-computed derived field.)
    """
    raw_3d = deskew.get_3d_slice()
    target_shape = None
    try:
        zd, yd, xd = deskew.derived.deskew_vol_shape
        target_shape = (int(yd), int(xd))  # MIP collapses the deskewed Z axis
    except Exception:
        target_shape = None
    return deskew_mip(
        raw_3d.data,
        angle_in_degrees=deskew.angle,
        voxel_size_z=deskew.dz,
        voxel_size_y=deskew.dy,
        voxel_size_x=deskew.dx,
        skew=deskew.skew,
        interpolation=interpolation,
        target_shape=target_shape,
        frame="shear_only" if not deskew.coverslip_rotation else "objective",
    )
