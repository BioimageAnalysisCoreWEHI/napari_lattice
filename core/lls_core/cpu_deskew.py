"""CPU (Numba) implementation of the orthogonal-interpolation deskew algorithm.

This is a CPU-only counterpart to `affine_transform_deskew.py`'s OpenCL kernel
(Sapoznik et al. 2020, https://doi.org/10.7554/eLife.57681), for users without a
working GPU/OpenCL device. It exists as an alternative *engine* for the same
standard (coverslip-rotation) deskew, not as a separate deskewing mode -- see
`DeskewEngine` in `lls_core/__init__.py`.

The Numba loop structure (row-vectorised bilinear interpolation between two
scan planes) is adapted from the qi2lab opm-processing-v2 project's
`opm_processing.imageprocessing.opmtools.orthogonal_deskew`
(https://github.com/QI2lab/opm-processing-v2). Two things are changed from
that source so the CPU engine is a drop-in swap for the existing GPU engine:

- The per-voxel interpolation weights match this project's own
  `affine_transform_deskew_y_3d_x.cl` / `affine_transform_deskew_x_3d_x.cl`
  kernels (weighted by `l_before`/`l_after`, not opm-processing-v2's unweighted
  variant), and the output Z axis is flipped the same way those kernels flip it
  unconditionally. This keeps CPU and GPU output numerically equivalent for the
  same input, rather than introducing a second, subtly different deskew result.
- The output shape is supplied by the caller (`DerivedDeskewFields.deskew_vol_shape`,
  the same affine-bounding-box shape the GPU path produces) instead of being
  recomputed from opm-processing-v2's own shape-estimator/padding conventions, so
  downstream code (writers, MIP, cropping) sees one consistent shape regardless
  of which engine ran.

Public API
----------
cpu_deskew(source, angle_in_degrees, voxel_size_x, voxel_size_y, voxel_size_z,
           deskew_direction, output_shape) -> np.ndarray
    Deskewed volume (float32) in the objective frame, in the same orientation as
    `affine_transform_deskew_3d`'s GPU output. Call sites treat it like a plain
    numpy array (`cle.pull` passes plain arrays through unchanged).
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import ArrayLike

from lls_core import DeskewDirection

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

_warned_no_numba = False


def _warn_if_no_numba() -> None:
    global _warned_no_numba
    if not _HAVE_NUMBA and not _warned_no_numba:
        logger.warning(
            "numba is not installed; CPU deskewing is running as a pure-Python "
            "loop, which is orders of magnitude slower and impractical for large "
            "volumes. Install numba (a declared dependency) for the JIT-compiled kernel."
        )
        _warned_no_numba = True


@njit(parallel=True, cache=True)
def _orthogonal_deskew_y(data, out_nz, out_ny, out_nx, pixel_step, tantheta, sintheta, costheta):
    num_images, ny, nx = data.shape
    ncols = out_nx if out_nx < nx else nx
    output = np.zeros((out_nz, out_ny, out_nx), dtype=np.float32)
    one = np.float32(1.0)

    for z in prange(out_nz):
        # Every scalar here is kept in float32, matching the GPU kernel's `float`
        # arithmetic bit-for-bit as closely as numba allows. A stray float64
        # promotion (e.g. from a bare `1.0` literal or an un-cast loop index)
        # can shift a floor() result by one grid cell right at a boundary,
        # silently swapping in a different source voxel from the GPU engine.
        zf = np.float32(z)
        za = zf / sintheta
        for y in range(out_ny):
            virtual_plane = np.float32(y) - zf / tantheta
            plane_before = int(math.floor(virtual_plane / pixel_step))
            plane_after = plane_before + 1
            if plane_before < 0 or plane_after >= num_images:
                continue

            l_before = virtual_plane - np.float32(plane_before) * pixel_step
            l_after = pixel_step - l_before

            virtual_pos_before = za + l_before * costheta
            virtual_pos_after = za - l_after * costheta

            pos_before = int(math.floor(virtual_pos_before))
            pos_after = int(math.floor(virtual_pos_after))

            if (pos_before < 0 or pos_after < 0 or
                    pos_before + 1 >= ny or pos_after + 1 >= ny):
                continue

            dz_before = virtual_pos_before - np.float32(pos_before)
            dz_after = virtual_pos_after - np.float32(pos_after)

            pix_1 = data[plane_after, pos_after + 1, :ncols]
            pix_2 = data[plane_after, pos_after, :ncols]
            pix_3 = data[plane_before, pos_before + 1, :ncols]
            pix_4 = data[plane_before, pos_before, :ncols]

            values = (
                l_before * dz_after * pix_1
                + l_before * (one - dz_after) * pix_2
                + l_after * dz_before * pix_3
                + l_after * (one - dz_before) * pix_4
            ) / pixel_step

            # Match the GPU kernel's unconditional Z flip (camera <-> stage orientation).
            output[out_nz - 1 - z, y, :ncols] = values

    return output


@njit(parallel=True, cache=True)
def _orthogonal_deskew_x(data, out_nz, out_ny, out_nx, pixel_step, tantheta, sintheta, costheta):
    num_images, ny, nx = data.shape
    nrows = out_ny if out_ny < ny else ny
    output = np.zeros((out_nz, out_ny, out_nx), dtype=np.float32)
    one = np.float32(1.0)

    for z in prange(out_nz):
        zf = np.float32(z)
        za = zf / sintheta
        for x in range(out_nx):
            virtual_plane = np.float32(x) - zf / tantheta
            plane_before = int(math.floor(virtual_plane / pixel_step))
            plane_after = plane_before + 1
            if plane_before < 0 or plane_after >= num_images:
                continue

            l_before = virtual_plane - np.float32(plane_before) * pixel_step
            l_after = pixel_step - l_before

            virtual_pos_before = za + l_before * costheta
            virtual_pos_after = za - l_after * costheta

            pos_before = int(math.floor(virtual_pos_before))
            pos_after = int(math.floor(virtual_pos_after))

            if (pos_before < 0 or pos_after < 0 or
                    pos_before + 1 >= nx or pos_after + 1 >= nx):
                continue

            dz_before = virtual_pos_before - np.float32(pos_before)
            dz_after = virtual_pos_after - np.float32(pos_after)

            pix_1 = data[plane_after, :nrows, pos_after + 1]
            pix_2 = data[plane_after, :nrows, pos_after]
            pix_3 = data[plane_before, :nrows, pos_before + 1]
            pix_4 = data[plane_before, :nrows, pos_before]

            values = (
                l_before * dz_after * pix_1
                + l_before * (one - dz_after) * pix_2
                + l_after * dz_before * pix_3
                + l_after * (one - dz_before) * pix_4
            ) / pixel_step

            output[out_nz - 1 - z, :nrows, x] = values

    return output


def cpu_deskew(
    source: ArrayLike,
    angle_in_degrees: float,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    deskew_direction: DeskewDirection,
    output_shape: "tuple[int, int, int]",
) -> np.ndarray:
    """Deskew *source* on the CPU using orthogonal interpolation.

    Parameters
    ----------
    source: ArrayLike
        3D raw (oblique) volume, in (Z, Y, X) order.
    angle_in_degrees: float
        Deskewing angle.
    voxel_size_x, voxel_size_y, voxel_size_z: float
        Physical voxel sizes, in the same units.
    deskew_direction: DeskewDirection
        Which raw axis (X or Y) the scan is skewed along.
    output_shape: tuple[int, int, int]
        Target (Z, Y, X) shape of the deskewed output. This should be
        `DerivedDeskewFields.deskew_vol_shape`, so the CPU and GPU engines agree
        on the output geometry.

    Returns
    -------
    np.ndarray
        Deskewed volume, float32, shape `output_shape`.
    """
    _warn_if_no_numba()

    source_arr = np.ascontiguousarray(np.asarray(source, dtype=np.float32))
    assert source_arr.ndim == 3, f"Image needs to be 3D, got shape of {source_arr.shape}"

    theta_rad = math.radians(angle_in_degrees)
    # Narrow to float32 once, here, the same way the GPU path computes these in
    # float64 and then hands them to the OpenCL kernel's `float` parameters -
    # so the two engines round to the same single-precision values before any
    # boundary (floor()) decisions are made downstream.
    tantheta = np.float32(math.tan(theta_rad))
    sintheta = np.float32(math.sin(theta_rad))
    costheta = np.float32(math.cos(theta_rad))

    out_nz, out_ny, out_nx = (int(s) for s in output_shape)

    if deskew_direction == DeskewDirection.Y:
        pixel_step = np.float32(voxel_size_z / voxel_size_y)
        return _orthogonal_deskew_y(source_arr, out_nz, out_ny, out_nx, pixel_step, tantheta, sintheta, costheta)
    elif deskew_direction == DeskewDirection.X:
        pixel_step = np.float32(voxel_size_z / voxel_size_x)
        return _orthogonal_deskew_x(source_arr, out_nz, out_ny, out_nx, pixel_step, tantheta, sintheta, costheta)
    else:
        raise ValueError(f"Unknown deskew_direction {deskew_direction!r}")
