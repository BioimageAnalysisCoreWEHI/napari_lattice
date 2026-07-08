"""Run the custom shear-only (coverslip-frame) deskew OpenCL kernel.

Sibling of the stock cle.deskew_y / cle.deskew_x; reached only when
coverslip_rotation is False (OPM/SOPi mode).  The kernel is a single-pass inverse gather
implementing the frozen shear-only map (validated vs the two-pass oracle).

Public API
----------
shear_only_deskew(source, angle_in_degrees, voxel_size_z, voxel_size_y,
                 voxel_size_x, skew="Y") -> OCLArray
    GPU array in the coverslip (level) frame; call cle.pull() to convert to numpy.
    Output shape comes from shear_only_geometry.shear_only_output_shape.
"""
from __future__ import annotations

import numpy as np
import pyclesperanto_prototype as cle
from pyclesperanto_prototype._tier0 import execute, create, push

from lls_core.shear_only_geometry import shear_only_output_shape, deskew_trig, pixel_step


def shear_only_deskew(
    source,
    angle_in_degrees: float,
    voxel_size_z: float,
    voxel_size_y: float,
    voxel_size_x: float,
    skew: str = "Y",
):
    """Deskew *source* into the coverslip (level) frame using a single-pass OpenCL kernel.

    Voxel sizes must share the same units. Returns an OCLArray; call ``cle.pull(dest)``
    to get a numpy array. Output shape from ``shear_only_geometry.shear_only_output_shape``.
    """
    # Push source to GPU (no-op if already an OCLArray)
    src = push(np.ascontiguousarray(np.asarray(source, dtype=np.float32)))
    raw_shape = tuple(int(s) for s in src.shape)  # (Nz, Ny, Nx)

    # Compute output shape and allocate destination
    out_shape = shear_only_output_shape(
        raw_shape, angle_in_degrees, voxel_size_z, voxel_size_y, voxel_size_x, skew
    )
    dest = create(out_shape)  # (nz, ny, nx)

    # Trig values from the deskew angle (cos not used in the frozen map)
    tan_t, sin_t, _ = deskew_trig(angle_in_degrees)

    # Pixel step (scan-axis voxels per lateral pixel)
    if skew == "Y":
        suffix = "y"
        step = pixel_step(voxel_size_z, voxel_size_y)
    elif skew == "X":
        suffix = "x"
        step = pixel_step(voxel_size_z, voxel_size_x)
    else:
        raise ValueError(f"Unknown skew {skew!r}; expected 'Y' or 'X'")

    params = {
        "input": src,
        "output": dest,
        "pixel_step": np.float32(step),
        "tantheta": np.float32(tan_t),
        "sintheta": np.float32(sin_t),
    }

    execute(
        __file__,
        f"kernels/affine_transform_deskew_{suffix}_shear_only_3d_x.cl",
        f"affine_transform_deskew_{suffix}_shear_only_3d",
        dest.shape,
        params,
    )
    return dest
