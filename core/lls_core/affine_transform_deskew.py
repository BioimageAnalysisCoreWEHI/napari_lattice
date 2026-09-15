"""Run the custom orthogonal-interpolation deskew OpenCL kernel (Sapoznik et al. 2020).

This is the objective-frame counterpart to `shear_only_deskew.py`'s coverslip-frame
kernel, used when `coverslip_rotation` is True. It has no equivalent in the new
`pyclesperanto` library (public or private) so, like `shear_only_deskew.py`, it is
vendored here and run directly through `pyclesperanto.execute`.

Ported from pyclesperanto_prototype._tier8._affine_transform_deskew_3d.affine_transform_deskew_3d.

Public API
----------
affine_transform_deskew_3d(source, transform, deskewing_angle_in_degrees,
                            voxel_size_x, voxel_size_y, voxel_size_z,
                            deskew_direction) -> Array
    GPU array in the deskewed (objective) frame; call cle.pull() to convert to numpy.
"""
from __future__ import annotations

import numpy as np
# `import pyclesperanto as cle` and access `cle.create`/etc. at call time rather than
# `from pyclesperanto import create, execute, push`: those names are only bound once a
# GPU/OpenCL backend is available, so a `from`-import fails at module-import time in
# backend-less environments (e.g. building the docs, which imports this module transitively
# but never calls it).
import pyclesperanto as cle

from lls_core import DeskewDirection
from lls_core.affine import AffineTransform3D, determine_translation_and_bounding_box


def affine_transform_deskew_3d(
    source,
    transform: AffineTransform3D,
    deskewing_angle_in_degrees: float = 30,
    voxel_size_x: float = 0.1449922,
    voxel_size_y: float = 0.1449922,
    voxel_size_z: float = 0.3,
    deskew_direction: DeskewDirection = DeskewDirection.Y,
):
    """Applies an affine transform to deskew an image using orthogonal interpolation
    (Sapoznik et al. (2020) https://doi.org/10.7554/eLife.57681).
    """
    assert len(source.shape) == 3, f"Image needs to be 3D, got shape of {len(source.shape)}"

    new_size, transform, _ = determine_translation_and_bounding_box(source, transform)
    # `create()` defaults to float64 (Python's `float`) when no dtype is given, but
    # `source` below and the kernel's float-typed parameters are float32. A float64
    # destination against a kernel compiled for float32 is a real buffer/kernel
    # mismatch - harmless on backends that silently reinterpret it, but on backends
    # without double-precision support (e.g. CI's pocl CPU backend) it silently
    # produces garbage/zeroed output instead of raising.
    destination = cle.create(new_size, dtype=np.float32)
    # Unlike the old backend, pyclesperanto.execute doesn't auto-push plain numpy
    # arrays passed in the parameters dict - push explicitly.
    source = cle.push(np.ascontiguousarray(np.asarray(source, dtype=np.float32)))

    # we invert the transform because we go from the target image to the source image to read pixels
    # The kernel's `mat` buffer is read as float (see the .cl file); numpy's affine inverse is
    # float64 by default, so cast explicitly - an uncast float64 buffer being read back as
    # float32 corrupts the GPU memory pool for later, unrelated pyclesperanto calls.
    transform_matrix = np.ascontiguousarray(np.asarray(transform.copy().inverse()), dtype=np.float32)

    tantheta = float(np.tan(deskewing_angle_in_degrees * np.pi / 180))
    sintheta = float(np.sin(deskewing_angle_in_degrees * np.pi / 180))
    costheta = float(np.cos(deskewing_angle_in_degrees * np.pi / 180))

    gpu_transform_matrix = cle.push(transform_matrix)

    if deskew_direction == DeskewDirection.Y:
        kernel_suffix = "deskew_y_"
        pixel_step = float(voxel_size_z / voxel_size_y)
    else:
        kernel_suffix = "deskew_x_"
        pixel_step = float(voxel_size_z / voxel_size_x)

    parameters = {
        "input": source,
        "output": destination,
        "mat": gpu_transform_matrix,
        "pixel_step": pixel_step,
        "tantheta": tantheta,
        "costheta": costheta,
        "sintheta": sintheta,
    }

    cle.execute(
        __file__,
        f"kernels/affine_transform_{kernel_suffix}{len(destination.shape)}d_x.cl",
        f"affine_transform_{kernel_suffix}{len(destination.shape)}d",
        destination.shape,
        parameters=parameters,
    )

    return destination
