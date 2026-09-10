"""
Minimal, local replacement for `pyclesperanto_prototype`'s private
`AffineTransform3D` and `_determine_translation_and_bounding_box`.

Neither exists in the new `pyclesperanto` library (a from-scratch rewrite with a much
thinner Python surface), but both are pure numpy/`transforms3d` matrix math with no
OpenCL dependency, so they're ported here verbatim from
`pyclesperanto_prototype._tier8._AffineTransform3D` and `._affine_transform` to keep
the deskew geometry byte-for-byte identical. Only the methods actually used elsewhere
in this codebase are included.
"""
from __future__ import annotations

import math
from itertools import product
from typing import Tuple
from warnings import warn

import numpy as np
import transforms3d


class AffineTransform3D:
    """Convenience class to build up a 4x4 affine transform matrix.

    Ported from pyclesperanto_prototype._tier8._AffineTransform3D.AffineTransform3D.
    """

    def __init__(self):
        self._matrix = transforms3d.zooms.zfdir2aff(1)

    def scale(self, scale_x: float = None, scale_y: float = None, scale_z: float = None) -> AffineTransform3D:
        if scale_x == 0:
            warn('scale_x must not be 0')
            scale_x = 1
        if scale_y == 0:
            warn('scale_y must not be 0')
            scale_y = 1
        if scale_z == 0:
            warn('scale_z must not be 0')
            scale_z = 1
        if scale_x is not None:
            self._concatenate(transforms3d.zooms.zfdir2aff(scale_x, direction=(1, 0, 0), origin=(0, 0, 0)))
        if scale_y is not None:
            self._concatenate(transforms3d.zooms.zfdir2aff(scale_y, direction=(0, 1, 0), origin=(0, 0, 0)))
        if scale_z is not None:
            self._concatenate(transforms3d.zooms.zfdir2aff(scale_z, direction=(0, 0, 1), origin=(0, 0, 0)))
        return self

    def rotate(self, axis: int = 2, angle_in_degrees: float = 0) -> AffineTransform3D:
        angle_in_rad = angle_in_degrees * np.pi / 180.0
        if axis == 0:
            self._concatenate(self._3x3_to_4x4(transforms3d.euler.euler2mat(angle_in_rad, 0, 0)))
        if axis == 1:
            self._concatenate(self._3x3_to_4x4(transforms3d.euler.euler2mat(0, angle_in_rad, 0)))
        if axis == 2:
            self._concatenate(self._3x3_to_4x4(transforms3d.euler.euler2mat(0, 0, angle_in_rad)))
        return self

    def translate(self, translate_x: float = 0, translate_y: float = 0, translate_z: float = 0) -> AffineTransform3D:
        self._concatenate(np.asarray([
            [1, 0, 0, translate_x],
            [0, 1, 0, translate_y],
            [0, 0, 1, translate_z],
            [0, 0, 0, 1],
        ]))
        return self

    def _deskew_y(self, angle_in_degrees: float, voxel_size_x: float = 1,
        voxel_size_y: float = 1, voxel_size_z: float = 1, scale_factor: float = 1) -> AffineTransform3D:
        shear_factor = math.sin((90 - angle_in_degrees) * math.pi / 180.0) * (voxel_size_z / voxel_size_y)
        self._matrix[1, 2] = shear_factor

        new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
        scale_factor_z = (new_dz / voxel_size_y) * scale_factor
        self.scale(scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z)

        self.rotate(angle_in_degrees=0 - angle_in_degrees, axis=0)
        return self

    def _deskew_x(self, angle_in_degrees: float, voxel_size_x: float = 1,
                    voxel_size_y: float = 1, voxel_size_z: float = 1, scale_factor: float = 1) -> AffineTransform3D:
        shear_factor = math.sin((90 - angle_in_degrees) * math.pi / 180.0) * (voxel_size_z / voxel_size_x)
        self._matrix[0, 2] = shear_factor

        new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
        scale_factor_z = (new_dz / voxel_size_x) * scale_factor
        self.scale(scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z)

        self.rotate(angle_in_degrees=angle_in_degrees, axis=1)
        return self

    def _3x3_to_4x4(self, matrix):
        mat = np.pad(matrix, (0, 1), 'constant', constant_values=(0, 0))
        mat[3, 3] = 1
        return mat

    def concatenate(self, transform: AffineTransform3D) -> AffineTransform3D:
        self._concatenate(transform._matrix)
        return self

    def _concatenate(self, matrix):
        self._matrix = np.matmul(matrix, self._matrix)

    def inverse(self) -> AffineTransform3D:
        self._matrix = np.linalg.inv(self._matrix)
        return self

    def copy(self) -> AffineTransform3D:
        a_copy = AffineTransform3D()
        a_copy._matrix = np.copy(self._matrix)
        return a_copy

    def __array__(self):
        return self._matrix


def determine_translation_and_bounding_box(source, affine_transformation: AffineTransform3D):
    """Starting from a given input image (or any object with a `.shape` attribute) and
    an affine transform, compute the output size of the new image and a translation
    vector necessary to keep all pixels in positive coordinates.

    Ported from pyclesperanto_prototype._tier8._affine_transform._determine_translation_and_bounding_box.
    Only `source.shape` is read, so a lightweight shape-only stand-in works too.

    Parameters
    ----------
    source: Any
        Object exposing a `.shape` attribute for the image to be transformed
    affine_transformation: AffineTransform3D
        The transform to be applied

    Returns
    -------
    new_shape: List[int]
        Size of output image (z, y, x)
    new_affine_transform: AffineTransform3D
        Modified transform so that all pixels remain in positive coordinates
    translation: np.ndarray
        Translation vector that is necessary to keep all pixels in positive coordinates
    """
    source_shape = source.shape
    if len(source_shape) == 2:
        ny, nz = source_shape
        nx = 1
    else:
        nx, ny, nz = source_shape

    original_bounding_box = [list(x) + [1] for x in product((0, nz), (0, ny), (0, nx))]
    transformed_bounding_box = np.asarray(
        list(map(lambda x: affine_transformation._matrix @ x, original_bounding_box)))

    min_coordinate = transformed_bounding_box.min(axis=0)
    max_coordinate = transformed_bounding_box.max(axis=0)
    new_shape = np.around((max_coordinate - min_coordinate)[0:3]).astype(int).tolist()[::-1]

    new_affine_transform = AffineTransform3D()
    new_affine_transform.concatenate(affine_transformation)

    translation = -min_coordinate
    new_affine_transform.translate(
        translate_x=translation[0],
        translate_y=translation[1],
        translate_z=translation[2],
    )

    if len(source_shape) == 2:
        return new_shape[1:], new_affine_transform, translation[1:3]
    else:
        return new_shape, new_affine_transform, translation[0:3]
