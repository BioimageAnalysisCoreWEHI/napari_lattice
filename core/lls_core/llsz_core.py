from __future__ import annotations

import numpy as np
import pyclesperanto_prototype as cle
from dask.array.core import Array as DaskArray
import dask.array as da
from resource_backed_dask_array import ResourceBackedDaskArray
from typing import Any, Optional, Union, TYPE_CHECKING, overload, Literal, Tuple
from typing_extensions import Unpack, TypedDict, Required
from pyclesperanto_prototype._tier8._affine_transform_deskew_3d import (
    affine_transform_deskew_3d,
)
from numpy.typing import NDArray
 
from lls_core.utils import calculate_crop_bbox
from lls_core import config, DeskewDirection
from lls_core.types import ArrayLike
from lls_core.deconvolution import pycuda_decon, skimage_decon, DeconvolutionChoice

# Enable Logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(config.log_level)
# pass shapes data from single ROI to crop the volume from original data

if TYPE_CHECKING:
    from napari.layers.shapes import Shapes

Psf = Union[
        NDArray,
        DaskArray,
        ResourceBackedDaskArray,
        cle._tier0._pycl.OCLArray,
]

class CommonArgs(TypedDict, total=False):
    original_volume: Required[ArrayLike]
    deskewed_volume: Union[ ArrayLike, None ]
    roi_shape: Union[list, NDArray, None]
    angle_in_degrees: float
    voxel_size_x: float
    voxel_size_y: float
    voxel_size_z: float
    z_start: int
    z_end: int
    deconvolution: bool
    decon_processing: Optional[DeconvolutionChoice]
    psf: Union[Psf, None]
    num_iter: int
    linear_interpolation: bool
    skew_dir: DeskewDirection

@overload
def crop_volume_deskew(*, debug: Literal[True], get_deskew_and_decon: bool = False, **kwargs: Unpack[CommonArgs]) -> Tuple[NDArray, NDArray]:
    ...
@overload
def crop_volume_deskew(*, debug: Literal[False] = False, get_deskew_and_decon: Literal[True], **kwargs: Unpack[CommonArgs]) -> Tuple[NDArray, NDArray]:
    ...
@overload
def crop_volume_deskew(*, debug: Literal[False] = False, get_deskew_and_decon: Literal[False] = False, **kwargs: Unpack[CommonArgs]) -> NDArray:
    ...
def crop_volume_deskew(
    original_volume: ArrayLike,
    deskewed_volume: Union[ ArrayLike, None ] = None,
    roi_shape: Union[list, NDArray, None] = None,
    angle_in_degrees: float = 30,
    voxel_size_x: float = 1,
    voxel_size_y: float = 1,
    voxel_size_z: float = 1,
    z_start: int = 0,
    z_end: int = 1,
    debug: bool = False,
    deconvolution: bool = False,
    decon_processing: Optional[DeconvolutionChoice]=None,
    psf: Union[Psf, None]=None,
    num_iter: int = 10,
    linear_interpolation: bool=True,
    skew_dir: DeskewDirection=DeskewDirection.Y,
    get_deskew_and_decon: bool = False,
):
    """Crop the volume from original data and deskew the cropped volume
    Args:
        original_volume (Union[da.core.Array,np.ndarray,cle._tier0._pycl.OCLArray,resource_backed_dask_array.ResourceBackedDaskArray]): Volume to deskew (zyx)
        deskewed_volume:DEPRECATED
        roi_shape (Union[shapes.Shapes,list,np.array]): shapes layer or rois
        angle_in_degrees (float, optional): deskewing angle in degrees. Defaults to 30.
        voxel_size_x (float, optional): microns. Defaults to 1.
        voxel_size_y (float, optional): microns.  Defaults to 1.
        voxel_size_z (float, optional): microns.  Defaults to 1.
        z_start (int, optional): Currently not used, but can be used to crop the volume in z. Defaults to 0.
        z_end (int, optional): _description_. Currently not used, but can be used to crop the volume in z. Defaults to 1.
        debug (bool, optional): If True, returns the cropped volume and the cropped volume with extra bounds. Defaults to False.
        deconvolution (bool, optional): Perform deconvolution. Defaults to False.
        decon_processing (str, optional): Choose decon option, cuda_gpu or cpu. Defaults to None.
        psf (_type_, optional): Pass a psf array for deconvolution. Defaults to None.
        num_iter (int, optional): Number of Iterations for Richardson Lucy deconvolution. Defaults to 10.
        linear_interpolation (bool, optional): Linear Interpolation after deskewing. Defaults to True.
        skew_direct (DeskewDirection, optional): Deskew direction. Defaults to DeskewDirection.Y.
        get_deskew_no_decon (bool, optional): Return both deconvolved data and deskewed data with no deconvolution. Defaults to False.

    Returns:
        _type_: _description_
    """

    assert len(original_volume.shape) == 3, print(
        "Shape of original volume must be 3"
    )
    # assert len(deskewed_volume.shape) == 3, print("Shape of deskewed volume must be 3")
    # assert len(shape) == 4, print("Shape must be an array of shape 4 ")
    shape = None

    # if shapes layer, get first one
    # TODO: test this
    # if is_napari_shape(roi_shape):
    #     shape = roi_shape.data[0]
    # if its a list and each element has a shape of 4, its a list of rois
    if isinstance(roi_shape, list) and len(roi_shape[0]) == 4:
        # TODO:change to accept any roi by passing index
        shape = roi_shape[0]
        # len(roi_shape) >= 1:
        # if its a array or list with shape of 4, its a single ROI
    elif len(roi_shape) == 4 and isinstance(roi_shape, (np.ndarray, list)):
        shape = roi_shape

    assert len(shape) == 4, print("Shape must be an array of shape 4")

    crop_bounding_box, crop_vol_shape = calculate_crop_bbox(
        shape, z_start, z_end
    )

    # get reverse transform by rotating around original volume
    (
        reverse_aff,
        excess_bounds,
        deskew_transform,
    ) = get_inverse_affine_transform(
        original_volume,
        angle_in_degrees,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        skew_dir,
    )

    # apply the transform to get corresponding bounding boxes in original volume
    crop_transform_bbox = np.asarray(
        list(map(lambda x: reverse_aff._matrix @ x, crop_bounding_box))
    )

    # get shape of original volume in xyz
    orig_img_shape = original_volume.shape[::-1]

    # Take min and max of the cropped bounding boxes to define min and max coordinates
    # crop_transform_bbox is in the form xyz

    min_coordinate = np.around(crop_transform_bbox.min(axis=0))
    max_coordinate = np.around(crop_transform_bbox.max(axis=0))

    # get min and max in each position
    # clip them to avoid negative values and any values outside the bounding box of original volume
    x_start = min_coordinate[0].astype(int)
    x_start = np.clip(x_start, 0, orig_img_shape[0])
    x_end = max_coordinate[0].astype(int)
    x_end = np.clip(x_end, 0, orig_img_shape[0])

    y_start = min_coordinate[1].astype(int)
    y_start = np.clip(y_start, 0, orig_img_shape[1])

    y_end = max_coordinate[1].astype(int)
    y_end = np.clip(y_end, 0, orig_img_shape[1])

    z_start_vol_prelim = min_coordinate[2].astype(int)
    # clip to z bounds of original volume
    z_start_vol = np.clip(z_start_vol_prelim, 0, orig_img_shape[2])

    z_end_vol_prelim = max_coordinate[2].astype(int)
    # clip to z bounds of original volume
    z_end_vol = np.clip(z_end_vol_prelim, 0, orig_img_shape[2])

    # If the coordinates are out of bound, then the final volume needs adjustment in Y axis
    # if skew in X direction, then use y axis for finding correction factor instead
    if z_end_vol_prelim != z_end_vol:
        out_bounds_correction = z_end_vol_prelim - z_end_vol
    elif z_start_vol_prelim != z_start_vol:
        out_bounds_correction = z_start_vol_prelim - z_start_vol
    else:
        out_bounds_correction = 0

    # make sure z_start < z_end
    if z_start_vol > z_end_vol:
        # tuple swap  #https://docs.python.org/3/reference/expressions.html#evaluation-order
        z_start_vol, z_end_vol = z_end_vol, z_start_vol

    # After getting the coordinates, crop from original volume and deskew only the cropped volume

    if isinstance(original_volume, (
        DaskArray,
        ResourceBackedDaskArray,
    )):
        # If using dask, use .map_blocks(np.copy) to copy subset (faster)
        crop_volume = (
            original_volume[
                z_start_vol:z_end_vol, y_start:y_end, x_start:x_end
            ]
            .map_blocks(np.copy)
            .squeeze()
        )
    else:
        crop_volume = original_volume[
            z_start_vol:z_end_vol, y_start:y_end, x_start:x_end
        ]

    # check if deconvolution is checked
    if deconvolution:
        if decon_processing == DeconvolutionChoice.cuda_gpu:
            crop_volume_processed = pycuda_decon(
                image=crop_volume,
                psf=psf,
                dzdata=voxel_size_z,
                dxdata=voxel_size_x,
                dzpsf=voxel_size_z,
                dxpsf=voxel_size_x,
                num_iter=num_iter,
                cropping=True,
            )
        else:
            crop_volume_processed = skimage_decon(
                vol_zyx=crop_volume,
                psf=psf,
                num_iter=num_iter,
                clip=False,
                filter_epsilon=0,
                boundary="nearest",
            )

        deskewed_prelim = affine_transform_deskew_3d(
            crop_volume_processed,
            transform=deskew_transform,
            deskewing_angle_in_degrees=angle_in_degrees,
            voxel_size_x=voxel_size_x,
            voxel_size_y=voxel_size_y,
            voxel_size_z=voxel_size_z,
            deskew_direction=skew_dir,
        )
        if get_deskew_and_decon:
            deskewed_no_decon = affine_transform_deskew_3d(
                crop_volume,
                transform=deskew_transform,
                deskewing_angle_in_degrees=angle_in_degrees,
                voxel_size_x=voxel_size_x,
                voxel_size_y=voxel_size_y,
                voxel_size_z=voxel_size_z,
                deskew_direction=skew_dir,
            )
    else:
        deskewed_prelim = affine_transform_deskew_3d(
            crop_volume,
            transform=deskew_transform,
            deskewing_angle_in_degrees=angle_in_degrees,
            voxel_size_x=voxel_size_x,
            voxel_size_y=voxel_size_y,
            voxel_size_z=voxel_size_z,
            deskew_direction=skew_dir,
        )

    # The height of deskewed_prelim will be larger than specified shape
    # as the coordinates of the ROI are skewed in the original volume
    # IF CLIPPING HAPPENS FOR Y_START or Y_END, use difference to calculate offset
    if skew_dir == DeskewDirection.Y:
        deskewed_height = deskewed_prelim.shape[1]
        crop_height = crop_vol_shape[1]

        # Find "excess" volume on both sides due to deskewing
        crop_excess: int = max(
            int(round((deskewed_height - crop_height) / 2)) + out_bounds_correction,
            0
        )
        # Crop in Y
        deskewed_prelim = np.asarray(deskewed_prelim)
        deskewed_crop = deskewed_prelim[
            :, crop_excess : crop_height + crop_excess, :
        ]
    # IF CLIPPING HAPPENS FOR X_START or X_END, use difference to calculate offset
    elif skew_dir == DeskewDirection.X:
        deskewed_width = deskewed_prelim.shape[2]
        crop_width = crop_vol_shape[2]
        
        # Find "excess" volume on both sides due to deskewing
        crop_excess = max(
            int(round((deskewed_width - crop_width) / 2)) + out_bounds_correction,
            0
        )
        # Crop in X
        deskewed_prelim = np.asarray(deskewed_prelim)
        deskewed_crop = deskewed_prelim[
            :, :, crop_excess : crop_width + crop_excess
        ]

    # For debugging, ,deskewed_prelim will also be returned which is the uncropped volume
    if debug:
        return deskewed_crop, deskewed_prelim
    elif get_deskew_and_decon:
        if skew_dir == DeskewDirection.Y:
            deskewed_crop_no_decon = deskewed_no_decon[
                :, crop_excess : crop_height + crop_excess, :
            ]
        elif skew_dir == DeskewDirection.X:
            deskewed_crop_no_decon = deskewed_no_decon[
                :, :, crop_excess : crop_width + crop_excess
            ]
        return deskewed_crop, deskewed_crop_no_decon
    else:
        return deskewed_crop


# Get reverse affine transform by rotating around a user-specified volume


def get_inverse_affine_transform(
    original_volume,
    angle_in_degrees,
    voxel_x,
    voxel_y,
    voxel_z,
    skew_dir=DeskewDirection.Y,
):
    """
    Calculate the inverse deskew transform and the excess z_bounds

    Args:
        original_volume (_type_): unprocessed volume
        angle_in_degrees (_type_): _description_
        voxel_x (_type_): _description_
        voxel_y (_type_): _description_
        voxel_z (_type_): _description_
        skew_dir: Direction of skew

    Returns:
        Inverse Affine transform (cle.AffineTransform3D), int: Excess z slices, Deskew transform (cle.AffineTransform3D)
    """
    # calculate the deskew transform for specified volume
    if skew_dir == DeskewDirection.Y:
        deskew_transform = _deskew_y_vol_transform(
            original_volume, angle_in_degrees, voxel_x, voxel_y, voxel_z
        )
    elif skew_dir == DeskewDirection.X:
        deskew_transform = _deskew_x_vol_transform(
            original_volume, angle_in_degrees, voxel_x, voxel_y, voxel_z
        )

    # Get the deskew transform after bringing the volume into bounds
    (
        deskewed_shape,
        new_deskew_transform,
        _,
    ) = cle._tier8._affine_transform._determine_translation_and_bounding_box(
        original_volume, deskew_transform
    )

    # Get the inverse of adjusted deskew transform
    deskew_inverse = new_deskew_transform.inverse()

    # We use the shape of deskewed volume to get the new vertices of deskewed volume in x,y and z
    from itertools import product

    nz, ny, nx = deskewed_shape
    deskewed_bounding_box = [
        list(x) + [1] for x in product((0, nx), (0, ny), (0, nz))
    ]

    # transform the corners of deskewed volume using the reverse affine transform
    undeskew_bounding_box = np.asarray(
        list(map(lambda x: deskew_inverse._matrix @ x, deskewed_bounding_box))
    )

    # Get the maximum z value and subtract it from shape of original volume to get excess bounds of bounding box
    max_bounds = undeskew_bounding_box.max(axis=0).astype(int)
    rev_deskew_z = max_bounds[2]
    extra_bounds = int((rev_deskew_z - original_volume.shape[0]))

    return deskew_inverse, extra_bounds, deskew_transform


# Get deskew transform where rotation is around centre of "original_volume"


def _deskew_y_vol_transform(
    original_volume,
    angle_in_degrees: float = 30,
    voxel_size_x: float = 1,
    voxel_size_y: float = 1,
    voxel_size_z: float = 1,
    scale_factor: float = 1,
):
    """Return deskew transform for specified volume when skew direction is Y
       Rotation is performed around centre of "original_volume"
    Args:
        crop ([type]): Volume to deskew (zyx)
        original_volume ([type]): Reference volume around with to perform rotation (zyx)
        angle_in_degrees (float): Deskewing angle
        voxel_size_x (float, optional): [description]. Defaults to 1.
        voxel_size_y (float, optional): [description]. Defaults to 1.
        voxel_size_z (float, optional): [description]. Defaults to 1.
        scale_factor (float, optional): [description]. Defaults to 1.

    Returns:
        cle.AffineTransform3D
    """
    import math

    transform = cle.AffineTransform3D()

    # shear factor for deskewing
    shear_factor = math.sin((90 - angle_in_degrees) * math.pi / 180.0) * (
        voxel_size_z / voxel_size_y
    )
    transform._matrix[1, 2] = shear_factor

    # make voxels isotropic, calculate the new scaling factor for Z after shearing
    # https://github.com/tlamberimage3/napari-ndtiffs/blob/092acbd92bfdbf3ecb1eb9c7fc146411ad9e6aae/napari_ndtiffs/affine.py#L57
    new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
    scale_factor_z = (new_dz / voxel_size_y) * scale_factor
    transform.scale(
        scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z
    )

    # rotation around centre of ref_vol
    # transform._concatenate(rotate_around_vol_mat(original_volume, (0-angle_in_degrees)))
    transform.rotate(angle_in_degrees=0 - angle_in_degrees, axis=0)
    # correct orientation so that the new Z-plane goes proximal-distal from the objective.

    return transform


# Get deskew transform where rotation is around centre of "original_volume"
def _deskew_x_vol_transform(
    original_volume,
    angle_in_degrees: float = 30,
    voxel_size_x: float = 1,
    voxel_size_y: float = 1,
    voxel_size_z: float = 1,
    scale_factor: float = 1,
):
    """Return deskew transform for specified volume when skew direction is X
       Rotation is performed around centre of "original_volume"
    Args:
        crop ([type]): Volume to deskew (zyx)
        original_volume ([type]): Reference volume around with to perform rotation (zyx)
        angle_in_degrees (float): Deskewing angle
        voxel_size_x (float, optional): [description]. Defaults to 1.
        voxel_size_y (float, optional): [description]. Defaults to 1.
        voxel_size_z (float, optional): [description]. Defaults to 1.
        scale_factor (float, optional): [description]. Defaults to 1.

    Returns:
        cle.AffineTransform3D
    """
    import math

    transform = cle.AffineTransform3D()

    # shear factor for deskewing
    shear_factor = math.sin((90 - angle_in_degrees) * math.pi / 180.0) * (
        voxel_size_z / voxel_size_x
    )
    transform._matrix[0, 2] = shear_factor

    # make voxels isotropic, calculate the new scaling factor for Z after shearing
    # https://github.com/tlamberimage3/napari-ndtiffs/blob/092acbd92bfdbf3ecb1eb9c7fc146411ad9e6aae/napari_ndtiffs/affine.py#L57
    new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
    scale_factor_z = (new_dz / voxel_size_x) * scale_factor

    transform.scale(
        scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z
    )

    # rotation around centre of ref_vol
    transform.rotate(angle_in_degrees=angle_in_degrees, axis=1)
    # correct orientation so that the new Z-plane goes proximal-distal from the objective.

    return transform


# deprecated
# Calculate rotation transform around a volume
def rotate_around_vol_mat(ref_vol, angle_in_degrees: float = 30.0):
    """Return the rotation matrix , so its rotated around centre of ref_vol

    Args:
        ref_vol (tuple): Shape of the ref volume (zyx)
        angle_in_degrees (float, optional): [description]. Defaults to 30.0.

    Returns:
        Rotation matrix: Will be returned in the form xyz for clesperanto affine transforms
    """
    angle_in_rad = angle_in_degrees * np.pi / 180.0
    # rotate_transform = cle.AffineTransform3D()
    # rotate_transform._matrix
    # first translate the middle of the image to the origin
    nz, ny, nx = ref_vol.shape
    T1 = np.array(
        [[1, 0, 0, nx / 2], [0, 1, 0, ny / 2], [0, 0, 1, nz / 2], [0, 0, 0, 1]]
    )

    R = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
            [0, -np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
            [0, 0, 0, 1],
        ]
    )

    T2 = np.array(
        [
            [1, 0, 0, -nx / 2],
            [0, 1, 0, -ny / 2],
            [0, 0, 1, -nz / 2],
            [0, 0, 0, 1],
        ]
    )
    T = np.eye(4)
    rotate_mat = np.dot(np.dot(np.dot(T, T1), R), T2)
    # print(rotate_mat)
    return rotate_mat


def _yield_arr_slice(img):
    """
    Create an array generator that yields each z slice
    """
    img = np.squeeze(img)
    assert img.ndim == 3, f"Image needs to be 3D. Got {img.ndim}"

    for slice in img:
        yield slice
