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
import math
 
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
    background: Union[float, str]
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
    background: Union[float, str] = 0,
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
        background (float, str, optional): Background value to subtract for deconvolution. Defaults to 0.
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
                background=background,
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


def _fit_to_shape(result: NDArray, expected_shape: Tuple[int, ...]) -> NDArray:
    """
    Pad or crop `result` so its shape matches `expected_shape`.

    This handles the edge-of-volume case where `crop_volume_deskew` may
    return a slightly different shape than requested.
    """
    if result.shape == expected_shape:
        return result
    output = np.zeros(expected_shape, dtype=result.dtype)
    slicers = tuple(
        slice(0, min(result.shape[i], expected_shape[i]))
        for i in range(result.ndim)
    )
    output[slicers] = result[slicers]
    return output


def _cast_dtype(data: NDArray, target: np.dtype) -> NDArray:
    """Clip (for integers) and cast `data` to `target` dtype."""
    if data.dtype == target:
        return data
    if np.issubdtype(target, np.integer):
        iinfo = np.iinfo(target)
        np.clip(data, float(iinfo.min), float(iinfo.max), out=data)
    return data.astype(target)


def _write_xy_tile(output: NDArray, y_start: int, x_start: int, tile: NDArray) -> None:
    """Write a non-overlapping XY tile (full Z) into the output array."""
    y_end = y_start + tile.shape[1]
    x_end = x_start + tile.shape[2]
    output[:tile.shape[0], y_start:y_end, x_start:x_end] = tile


def deskew_xy_tiles(
    input_volume: NDArray,
    deskew_vol_shape: Tuple[int, ...],
    angle_in_degrees: float,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    skew_dir: DeskewDirection,
    output_dtype: Any = None,
) -> NDArray:
    """Deskew a volume by tiling in XY only, keeping the full Z per tile.

    Each tile will call ``crop_volume_deskew`` with z_start=0, z_end=oz and a
    small YX ROI.  This avoids Z-boundary artifacts caused by the deskew
    shear. Tile writing of tile N will overlap with GPU procesing of tile N+1 GPU using a background thread.
    This will make it faster.

    Args:
        input_volume: 3D numpy array (ZYX), already in memory.
        deskew_vol_shape: Expected shape of the full deskewed output (Z, Y, X).
        angle_in_degrees: Deskewing angle.
        voxel_size_x, voxel_size_y, voxel_size_z: Pixel sizes in microns.
        skew_dir: Deskew direction (DeskewDirection.Y or DeskewDirection.X).
        output_dtype: If set, convert the output to this dtype.

    Returns:
        NDArray: Deskewed volume in output_dtype (or float32 if not specified).
    """
    #TODO: What if input volume is large on Z and does not fit; Perhaps we'll need to tile in XYZ then
    from concurrent.futures import ThreadPoolExecutor
    import tempfile as _tempfile
    import os as _os

    oz, oy, ox = deskew_vol_shape
    target = np.dtype(output_dtype) if output_dtype is not None else np.dtype(np.float32)

    tile_y, tile_x = get_xy_tile_sizes(
        input_volume.shape, deskew_vol_shape, skew_dir
    )

    n_tiles_y = math.ceil(oy / tile_y)
    n_tiles_x = math.ceil(ox / tile_x)

    # Overlap margin on the shear axis — only when there are multiple tiles
    if skew_dir == DeskewDirection.Y:
        y_margin = _compute_overlap_margin(tile_y) if n_tiles_y > 1 else 0
        x_margin = 0
    else:
        y_margin = 0
        x_margin = _compute_overlap_margin(tile_x) if n_tiles_x > 1 else 0
    logger.info(
        f"XY tiling: {n_tiles_y}x{n_tiles_x} tiles of ({oz}, {tile_y}, {tile_x}), "
        f"Y margin={y_margin}, X margin={x_margin}"
    )

    # Allocate output (memmap for >2 GB, ndarray otherwise)
    _memmap_path = None
    output_bytes = math.prod(deskew_vol_shape) * target.itemsize
    if output_bytes > 2 * 1024**3:
        _tmpfd, _memmap_path = _tempfile.mkstemp(suffix='.mmap')
        _os.close(_tmpfd)
        output = np.memmap(
            _memmap_path, dtype=target, mode='w+', shape=deskew_vol_shape
        )
        logger.info(
            f"Using memory-mapped output ({output_bytes / 1e9:.1f} GB, {target})"
        )
    else:
        output = np.zeros(deskew_vol_shape, dtype=target)

    # Pipeline: GPU on main thread, write on background thread
    prev_future = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        for yi in range(0, oy, tile_y):
            for xi in range(0, ox, tile_x):
                y_start = yi
                y_end = min(yi + tile_y, oy)
                x_start = xi
                x_end = min(xi + tile_x, ox)

                # Wait for previous write before submitting the next GPU result
                if prev_future is not None:
                    prev_future.result()

                # Expand shear axis by margin
                y_start_exp = max(0, y_start - y_margin)
                y_end_exp = min(oy, y_end + y_margin)
                x_start_exp = max(0, x_start - x_margin)
                x_end_exp = min(ox, x_end + x_margin)

                roi_shape = [
                    [y_start_exp, x_start_exp],
                    [y_start_exp, x_end_exp],
                    [y_end_exp, x_end_exp],
                    [y_end_exp, x_start_exp],
                ]

                result = crop_volume_deskew(
                    original_volume=input_volume,
                    roi_shape=roi_shape,
                    z_start=0,
                    z_end=oz,
                    angle_in_degrees=angle_in_degrees,
                    voxel_size_x=voxel_size_x,
                    voxel_size_y=voxel_size_y,
                    voxel_size_z=voxel_size_z,
                    skew_dir=skew_dir,
                    deconvolution=False,
                )

                # Fit to expanded shape, then trim margins
                expanded_shape = (oz, y_end_exp - y_start_exp, x_end_exp - x_start_exp)
                result = _fit_to_shape(result, expanded_shape)

                y_trim = y_start - y_start_exp
                x_trim = x_start - x_start_exp
                result = result[
                    :,
                    y_trim : y_trim + (y_end - y_start),
                    x_trim : x_trim + (x_end - x_start),
                ]

                result = _cast_dtype(result, target)
                prev_future = pool.submit(
                    _write_xy_tile, output, y_start, x_start, result
                )

        if prev_future is not None:
            prev_future.result()

    if _memmap_path is not None:
        output.flush()
        # Copy to a regular numpy array and clean up the temp file.
        # The memmap was only needed during tiling to avoid holding
        # the full output + GPU buffers in RAM simultaneously.
        result = np.array(output)
        del output
        try:
            _os.unlink(_memmap_path)
            logger.info(f"Cleaned up memmap temp file: {_memmap_path}")
        except OSError:
            logger.warning(f"Could not remove memmap temp file: {_memmap_path}")
        return result

    return output


def _should_tile_on_gpu(input_shape, output_shape: Tuple[int, ...], dtype: Any) -> bool:
    """
    Check if an image of a given shape and dtype should be tiled for GPU processing.
    Tiling is recommended if the image size is > 95% of the GPU's max allocation size.
    or if the combined size of input and output buffers exceeds available VRAM.
    """
    from lls_core.utils import get_max_allocation_size, get_global_mem_size

    max_alloc = get_max_allocation_size()
    global_mem = get_global_mem_size()

    if not max_alloc or not global_mem:
        # If we can't get GPU memory, it's safer to tile.
        logger.warning("Could not determine GPU memory information. Falling back to tiling.")
        return True

    # All GPU buffers are float32 (pyclesperanto converts internally)
    BYTES_PER_ELEMENT = 4
    output_bytes = math.prod(output_shape) * BYTES_PER_ELEMENT
    input_bytes = math.prod(input_shape) * BYTES_PER_ELEMENT

    # Check 1: A single buffer cannot exceed MAX_MEM_ALLOC_SIZE.
    if output_bytes > max_alloc:
        logger.info(
            f"Output volume ({output_bytes / 1e6:.2f} MB) exceeds GPU max single allocation "
            f"({max_alloc / 1e6:.2f} MB). Tiling will be used."
        )
        return True

    # Check 2: Input + output (with 2x safety factor for internal temp iages)
    # should not exceed total VRAM.
    total_required_bytes = (input_bytes + output_bytes) * 2
    if total_required_bytes > global_mem:
        logger.info(
            f"Estimated GPU memory ({total_required_bytes / 1e6:.2f} MB) exceeds "
            f"total GPU memory ({global_mem / 1e6:.2f} MB). Tiling will be used."
        )
        return True

    logger.info(f"Output volume ({output_bytes / 1e6:.2f} MB) fits in GPU memory. Processing as a whole.")
    return False


def get_xy_tile_sizes(
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    skew_dir: DeskewDirection = DeskewDirection.Y,
) -> Tuple[int, int]:
    """Compute (tile_y, tile_x) for XY-only tiling with full Z per tile.

    Each OpenCL buffer must fit in MAX_MEM_ALLOC_SIZE, and the total of
    input + output (with 2x safety for temporaries) must fit in VRAM.

    For full-Z tiles the input subvolume spans nearly the full input
    along the shear axis (Y for DeskewDirection.Y, X for DeskewDirection.X)
    because the inverse affine maps the full Z range across it.  Reducing
    the *non-shear* axis shrinks both input and output proportionally, so
    it is halved first.

    Returns:
        (tile_y, tile_x) — full output Y/X if no tiling is needed.
    """
    from lls_core.utils import get_max_allocation_size, get_global_mem_size

    BYTES_PER_ELEMENT = 4  # float32 on GPU
    max_alloc = get_max_allocation_size() or (500 * 1024 * 1024)
    global_mem = get_global_mem_size() or (2 * 1024 * 1024 * 1024)

    iz, iy, ix = input_shape
    oz, oy, ox = output_shape
    tile_y, tile_x = oy, ox

    def _exceeds_limits(ty: int, tx: int) -> bool:
        output_bytes = oz * ty * tx * BYTES_PER_ELEMENT
        # Input subvolume: shear axis spans full input extent,
        # non-shear axis tracks the tile size proportionally.
        if skew_dir == DeskewDirection.Y:
            input_bytes = iz * iy * tx * BYTES_PER_ELEMENT
        else:  # DeskewDirection.X
            input_bytes = iz * ty * ix * BYTES_PER_ELEMENT
        if output_bytes > max_alloc or input_bytes > max_alloc:
            return True
        if (input_bytes + output_bytes) * 2 > global_mem:
            return True
        return False

    while _exceeds_limits(tile_y, tile_x) and (tile_y > 32 or tile_x > 32):
        # Halve the non-shear axis first (shrinks both input and output).
        if skew_dir == DeskewDirection.Y:
            # X is non-shear; halve X first, then Y.
            if tile_x > 32:
                tile_x = max(32, tile_x // 2)
            else:
                tile_y = max(32, tile_y // 2)
        else:  # DeskewDirection.X
            # Y is non-shear; halve Y first, then X.
            if tile_y > 32:
                tile_y = max(32, tile_y // 2)
            else:
                tile_x = max(32, tile_x // 2)

    return (tile_y, tile_x)


def _compute_overlap_margin(chunk_size: int, fraction: float = 0.1) -> int:
    """Compute overlap margin as a percentage of the tile's chunk size.

    Args:
        chunk_size: Number of pixels in this dimension's chunk.
        fraction: Fraction of chunk_size to use as margin on each side.
            Defaults to 10%.

    Returns:
        Margin in pixels (at least 2).
    """
    #TODO: Could make this configurable
    return max(2, int(math.ceil(chunk_size * fraction)))


def _deskew_tile(
    block: NDArray,
    block_info: dict | None = None,
    original_volume: ArrayLike | None = None,
    angle_in_degrees: float = 30,
    voxel_size_x: float = 1.0,
    voxel_size_y: float = 1.0,
    voxel_size_z: float = 1.0,
    skew_dir: DeskewDirection = DeskewDirection.Y,
    deconvolution: bool = False,
    decon_processing: DeconvolutionChoice | None = None,
    psf: Psf | None = None,
    num_iter: int = 10,
    background: Union[float, str] = 0,
    offset: Optional[Tuple[int, int, int]] = None,
    y_margin: int = 0,
    x_margin: int = 0,
    total_output_shape: Tuple[int, int, int] = (0, 0, 0),
) -> NDArray:
    """Process one XY tile (full Z) via ``crop_volume_deskew``.

    Called by dask ``map_blocks`` from :func:`deskew_large_image`.
    Chunks are ``(full_z, tile_y, tile_x)`` so Z is never tiled.
    A small margin is applied on the shear axis (Y or X depending on
    ``skew_dir``) to handle rounding at tile boundaries.
    """
    if block_info is None or original_volume is None:
        raise ValueError("block_info and original_volume must be provided")

    total_oz, total_oy, total_ox = total_output_shape

    expected_shape = block.shape
    loc = block_info[0]["array-location"]
    z_start, z_end = loc[0]
    y_start, y_end = loc[1]
    x_start, x_end = loc[2]

    if offset:
        z_start += offset[0]
        z_end += offset[0]
        y_start += offset[1]
        y_end += offset[1]
        x_start += offset[2]
        x_end += offset[2]

    # Expand shear axis by margin, trim afterwards
    y_start_exp = max(0, y_start - y_margin)
    y_end_exp = min(total_oy, y_end + y_margin) if total_oy > 0 else y_end + y_margin
    x_start_exp = max(0, x_start - x_margin)
    x_end_exp = min(total_ox, x_end + x_margin) if total_ox > 0 else x_end + x_margin

    roi_shape = [
        [y_start_exp, x_start_exp],
        [y_start_exp, x_end_exp],
        [y_end_exp, x_end_exp],
        [y_end_exp, x_start_exp],
    ]

    result = crop_volume_deskew(
        original_volume=original_volume,
        roi_shape=roi_shape,
        z_start=z_start,
        z_end=z_end,
        angle_in_degrees=angle_in_degrees,
        voxel_size_x=voxel_size_x,
        voxel_size_y=voxel_size_y,
        voxel_size_z=voxel_size_z,
        skew_dir=skew_dir,
        deconvolution=deconvolution,
        decon_processing=decon_processing,
        psf=psf,
        num_iter=num_iter,
        background=background,
        debug=False,
        get_deskew_and_decon=False,
    )

    # Fit to expanded shape, then trim margins
    expanded_expected = (
        z_end - z_start,
        y_end_exp - y_start_exp,
        x_end_exp - x_start_exp,
    )
    result = _fit_to_shape(result, expanded_expected)

    y_trim = y_start - y_start_exp
    x_trim = x_start - x_start_exp
    result = result[
        :,
        y_trim : y_trim + (y_end - y_start),
        x_trim : x_trim + (x_end - x_start),
    ]

    if result.shape != expected_shape:
        output_block = np.zeros(expected_shape, dtype=result.dtype)
        slicers = tuple(slice(0, min(result.shape[i], expected_shape[i])) for i in range(result.ndim))
        output_block[slicers] = result[slicers]
        return output_block
    return result


def deskew_large_image(
    original_volume: ArrayLike,
    deskewed_shape: Tuple[int, int, int],
    angle_in_degrees: float,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    skew_dir: DeskewDirection,
    deconvolution: bool = False,
    decon_processing: Optional[DeconvolutionChoice] = None,
    psf: Optional[Psf] = None,
    num_iter: int = 10,
    background: Union[float, str] = 0,
    offset: Optional[Tuple[int, int, int]] = None,
) -> DaskArray:
    oz, oy, ox = deskewed_shape

    # XY-only tile sizes — full Z per tile
    tile_y, tile_x = get_xy_tile_sizes(
        original_volume.shape, deskewed_shape, skew_dir
    )
    chunks = (oz, tile_y, tile_x)
    n_tiles_y = math.ceil(oy / tile_y)
    n_tiles_x = math.ceil(ox / tile_x)

    # Overlap margin on the shear axis — only when there are multiple tiles
    if skew_dir == DeskewDirection.Y:
        y_margin = _compute_overlap_margin(tile_y) if n_tiles_y > 1 else 0
        x_margin = 0
    else:
        y_margin = 0
        x_margin = _compute_overlap_margin(tile_x) if n_tiles_x > 1 else 0

    logger.info(
        f"Tiling: chunks={chunks}, Y margin={y_margin}, X margin={x_margin}"
    )

    template = da.zeros(deskewed_shape, chunks=chunks, dtype=np.float32)

    deskewed = template.map_blocks(
        _deskew_tile,
        dtype=np.float32,
        chunks=chunks,
        original_volume=original_volume,
        angle_in_degrees=angle_in_degrees,
        voxel_size_x=voxel_size_x,
        voxel_size_y=voxel_size_y,
        voxel_size_z=voxel_size_z,
        skew_dir=skew_dir,
        deconvolution=deconvolution,
        decon_processing=decon_processing,
        psf=psf,
        num_iter=num_iter,
        background=background,
        offset=offset,
        y_margin=y_margin,
        x_margin=x_margin,
        total_output_shape=deskewed_shape,
    )

    return deskewed
