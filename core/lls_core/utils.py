from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull, path
from typing import Collection, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pyclesperanto_prototype as cle
from lls_core.types import ArrayLike
from numpy.typing import NDArray
from read_roi import read_roi_file, read_roi_zip
from typing_extensions import TYPE_CHECKING, Any, TypeGuard

from . import DeskewDirection, config

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element
    from dask.array.core import Array as DaskArray
    from napari.layers import Shapes

# Enable Logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(config.log_level)

# get bounding box of ROI in 3D and the shape of the ROI

def check_subclass(obj: Any, pkg_name: str, cls_name: str) -> bool:
    """
    Like `isinstance`, but doesn't require that the class in question is imported
    """
    # TODO: make this work for subclasses
    cls = obj.__class__
    module = cls.__module__
    return cls.__name__ == cls_name and module.__qualname__ == pkg_name

def is_napari_shape(obj: Any) -> TypeGuard[Shapes]:
    return check_subclass(obj, "napari.shapes", "Shapes")

def calculate_crop_bbox(shape: list, z_start: int, z_end: int) -> tuple[List[List[Any]], List[int]]:
    """Get bounding box as vertices in 3D in the form xyz

    Args:
        shape (int): [description]
        z_start (int): Start of slice
        z_end (int): End of slice

    Returns:
        list,np.array: Bounding box of ROI in 3D (xyz), the shape of the ROI in 3D (zyx)
    """
    # calculate bounding box and shape of cropped 3D volume
    # shape is 3D ROI from shape layer
    # get ROI coordinates,clip to zero, get shape of ROI volume and vertices of 3D ROI
    start = np.rint(np.min(shape, axis=0)).clip(0)
    stop = np.rint(np.max(shape, axis=0)).clip(0)

    # Shapes layer can return 3D coordinates, so only take last two
    if len(start > 2):
        start = start[-2:]
    if len(stop > 2):
        stop = stop[-2:]

    start = np.insert(start, 0, z_start)  # start[0]=z_start
    stop = np.insert(stop, 0, z_end)  # =z_start

    z0, y0, x0 = np.stack([start, stop])[0].astype(int)
    z1, y1, x1 = np.stack([start, stop])[1].astype(int)

    crop_shape = (stop - start).astype(int).tolist()

    from itertools import product

    crop_bounding_box = [list(x)+[1]
                         for x in product((x0, x1), (y0, y1), (z0, z1))]
    return crop_bounding_box, crop_shape


def get_deskewed_shape(volume: ArrayLike,
                       angle: float,
                       voxel_size_x_in_microns: float,
                       voxel_size_y_in_microns: float,
                       voxel_size_z_in_microns: float,
                       skew_dir: DeskewDirection=DeskewDirection.Y) -> Tuple[Tuple[int], cle.AffineTransform3D]:
    """
    Calculate shape of deskewed volume 
    Also, returns affine transform

    Args:
        volume ([type]): Volume to deskew
        angle ([type]): Deskewing Angle
        voxel_size_x_in_microns ([type])
        voxel_size_y_in_microns ([type])
        voxel_size_z_in_microns ([type])
        skew_dir

    Returns:
        tuple: Shape of deskewed volume in zyx
        np.array: Affine transform for deskewing
    """
    from pyclesperanto_prototype._tier8._affine_transform import (
        _determine_translation_and_bounding_box,
    )

    deskew_transform = cle.AffineTransform3D()

    if skew_dir == DeskewDirection.Y:
        deskew_transform._deskew_y(angle_in_degrees=angle,
                                   voxel_size_x=voxel_size_x_in_microns,
                                   voxel_size_y=voxel_size_y_in_microns,
                                   voxel_size_z=voxel_size_z_in_microns)
    elif skew_dir == DeskewDirection.X:
        deskew_transform._deskew_x(angle_in_degrees=angle,
                                   voxel_size_x=voxel_size_x_in_microns,
                                   voxel_size_y=voxel_size_y_in_microns,
                                   voxel_size_z=voxel_size_z_in_microns)

    # TODO:need better handling of aics dask array
    if len(volume.shape) == 5:
        volume = volume[0, 0, ...]

    new_shape, new_deskew_transform, _ = _determine_translation_and_bounding_box(
        volume, deskew_transform)
    return new_shape, new_deskew_transform


# https://stackoverflow.com/a/10076823
# Credit: Antoine Pinsard
# Convert xml elementree to dict
def etree_to_dict(t: Element) -> dict:
    """Parse an XML file and convert to dictionary 
    This can be used to access the Zeiss metadata
    Access it from ["ImageDocument"]["Metadata"]

    Args:
        xml object : XML document (ImageDocument) containing Zeiss metadata

    Returns:
        dictionary: Zeiss czi file metadata 
    """
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d

# block printing to console


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# dask implementation for expand_dims not in latest release yet, so copying from their repo
# https://github.com/dask/dask/blob/dca10398146c6091a55c54db3778a06b485fc5ce/dask/array/routines.py#L1889


def dask_expand_dims(a: ArrayLike, axis: Union[Collection[int], int]):
    _axis: Collection[int]

    if isinstance(axis, int):
        _axis = (axis,)
    else:
        _axis = axis

    out_ndim = len(_axis) + a.ndim

    shape_it = iter(a.shape)
    shape = [1 if ax in _axis else next(shape_it) for ax in range(out_ndim)]

    return a.reshape(shape)



def pad_image_nearest_multiple(img: NDArray, nearest_multiple: int) -> NDArray:
    """pad an Image to the nearest multiple of provided number

    Args:
        img (np.ndarray): 
        nearest_multiple (int): Multiple of number to be padded

    Returns:
        np.ndarray: Padded image
    """
    import math

    rounded_shape = tuple(
        [math.ceil(dim/nearest_multiple)*nearest_multiple for dim in img.shape])
    # get required padding
    padding = np.array(rounded_shape) - np.array(img.shape)
    padded_img = np.pad(
        img, ((0, padding[0]), (0, padding[1]), (0, padding[2])), mode="reflect")
    return padded_img


def check_dimensions(user_time_start: int, user_time_end: int, user_channel_start: int, user_channel_end: int, total_channels: int, total_time: int):

    if total_time == 1 or total_time == 2:
        max_time = 1
    else:  # max time should be index - 1
        max_time = total_time - 1

    if total_channels == 1 or total_channels == 2:
        max_channels = 1
    else:  # max time should be index - 1
        max_channels = total_channels - 1

    # Assert time and channel starts are valid
    assert 0 <= user_time_start <= max_time, f"Time start should be 0 or end time: {total_time-1}"
    assert 0 <= user_channel_start <= max_channels, f"Channel start should be 0 or end channels: {total_channels-1}"

    # Not everyone will be aware that indexing starts at zero and ends at channel-1 or time -1
    # below only accounts for ending

    # If user enters total channel number as last channel, correct for indexing by subtracting it by 1
    if user_time_end == total_time:
        print(
            f"Detected end time as {user_time_end}, but Python indexing starts at zero so last timepoint should be {total_time-1}")
        user_time_end = user_time_end-1
        print(f"Adjusting end time to {user_time_end}")
    elif user_time_end == 0:
        user_time_end = user_time_end+1

    # If user enters total time as last time, correct for indexing by subtracting it by 1
    if user_channel_end == total_channels:
        print(
            f"Detected end channel as {user_channel_end}, but Python indexing starts at zero so last channel should be {total_channels-1}")
        user_channel_end = user_channel_end-1
        print(f"Adjusting channel end to {user_channel_end}")
    elif user_channel_end == 0:
        user_channel_end = user_channel_end+1

    assert 0 < user_time_end <= max_time, f"Time is out of range. End time: {max_time}"
    assert 0 < user_channel_end <= max_channels, f"Channel is out of range. End channels: {max_channels}"

    #logging.debug(f"Time start,end: {user_time_start,user_time_end}, Channel start,end: {user_channel_start,user_channel_end}")

    return None

# Reference: https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/blob/master/examples/find_PSF_support.ipynb


def crop_psf(psf_img: np.ndarray, threshold: float = 3e-3):
    """ Crop a PSF image based on the threshold specifiied

    Args:
        psf_img (np.ndarray): PSF image
        threshold (float): Threshold value (0 to 1) as its a 32-bit image

    Returns:
        np.ndarray: Cropped PSF image
    """
    # get max value
    psf_max = psf_img.max()
    psf_threshold = psf_max * threshold
    # print(psf_threshold)
    psf_filtered = psf_img > psf_threshold
    # Get dimensions for min and max, where psf is greater than threshold
    min_z, min_y, min_x = np.min(np.where(psf_filtered), axis=1)
    max_z, max_y, max_x = np.max(np.where(psf_filtered), axis=1)

    # info for debugging
    psf_shape = np.max(np.where(psf_filtered), axis=1) - \
        np.min(np.where(psf_filtered), axis=1)
    logging.debug(f"Shape of cropped psf is {psf_shape}")

    psf_crop = psf_img[min_z:max_z, min_y:max_y, min_x:max_x]
    return psf_crop

def as_type(img, ref_vol):
    """return image same dtype as ref_vol

    Args:
        img (_type_): _description_
        ref_vol (_type_): _description_

    Returns:
        _type_: _description_
    """
    img.astype(ref_vol.dtype)
    return img

T = TypeVar("T")
def raise_if_none(obj: Optional[T], message: str) -> T:
    """
    Asserts that `obj` is not None
    """
    if obj is None:
        raise TypeError(message)
    return obj

def array_to_dask(arr: ArrayLike) -> DaskArray:
    from dask.array.core import Array as DaskArray, from_array
    from xarray import DataArray
    from resource_backed_dask_array import ResourceBackedDaskArray 

    if isinstance(arr, DataArray):
        arr = arr.data
    if isinstance(arr, (DaskArray, ResourceBackedDaskArray)):
        return arr
    else:
        return from_array(arr)

def make_filename_suffix(prefix: Optional[str] = None, roi_index: Optional[int] = None, channel: Optional[str] = None, time: Optional[str] = None) -> str:
    """
    Generates a filename for this result
    """
    components: List[str] = []
    if prefix is not None:
        components.append(prefix)
    if roi_index is not None:
        components.append(f"ROI_{roi_index}")
    if channel is not None:
        components.append(f"C{channel}")
    if time is not None:
        components.append(f"T{time}")
    return "_".join(components)
