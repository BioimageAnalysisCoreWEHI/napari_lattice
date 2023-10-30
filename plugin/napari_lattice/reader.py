"""
reader plugin for h5 saved using np2bdv
https://github.com/nvladimus/npy2bdv
#TODO: pass pyramidal layer to napari
##use ilevel parameter in read_view to access different subsamples/pyramids
#pass a list of images with different resolution for pyramid; use is_pyramid=True flag in napari.add_image
, however pyramidal support for 3D not available yet
"""
from __future__ import annotations

import dask.array as da
import dask.delayed as delayed
import os 
import numpy as np
from napari.layers import Image
from aicsimageio.aics_image import AICSImage

from typing import List, Optional, Tuple, Collection, TYPE_CHECKING, TypedDict

from aicsimageio.types import PhysicalPixelSizes
from lls_core.models.deskew import DefinedPixelSizes

from logging import getLogger
logger = getLogger(__name__)

if TYPE_CHECKING:
    from aicsimageio.types import ImageLike
    from xarray import DataArray

class NapariImageParams(TypedDict):
    data: DataArray
    physical_pixel_sizes: DefinedPixelSizes
    save_name: str

def lattice_params_from_napari(
    imgs: Collection[Image],
    stack_along: str,
    dimension_order: Optional[str] = None,
    physical_pixel_sizes: Optional[PhysicalPixelSizes] = None,
) -> NapariImageParams:
    """
    Factory function for generating a LatticeData from a Napari Image
    """
    from xarray import DataArray, concat

    if len(imgs) < 1:
        raise ValueError("At least one image must be provided.")

    if len(set(len(it.data.shape) for it in imgs)) > 1:
        size_message = ",".join(f"{img.name}: {len(img.data.shape)}" for img in imgs)
        raise ValueError(f"The input images have multiple different dimensions, which napari lattice doesn't support: {size_message}")

    save_name: str
    # This is a set of all pixel sizes that we have seen so far
    metadata_pixel_sizes: set[PhysicalPixelSizes] = set()
    save_names = []
    # The pixel sizes according to the AICS metadata, if any
    final_imgs: list[DataArray] = []

    for img in imgs:
        if img.source.path is None:
            # remove colon (:) and any leading spaces
            save_name = img.name.replace(":", "").strip()
            # replace any group of spaces with "_"
            save_name = '_'.join(save_name.split())
        else:
            file_name_noext = os.path.basename(img.source.path)
            file_name = os.path.splitext(file_name_noext)[0]
            # remove colon (:) and any leading spaces
            save_name = file_name.replace(":", "").strip()
            # replace any group of spaces with "_"
            save_name = '_'.join(save_name.split())

        save_names.append(save_name)
            
        if 'aicsimage' in img.metadata.keys():
            img_data_aics: AICSImage = img.metadata['aicsimage']
            # If the user has not provided pixel sizes, we extract them fro the metadata
            # Only process pixel sizes that are not none
            if physical_pixel_sizes is None and all(img_data_aics.physical_pixel_sizes):
                metadata_pixel_sizes.add(img_data_aics.physical_pixel_sizes)

            metadata_order = list(img_data_aics.dims.order)
            metadata_shape = list(img_data_aics.dims.shape)
            while len(metadata_order) > len(img.data.shape):
                logger.info(f"Image metadata implies there are more dimensions ({len(metadata_order)}) than the image actually has ({len(img.data.shape)})")
                for i, size in enumerate(metadata_shape):
                    if size not in img.data.shape:
                        logger.info(f"Excluding the {metadata_order[i]} dimension to reconcile dimension order")
                        del metadata_order[i]
                        del metadata_shape[i]
            calculated_order = metadata_order
        elif dimension_order is None:
            raise ValueError("Either the Napari image must have dimensional metadata, or a dimension order must be provided")
        else:
            calculated_order = tuple(dimension_order)

        final_imgs.append(DataArray(img.data, dims=calculated_order))

    if physical_pixel_sizes:
        final_pixel_size = DefinedPixelSizes.from_physical(physical_pixel_sizes)
    else:
        if len(metadata_pixel_sizes) > 1:
            raise Exception(f"Two or more layers that you have tried to merge have different pixel sizes according to their metadata! {metadata_pixel_sizes}")
        elif len(metadata_pixel_sizes) < 1:
            raise Exception("No pixel sizes could be determined from the image metadata. Consider manually specifying the pixel sizes.")
        else:
            final_pixel_size = DefinedPixelSizes.from_physical(metadata_pixel_sizes.pop())

    final_img = concat(final_imgs, dim=stack_along)
    return NapariImageParams(save_name=save_names[0], physical_pixel_sizes=final_pixel_size, data=final_img, dims=final_img.shape)

def napari_get_reader(path: list[str] | str):
    """Check if file ends with h5 and returns reader function if true
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
    Returns
    -------
    function 
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, we are only going to open first file
        path = path[0]

    tiff_formats = (".tif",".tiff")
    
    if path.endswith(".h5"):
        return bdv_h5_reader
    elif path.endswith(tiff_formats):
        return tiff_reader
    # if we know we cannot read the file, we immediately return None.
    else:
        return None


def bdv_h5_reader(path):
    """Take a path and returns a list of LayerData tuples."""
    
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    #print(path)
    import npy2bdv
    h5_file = npy2bdv.npy2bdv.BdvEditor(path)

    img = []

    #get dimensions of first image
    first_timepoint = h5_file.read_view(time=0,channel=0)

    #Threshold to figure out when to use out-of-memory loading/dask 
    #Got the idea from napari-aicsimageio 
    #https://github.com/AllenCellModeling/napari-aicsimageio/blob/22934757c2deda30c13f39ec425343182fa91a89/napari_aicsimageio/core.py#L222
    mem_threshold_bytes = 4e9
    mem_per_threshold = 0.3
    
    from psutil import virtual_memory
    
    file_size = os.path.getsize(path)
    avail_mem = virtual_memory().available

    #if file size <30% of available memory and <4GB, open 
    if file_size<=mem_per_threshold*avail_mem and file_size<mem_threshold_bytes:
        in_memory=True
    else:
        in_memory=False
    
    if in_memory:
        for time in range(h5_file.ntimes):
            for ch in range(h5_file.nchannels):
                image = h5_file.read_view(time=time,channel=ch)
                img.append(image)
        images=np.stack(img)
        
    else:
        for time in range(h5_file.ntimes):
            for ch in range(h5_file.nchannels):         
                image = da.from_delayed(
                            delayed(h5_file.read_view)(time=time,channel=ch), shape=first_timepoint.shape, dtype=first_timepoint.dtype
                        )
                img.append(image)
            
        images = da.stack(img)

    
    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(images, add_kwargs, layer_type)]

def tiff_reader(path: ImageLike) -> List[Tuple[AICSImage, dict, str]]:
    """Take path to tiff image and returns a list of LayerData tuples.
    Specifying tiff_reader to have better control over tifffile related errors when using AICSImage
    """
    
    try:
        image = AICSImage(path)
    except Exception as e:
        raise Exception("Error reading TIFF. Try upgrading tifffile library: pip install tifffile --upgrade.") from e

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(image, add_kwargs, layer_type)]
