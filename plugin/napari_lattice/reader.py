"""
reader plugin for h5 saved using np2bdv
https://github.com/nvladimus/npy2bdv
#TODO: pass pyramidal layer to napari
##use ilevel parameter in read_view to access different subsamples/pyramids
#pass a list of images with different resolution for pyramid; use is_pyramid=True flag in napari.add_image
, however pyramidal support for 3D not available yet
"""
from __future__ import annotations
from pathlib import Path

import dask.array as da
import dask.delayed as delayed
import os 
import numpy as np
from napari.layers import Layer, Image
from napari.layers._data_protocols import LayerDataProtocol
from aicsimageio.dimensions import Dimensions
from aicsimageio.aics_image import AICSImage

from typing_extensions import Literal
from typing import Any, Optional, cast, Tuple, Collection

from lls_core.lattice_data import DefinedPixelSizes, lattice_params_from_aics, img_from_array, AicsLatticeParams, PhysicalPixelSizes
from lls_core.types import ArrayLike

class NapariImageParams(AicsLatticeParams):
    save_name: str

def lattice_params_from_napari(
    imgs: Collection[Image],
    dimension_order: Optional[str],
    physical_pixel_sizes: PhysicalPixelSizes,
    stack_along: str
) -> NapariImageParams:
    """
    Factory function for generating a LatticeData from a Napari Image
    """
    from xarray import DataArray, concat

    if len(imgs) < 1:
        raise ValueError("At least one image must be provided.")

    save_name: str
    pixel_sizes: set[PhysicalPixelSizes] = {physical_pixel_sizes}
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
            # Only process pixel sizes that are not none
            if all(img_data_aics.physical_pixel_sizes):
                pixel_sizes.add(img_data_aics.physical_pixel_sizes)
                # if pixel_size_metadata is not None and pixel_sizes != img_data_aics.physical_pixel_sizes:
                #     raise Exception(f"Two or more layers that you have tried to merge have different pixel sizes according to their metadata! A previous image has size {physical_pixel_sizes}, whereas {img.name} has size {img_data_aics.physical_pixel_sizes}.")
                # else:
                #     pixel_size_metadata = img_data_aics.physical_pixel_sizes

            calculated_order = img_data_aics.dims.order
        elif dimension_order is None:
            raise ValueError("Either the Napari image must have dimensional metadata, or a dimension order must be provided")
        else:
            calculated_order = list(dimension_order)

        final_imgs.append(DataArray(img.data, dims=calculated_order))

    if len(pixel_sizes) > 1:
        raise Exception(f"Two or more layers that you have tried to merge have different pixel sizes according to their metadata! {pixel_sizes}")
    elif len(pixel_sizes) == 1:
        final_pixel_size = DefinedPixelSizes.from_physical(pixel_sizes.pop())
    else:
        final_pixel_size = DefinedPixelSizes.from_physical(physical_pixel_sizes)

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

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".h5"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return bdv_h5_reader

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
