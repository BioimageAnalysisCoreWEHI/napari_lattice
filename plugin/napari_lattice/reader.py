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
from napari.layers import image, Layer
from napari.layers._data_protocols import LayerDataProtocol
from aicsimageio.dimensions import Dimensions
from aicsimageio.aics_image import AICSImage

from typing_extensions import Literal
from typing import Any, Optional, cast, Tuple

from lls_core.lattice_data import lattice_params_from_aics, img_from_array, AicsLatticeParams, PhysicalPixelSizes
from lls_core.types import ArrayLike

class NapariImageParams(AicsLatticeParams):
    save_name: str

def lattice_params_from_napari(
    img: Layer,
    dimension_order: Optional[str],
    # last_dimension: Optional[Literal["channel", "time"]],
    physical_pixel_sizes: PhysicalPixelSizes = PhysicalPixelSizes(None, None, None)
) -> NapariImageParams:
    """
    Factory function for generating a LatticeData from a Napari Image

    Arguments:
        kwargs: Extra arguments to pass to the LatticeData constructor
    """

    if not isinstance(img, Layer):
        raise Exception("img must be a napari layer object")

    img_data_aics: AICSImage

    if 'aicsimage' in img.metadata.keys():
        img_data_aics = img.metadata['aicsimage']
    else:
        if not dimension_order:
            raise ValueError("Either the Napari image must have dimensional metadata, or a dimension order must be provided")
        img_data_aics = img_from_array(cast(ArrayLike, img.data), dimension_order=dimension_order, physical_pixel_sizes=physical_pixel_sizes)

    save_name: str
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

    return NapariImageParams(save_name=save_name, **lattice_params_from_aics(img_data_aics, physical_pixel_sizes=physical_pixel_sizes))

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
