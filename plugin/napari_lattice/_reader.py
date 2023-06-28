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
from napari.layers import image

from lattice_lightsheet_core.io import LatticeData

def lattice_from_napari(img: image.Image, last_dimension) -> LatticeData:
    data = LatticeData()
    # check if its an aicsimageio object and has voxel size info
    if 'aicsimage' in img.metadata.keys() and img.metadata['aicsimage'].physical_pixel_sizes != (None, None, None):
        img_data_aics = img.metadata['aicsimage']
        data.data = img_data_aics.dask_data
        data.dims = img_data_aics.dims
        data.time = img_data_aics.dims.T
        data.channels = img_data_aics.dims.C
        data.dz, data.dy, data.dx = img_data_aics.physical_pixel_sizes
    else:
        print("Cannot read voxel size from metadata")
        if 'aicsimage' in img.metadata.keys():
            img_data_aics = img.metadata['aicsimage']
            data.data = img_data_aics.dask_data
            data.dims = img_data_aics.dims
            # if aicsimageio tiffreader assigns last dim as time when it should be channel, user can override this
            if last_dimension:
                if len(img.data.shape) == 4:
                    if last_dimension.lower() == "channel":
                        data.channels = img.data.shape[0]
                        data.time = 0
                    elif last_dimension.lower() == "time":
                        data.time = img.data.shape[0]
                        data.channels = 0
                elif len(img.data.shape) == 5:
                    if last_dimension.lower() == "channel":
                        data.channels = img.data.shape[0]
                        data.time = img.data.shape[1]
                    elif last_dimension.lower() == "time":
                        data.time = img.data.shape[0]
                        data.channels = img.data.shape[1]
            else:
                data.time = img_data_aics.dims.T
                data.channels = img_data_aics.dims.C
        else:
            # if no aicsimageio key in metadata
            # get the data and convert it into an aicsimage object
            img_data_aics = aicsimageio.AICSImage(img.data)
            data.data = img_data_aics.dask_data
            # if user has specified ch
            if last_dimension:
                if len(img.data.shape) == 4:
                    if last_dimension.lower() == "channel":
                        data.channels = img.data.shape[0]
                        data.time = 0
                    elif last_dimension.lower() == "time":
                        data.time = img.data.shape[0]
                        data.channels = 0
                elif len(img.data.shape) == 5:
                    if last_dimension.lower() == "channel":
                        data.channels = img.data.shape[0]
                        data.time = img.data.shape[1]
                    elif last_dimension.lower() == "time":
                        data.time = img.data.shape[0]
                        data.channels = img.data.shape[1]
            else:
                if last_dimension:
                    if len(img.data.shape) == 4:
                        if last_dimension.lower() == "channel":
                            data.channels = img.data.shape[0]
                            data.time = 0
                        elif last_dimension.lower() == "time":
                            data.time = img.data.shape[0]
                            data.channels = 0
                    elif len(img.data.shape) == 5:
                        if last_dimension.lower() == "channel":
                            data.channels = img.data.shape[0]
                            data.time = img.data.shape[1]
                        elif last_dimension.lower() == "time":
                            data.time = img.data.shape[0]
                            data.channels = img.data.shape[1]
                else:

                    data.time = img.data.shape[0]
                    data.channels = img.data.shape[1]

        # read metadata for pixel sizes
        if None in img_data_aics.physical_pixel_sizes or img_data_aics.physical_pixel_sizes == False:
            data.dx = dx
            data.dy = dy
            data.dz = dz
        else:
            data.dz, data.dy, data.dx = img.data.physical_pixel_sizes
        # if not last_dimension:
        # if xarray, access data using .data method
            # if type(img.data) in [xarray.core.dataarray.DataArray,np.ndarray]:
            #img = img.data
        # img = dask_expand_dims(img,axis=1) ##if no channel dimension specified, then expand axis at index 1
    # if no path returned by source.path, get image name with colon and spaces removed
    # if last axes of "aicsimage data" shape is not equal to time, then swap channel and time
    if data.data.shape[0] != data.time or data.data.shape[1] != data.channels:
        data.data = np.swapaxes(data.data, 0, 1)

    if img.source.path is None:
        # remove colon (:) and any leading spaces
        data.save_name = img.name.replace(":", "").strip()
        # replace any group of spaces with "_"
        data.save_name = '_'.join(data.save_name.split())

    else:
        file_name_noext = os.path.basename(img.source.path)
        file_name = os.path.splitext(file_name_noext)[0]
        # remove colon (:) and any leading spaces
        data.save_name = file_name.replace(":", "").strip()
        # replace any group of spaces with "_"
        data.save_name = '_'.join(data.save_name.split())



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