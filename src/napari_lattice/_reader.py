"""
reader plugin for h5 saved using np2bdv
https://github.com/nvladimus/npy2bdv
#TODO: pass pyramidal layer to napari
##use ilevel parameter in read_view to access different subsamples/pyramids
#pass a list of images with different resolution for pyramid; use is_pyramid=True flag in napari.add_image
"""

import dask.array as da
import dask.delayed as delayed
import os 
def napari_get_reader(path):
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
    
    print(path)
    import npy2bdv
    h5_file = npy2bdv.npy2bdv.BdvEditor(path)

    img = []

    #get dimensions of first image
    first_timepoint = h5_file.read_view(time=0,channel=0)

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