import numpy as np
from collections import defaultdict
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import numpy as np
import pyclesperanto_prototype as cle


#get bounding box of ROI in 3D and the shape of the ROI
def calculate_crop_bbox(shape, z_start:int, z_end:int):
    """Get bounding box as vertices in 3D in the form xyz

    Args:
        shape (int): [description]
        z_start (int): Start of slice
        z_end (int): End of slice

    Returns:
        list,np.array: Bounding box of ROI in 3D (xyz), the shape of the ROI in 3D (zyx)
    """    
    #calculate bounding box and shape of cropped 3D volume
    #shape is 3D ROI from shape layer
    #get ROI coordinates,clip to zero, get shape of ROI volume and vertices of 3D ROI
    start = np.rint(np.min(shape, axis=0)).clip(0)
    stop = np.rint(np.max(shape, axis=0)).clip(0)

    #Shapes layer can return 3D coordinates, so only take last two
    if len(start>2): start = start[-2:]
    if len(stop>2): stop = stop[-2:]

    start=np.insert(start,0,z_start) # start[0]=z_start
    stop = np.insert(stop,0,z_end) #  =z_start

    z0,y0,x0 = np.stack([start,stop])[0].astype(int)
    z1,y1,x1 = np.stack([start,stop])[1].astype(int)

    crop_shape = (stop - start).astype(int).tolist()

    from itertools import product

    crop_bounding_box = [list(x)+[1] for x in product((x0,x1),(y0,y1),(z0,z1))] 
    return crop_bounding_box,crop_shape

def get_deskewed_shape(volume,
                        angle,
                        voxel_size_x_in_microns:float,
                        voxel_size_y_in_microns:float,
                        voxel_size_z_in_microns:float):
    """
    Calculate shape of deskewed volume

    Args:
        volume ([type]): Volume to deskew
        angle ([type]): Deskewing Angle
        voxel_size_x_in_microns ([type])
        voxel_size_y_in_microns ([type])
        voxel_size_z_in_microns ([type])

    Returns:
        tuple: Shape of deskewed volume in zyx
    """
    from pyclesperanto_prototype._tier8._affine_transform import _determine_translation_and_bounding_box

    deskew_transform = cle.AffineTransform3D()

    deskew_transform._deskew_y(angle_in_degrees=angle,
                                voxel_size_x=voxel_size_x_in_microns,
                                voxel_size_y=voxel_size_y_in_microns,
                                voxel_size_z=voxel_size_z_in_microns)
    
    #TODO:need better handling of aics dask array
    if len(volume.shape) == 5:
        volume = volume[0,0,...]

    new_shape, new_deskew_transform, _ = _determine_translation_and_bounding_box(volume, deskew_transform)
    return new_shape


#https://stackoverflow.com/a/10076823
#Credit: Antoine Pinsard
#Convert xml elementree to dict
def etree_to_dict(t):
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

#block printing to console
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
        
#dask implementation not available yet, so copying from their repo
#https://github.com/dask/dask/blob/dca10398146c6091a55c54db3778a06b485fc5ce/dask/array/routines.py#L1889        
def dask_expand_dims(a,axis):
    if type(axis) not in (tuple, list):
        axis = (axis,)

    out_ndim = len(axis) + a.ndim

    shape_it = iter(a.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]

    return a.reshape(shape)