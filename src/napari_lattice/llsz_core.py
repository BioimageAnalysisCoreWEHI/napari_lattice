
import numpy as np
import pyclesperanto_prototype as cle
import dask.array as da
from typing import Union
from napari.layers.shapes import shapes
import RedLionfishDeconv as rl

from .utils import calculate_crop_bbox


#pass shapes data from single ROI to crop the volume from original data
def crop_volume_deskew(original_volume:Union[da.core.Array,np.ndarray,cle._tier0._pycl.OCLArray], 
                        deskewed_volume:Union[da.core.Array,np.ndarray,cle._tier0._pycl.OCLArray], 
                        roi_shape:Union[shapes.Shapes,list,np.array], 
                        angle_in_degrees:float=30, 
                        voxel_size_x:float=1, 
                        voxel_size_y:float=1, 
                        voxel_size_z:float=1, 
                        z_start:int=0, 
                        z_end:int=1,
                        debug:bool=False):

    """
        Uses coordinates from deskewed space to find corresponding coordinates in original volume 
        and deskew only the specific volume
    Args:
        original_volume (np.array): Volume to deskew (zyx)
        deskewed_volume (np.array): Deskewed volume (zyx)
        roi_shape ((Union[shapes.Shapes,list,np.array])): if shapes layer or data passed, use onl
        angle_in_degrees ([float): deskewing angle in degrees
        voxel_size_x (float): [description]
        voxel_size_y (float): [description]
        voxel_size_z (float): [description]
        z_start (int): [description]
        z_end (int): [description]
        debug (bool) : False , Can be used to return the uncropped volume for debugging purposes
    """

    assert len(original_volume.shape) == 3, print("Shape of original volume must be 3")
    assert len(deskewed_volume.shape) == 3, print("Shape of deskewed volume must be 3")
    #assert len(shape) == 4, print("Shape must be an array of shape 4 ")
    shape = None


    #if shapes layer, get first one
    if type(roi_shape) is shapes.Shapes:
        shape = roi_shape.data[0]
    #if its a list and each element has a shape of 4, its a list of rois
    elif type(roi_shape) is list and len(roi_shape[0])==4:
        #TODO:change to accept any roi by passing index
        shape = roi_shape[0]
        #len(roi_shape) >= 1:  
        #if its a array or list with shape of 4, its a single ROI
    elif len(roi_shape) == 4 and type(roi_shape) in(np.ndarray,list):
        shape = roi_shape
        

    assert len(shape) == 4, print("Shape must be an array of shape 4") 


    crop_bounding_box, crop_vol_shape = calculate_crop_bbox(shape,z_start,z_end)
    
    #get reverse transform by rotating around original volume
    reverse_aff, excess_bounds, deskew_transform = get_inverse_affine_transform(original_volume,angle_in_degrees,voxel_size_x,voxel_size_y,voxel_size_z)

    #apply the transform to get corresponding bounding boxes in original volume
    crop_transform_bbox = np.asarray(list(map(lambda x: reverse_aff._matrix @ x,crop_bounding_box)))

    #get shape of original volume in xyz
    orig_shape = original_volume.shape[::-1]


    #Take min and max of the cropped bounding boxes to define min and max coordinates
    #crop_transform_bbox is in the form xyz
    
    min_coordinate = np.around(crop_transform_bbox.min(axis=0))
    max_coordinate = np.around(crop_transform_bbox.max(axis=0))

    #get min and max in each position
    #clip them to avoid negative values and any values outside the bounding box of original volume
    x_start = min_coordinate[0].astype(int)
    x_start = np.clip(x_start, 0,orig_shape[0])
    x_end = max_coordinate[0].astype(int)
    x_end = np.clip(x_end, 0,orig_shape[0])

    y_start = min_coordinate[1].astype(int)
    y_start = np.clip(y_start, 0,orig_shape[1])
    
    y_end = max_coordinate[1].astype(int)
    y_end = np.clip(y_end, 0,orig_shape[1])
    
    z_start_vol_prelim = min_coordinate[2].astype(int)
    z_start_vol = np.clip(z_start_vol_prelim, 0,orig_shape[2]) #clip to z bounds of original volume
    
    z_end_vol_prelim = max_coordinate[2].astype(int)
    z_end_vol = np.clip(z_end_vol_prelim, 0,orig_shape[2]) #clip to z bounds of original volume
    
    #If the coordinates are out of bound, then the final volume needs adjustment in Y axis
    #if skew in X direction, then use y axis for finding correction factor instead
    if z_end_vol_prelim!=z_end_vol:
        out_bounds_correction = z_end_vol_prelim - z_end_vol
    elif z_start_vol_prelim!=z_start_vol:
        out_bounds_correction = z_start_vol_prelim - z_start_vol
    else:
        out_bounds_correction = 0
        
    #make sure z_start < z_end
    if z_start_vol > z_end_vol:
        #tuple swap  #https://docs.python.org/3/reference/expressions.html#evaluation-order
        z_start_vol,z_end_vol = z_end_vol,z_start_vol
    
    #After getting the coordinates, crop from original volume and deskew only the cropped volume
    
    if type(original_volume) is da.core.Array:
        #If using dask, use .map_blocks(np.copy) to copy subset (faster)
        crop_volume = original_volume[z_start_vol:z_end_vol,y_start:y_end,x_start:x_end].map_blocks(np.copy).squeeze()
    else:
        crop_volume = original_volume[z_start_vol:z_end_vol,y_start:y_end,x_start:x_end]

    deskewed_prelim = cle.affine_transform(crop_volume, 
                                           transform =deskew_transform,
                                           auto_size=True)
    #The height of deskewed_prelim will be larger than specified shape
    # as the coordinates of the ROI are skewed in the original volume
    #IF CLIPPING HAPPENS FOR Y_START or Y_END, use difference to calculate offset
    
    deskewed_height = deskewed_prelim.shape[1]
    crop_height = crop_vol_shape[1]
    
    #Find "excess" volume on both sides due to deskewing
    crop_excess = int(round((deskewed_height  - crop_height)/2)) + out_bounds_correction
    #Crop in Y
    deskewed_crop = deskewed_prelim[:,crop_excess:crop_height+crop_excess,:]
    
    # For debugging, ,deskewed_prelim will also be returne which is the uncropped volume
    if debug:
        return deskewed_crop,deskewed_prelim
    else:
        return deskewed_crop 

#Get reverse affine transform by rotating around a user-specified volume
def get_inverse_affine_transform(original_volume,angle_in_degrees,voxel_x,voxel_y,voxel_z):
    """
    Calculate the inverse deskew transform and the excess z_bounds 
    Difference from using inverse on deskew_y transform is the rotation here is fixed around a
    specified volume and final affine matrix will be based on the ref volume used
    Args:
        original_volume (_type_): _description_
        angle_in_degrees (_type_): _description_
        voxel_x (_type_): _description_
        voxel_y (_type_): _description_
        voxel_z (_type_): _description_

    Returns:
        Inverse Affine transform (cle.AffineTransform3D), int: Excess z slices, Deskew transform (cle.AffineTransform3D)
    """    
    #calculate the deskew transform for specified volume
    deskew_transform = _deskew_y_vol_transform(original_volume,angle_in_degrees,voxel_x,voxel_y,voxel_z)

    #Get the deskew transform after bringing the volume into bounds
    deskewed_shape, new_deskew_transform, _ = cle._tier8._affine_transform._determine_translation_and_bounding_box(
                                        original_volume, deskew_transform)
    
    #Get the inverse of adjusted desnew transform
    deskew_inverse = new_deskew_transform.inverse()

    #We use the shape of deskewed volume to get the new vertices of deskewed volume in x,y and z
    from itertools import product
    nz, ny, nx = deskewed_shape
    deskewed_bounding_box = [list(x) + [1] for x in product((0, nx), (0, ny), (0, nz))]

    # transform the corners of deskewed volume using the reverse affine transform
    undeskew_bounding_box = np.asarray(list(map(lambda x: deskew_inverse._matrix @ x, deskewed_bounding_box)))

    #Get the maximum z value and subtract it from shape of original volume to get excess bounds of bounding box
    max_bounds = undeskew_bounding_box.max(axis=0).astype(int)
    rev_deskew_z = max_bounds[2]
    extra_bounds = int((rev_deskew_z - original_volume.shape[0]))

    return deskew_inverse, extra_bounds, deskew_transform

#Get deskew transform where rotation is around centre of "original_volume"
def _deskew_y_vol_transform(original_volume, angle_in_degrees:float = 30, voxel_size_x: float = 1,
              voxel_size_y: float = 1, voxel_size_z: float = 1, scale_factor: float = 1):
    """Return deskew transform for specified volume
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
    
    #shear factor for deskewing
    shear_factor = math.sin((90 - angle_in_degrees) * math.pi / 180.0) * (voxel_size_z/voxel_size_y)
    transform._matrix[1, 2] = shear_factor
    
    
    # make voxels isotropic, calculate the new scaling factor for Z after shearing
    # https://github.com/tlambert03/napari-ndtiffs/blob/092acbd92bfdbf3ecb1eb9c7fc146411ad9e6aae/napari_ndtiffs/affine.py#L57
    new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
    scale_factor_z = (new_dz / voxel_size_y) * scale_factor
    transform.scale(scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z)
    
    #rotation around centre of ref_vol
    #transform._concatenate(rotate_around_vol_mat(original_volume, (0-angle_in_degrees)))
    transform.rotate(angle_in_degrees= 0 - angle_in_degrees, axis=0)
    # correct orientation so that the new Z-plane goes proximal-distal from the objective.
    

    return transform

#deprecated
#Calculate rotation transform around a volume
def rotate_around_vol_mat(ref_vol,angle_in_degrees:float=30.0):
    """Return the rotation matrix , so its rotated around centre of ref_vol

    Args:
        ref_vol (tuple): Shape of the ref volume (zyx)
        angle_in_degrees (float, optional): [description]. Defaults to 30.0.

    Returns:
        Rotation matrix: Will be returned in the form xyz for clesperanto affine transforms
    """    
    angle_in_rad = angle_in_degrees * np.pi / 180.0
    #rotate_transform = cle.AffineTransform3D()
    #rotate_transform._matrix
    # first translate the middle of the image to the origin
    nz,ny,nx = ref_vol.shape
    T1 = np.array([
            [1, 0, 0, nx / 2],
            [0, 1, 0, ny / 2],
            [0, 0, 1, nz / 2],
            [0, 0, 0, 1]
            ])

    R = np.array([
                [1, 0, 0, 0],
                [0, np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
                [0, -np.sin(angle_in_rad), np.cos(angle_in_rad),0],
                [0, 0, 0, 1]
                ])
                
    T2 = np.array([
            [1, 0, 0, -nx / 2],
            [0, 1, 0, -ny / 2],
            [0, 0, 1, -nz / 2],
            [0, 0, 0, 1]
            ])
    T = np.eye(4)
    rotate_mat = np.dot(np.dot(np.dot(T, T1), R), T2)
    #print(rotate_mat)
    return rotate_mat

#https://github.com/rosalindfranklininstitute/RedLionfish/blob/19ff16fe307343e417039627240224b29b4f4a95/RedLionfishDeconv/napari_plugin.py#L97
#adapted the redfishlion napari plugin
def rl_decon(image,psf,niter:int=10,method:str="gpu",resAsUint8=False,useBlockAlgorithm=True):
    """Apply Richardson Lucy Deconvolution using the redfishlion library

    Args:
        image (_type_): Image to deconvolve
        psf (_type_): PSF to be used for deconvolution
        niter (int): No. of iterations
        method (str, optional): . Defaults to "gpu".
        resAsUint8 (bool, optional): int8 as output.
        useBlockAlgorithm (bool, optional): process images as blocks. Allows large arrays to be processed in gpu memory

    Returns:
        np.array: Deconvolved image
    """    
    assert image.ndim == 3, f"Image needs to be 3D. Got {image.ndim}"
    assert psf.ndim == 3, f"PSF needs to be 3D. Got {psf.ndim}"
    
    if method.lower() == "cpu":
        method = "cpu"
    else:
        method = "gpu"
    
    decon_data = rl.doRLDeconvolutionFromNpArrays(data = image, 
                                                  psfdata = psf, 
                                                  niter= niter, 
                                                  method = method ,
                                                  resAsUint8=resAsUint8,
                                                  useBlockAlgorithm=useBlockAlgorithm)
    return decon_data