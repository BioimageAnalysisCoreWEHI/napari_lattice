from datetime import time
import numpy as np
from dask_image import ndinterp
from gputools import affine as affineGPU

import pyclesperanto_prototype as cle

def affine_transform(source, at:cle.AffineTransform3D=None):
    """
    Applies an AffineTransform3D to a given soure image. What's different to the affine_transform in clesperanto:
    Assuming the transform rotates the image out of the field of view (negative coordinates), it will be moved back 
    so that all pixels are visible. Thus, the applied transform actually contains a translation, that was not passed
    as parameter.
    """
    
    # define coordinates of all corners of the current stack
    from itertools import product
    nx,ny,nz=source.shape
    original_bounding_box=[list(x)+[1] for x in product((0,nz),(0,ny),(0,nx))] 
    # transform the corners using the given affine transform
    transformed_bounding_box=np.asarray(list(map(lambda x:at._matrix@x,original_bounding_box)))

    # the min and max coordinates tell us from where to where the image ranges (bounding box)
    min_coordinate = transformed_bounding_box.min(axis=0)
    max_coordinate = transformed_bounding_box.max(axis=0)
    # determin the size of the transformed bounding box
    new_size =(max_coordinate - min_coordinate)[0:3].astype(int).tolist()[::-1]

    # create a new stack on GPU
    destination = cle.create(new_size)
    
    # we make a copy to not modify the original transform
    transform_copy = cle.AffineTransform3D()
    transform_copy._concatenate(at._matrix)
    
    # if the new minimum-coordinate is `-x`, we need to 
    # translate the stack by `x` so that the new origin is (0,0,0)
    translation = -min_coordinate
    transform_copy.translate(
        translate_x = translation[0],
        translate_y = translation[1],
        translate_z = translation[2]
    )
    
    # apply transform and return result
    return cle.affine_transform(source, destination, transform=transform_copy)
    

def deskew_y(raw, rotation_angle : float = 30,dx: float = 1.0 , dy:float = 1.0, dz:float = 1.0, reverse:bool = False, keep_orientation:bool = False, viewer:"napari.Viewer"=None) -> "napari.types.ImageData":
    """
    Deskew an image stack

    Returns:
        Deskewed volume
    """    
    
    # from https://github.com/SpimCat/unsweep/blob/6592b2667bda304336360e099ac015654a87787a/src/main/java/net/haesleinhuepf/spimcat/unsweep/Unsweep.java#L45
    import math
    
    deskew_transform = cle.AffineTransform3D()

    #scaling 
    new_dz=np.sin(rotation_angle * np.pi/180.0)*dz
    scale_factor=(new_dz/dy)
    deskew_transform.scale(scale_z=scale_factor)

    # shearing
    deskew_factor = 1.0 / math.tan(rotation_angle * math.pi / 180)
    
    shear_mat = np.array([
                        [1.0, 0 ,0 , 0],
                        [0, 1.0, deskew_factor, 0],
                        [0, 0, 1.0, 0],  
                        [0, 0.0, 0.0, 1.0]          
                        ])
    deskew_transform._concatenate(shear_mat)
    
    delta = 0

    # rotation
    deskew_transform.rotate(angle_in_degrees=(delta-rotation_angle), axis=0)

    # apply transform
    return affine_transform(raw, at=deskew_transform)

def deskew_affine_matrix(deskew_factor:float=1.7321,skew_dir:str="Y",reverse:bool=False):
    """Take deskew factor and skew direction and return coresponding affine matrix
    Info about transformation matrices: https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
    Volumes are assumed to be in the form zyx. This is critical or it will perform the wrong transformation!

    Args:
        deskew_factor (float, optional): Shear factor to skew the image. Defaults to 1.7321.
        skew_dir (str, optional): [Direction of skew]. Defaults to "Y".
        reverse (bool, optional): [Option to reverse the transformation. reverse=true can be used if going from deskewed to original volume]. Defaults to False.

    Returns:
        [np.array]: [Shear affine transformation matrix]
    """    
 
    ##Use negative value for when deriving coordinates from deskewed volume
    deskew_factor = deskew_factor if not reverse else -deskew_factor
    #Zeiss, deskew_factor is at (1,0)
    #home-built lattice: deskew_factor is at (2,0)
    if skew_dir=="Y":
        shear_mat = np.array([
                    [1.0, 0 ,0 , 0],
                    [deskew_factor, 1.0, 0.0, 0],
                    [0, 0.0, 1.0, 0],  
                    [0.0, 0.0, 0.0, 1.0]          
                    ])                                 
    
    elif skew_dir=="X":
        shear_mat = np.array([
                    [1.0, 0 ,0 , 0],
                    [0, 1.0, 0.0, 0],
                    [deskew_factor, 0.0, 1.0, 0],  
                    [0.0, 0.0, 0.0, 1.0]   
                    ])        
    
    return shear_mat


def rotate_affine_matrix(vol_shape,skew_dir:str="Y",angle:float=30.0,reverse:bool=False):
    """Take angle of rotation and skew direction, return the rotation affine matrix
    Info about transformation matrices: https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
    Volumes are assumed to be in the form zyx. This is critical or it will perform the wrong transformation!

    Args:
        vol_shape ([list, tuple or np.array]): [Shape of the 3D volume. It can be passed as (z,y,x,1) or (z,y,x)]
        skew_dir (str, optional): [Direction of skew as this dictates the axis of rotation]. Defaults to "Y" -> Rotation around Z.
        angle (float, optional): [Lightsheet angle. Default values is for Zeiss Lattice]
        reverse (bool, optional): [Option to reverse the transformation. reverse=true can be used if going from deskewed to original volume]. Defaults to False.

    Raises:
        ValueError: [description]

    Returns:
        [np.array]: [Rotation affine transformation matrix]
    """
    
    if len(vol_shape)==4: #if coordinate 
        nz,ny,nx,_=vol_shape
    elif len(vol_shape)==3: #if passing shape of volume
        nz,ny,nx=vol_shape
    elif vol_shape.ndim==3: #if volume
        nz,ny,nx=vol_shape.shape
    else:
        raise ValueError("Array is neither shape nor volume. Check the format of array")
    
    #convert angle from degrees to radians
    theta = angle * np.pi / 180
    
    ##Use negative value for when deriving coordinates from deskewed volume
    theta = theta if not reverse else -theta

    
    #To rotate a 3D volume, need to translate the image to origin
    #rotate and then translate back
    
    # first translate the middle of the image to the origin
    T1 = np.array([
         [1, 0, 0, nz / 2],
         [0, 1, 0, ny / 2],
         [0, 0, 1, nx / 2],
         [0, 0, 0, 1]
         ])
    # then rotate theta degrees about the X axis if skew direction is Y
    # rotate around Y axis if skew direction is X
    #Rotation matrix usually specified in literature is for the order (x,y,z)
    #as aicsimageio returns dimensions as z,y,x, we use corresponding rotation matrices
    #For rotation around X axis, we use affine matrix for rotation round Z 
    #For rotation around Y, as Y is in the middle for (x,y,z) and (z,y,x)
    #it remains the same (rotataion around Y affine transform)
    #Essentially, we are rotating around the order for the coordinate of interest!
    
    if skew_dir=="Y":
        # as X is in the 3rd dimension, use rotation around Z
        R = np.array([
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta),0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ])
    elif skew_dir=="X":
        # as Y is in the 2nd dimension, use rotation around Y
        R = np.array([
                [np.cos(theta),0,  -np.sin(theta), 0],
                [0, 1, 0, 0],
                [np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]
                ])    
    elif skew_dir=="Z":
        # as Z is in the 1st dimension, use rotation around X matrix
        R = np.array([
                [np.cos(theta),-np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ])  
    # then translate back to the original origin
    T2 = np.array([
             [1, 0, 0, -nz / 2],
             [0, 1, 0, -ny / 2],
             [0, 0, 1, -nx / 2],
             [0, 0, 0, 1]
             ])
    T = np.eye(4)
    rotate_mat = np.dot(np.dot(np.dot(T, T1), R), T2)
    
    return rotate_mat

#Affine matrix for scaling
def scale_Z_affine_matrix(scale_factor:float, reverse:bool=False):
    """Take scale factor and return the scaling matrix
    Info about transformation matrices: https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html

    Args:
        scale_factor ([float]): Enter the scale value for Z-scaling
        reverse (bool, optional): [Option to reverse the transformation. reverse=true can be used if going from deskewed to original volume]. Defaults to False.

    Returns:
        [np.array]: affine matrix for scaling
    """
    scale_mat=np.array([
                [scale_factor, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ])
    #Use inverse matrix for when deriving coordinates from deskewed volume
    scale_mat = scale_mat if not reverse else np.linalg.inv(scale_mat) 
    return scale_mat

#Affine matrix for translation
def translate_Y_matrix(translation, reverse:bool=False):
    """Take translation and return the translation affine matrix in Y
    Info about transformation matrices: https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html

    Args:
        translation ([type]): Enter the translation value (Y direction)
        reverse (bool, optional): [Option to reverse the transformation. reverse=true can be used if going from deskewed to original volume]. Defaults to False.

    Returns:
        [np.array]: [Translation affine matrix]
    """    
    """"
    """
    translation = translation if not reverse else -translation
    translate_mat=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, translation],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ])
    return translate_mat
    
def apply_deskew_transformation(vol,angle:float=30.0,shear_factor:float=1.7321,scale_factor:float=0,translation:float=0,skew_dir:str="Y",reverse:bool=False,dask:bool=False):
    """Performs Scaling, Deskew, Rotation and Translation of the raw data from Zeiss lattice and returns a processed image

    Args:
        vol ([type]): [description]
        angle (float, optional): [description]. Defaults to 30.0.
        shear_factor (float, optional): [description]. Defaults to 1.7321.
        scale_factor (float, optional): [description]. Defaults to 0.
        translation (float, optional): [description]. Defaults to 0.
        skew_dir (str, optional): [description]. Defaults to "Y".
        reverse (bool, optional): [description]. Defaults to False.
        dask (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    """Performs scale, deskew and rotation on data from Zeiss lattice
       vol=array of shape (z,y,x)
    """

    deskew_mat=deskew_affine_matrix(shear_factor,skew_dir=skew_dir,reverse=reverse)
    rotate_mat=rotate_affine_matrix(vol_shape=vol.shape,skew_dir=skew_dir,angle=angle,reverse=reverse)
    scale_mat=scale_Z_affine_matrix(scale_factor,reverse)
    translate_mat_y=translate_Y_matrix(translation, reverse)
    
    #normal deskewing affine transformation
    if not reverse:
        deskew_rotation_mat=translate_mat_y@rotate_mat@deskew_mat@scale_mat 
    else:
        #reverse transformation order
        deskew_rotation_mat=scale_mat@deskew_mat@rotate_mat@translate_mat_y
    
    #If using dask, then apply dask image ndinterp affine transformation
    if dask:
        #rotate_axes=(0,1) #for skew Y, if X (0,2)
        deskewed_vol = ndinterp.affine_transform(vol,np.linalg.inv(deskew_rotation_mat), interpolation="linear",dtype=vol.dtype)
        #rotate_coverslip=deskew.map_blocks(ndimage.rotate, angle=angle, axes=rotate_axes,dtype=vol.dtype)
        deskewed_vol=deskewed_vol.rechunk(vol.shape)
    else:
        deskewed_vol = affineGPU(vol, np.linalg.inv(deskew_rotation_mat),output_shape=vol.shape,mode="constant")
        #scipy deskewed_vol = affine_transform(vol, n
    return deskewed_vol #img_as_uint(processed)


