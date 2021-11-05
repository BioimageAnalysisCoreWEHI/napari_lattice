from itertools import product
import numpy as np
from .transformations import deskew_affine_matrix,rotate_affine_matrix,scale_Z_affine_matrix,translate_Y_matrix
from collections import defaultdict


def get_vertices_volume(vol_shape):
    """Returns vertices of volume given the shape

    Args:
        vol_shape (list,np.array): Shape of the volume (z,y,x) or (z,y,x,1)

    Returns:
        list: List of coordinates for 3D volume
    """    

    if(len(vol_shape)==4):
        nz,ny,nx,_=vol_shape
    else:
        nz,ny,nx=vol_shape
    #generating list of coordinates using itertools product (Cartesian product of input iterables)
    #adding a 1 at the end; #For appending a value to,  [1] +[2] actually concatenates them to [1,2]
    coord_list=[list(x)+[1] for x in product((0,nz),(0,ny),(0,nx))] 
    return coord_list

def get_vertices_coord(coord):
    """Returns vertices of a volume given the coordinate bounds of the volume (z1,z2),(y1,y2),(x1,x2)

    Args:
        coord (tuple or list): Coordinate bounds of the volume (z1,z2),(y1,y2),(x1,x2)

    Returns:
        list: Returns vertices of a volume 
    """    

    (z1,z2),(y1,y2),(x1,x2)=coord
    #generating list of coordinates using itertools product (Cartesian product of input iterables)
    #adding a 1 at the end; #For appending a value to,  [1] +[2] actually concatenates them to [1,2]
    coord_list=[list(x)+[1] for x in product((z1,z2),(y1,y2),(x1,x2))] 
    return coord_list


def transform_coordinates(affine_mat,coord_list):
    """Apply transformation matrix to a coordinate list
       Returns transformed coordinates when affine transformation matrix is applied to vertices of volume

    Args:
        affine_mat (np.array): Affine matrix 
        coord_list (list): Coordinate list

    Returns:
        [list]: Transformed coordinate list
    """    

    transformed_coordinates=list(map(lambda x:affine_mat@x,coord_list))
    return transformed_coordinates

#Scale,deskew,rotate and translate
def get_new_coordinates(vol_shape,shape_coord,angle:float,dx_y:float,dz:float,translation:float=0,skew_dir:str="Y",reverse:bool=False):
    
    """Get the new coordinates of the volume after applying a transformation
    vol_shape is the shape around with rotation is performed

    Args:
        vol_shape (tuple,np.array): Shape of final volume around with rotation performed (z,y,x,1) 
        shape_coord ([type]): Coordinates of the volume that will be transformed
        angle (float): Angle of acquisition
        dx_y (float): X or Y spacing (microns)
        dz (float): Voxel spacing/distance between Z slices (microns)
        translation (float, optional): Translations to keep final volume in bounds of the image. Defaults to 0.
        skew_dir (float, optional): Direction of skew; Zeiss lattice is "Y", Janelia is "X". Defaults to "Y".
        reverse (bool, optional): [Option to reverse the transformation. reverse=true can be used if going from deskewed to original volume]. Defaults to False.
    Raises:
        ValueError: Coordinates (shape_coord) needs to be (z,y,x) or (z,y,x,1) or (z1,z2),(y1,y2),(x1,x2)"

    Returns:
        Transformed coordinates [type]: Transformed coordinates of shape_coord after affine transformation with rotation around vol_shape
    """    
    shear_factor=get_shear_factor(angle)

    scale_factor=get_scale_factor(angle,dx_y,dz)

    #Deskew matrix
    deskew_mat=deskew_affine_matrix(shear_factor,skew_dir,reverse)

    #rotation affine matrix using original dimensions
    rotate_mat=rotate_affine_matrix(vol_shape,skew_dir,angle,reverse)

    #scaling matrix
    scale_mat=scale_Z_affine_matrix(scale_factor,reverse)
    
    #translation matrix
    translate_mat_y=translate_Y_matrix(translation, reverse)

    #flip_mat_z=rotate_affine_matrix(vol_shape,angle=180,skew_dir="Y",reverse=reverse)
    
    #determine if shape_coord is a shape or set of coordinates
    flattened_coord=np.array(shape_coord).flatten()
    
    if len(flattened_coord) in (3, 4):
        print("Transforming shape info to vertices")
        coord_list=get_vertices_volume(shape_coord)
    elif len(flattened_coord)== 6:
        print("Transforming coordinates to vertices")
        coord_list=get_vertices_coord(shape_coord)
    else: 
        print("\nCoordinates to transform are not in the right shape, they should be in the form (z,y,x) or (z,y,x,1) or (z1,z2),(y1,y2),(x1,x2)")
        #print("You have passed: ",shape_coord)
        raise ValueError(shape_coord)
    
    #affine transform by multiplying the affine matrix with the coordinate
    #we give the inverse of the matrix to the affine transform function
    if not reverse:
        affine_mat=translate_mat_y@rotate_mat@deskew_mat@scale_mat#@flip_mat_z #scale, deskew and then rotate 
    else:
        affine_mat=scale_mat@deskew_mat@rotate_mat@translate_mat_y #reverse ###flip_mat_z@
    #print(affine_mat)
    transformed_coordinates=transform_coordinates(affine_mat,coord_list)
    return transformed_coordinates

#Used to calculate the new Y dimension
def transform_dim(vol_shape,shape_coord,angle:float,dx_y:float,dz:float, skew_dir:str="Y",reverse:bool=False):
#,deskew_factor:float,angle:float,scale_factor:float,
    """Return the new dimension after deskewing/transformation 
    For Zeiss lattice, it will be the new Y axis

    Args:
        vol_shape (tuple,np.array): Shape of final volume around with rotation performed (z,y,x,1) 
        shape_coord ([type]): Coordinates of the volume that will be transformed
        dx_y (float): X or Y spacing (microns)
        dz (float): Voxel spacing/distance between Z slices (microns)
        translation (float, optional): Translations to keep final volume in bounds of the image. Defaults to 0.
        skew_dir (float, optional): Direction of skew; Zeiss lattice is "Y", Janelia is "X". Defaults to "Y".
        reverse (bool, optional): [Option to reverse the transformation. reverse=true can be used if going from deskewed to original volume]. Defaults to False.
    Returns:
        deskewed dimension [int]: New dimension after deskewing
    """    
    #shear_factor=get_shear_factor(angle)
    #scale_factor=get_scale_factor(angle,dx_y,dz)
    translation=0 #no translation of the volume yet
    #get coordinates without translation
    transformed_coordinates=get_new_coordinates(vol_shape,shape_coord,angle,dx_y,dz,translation,skew_dir,reverse)
    deskewed_y=(transformed_coordinates[7][1] - transformed_coordinates[0][1])
    return round(deskewed_y)

def get_translation_y(vol_shape_deskew,shape_coord,angle:float,dx_y:float,dz:float,skew_dir:str="Y",reverse:bool=False):
    #,deskew_factor:float,,scale_factor:float
    """Calculate the translation in Y direction required to keep the deskewed volume within frame
    to avoid clipping

    Args:
        vol_shape_deskew (tuple,np.array): Shape of final volume,usually deskewed volume around with rotation performed (z,y,x,1) 
        shape_coord ([type]): Coordinates of the volume that will be transformed
        angle (float): Angle of acquisition
        dx_y (float): X or Y spacing (microns)
        dz (float): Voxel spacing/distance between Z slices (microns)
        scale_factor (float): Scaling factor
        translation (float, optional): Translations to keep final volume in bounds of the image. Defaults to 0.
        skew_dir (float, optional): Direction of skew; Zeiss lattice is "Y", Janelia is "X". Defaults to "Y".
        reverse (bool, optional): [Option to reverse the transformation. reverse=true can be used if going from deskewed to original volume]. Defaults to False.

    Returns:
        translate_y [float]: Pixels to translate vol_shape_deskew after deskewing
    """   
    translation=0 #no translation of the volume yet
    #If deskewing leads to volume being out of bounds in the final image, 
    # first y coordinate will be negative, so we translate volume by this value to avoid clipping
    #get coordinates without translation
    deskew_coordinates_no_translation=get_new_coordinates(vol_shape_deskew,shape_coord,angle,dx_y,dz,translation,skew_dir,reverse)
    #get_new_coordinates(vol_shape_deskew,shape_coord,shear_factor,angle,scale_factor,\ #translation,skew_dir,reverse=False)
    translate_y= -deskew_coordinates_no_translation[0][1] #np.abs?
    return translate_y

def get_shear_factor(angle:float = 30.0):
    """[summary]

    Args:
        angle (float, optional): [description]. Defaults to 30.0.

    Returns:
        [type]: [description]
    """    
    shear_factor=np.tan((90-angle) * np.pi / 180)
    return shear_factor


def get_scale_factor(angle:float,dx_y:float,dz:float):
    """[summary]

    Args:
        angle (float): [description]
        dx_y (float): [description]
        dz (float): [description]

    Returns:
        [type]: [description]
    """    
    new_dz=np.sin(angle * np.pi/180.0)*dz
    scale_factor=new_dz/dx_y
    return scale_factor

#https://stackoverflow.com/questions/7684333/converting-xml-to-dictionary-using-elementtree
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

