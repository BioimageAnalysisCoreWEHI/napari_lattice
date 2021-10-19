
#llsz imports
from .utils import transform_dim, get_translation_y,get_new_coordinates,\
    get_shear_factor,get_scale_factor
import numpy as np

#TODO: break down into smaller functions if needed

def process_czi(stack,angle,skew_direction):
    """Process a AICSImage dask array to calculate the shape and coordinates of the final deskewed array
    Returns the shape of the deskewed array, any translations to keep volume in bounds
    start and end z positions of deskewed array after rotation

    Args:
        stack ([type]): [description]
        angle ([type]): [description]
        skew_direction ([type]): [description]

    Returns:
        deskew_shape (tuple/array): Final shape of the deskewed array (1,z,y,x)
        vol_shape (tuple/array): Shape of the volume (1,z,y,x)
        translate_y: Any translations required to keep deskewed array in bounds
        z_start: Starting Z slice
        z_end: Ending Z slice
    """    
    #Get all metadata AICSIMAGEIO returns the data consistently as TCZYX regardless of image dimensions
    print("Image is read as ",stack.dims.order)

    dz,dy,dx=stack.physical_pixel_sizes
    #channels=stack.dims.C

    #if scenes are present
    if "S" in stack.dims.order:
        print("Image has scenes. Currently does not support different scenes")
        scenes=stack.dims.S
    else:
        scenes=0
        
    #time=stack.dims.T
    nz=stack.dims.Z
    ny=stack.dims.Y
    nx=stack.dims.X

    print("Dimensions of image (X,Y,Z)",nx,ny,nz)
    print("Pixel size of image (dX,dY,dZ) in microns",dx,dy,dz)

    #calculate deskew factor
    #Using tan of angle subtracted by 90 gives accurate deskew factor; verified on FIJI with CLIJ
    deskew_factor=get_shear_factor(angle=30.0)
    print("Using deskew factor of: ", deskew_factor)

    #Calculating scale factor
    scale_factor=get_scale_factor(angle,dx,dz)
    print("Using scaling factor of: ", scale_factor)

    #original/raw volume shape
    vol_shape=(nz,ny,nx,1)

    #Find new y dimension by performing an affine transformation of original volume
    deskewed_y=transform_dim(vol_shape,vol_shape,angle,dy,dz,skew_dir=skew_direction,reverse=False)

    vol_shape_deskew=(nz,deskewed_y,nx,1)
    print("New shape after deskewing is: ",vol_shape_deskew)

    #The volume above maybe outside of bounds, which can be determined by checking the value of Y-coordinate at origin
    #If it isn't, then we use the value to translate the volume within bounds of the image frame
    #Take raw volume, perform deskew, and then rotate around deskewed volume to get the Y coordinate value
    #THe shape you give for rotation is key in getting the right coordinates
    translate_y=get_translation_y(vol_shape_deskew,vol_shape,angle,dy,dz,skew_dir=skew_direction,reverse=False)
    #deskew_factorscale_factor,translation=0

    print("Volume will be translated by: ",translate_y," pixels to keep in bounds.")

    #Calculate new deskew coordinates based on translation
    deskew_coordinates=get_new_coordinates(vol_shape_deskew,vol_shape,angle,dx,dz,translate_y,skew_direction,reverse=False)

    #Values to use within CLIJ affine transform 3D
    print("\nThe transformation within this notebook can be used within FIJI using the CLIJ affinetransform 3D method")
    print("scaleZ=",scale_factor," shearYZ=-",deskew_factor," -center rotateX=-",angle," center translateY=-",translate_y, sep='')

    print("\nAfter deskewing, rotation and translation, new coordinates for the deskewed volume are:")
    print(deskew_coordinates)

    #Calculate the no of z slices
    z_end=np.abs(round(deskew_coordinates[0][0]))
    z_start=np.abs(round(deskew_coordinates[7][0]))
    print("Start slice is: ",z_start)
    print("End slice is: ",z_end)
    no_slices=np.absolute(z_end-z_start)
    #print("No of slices: ",no_slices)
    print("Dimensions of deskewed stack: ",(no_slices,deskewed_y,nx))

    deskew_shape=tuple((nz,deskewed_y,nx))

    return deskew_shape,vol_shape,translate_y,z_start,z_end



    #channel_range=range(channels)
    #raw_data_dask=stack.get_image_dask_data("TCZYX",C=channel_range,S=0)