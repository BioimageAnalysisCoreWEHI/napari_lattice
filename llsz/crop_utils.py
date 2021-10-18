#File to add utilities to enable cropping
#ROIs must be in (z,y,x format)
from napari import Viewer
from napari.types import ImageData
import numpy as np
from .utils import get_new_coordinates, get_scale_factor, get_shear_factor, get_vertices_volume, transform_dim,get_translation_y
from .transformations import deskew_zeiss
import dask.array as da
from pprint import pprint #pretty print!
import itertools

#define function to calculate deskew volume based on input or reference volume
#function for cropping and getting rois, 
#distribute it to rest of the functions here
#return a list of cropped volumes

#provide raw volume shape or enter raw volume shape
#Need to provide raw volume shape anyway to crop from it
#or get deskew volume from existing analysis

#one function for crop and deskew single ROI? -> Can use this for testing
#one function for taking a list of ROIs and passing to the above function?

def crop_deskew_roi(crop_roi,vol_shape,vol,angle,dx_y,dz,z_start,z_end,time,channel,skew_dir,reverse=True):
    #vol will be just  dask volume, pass channel, time

    #Get 3D shape of the ROI
    #Get 3D coordinates of the ROI
    #Transform the ROI using the 3D ROI coordinates and shape of the deskewed volume
    #This will return the ROI coordinates for cropping in the non-deskewed volume
    #Get the shape of this new ROI to be cropped
    #As the ROI shape is skewed in non-deskewed volume; extend Y coordinate to get the entire ROI

    shear_factor=get_shear_factor(angle)
    scale_factor=get_scale_factor(angle,dx_y,dz)


    #Find new y dimension by performing an affine transformation of original volume
    deskewed_y=transform_dim(vol_shape,vol_shape,angle,dx_y,dz,skew_dir,reverse=False)
    vol_shape_deskew=(vol_shape[0],deskewed_y,vol_shape[2],1)

    #Calculate translation to get new vol_shape_deskew
    translate_y=get_translation_y(vol_shape_deskew,vol_shape,angle,dx_y,dz,skew_dir=skew_dir,reverse=False)
    
    #get 3d shape and coordinates for the roi from MIP or reference image
    roi_shape, roi_coord=get_roi_3D_shape_coord(crop_roi,z_start,z_end)
    print("ROI shape in reference image", roi_shape)
    
    #transform the roi using the deskew shape as we are going from deskew volume to original volume 
    #reverse=True
    new_roi=get_transformed_roi_coord(vol_shape_deskew,roi_coord,angle,dx_y,dz,translate_y,skew_dir=skew_dir,reverse=reverse) #reverse=True
    raw_vol_shape=get_roi_3D_shape(new_roi)
    print("Transformed ROI coordinates in raw volume",new_roi)

    #check coordinates are >0 
    new_rois_flatten=list(itertools.chain.from_iterable(new_roi))
    if not all([r>=0 for r in new_rois_flatten]):
        raise ValueError("New ROI for cropping: Coordinates should not have negative values.")

    #As the shape changes after deskewing, we take the maximum across all dimensions ensuring that nothing gets clipped
    max_roi_shape=get_roi_skew_shape(raw_vol_shape,roi_shape,angle,dx_y,dz,skew_dir=skew_dir)
    print("ROI shape that will be used in raw image to get entire volume (due to skew)", max_roi_shape)

    transformed_roi_depth=max_roi_shape[0]
    transformed_roi_height=max_roi_shape[1]
    transformed_roi_width=max_roi_shape[2]

    #actual height of ROI from MIP image
    #roi_height=roi_shape[1]

    #Height of the ROI from original image
    roi_skew_height=raw_vol_shape[1]
    #roi_skew_depth=raw_vol_shape[0]
    
    (z_roi_1,z_roi_2),(y_roi_1,y_roi_2),(x_roi_1,x_roi_2)=new_roi

    if z_roi_1<0: 
        #z_roi_2=z_roi_2-z_roi_1
        z_roi_2=z_roi_2+z_roi_1
        transformed_roi_depth=z_roi_2
        z_roi_1=0

    #Crop from raw volume based on coordinates determined above
    crop_dask_stack=vol[time,channel,z_roi_1:z_roi_2,y_roi_1:y_roi_2,x_roi_1:x_roi_2].map_blocks(np.copy).squeeze()

    print("Shape of original volume",crop_dask_stack.shape)
    print("Transformed ROI intended shape",transformed_roi_depth,transformed_roi_height,transformed_roi_width)

    #Confirm the cropped volume depth matches that of the transformed roi depth we calculated above
    #It may not match if the crop is close to the ends of the image

    if crop_dask_stack.shape[0]!= transformed_roi_depth:
        z_diff = max_roi_shape[0] - crop_dask_stack.shape[0] #transformed_roi_depth
        #max_roi_shape[0]=crop_dask_stack.shape[0]  
        print("Adjusting for empty array at ends of the stack")
        print("New ROI shape extracted from raw volume:",max_roi_shape)
    else:
        z_diff=0


    #create empty dask array with same size as the transformed roi from above
    deskew_roi_img=da.zeros(max_roi_shape,dtype=vol.dtype,chunks=tuple(max_roi_shape))
    
    #hold the values for the image in this array
    deskew_roi_img[:transformed_roi_depth-z_diff,:roi_skew_height,:transformed_roi_width]=crop_dask_stack
    
    #deskew the cropped roi
    deskew_roi=deskew_zeiss(deskew_roi_img,angle,shear_factor,scale_factor,translate_y,reverse=False,dask=False)
    deskew_roi=deskew_roi.astype("uint16")

    #Get the bounds for cropping
    #Transform our volume of interest within the bounds of the extended volume to get coordinates
    crop_y_top,crop_y_bottom=get_ROI_bounds(deskew_roi.shape,raw_vol_shape,angle,dx_y,dz,translate_y,skew_dir,False)
    
    if crop_y_top<0:
        crop_y_top=0 #if negative, then make it zero
    
    #print(crop_y_top , crop_y_bottom)

    #crop_y_bottom=round(crop_y_top+roi_height)+(pad*2)
    deskew_roi=deskew_roi[:,crop_y_top:crop_y_bottom,:]
    return deskew_roi


#TODO:MODIFY TO ACCEPT ROI LISTS and link to above
def crop_roi_list(crop_rois,vol_shape,angle,dx_y,dz,z_start,z_end,skew_dir,reverse=True):
    ## Add option so user can specify z and t range
    roi_mip_shape=[]
    roi_transformed_shape=[]

    #mip roi list
    roi_mip_coord_list=[]
    roi_transformed_coord_list=[]

    transformed_roi=[]

    roi_orig_shape=[]

    z_min=z_start
    z_max=z_end
    #Get 3D shape of the ROI
    #Get 3D coordinates of the ROI
    #Transform the ROI using the 3D ROI coordinates and shape of the deskewed volume
    #This will return the ROI coordinates for cropping in the non-deskewed volume
    #Get the shape of this new ROI to be cropped
    #As the ROI shape is skewed in non-deskewed volume; extend Y coordinate to get the entire ROI

    shear_factor=get_shear_factor(angle)
    scale_factor=get_scale_factor(angle,dx_y,dz)


    #Find new y dimension by performing an affine transformation of original volume
    deskewed_y=transform_dim(vol_shape,vol_shape,angle,dx_y,dz,skew_dir,reverse=False)
    vol_shape_deskew=(vol_shape[0],deskewed_y,vol_shape[2],1)

    #Calculate translation to get new vol_shape_deskew
    translate_y=get_translation_y(vol_shape_deskew,vol_shape,angle,dx_y,dz,skew_dir=skew_dir,reverse=False)

    print("New shape after deskewing is: ",vol_shape_deskew)

    for coord in crop_rois:
        #get shape of roi in 3d and roi coord as 3d roi volume bounds
        roi_shape,roi_coord=get_roi_3D_shape_coord(coord,z_min,z_max)
        roi_mip_shape.append(roi_shape)#(depth,height,width))
        roi_mip_coord_list.append(roi_coord)
            
        #transform the roi using the deskew shape as we are going from deskew volume to original volume 
        #reverse=True
        new_roi=get_transformed_roi_coord(vol_shape_deskew,roi_coord,angle,dx_y,dz,translate_y,skew_dir=skew_dir,reverse=reverse)
        roi_transformed_coord_list.append(new_roi)
        temp_shape=get_roi_3D_shape(new_roi)
        roi_orig_shape.append(temp_shape)
            
        #As the shape changes after deskewing, we take the maximum across all dimensions ensuring that nothing gets clipped
        max_roi_shape=get_roi_skew_shape(temp_shape,roi_shape,angle,dx_y,dz,skew_dir=skew_dir)
        roi_transformed_shape.append(max_roi_shape)

    #Now we have the coord as (z,y,x) for each roi in the max project
    #We have the 3D shape for each ROI
    #Apply transformation and we have new coordinats (roi_transformed_coord_list)
    #We have new shape of the ROI
            
    print("Shape of MIP ROI is: ",roi_mip_shape)
    print("Shape of transformed ROI (in raw lattice image) is: ",roi_transformed_shape)

    print("Coord of MIP ROI is: ",roi_mip_coord_list)
    print("Coord of transformed ROI (in raw lattice image) is: ",roi_transformed_coord_list)

    #Transformed ROI coordinates -> z,y,x coordinates used to crop from original volume
    #Transformed ROI shape -> used the 3d volume shape for deskewing
    img_stack_dask=vol.get_image_dask_data("TCZYX",C=channel_range,S=0)


    for roi_no in range(1):
        transformed_roi_depth=roi_transformed_shape[roi_no][0]
        transformed_roi_height=roi_transformed_shape[roi_no][1]
        transformed_roi_width=roi_transformed_shape[roi_no][2]
        #actual height of ROI from MIP image
        roi_height=roi_mip_shape[roi_no][1]
        #Height of the ROI from original image
        roi_skew_height=roi_orig_shape[roi_no][1]
        (z_roi_1,z_roi_2),(y_roi_1,y_roi_2),(x_roi_1,x_roi_2)=roi_transformed_coord_list[roi_no]
        #Catch Value Error if last slice
        #Change shape if the z_roi_1 is -ve
        if(z_roi_1<0): 
            #z_roi_2=z_roi_2-z_roi_1
            z_roi_2=z_roi_2+z_roi_1
            transformed_roi_depth=z_roi_2
            z_roi_1=0
        crop_dask_stack=img_stack_dask[t,ch,z_roi_1:z_roi_2,y_roi_1:y_roi_2,x_roi_1:x_roi_2].map_blocks(np.copy).squeeze()
        #create empty dask array with same size as the transformed roi from above
        deskew_roi_img=da.zeros(roi_transformed_shape[roi_no],dtype=img_stack_dask.dtype,chunks=tuple(roi_transformed_shape[roi_no]))
        #hold the values for the image in this array
        deskew_roi_img[:transformed_roi_depth,:roi_skew_height,:transformed_roi_width]=crop_dask_stack
        #deskew the cropped roi
        deskew_roi=deskew_rotate_zeiss(deskew_roi_img,angle,shear_factor,scale_factor,reverse=False,dask=False)
        deskew_roi=img_as_uint(deskew_roi/65535.0)
        #Crop image
        #Get the bounds for cropping
        excess_height=np.abs(transformed_roi_height-roi_height)
        crop_y_top=int(np.floor(0+(excess_height/2)))
        crop_y_bottom=round(crop_y_top+roi_height)
        deskew_roi=deskew_roi[:,crop_y_top:crop_y_bottom,:]
    return

def get_transformed_roi_coord(vol_shape,roi_coord,angle,dx_y,dz,translation,skew_dir="Y",reverse=True):
    """Return the transformed coordinates of ROI based on a volume shape
    reverse=True by default
    """
    #calculate the roi coordinates transformed to the original image/skewed image
    #can pass shape of volume or a list of coordinates as second parameter

    roi_transformed_coordinates=get_new_coordinates(vol_shape,roi_coord,angle,dx_y,dz,translation,skew_dir,reverse=reverse)
    roi_transformed_coordinates=np.array(roi_transformed_coordinates)
    pprint(roi_transformed_coordinates)
    #print(roi_transformed_coordinates)
    #get each vertex of the transformed roi
    #if y is translated in deskew volume, equivalent translation in raw is in z direction
    # y translation is subtracted, but z
    z_roi_1=int(round(roi_transformed_coordinates[:,0][0]))#roi_transformed_coordinates.min(axis=0)[0]))# - translation)
    z_roi_2=int(round(roi_transformed_coordinates[:,0][6]))#roi_transformed_coordinates.max(axis=0)[0]))# - translation)
    
    print("ROI coordinates from raw image are:")
    print("Start and end Z positions are: ",z_roi_1, z_roi_2)
    print("Non rounded Z positions are: ",roi_transformed_coordinates[:,0][0],roi_transformed_coordinates[:,0][6])
    #roi_transformed_coordinates.min(axis=0)[0],roi_transformed_coordinates.max(axis=0)[0])

    y_roi_1=int((roi_transformed_coordinates.min(axis=0)[1]))
    y_roi_2=int(np.ceil(roi_transformed_coordinates.max(axis=0)[1]))

    print("Start and end Y positions are: ",y_roi_1, y_roi_2)
    print("Non rounded Y positions are: ",roi_transformed_coordinates.min(axis=0)[1],roi_transformed_coordinates.max(axis=0)[1])

    x_roi_1=int((roi_transformed_coordinates.min(axis=0)[2]))
    x_roi_2=int(round(roi_transformed_coordinates.max(axis=0)[2]))

    print("Start and end X positions are: ",x_roi_1, x_roi_2)
    print("Non rounded X positions are: ",roi_transformed_coordinates.min(axis=0)[2], roi_transformed_coordinates.max(axis=0)[2])
    #See if a better way to get these corodinates? as a loop?
    #flattened_f=roi_transformed_coordinates.flatten(order='F').reshape(4,8)
    return (z_roi_1,z_roi_2),(y_roi_1,y_roi_2),(x_roi_1,x_roi_2)

def get_ROI_bounds(deskew_roi_shape,coord,angle:float,dx_y:float,dz:float,translation:float=0,skew_dir:str="Y",reverse:bool=False):
    """get the bounds of the roi for cropping
    deskew_roi_shape is the extended volume
    coord are coordinates within the volume which are being transformed
    output will be coord tranformed within deskew_roi_shape
    """
    deskew_coord = get_new_coordinates(deskew_roi_shape,coord,angle,dx_y,dz,translation,skew_dir,False)
    y_top=int(deskew_coord[2][1])
    y_bottom=int(deskew_coord[4][1])
    return y_top,y_bottom


def get_roi_skew_shape(roi_skew_shape,roi_shape,angle,dx_y,dz,skew_dir="Y"):
    #Get shape of the ROI after it has been deskewed
    
    #get vertices for the extracted roi based on its shape
    #skew_roi_coord_list=get_vertices_volume(roi_skew_shape)

    #find max for all dimensions
    shape=np.vstack((roi_shape,roi_skew_shape))
    max_roi_shape=np.max(shape,axis=0)

    #The Z and Y positions get swapped between raw and deskewed image
    #So, we find the shape of the deskewed ROI, get the shape of the crop image from above
    #As we use maximum bounds of the roi and its going to be bigger than original roi, we need to find new shape
    #We use the shape of the skew roi and use bounds of our max_roi as coordinates

    roi_deskew_y=transform_dim(roi_skew_shape,max_roi_shape,angle,dx_y,dz,skew_dir)
    print("New Y dimension after deskewing: ",roi_deskew_y)

    if(roi_deskew_y>max_roi_shape[1]):
        max_roi_shape[1]=roi_deskew_y
    print("New shape to avoid clipping during transformation", max_roi_shape)
    return max_roi_shape.astype(int)

def get_roi_3D_shape_coord(roi_coord,z_min,z_max):
    #convert rois in (x,y..); usually rois only have xy, so adding arguments for z
    y_min,x_min=roi_coord.min(axis=0).astype(int)
    y_max,x_max=roi_coord.max(axis=0).astype(int)
    height=int(np.floor(y_max-y_min))
    width=int(x_max-x_min)
    depth=int(z_max-z_min)
    shape=(depth,height,width)
    coord=(z_min,z_max),(y_min,y_max),(x_min,x_max)
    return shape,coord

#just return shape of a 3D 
def get_roi_3D_shape(roi_coord):
    #get 3d shape if given z,y,x coord of roi takes roi:(z1,z2),(y1,y2),x1,x2
    #returns shape (depth,height,width)
    if type(roi_coord) is not np.array:
        roi_coord=np.array(roi_coord)
    z_min,y_min,x_min=roi_coord.min(axis=1).astype(int)
    z_max,y_max,x_max=roi_coord.max(axis=1).astype(int)
    height=np.abs(y_max-y_min)
    width=np.abs(x_max-x_min)
    depth=np.abs(z_max-z_min)
    
    shape=(depth,height,width)
    return shape

