from os import read
import numpy as np
from pprint import pprint #pretty print!
from tqdm.dask import TqdmCallback

#Reading metadata of czi file
import aicspylibczi

#RTX 3080: Use pyopencl-2021.2.8+cl21-cp39-cp39-win_amd64.whl
from gputools.transforms import affine as affineGPU
from gputools.transforms import rotate as rotate_gputools

import napari
from napari.types import ImageData,ShapesData
from magicgui import magicgui
from napari import Viewer
from napari.layers import Layer


#DASK imports
from dask_image import ndinterp
import dask.array as da
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, ProgressBar
from tqdm import tqdm

#llsz imports
from llsz.utils import transform_dim, get_translation_y,get_new_coordinates,\
    get_shear_factor,get_scale_factor, etree_to_dict
from llsz.transformations import deskew_zeiss
from llsz.array_processing import get_deskew_arr
from llsz.io import read_czi
from llsz.crop_utils import crop_deskew_roi

img_location="C:\\WTB6-02-Create Image Subset-02.czi"#C:\\RAPA_treated-01_resaved_c02_t_100.czi" #Z://LLS//LLSZ//Lung-Yu//20210730//WTB6-02_decon_deskew.czi"

#Read lattice file
stack=read_czi(img_location)

#Get all metadata AICSIMAGEIO returns the data consistently as TCZYX regardless of image dimensions
print("Image is read as ",stack.dims.order)

dz,dy,dx=stack.physical_pixel_sizes
channels=stack.dims.C

#if scenes are present
if "S" in stack.dims.order:
    print("Image has scenes. Currently does not support different scenes")
    scenes=stack.dims.S
else:
    scenes=0
    
time=stack.dims.T
nz=stack.dims.Z
ny=stack.dims.Y
nx=stack.dims.X

print("Dimensions of image (X,Y,Z)",nx,ny,nz)
print("Pixel size of image (dX,dY,dZ) in microns",dx,dy,dz)


#Accessing Metadata
#Use Zeiss metadata to extract parameters for deskewing. Use this later when LLSZ data is in the metadata
#to access skew direction, angle etc..
metadatadict_czi = etree_to_dict(aicspylibczi.CziFile(img_location).meta)
metadatadict_czi = metadatadict_czi["ImageDocument"]["Metadata"]
#metadatadict_czi = etree_to_dict(t) #czi_obj.metadata(raw=False)


#Lattice acquisition details
#Get from metadata or user
angle= 30.0
skew_direction="Y"
print("Skew direction is: ",skew_direction)

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
pprint(deskew_coordinates)

#Calculate the no of z slices
z_end=np.abs(round(deskew_coordinates[0][0]))
z_start=np.abs(round(deskew_coordinates[7][0]))
print("Start slice is: ",z_start)
print("End slice is: ",z_end)
no_slices=np.absolute(z_end-z_start)
#print("No of slices: ",no_slices)
print("Dimensions of deskewed stack: ",(no_slices,deskewed_y,nx))


deskew_shape=tuple((nz,deskewed_y,nx))
channel_range=range(channels)
raw_data_dask=stack.get_image_dask_data("TCZYX",C=channel_range,S=0)

@magicgui(
        title=dict(
            widget_type="Label",
            label="<h3>Deskewing</h3>",
        ),
    call_button="Run Deskew",
    #mode={"choices": ["reflect", "constant", "nearest", "mirror", "wrap"]},
    layout='vertical'
)
def run_deskew(title,layer:ImageData,viewer:Viewer) -> ImageData:
    #print(type(layer))
    #print(layer.shape)
    print("Processing ",viewer.dims.current_step[0])
    time=viewer.dims.current_step[0]
    #z is viewer.dims.current_step[2]
    #time is viewer.dims.current_step[0]
    deskew_img=get_deskew_arr(layer,deskew_shape,vol_shape,time=time,channel=0,scene=0,skew_dir=skew_direction)
    deskew_final=deskew_zeiss(deskew_img,angle,deskew_factor,scale_factor,translate_y,reverse=False,dask=False)
    deskew_crop_z=deskew_final[z_start:z_end].astype('uint16') 
    return deskew_crop_z

@magicgui(
        title=dict(
            widget_type="Label",
            label="<h3>Cropping and deskewing</h3>"),
        reference_title=dict(
            widget_type="Label",
            label="<h3>Reference Image to draw ROI</h3>"),
        crop_title=dict(
            widget_type="Label",
            label="<h3>Image to crop from:</h3>"),
        crop_roi_title=dict(
            widget_type="Label",
            label="<h3>Use ROIs from layer:</h3>"),
        timepoint=dict(
            min=0,
            max=time,
            step=1,
            label="<h3>Time:</h3>"),
        chan=dict(
            min=0,
            max=channels,
            step=1,
            label="<h3>Channel:</h3>"),
        layout='vertical')
def crop_image(title,
                crop_roi_title,
                roi_layer:Layer,
                reference_title,
                img_layer:ImageData,
                crop_title,
                crop_img:ImageData,
                timepoint:int,
                chan:int,
                viewer:Viewer) -> ImageData:
    #viewer.add_shapes(shape_type='polygon', edge_width=5,edge_color='white',face_color=[1,1,1,0],text="BBOX")
    #crop_rois=[roi[idx] for idx, roi in enumerate(roi_data)]
    if not roi_layer.data:
        print("No coordinates found. Draw or Import ROI layer.")
    else:
        #print(roi_layer)
        print(roi_layer.data)
        print(chan,timepoint)
        crop_roi_vol=crop_deskew_roi(roi_layer.data[0],vol_shape,raw_data_dask,angle,dy,dz,z_start,z_end,timepoint,chan,skew_direction,reverse=True)
    #CROP ROI and return? how to deal with multiple roi images?
    return crop_roi_vol
    
#SHIFT Z_ROI by TRANSLATE Y WHEN CROPPING FROM CROP_DASK_STACL

# create a viewer and add some images
viewer = napari.Viewer()
viewer.add_image(stack.dask_data, name="Raw Image")
rois= viewer.add_shapes(shape_type='polygon', edge_width=5,edge_color='white',face_color=[1,1,1,0],name="Cropping BBOX layer")
# Add it to the napari viewer
viewer.window.add_dock_widget(run_deskew)
viewer.window.add_dock_widget(crop_image)

# update the layer dropdown menu when the layer list changes
#viewer.layers.events.changed.connect(run_deskew.reset_choices)

napari.run()

#CALL CROP

"""
#Sample first time point to display
#Wrap this in a function
#get a dask array for a single time point and channel with same shape as final 3D volume  
deskew_img=get_deskew_arr(stack,deskew_shape,vol_shape,time=0,channel=0,scene=0,skew_dir=skew_direction)

#GPUTOOLS OR DASK
deskew_final=deskew_zeiss(deskew_img,angle,deskew_factor,scale_factor,translate_y,reverse=False,dask=False)

#gputools converts image to 32 bit using np.astype, so we convert back to 16-bit.. 
#Otherwise always use skimage conversions
deskew_crop_z=deskew_final[z_start:z_end].astype('uint16') 

#max_proj=np.max(deskew_crop_z,axis=0)
#max_x_proj=np.max(deskew_final,axis=2)

#viewer=napari.view_image(deskew_final)
viewer.add_image(max_proj)
viewer.add_image(max_x_proj)

# create a viewer and add some images
viewer = napari.Viewer()

# Add it to the napari viewer
viewer.window.add_dock_widget(run_deskew)

napari.run()
"""