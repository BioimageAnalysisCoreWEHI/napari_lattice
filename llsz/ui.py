#UI elements for opening file
#Deskewing
#Cropping

import pathlib
import os

import numpy as np
from napari.types import ImageData, LayerDataTuple,ShapesData
from magicgui import magicgui
from napari import Viewer
from napari.layers import Layer

from .utils import get_scale_factor, get_shear_factor
from .array_processing import get_deskew_arr
from .transformations import deskew_zeiss
from .io import read_czi
from .llsz_core import process_czi
from llsz.crop_utils import crop_deskew_roi


@magicgui(File_Path={'mode': 'r'},
           call_button='Open')
def Open_czi_file(viewer:Viewer, File_Path:pathlib.Path.home):
    """Widget to open czi file

    Args:
        viewer (Viewer): napari viewer
        File_Path ([czi], optional): Path to raw/unprocessed Zeiss lattice data (czi). Defaults to pathlib.Path.home().
    """    
    print("Opening", File_Path)
    file_path=File_Path
    
    #check if a file
    assert os.path.isfile(file_path), "Not a file"
    name,ext=os.path.splitext(file_path)
    
    #check if czi file
    assert ext == ".czi", "Not a czi file"
    #global stack
    stack=read_czi(file_path)
    time=stack.dims.T
    channels=stack.dims.C
    viewer.add_image(stack.dask_data,name="Original Skewed Data")
    viewer.dims.axis_labels = ["Time", "Channels", "Z","Y","X"]
    viewer.window.add_function_widget(run_deskew,magic_kwargs=dict(header=dict(widget_type="Label",
                                                               label="<h3>Deskewing</h3>"),
                                                               stack=dict(value=stack),
                                                               #viewer=dict(value=Viewer),
                                                               angle=dict(value=30.0),
                                                               time_deskew=dict(min=0,max=time,step=1,label="<h3>Time:</h3>"),
                                                               chan_deskew=dict(min=0,max=channels,step=1,label="<h3>Channel:</h3>"),
                                                               skew_direction=dict(value="Y"),
                                                               call_button='Run Deskew')
                                                                )
    return 

def run_deskew(header,
                img_layer:ImageData,
                stack,
                angle:float,
                time_deskew:int,
                chan_deskew:int,
                skew_direction:str,
                viewer:Viewer)  -> LayerDataTuple:
    """Creates a widget for deskewing
    It is created after file is opened and called from the Open_czi_file widget
     Has option to select time and channel for deskewing

    Args:
        header (dict): Title of widget
        img_layer (ImageData): Image layer in napari to be deskewed (z,y,x) 
        stack (AICSImage object): original data as an AICSImage object
        angle (float): Lightsheet Angle
        time_deskew (int): Timepoint for deskewing
        chan_deskew (int): Channel for deskewing
        skew_direction (str): Skew direction, which is a property of the microscope
        viewer (Viewer): Napari Viewer instance

    Returns:
        LayerDataTuple: Returns the deskewed image (cropped in Z-direction)
    """                

    assert str.upper(skew_direction) in ('Y','X'), "Skew direction not recognised. Enter either Y or X"
    #curr_time=viewer.dims.current_step[0]

    #process the file to get parameters for deskewing
    deskew_shape,vol_shape,translate_y,z_start,z_end=process_czi(stack,angle,skew_direction)
    
    #print("Processing ",viewer.dims.current_step[0])
    #z is viewer.dims.current_step[2]
    #time is viewer.dims.current_step[0]
    print("Deskewing for time:",time_deskew,"and channel", chan_deskew)
    dz,dy,dx=stack.physical_pixel_sizes
    time=stack.dims.T
    channels=stack.dims.C
    shear_factor=get_shear_factor(angle)
    scaling_factor=get_scale_factor(angle,dy,dz)

    #Get a dask array with same shape as final deskewed image and containing the raw data (Essentially a scaled up version of the raw data)   
    deskew_img=get_deskew_arr(img_layer,deskew_shape,vol_shape,time=time_deskew,channel=chan_deskew,scene=0,skew_dir=skew_direction)
    #Perform deskewing on the skewed dask array 
    deskew_full=deskew_zeiss(deskew_img,angle,shear_factor,scaling_factor,translate_y,reverse=False,dask=False)
    #Crop the z slices to get only the deskewed array and not the empty area
    deskew_final=deskew_full[z_start:z_end].astype('uint16') 

    channel_range=range(channels)
    raw_data_dask=stack.get_image_dask_data("TCZYX",C=channel_range,S=0)
    #Add layer for cropping
    viewer.add_shapes(shape_type='polygon', edge_width=5,edge_color='white',face_color=[1,1,1,0],name="Cropping BBOX layer")
    max_proj_deskew=np.max(deskew_final,axis=0)
    viewer.add_image(max_proj_deskew,name="Max projection:deskew")
    
    #Add a function widget and pass parameters to teh crop_image widget
    viewer.window.add_function_widget(crop_image,magic_kwargs=dict(header=dict(widget_type="Label",
                                                                                label="<h3>Cropping and deskewing</h3>"),
                                                                    crop_roi_title=dict(widget_type="Label",
                                                                    label="<h3>Use ROIs drawn on this shapes layer:</h3>"),
                                                                    crop_title=dict(widget_type="Label",
                                                                    label="<h3>Image to crop from (NOT implemented; uses original stack):</h3>"),
                                                                    vol_shape=dict(value=vol_shape),
                                                                    raw_data_dask=dict(value=raw_data_dask),
                                                                    angle=dict(value=angle),
                                                                    dy=dict(value=dy),
                                                                    dz=dict(value=dz),
                                                                    z_start=dict(value=z_start),
                                                                    z_end=dict(value=z_end),
                                                                    timepoint=dict(min=0,max=time,step=1,label="<h3>Time:</h3>"),
                                                                    chan=dict(min=0,max=channels,step=1,label="<h3>Channel:</h3>"),
                                                                    skew_direction=dict(value=skew_direction),                                                                            
                                                                    layout='vertical')
                                                                    )                                     
    img_name="Deskewed image_c"+str(chan_deskew)+"_t"+str(time_deskew)
    #return (deskew_full, {"name":"Uncropped data"})
    return (deskew_final, {"name":img_name})


def crop_image(header,
                crop_roi_title,
                roi_layer:ShapesData,
                crop_title,
                crop_img:ImageData,
                vol_shape,
                raw_data_dask,
                angle,
                dy,
                dz,
                z_start,
                z_end,
                timepoint:int,
                chan:int,
                skew_direction,
                viewer:Viewer) -> LayerDataTuple:

                if not roi_layer:
                    print("No coordinates found. Draw or Import ROI layer.")
                else:
                    #TODO: Add assertion to check if bbox layer or coordinates
                    print("Using channel and time", chan,timepoint)
                    #if passsing roi layer as layer, use roi.data
                    crop_roi_vol=crop_deskew_roi(roi_layer[0],vol_shape,raw_data_dask,angle,dy,dz,z_start,z_end,timepoint,chan,skew_direction,reverse=True)
                    translate_x=int(roi_layer[0][0][0])
                    translate_y=int(roi_layer[0][0][1])
                    crop_img_layer= (crop_roi_vol , {'translate' : [0,translate_x,translate_y] })
                #CROP ROI and return? how to deal with multiple roi images?
                return crop_img_layer
"""


 
        dz,dy,dx=stack.physical_pixel_sizes
        deskew_shape,vol_shape,translation,z_start,z_end=process_czi(stack,angle,skew_direction)
        shear_factor=get_shear_factor(angle)
        scaling_factor=get_scale_factor(angle,dy,dz)

        # create a viewer and add some images
        #viewer = napari.Viewer()
        #viewer.window.add_dock_widget(Open_czi_file)
        #viewer.run()
        @magicgui(
                title=dict(
                    widget_type="Label",
                    label="<h3>Deskewing</h3>",
                ),
            call_button="Run Deskew",
            layout='vertical'
        )
        def run_deskew(title,
                        layer:ImageData,
                        deskew_shape,
                        vol_shape,
                        angle,
                        dx_y,
                        dz,
                        translation,
                        skew_dir="Y",
                        viewer:Viewer) -> ImageData:
            #print(type(layer))
            #print(layer.shape)
            print("Processing ",viewer.dims.current_step[0])
            time=viewer.dims.current_step[0]
            #z is viewer.dims.current_step[2]
            #time is viewer.dims.current_step[0]
            shear_factor=get_shear_factor(angle)
            scale_factor=get_scale_factor(angle,dx_y,dz)
            deskew_img=get_deskew_arr(layer,deskew_shape,vol_shape,time=time,channel=0,scene=0,skew_dir=skew_dir)
            deskew_final=deskew_zeiss(deskew_img,angle,shear_factor,scale_factor,translation,reverse=False,dask=False)
            deskew_crop_z=deskew_final[z_start:z_end].astype('uint16') 
            return deskew_crop_z

@magicgui(
        title=dict(
            widget_type="Label",
            label="<h3>Cropping and deskewing</h3>"),
        crop_roi_title=dict(
            widget_type="Label",
            label="<h3>Use ROIs drawn on this shapes layer:</h3>"),
        crop_title=dict(
            widget_type="Label",
            label="<h3>Image to crop from (NOT implemented; uses original stack):</h3>"),
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
                roi_layer:ShapesData,
                crop_title,
                crop_img:ImageData,
                timepoint:int,
                chan:int,
                viewer:Viewer) -> LayerDataTuple:
    #viewer.add_shapes(shape_type='polygon', edge_width=5,edge_color='white',face_color=[1,1,1,0],text="BBOX")
    #crop_rois=[roi[idx] for idx, roi in enumerate(roi_data)]
    if not roi_layer:
        print("No coordinates found. Draw or Import ROI layer.")
    else:
        #print(roi_layer)
        #print(roi_layer)
        #roi_layer
        print("Using channel and time", chan,timepoint)
        #print(crop_img.shape)
        #if passsing roi layer as layer, use roi.data
        crop_roi_vol=crop_deskew_roi(roi_layer[0],vol_shape,raw_data_dask,angle,dy,dz,z_start,z_end,timepoint,chan,skew_direction,reverse=True)
        #print((0, roi_layer[0][2][0], roi_layer[0][0][0]))
        #print(roi_layer[0])
        translate_x=int(roi_layer[0][0][0])
        translate_y=int(roi_layer[0][0][1])
        crop_img_layer= (crop_roi_vol , {'translate' : [0,translate_x,translate_y] })
    #CROP ROI and return? how to deal with multiple roi images?
    return crop_img_layer
    
#SHIFT Z_ROI by TRANSLATE Y WHEN CROPPING FROM CROP_DASK_STACL







    # create a viewer and add some images
    viewer = napari.Viewer()
    img_layer=viewer.add_image(dask_vol, name="Raw Image")
    viewer.add_shapes(shape_type='polygon', edge_width=5,edge_color='white',face_color=[1,1,1,0],name="Cropping BBOX layer")
    # Add it to the napari viewer
    deskew_widget=run_deskew(img_layer,)
    viewer.window.add_dock_widget(run_deskew(img_layer,))
    #viewer.window.add_function_widget(run_deskew())
    viewer.window.add_dock_widget(crop_image)
    # update the layer dropdown menu when the layer list changes
    #viewer.layers.events.changed.connect(run_deskew.reset_choices)
    napari.run()

def run_deskew(title,
                layer:ImageData,
                deskew_shape,
                vol_shape,
                angle,
                dx_y,
                dz,
                translation,
                skew_dir="Y",
                viewer:Viewer) -> ImageData:
    #print(type(layer))
    #print(layer.shape)
    print("Processing ",viewer.dims.current_step[0])
    time=viewer.dims.current_step[0]
    #z is viewer.dims.current_step[2]
    #time is viewer.dims.current_step[0]
    shear_factor=get_shear_factor(angle)
    scale_factor=get_scale_factor(angle,dx_y,dz)
    deskew_img=get_deskew_arr(layer,deskew_shape,vol_shape,time=time,channel=0,scene=0,skew_dir=skew_dir)
    deskew_final=deskew_zeiss(deskew_img,angle,shear_factor,scale_factor,translation,reverse=False,dask=False)
    deskew_crop_z=deskew_final[z_start:z_end].astype('uint16') 
    return deskew_crop_z


if __name__ == "__main__":
    start_llsz_ui()
"""