"""
import napari
from napari.types import ImageData, LayerDataTuple,ShapesData
from magicgui import magicgui
from napari import Viewer
from napari.layers import Layer

from llsz.utils import get_scale_factor, get_shear_factor
from .array_processing import get_deskew_arr
from .transformations import deskew_zeiss

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

def start_llsz_ui(dask_vol,deskew_shape,vol_shape, time:int,channel:int,scene:int, angle:float, translation:int, skew_dir:str="Y"):
    # create a viewer and add some images
    viewer = napari.Viewer()
    img_layer=viewer.add_image(dask_vol, name="Raw Image")
    viewer.add_shapes(shape_type='polygon', edge_width=5,edge_color='white',face_color=[1,1,1,0],name="Cropping BBOX layer")
    # Add it to the napari viewer
    viewer.window.add_dock_widget(run_deskew(img_layer,))
    #viewer.window.add_function_widget(run_deskew())
    viewer.window.add_dock_widget(crop_image)
    # update the layer dropdown menu when the layer list changes
    #viewer.layers.events.changed.connect(run_deskew.reset_choices)
    napari.run()


if __name__ == "__main__":
    start_llsz_ui()
"""