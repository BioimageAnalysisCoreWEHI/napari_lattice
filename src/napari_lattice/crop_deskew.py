import os
from pathlib import Path
from magicclass.wrappers import set_design
from magicgui import magicgui, widgets
from magicclass import magicclass, click, field, vfield, set_options
from qtpy.QtCore import Qt

import dask.array as da
import pyclesperanto_prototype as cle
from tqdm import tqdm

from napari.layers import Layer
from napari.types import ImageData, ShapesData

from napari.utils import history

from .io import LatticeData,  save_tiff
from .llsz_core import crop_volume_deskew
from .utils import read_imagej_roi


def _crop_deskew_widget():
    @magicclass
    class CropWidget:
        @magicclass
        class CropMenu:
            open_file = False
            lattice = None
            aics = None
            dask = False
            file_name = ""
            save_name = ""
            
            main_heading = widgets.Label(value="<h3>Napari Lattice: Cropping & Deskewing</h3>")
            heading1 = widgets.Label(value="Drag and drop an image file onto napari.\nChoose the corresponding Image layer by clicking 'Choose Existing Layer'.\n If choosing a czi file, no need to enter voxel sizes")
            
            @set_design(background_color="magenta", font_family="Consolas",visible=True) 
            @set_options(pixel_size_dx={"widget_type": "FloatSpinBox", "value":0.1449922,"step": 0.000000001},
                         pixel_size_dy={"widget_type": "FloatSpinBox", "value":0.1449922, "step": 0.000000001},
                         pixel_size_dz={"widget_type": "FloatSpinBox", "value":0.3, "step": 0.000000001}
                         )
            def Choose_Image_Layer(self, 
                                      img_layer:Layer,
                                      pixel_size_dx: float = 1.00000, 
                                      pixel_size_dy: float = 1.00000,
                                      pixel_size_dz: float = 1.00000,
                                      channel_dimension_present:bool=False, 
                                      skew_dir: str="Y"):
                
                print("Using existing image layer")
                skew_dir = str.upper(skew_dir)
                assert skew_dir in ('Y', 'X'), "Skew direction not recognised. Enter either Y or X"
                if skew_dir == "X":
                    CropWidget.CropMenu.deskew_func = cle.deskew_x
                elif skew_dir == "Y":
                    CropWidget.CropMenu.deskew_func = cle.deskew_y
                #TODO: Implement deskew in X direction; pass this to lattice function below
                CropWidget.CropMenu.lattice = LatticeData(img_layer, 30.0, skew_dir,pixel_size_dx, pixel_size_dy,
                                                          pixel_size_dz,channel_dimension_present)
                CropWidget.CropMenu.dask = False  # Use GPU by default
                CropWidget.CropMenu.save_name =  os.path.splitext(os.path.basename(img_layer.source.path))[0]
                
                #Flag to check if file has been initialised
                CropWidget.CropMenu.open_file = True
                self["Choose_Image_Layer"].background_color = "green"
                print("Pixel size (ZYX): ",(CropWidget.CropMenu.lattice.dz,CropWidget.CropMenu.lattice.dy,CropWidget.CropMenu.lattice.dx))
                print("Dimensions of image layer (ZYX): ",list(CropWidget.CropMenu.lattice.data.shape[-3:]))
                print("Dimensions of deskewed image (ZYX): ",CropWidget.CropMenu.lattice.deskew_vol_shape)
                print("Initialised")
                return
                    

            # Enter custom angle if needed
            # Will only update after choosing an image
            angle = vfield(float, options={"value": 30.0}, name="Deskew Angle")
            angle_value = 30.0

            @angle.connect
            def _set_angle(self):
                try:
                    CropWidget.CropMenu.lattice.set_angle(self.angle)
                    CropWidget.CropMenu.lattice.angle_value = self.angle
                    print("Angle is set to: ", CropWidget.CropMenu.lattice.angle)
                except AttributeError:
                    print("Open a file first before setting angles")
                #print(CropWidget.CropMenu.lattice.angle)
                #print(CropWidget.CropMenu.lattice.angle_value)
                return

            @magicgui(labels=False, auto_call=True)
            def use_GPU(self, use_GPU: bool = True):
                """Choose to use GPU or Dask

                Args:
                    use_GPU (bool, optional): Defaults to True.
                """
                print("Use GPU set to, ", use_GPU)
                CropWidget.CropMenu.dask = not use_GPU
                return

        @magicclass(widget_type="collapsible", name="Preview Deskew")
        class Preview:         
            @magicgui(#header=dict(widget_type="Label",label="<h3>Preview Deskew</h3>"),
                      time=dict(label="Time:"),
                      channel=dict(label="Channel:"),
                      call_button="Preview")
            def Preview_Deskew(self, 
                               #header,
                               time:int,
                               channel:int,
                               img_data: ImageData):
                """
                Preview the deskewing for a single timepoint

                Args:
                    header ([type]): [description]
                    img_data (ImageData): [description]
                """
                print("Previewing deskewed channel and time")
                assert img_data.size, "No image open or selected"
                assert time< CropWidget.CropMenu.lattice.time, "Time is out of range"
                assert channel < CropWidget.CropMenu.lattice.channels, "Channel is out of range"
                
                assert str.upper(CropWidget.CropMenu.lattice.skew) in ('Y', 'X'), \
                    "Skew direction not recognised. Enter either Y or X"

                print("Deskewing for Time:", time,"and Channel: ", channel)

                vol = CropWidget.CropMenu.lattice.data

                vol_zyx= vol[time,channel,...]

                # Deskew using pyclesperanto
                deskew_final = cle.deskew_y(vol_zyx, 
                                            angle_in_degrees=CropWidget.CropMenu.angle_value,
                                            voxel_size_x=CropWidget.CropMenu.lattice.dx,
                                            voxel_size_y=CropWidget.CropMenu.lattice.dy,
                                            voxel_size_z=CropWidget.CropMenu.lattice.dz).astype(vol_zyx.dtype)
                
                #deskew_final = cle.pull_zyx(deskewed)
                # TODO: Use dask
                if CropWidget.CropMenu.dask:
                    print("Using CPU for deskewing")
                    # use cle library for affine transforms, but use dask and scipy
                    # deskew_final = deskew_final.compute()
                
                max_proj_deskew = cle.maximum_z_projection(deskew_final) #np.max(deskew_final, axis=0)

                # add channel and time information to the name
                suffix_name = "_c" + str(channel) + "_t" + str(time)

                self.parent_viewer.add_image(max_proj_deskew, name="Deskew_MIP")

                # img_name="Deskewed image_c"+str(chan_deskew)+"_t"+str(time_deskew)
                self.parent_viewer.add_image(deskew_final, name="Deskewed image" + suffix_name)
                self.parent_viewer.layers[0].visible = False
                #print("Shape is ",deskew_final.shape)
                print("Preview: Deskewing complete")
                return
        
        #add function for previewing cropped image
        @magicclass(widget_type="collapsible", name="Preview Crop",popup_mode="below")
        class Preview_Crop_Menu:
              
            @click(enables =["Import_ImageJ_ROI","Crop_Preview"])
            def Initialize_Shapes_Layer(self):
                CropWidget.Preview_Crop_Menu.shapes_layer = self.parent_viewer.add_shapes(shape_type='polygon', edge_width=5, edge_color='white',
                                            face_color=[1, 1, 1, 0], name="Cropping BBOX layer")
                #TO select ROIs if needed
                CropWidget.Preview_Crop_Menu.shapes_layer.mode="SELECT"
                return
            
            @click(enabled =False)
            def Import_ImageJ_ROI(self, path: Path = Path(history.get_open_history()[0])):
                print("Opening", path)
                roi_list = read_imagej_roi(path)
                #print(CropWidget.Preview_Crop_Menu.shapes_layer)
                CropWidget.Preview_Crop_Menu.shapes_layer.add(roi_list,shape_type='polygon', edge_width=5, edge_color='yellow',
                                                                          face_color=[1, 1, 1, 0])
                return
            
            time_crop = field(int, options={"min": 0, "step": 1}, name="Time")
            chan_crop = field(int, options={"min": 0, "step": 1}, name="Channels")
            heading_roi = widgets.Label(value="Import or draw ROI, and then select the ROI using the cursor.")
            #roi_idx = field(int, options={"min": 0, "step": 1}, name="ROI number")

            #@magicgui
            
            @click(enabled =False)
            def Crop_Preview(self, roi_layer: ShapesData):  # -> LayerDataTuple:
                assert roi_layer, "No coordinates found for cropping. Check if right shapes layer or initialise shapes layer and draw ROIs."
                #assert self.roi_idx.value <len(CropWidget.Preview_Crop_Menu.shapes_layer.data), "ROI not present"
                assert len(CropWidget.Preview_Crop_Menu.shapes_layer.selected_data)>0, "ROI not selected"
                # TODO: Add assertion to check if bbox layer or coordinates
                time = self.time_crop.value
                channel = self.chan_crop.value

                assert time < CropWidget.CropMenu.lattice.time, "Time is out of range"
                assert channel < CropWidget.CropMenu.lattice.channels, "Channel is out of range"
                
                print("Using channel ", channel," and time", time)
                
                vol = CropWidget.CropMenu.lattice.data

                vol_zyx= vol[time,channel,...]

                deskewed_shape = CropWidget.CropMenu.lattice.deskew_vol_shape
                
                deskewed_volume = da.zeros(deskewed_shape)

                #Option for entering custom z values?
                z_start = 0
                z_end = deskewed_shape[0]
                roi_idx = list(CropWidget.Preview_Crop_Menu.shapes_layer.selected_data)[0]
                roi_choice = roi_layer[roi_idx]
                print("Previewing ROI ", roi_idx)
                
                crop_roi_vol_desk = crop_volume_deskew(original_volume = vol_zyx, 
                                                deskewed_volume=deskewed_volume, 
                                                roi_shape = roi_choice, 
                                                angle_in_degrees = CropWidget.CropMenu.lattice.angle, 
                                                voxel_size_x =CropWidget.CropMenu.lattice.dx, 
                                                voxel_size_y =CropWidget.CropMenu.lattice.dy, 
                                                voxel_size_z =CropWidget.CropMenu.lattice.dz, 
                                                z_start = z_start, 
                                                z_end = z_end).astype(vol_zyx.dtype)

                #crop_roi_vol = cle.pull_zyx(crop_roi_vol_desk)
                
                self.parent_viewer.add_image(crop_roi_vol_desk)
                return

        @magicclass(widget_type="collapsible", name="Crop and Save Data")
        class CropSaveData:
            @magicgui(time_start=dict(label="Time Start:"),
                      time_end=dict(label="Time End:", value=1),
                      ch_start=dict(label="Channel Start:"),
                      ch_end=dict(label="Channel End:", value=1),
                      save_path=dict(mode='d', label="Directory to save "))
            def Crop_Save(self, 
                          time_start: int, 
                          time_end: int, 
                          ch_start: int, 
                          ch_end: int,
                          roi_layer_list: ShapesData, 
                          save_path: Path = Path(history.get_save_history()[0])):

                if not roi_layer_list:
                    print("No coordinates found or cropping. Initialise shapes layer and draw ROIs.")
                else:
                    assert CropWidget.CropMenu.open_file, "Image not initialised"
                    assert 0<= time_start <=CropWidget.CropMenu.lattice.time, "Time start should be 0 or >0 or same as total time "+str(CropWidget.CropMenu.lattice.time)
                    assert 0< time_end <=CropWidget.CropMenu.lattice.time, "Time end should be >0 or same as total time "+str(CropWidget.CropMenu.lattice.time)
                    assert 0<= ch_start <= CropWidget.CropMenu.lattice.channels, "Channel start should be 0 or >0 or same as no. of channels "+str(CropWidget.CropMenu.lattice.channels)
                    assert 0< ch_end <= CropWidget.CropMenu.lattice.channels, "Channel end should be >0 or same as no. of channels " +str(CropWidget.CropMenu.lattice.channels)
              
                    angle = CropWidget.CropMenu.lattice.angle
                    dx = CropWidget.CropMenu.lattice.dx
                    dy = CropWidget.CropMenu.lattice.dy
                    dz = CropWidget.CropMenu.lattice.dz
                    
                    #get image data
                    img_data = CropWidget.CropMenu.lattice.data
                    #Get shape of deskewed image
                    deskewed_shape = CropWidget.CropMenu.lattice.deskew_vol_shape
                    deskewed_volume = da.zeros(deskewed_shape)
                    z_start = 0
                    z_end = deskewed_shape[0]
                    
                    print("Cropping and saving files...")           

                    
                    for idx, roi_layer in enumerate(tqdm(roi_layer_list, desc="ROI:", position=0)):
                        #pass arguments for save tiff, callable and function arguments
                        print("Processing ROI ",idx)
                        #pass parameters for the crop_volume_deskew function
                        save_tiff(img_data,
                            func = crop_volume_deskew,
                            time_start = time_start,
                            time_end = time_end,
                            channel_start = ch_start,
                            channel_end = ch_end,
                            save_name_prefix  = "ROI_" + str(idx)+"_",
                            save_path = save_path,
                            save_name= CropWidget.CropMenu.save_name,
                            dx = dx,
                            dy = dy,
                            dz = dz,
                            angle = angle,
                            deskewed_volume=deskewed_volume,
                            roi_shape = roi_layer,
                            angle_in_degrees = angle,
                            z_start = z_start,
                            z_end = z_end,
                            voxel_size_x=dx,
                            voxel_size_y=dy,
                            voxel_size_z=dz,
                            )

                    print("Cropping and Saving Complete -> ", save_path)
                    return        
                
    #Important to have this or napari won't recognize the classes and magicclass qidgets
    crop_widget = CropWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    crop_widget._widget._layout.setAlignment(Qt.AlignTop)

    return crop_widget   