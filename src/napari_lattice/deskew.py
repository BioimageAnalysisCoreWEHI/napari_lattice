import os
import numpy as np
from pathlib import Path

from magicclass.wrappers import set_design
from magicgui import magicgui, widgets
from magicclass import magicclass, vfield, set_options
from qtpy.QtCore import Qt

from napari.layers import Layer
from napari.types import ImageData
from napari.utils import history

import pyclesperanto_prototype as cle
from .io import LatticeData,  save_tiff


def _deskew_widget():
    @magicclass
    class LLSZWidget:
        @magicclass
        class LlszMenu:
            open_file = False
            lattice = None
            aics = None
            dask = False
            file_name = ""
            save_name = ""

            main_heading = widgets.Label(value="<h3>Napari Lattice: Deskewing</h3>")
            heading1 = widgets.Label(value="Drag and drop an image file onto napari.\nChoose the corresponding Image layer by clicking 'Choose Existing Layer'.\n If choosing a czi file, no need to enter voxel sizes")
            @set_design(background_color="magenta", font_family="Consolas",visible=True) 
            @set_options(pixel_size_dx={"widget_type": "FloatSpinBox", "value":0.1449922,"step": 0.000000001},
                         pixel_size_dy={"widget_type": "FloatSpinBox", "value":0.1449922, "step": 0.000000001},
                         pixel_size_dz={"widget_type": "FloatSpinBox", "value":0.3, "step": 0.000000001}
                         )
            def Choose_Image_Layer(self,
                                      img_layer:Layer,
                                      pixel_size_dx: float = 0.145, 
                                      pixel_size_dy: float = 0.145,
                                      pixel_size_dz: float = 0.3,
                                      channel_dimension_present:bool=False, 
                                      skew_dir: str="Y"):
                
                print("Using existing image layer")
                skew_dir = str.upper(skew_dir)
                assert skew_dir in ('Y', 'X'), "Skew direction not recognised. Enter either Y or X"
                if skew_dir == "X":
                    LLSZWidget.LlszMenu.deskew_func = cle.deskew_x
                elif skew_dir == "Y":
                    LLSZWidget.LlszMenu.deskew_func = cle.deskew_y
                #TODO:change LatticeData class to accept deskew direction, calculate shape and for croppng too
                LLSZWidget.LlszMenu.lattice = LatticeData(img_layer, 30.0, skew_dir,pixel_size_dx, pixel_size_dy,
                                                          pixel_size_dz,channel_dimension_present)
                #LLSZWidget.LlszMenu.aics = LLSZWidget.LlszMenu.lattice.data
                self["Choose_Image_Layer"].background_color = "green"
                LLSZWidget.LlszMenu.dask = False  # Use GPU by default
                save_name = os.path.splitext(os.path.basename(img_layer.source.path))[0]
                if save_name:
                    LLSZWidget.LlszMenu.save_name = os.path.splitext(os.path.basename(img_layer.source.path))[0]
                else:
                    LLSZWidget.LlszMenu.save_name = img_layer.name
                LLSZWidget.LlszMenu.open_file = True
                print("Pixel size (ZYX): ",(LLSZWidget.LlszMenu.lattice.dz,LLSZWidget.LlszMenu.lattice.dy,LLSZWidget.LlszMenu.lattice.dx))
                print("Dimensions of image layer (ZYX): ",list(LLSZWidget.LlszMenu.lattice.data.shape[-3:]))
                print("Dimensions of deskewed image (ZYX): ",LLSZWidget.LlszMenu.lattice.deskew_vol_shape)
                print("Initialised")
                return
                

            # Enter custom angle if needed
            # Will only update after choosing an image
            angle = vfield(float, options={"value": 30.0}, name="Deskew Angle")
            angle_value = 30.0

            @angle.connect
            def _set_angle(self):
                try:
                    LLSZWidget.LlszMenu.lattice.set_angle(self.angle)
                    LLSZWidget.LlszMenu.lattice.angle_value = self.angle
                    print("Angle is set to: ", LLSZWidget.LlszMenu.lattice.angle)
                except AttributeError:
                    print("Choose image layer first before setting angles")
                #print(LLSZWidget.LlszMenu.lattice.angle)
                #print(LLSZWidget.LlszMenu.lattice.angle_value)
                return

            @magicgui(labels=False, auto_call=True)
            def use_GPU(self, use_GPU: bool = True):
                """Choose to use GPU or Dask

                Args:
                    use_GPU (bool, optional): Defaults to True.
                """
                print("Use GPU set to, ", use_GPU)
                LLSZWidget.LlszMenu.dask = not use_GPU
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
                assert time< LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
                assert channel < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"
                
                assert str.upper(LLSZWidget.LlszMenu.lattice.skew) in ('Y', 'X'), \
                    "Skew direction not recognised. Enter either Y or X"

                print("Deskewing for Time:", time,"and Channel: ", channel)

                vol = LLSZWidget.LlszMenu.lattice.data

                vol_zyx= vol[time,channel,:,:,:]

                # Deskew using pyclesperanto
                #on some hardware, getting an error LogicError: clSetKernelArg failed:
                # INVALID_ARG_SIZE - when processing arg#13 (1-based), using dask map_blocks seems to work
                #try:
                    #deskew_final = cle.deskew_y(vol_zyx, 
                                            #angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                            #voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                            #voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                #                           voxel_size_z=LLSZWidget.LlszMenu.lattice.dz).astype(vol.dtype)
                #except Exception as e:
                #print("Got error: ",e,". Using tiling strategy")
                deskew_final = vol_zyx.map_blocks(cle.deskew_y,
                                                angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                                voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                                voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                                voxel_size_z=LLSZWidget.LlszMenu.lattice.dz,
                                                dtype=vol.dtype)
                    
                #deskew_final = cle.pull_zyx(deskewed)
                # TODO: Use dask
                if LLSZWidget.LlszMenu.dask:
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
        
        
        @magicclass(widget_type="collapsible", name="Save Data")
        class SaveData:
            
            @magicgui(header=dict(widget_type="Label", label="<h3>Deskew and Save</h3>"),
                      time_start=dict(label="Time Start:"),
                      time_end=dict(label="Time End:", value=1),
                      ch_start=dict(label="Channel Start:"),
                      ch_end=dict(label="Channel End:", value=1),
                      save_path=dict(mode='d', label="Directory to save"),
                      call_button="Save")
            def Deskew_Save(self, header, time_start: int, time_end: int, ch_start: int, ch_end: int,
                            save_path: Path = Path(history.get_save_history()[0])):

                #assert LLSZWidget.LlszMenu.open_file, "Image not initialised"
                assert 0<= time_start <=LLSZWidget.LlszMenu.lattice.time, "Time start should be 0 or same as total time: "+str(LLSZWidget.LlszMenu.lattice.time)
                assert 0< time_end <=LLSZWidget.LlszMenu.lattice.time, "Time end should be >0 or same as total time: "+str(LLSZWidget.LlszMenu.lattice.time)
                assert 0<= ch_start <= LLSZWidget.LlszMenu.lattice.channels, "Channel start should be 0 or same as no. of channels: "+str(LLSZWidget.LlszMenu.lattice.channels)
                assert 0< ch_end <= LLSZWidget.LlszMenu.lattice.channels, "Channel end should be >0 or same as no. of channels: " +str(LLSZWidget.LlszMenu.lattice.channels)
              
                #time_range = range(time_start, time_end)
                #channel_range = range(ch_start, ch_end)
                angle = LLSZWidget.LlszMenu.lattice.angle
                dx = LLSZWidget.LlszMenu.lattice.dx
                dy = LLSZWidget.LlszMenu.lattice.dy
                dz = LLSZWidget.LlszMenu.lattice.dz

                # Convert path to string
                #save_path = save_path.__str__()

                #get the image data as dask array
                img_data = LLSZWidget.LlszMenu.lattice.data
                
                #pass arguments for save tiff, callable and function arguments
                save_tiff(img_data,
                          func = cle.deskew_y,
                          time_start = time_start,
                          time_end = time_end,
                          channel_start = ch_start,
                          channel_end = ch_end,
                          save_path = save_path,
                          save_name= LLSZWidget.LlszMenu.save_name,
                          dx = dx,
                          dy = dy,
                          dz = dz,
                          angle = angle,
                          angle_in_degrees = angle,
                          voxel_size_x=dx,
                          voxel_size_y=dy,
                          voxel_size_z=dz
                          )
                
                print("Deskewing and Saving Complete -> ", save_path)
                return
    
    #Important to have this or napari won't recognize the classes and magicclass qidgets
    widget = LLSZWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    widget._widget._layout.setAlignment(Qt.AlignTop)

    return widget   