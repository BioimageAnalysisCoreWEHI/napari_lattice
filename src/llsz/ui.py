# UI for reading files, deskewing and cropping

import os
from pathlib import Path
from magicclass.wrappers import set_design
from magicgui import magicgui
from magicclass import magicclass, click, field, vfield
from qtpy.QtCore import Qt

import numpy as np
import dask.array as da
import pyclesperanto_prototype as cle
from tqdm import tqdm

from napari.types import ImageData, ShapesData
from napari.layers.image.image import Image
from napari_plugin_engine import napari_hook_implementation
from napari.utils import history

from llsz.io import LatticeData, LatticeData_czi, save_tiff
from llsz.llsz_core import crop_volume_deskew
from napari_workflows import Workflow, WorkflowManager
from napari_workflows._io_yaml_v1 import load_workflow, save_workflow

from llsz.utils import get_first_last_image_and_task


def plugin_wrapper():
    @magicclass(widget_type="scrollable", name="LLSZ analysis")
    class LLSZWidget:
        @magicclass(widget_type="frame")  # , close_on_run=False
        class LlszMenu:

            open_file = False
            lattice = None
            aics = None
            dask = False
            file_name = ""
            save_name = ""
            #default scale
            glob_scale = (1,1,1)

            @set_design(background_color="orange", font_family="Consolas", visible=True)
            @click(hides=["Choose_Existing_Layer","OR","Select_image_layer_below_to_process"])
            def Open_a_czi_File(self, path: Path = Path(history.get_open_history()[0])):
                print("Opening", path)
                # update the napari settings to use the opened file path as last opened path
                history.update_open_history(path.__str__())
                LLSZWidget.LlszMenu.lattice = LatticeData_czi(path, 30.0, "Y")
                LLSZWidget.LlszMenu.aics = LLSZWidget.LlszMenu.lattice.data
                LLSZWidget.LlszMenu.file_name = os.path.splitext(os.path.basename(path))[0]
                LLSZWidget.LlszMenu.save_name = os.path.splitext(os.path.basename(path))[0]
                self.parent_viewer.add_image(LLSZWidget.LlszMenu.aics.dask_data)
                self["Open_a_czi_File"].background_color = "green"
                LLSZWidget.LlszMenu.dask = False  # Use GPU by default
                LLSZWidget.LlszMenu.open_file = True  # if open button used
                #set scale
                self.glob_scale = LLSZWidget.LlszMenu.lattice.data.physical_pixel_sizes

            # Display text
            @click(enabled=False)
            def OR(self):
                pass

            # Display text
            @click(enabled=False)
            def Select_image_layer_below_to_process(self):
                pass

            @set_design(background_color="magenta", font_family="Consolas", visible=True)
            @click(hides=["Open_a_czi_File","OR"])
            def Choose_Existing_Layer(self, img_data:ImageData, skew_dir: str = "Y"):
                print("Using existing image layer")
                skew_dir = str.upper(skew_dir)
                assert skew_dir in ('Y', 'X'), "Skew direction not recognised. Enter either Y or X"
                
                #get scale from active iamge layers
                #if image layer not selected, then select the first image layer in the layers list
                try:
                    scale = self.parent_viewer.layers.selection.active.scale
                    LLSZWidget.LlszMenu.scale = scale
                except AttributeError: #scale is None
                    for layer in self.parent_viewer.layers:
                        if type(layer) is Image:
                            self.parent_viewer.layers.selection.active = self.parent_viewer.layers[0]
                            scale = self.parent_viewer.layers.selection.active.scale          
    
                if list(scale) in [[None], [1,1,1]]:
                    print("Pixel sizes are (zyx) are: ",scale)
                    print("Enter it manually if this is incorrect")
                    dz,dy,dx = (1,1,1)
                else:
                    dz,dy,dx = scale
                    print("Pixel sizes in microns (zyx): ", dz,dy,dx)
                
                LLSZWidget.LlszMenu.lattice = LatticeData(img_data, 30.0, skew_dir, dx, dy,dz)

                LLSZWidget.LlszMenu.aics = LLSZWidget.LlszMenu.lattice.data
                
                self.glob_scale = LLSZWidget.LlszMenu.lattice.data.PhysicalPixelSizes
                
                self["Choose_Existing_Layer"].background_color = "green"
                self["Enter_pixel_size"].background_color = "orange"
                
                LLSZWidget.LlszMenu.dask = False  # Use GPU by default
                LLSZWidget.LlszMenu.open_file = True  # if open button used
                return
            
            #Enter pixel size manually
            @set_design(text = "Manually Enter pixel size")
            def Enter_pixel_size(self,
                                pixel_size_dx: float = glob_scale[2],
                                pixel_size_dy: float = glob_scale[1],
                                pixel_size_dz: float = glob_scale[0] ):
                try:
                    LLSZWidget.LlszMenu.lattice.dx = pixel_size_dx
                    LLSZWidget.LlszMenu.lattice.dy = pixel_size_dy
                    LLSZWidget.LlszMenu.lattice.dz = pixel_size_dz
                    print("Pixel sizes (zyx) set to",LLSZWidget.LlszMenu.lattice.dx,
                                                     LLSZWidget.LlszMenu.lattice.dy,
                                                     LLSZWidget.LlszMenu.lattice.dz," microns")
                    self["Enter_pixel_size"].background_color = "green"
                except AttributeError:
                    print("Choose layer first before entering pixel sizes")
                
                return  

            # Enter custom angle if needed
            # Will only update after choosing an image
            angle = vfield(float, options={"value": 30.0}, name="Deskew Angle")
            angle_value = 30.0
            
            #Connect with vfield above
            @angle.connect
            def _set_angle(self):
                try:
                    LLSZWidget.LlszMenu.lattice.set_angle(self.angle)
                    LLSZWidget.LlszMenu.lattice.angle_value = self.angle
                    print("Angle is set to: ", LLSZWidget.LlszMenu.lattice.angle)
                except AttributeError:
                    print("Open a file first before setting angles")
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

            time_deskew = field(int, options={"min": 0, "step": 1}, name="Time")
            chan_deskew = field(int, options={"min": 0, "step": 1}, name="Channels")

            @magicgui(header=dict(widget_type="Label", label="<h3>Preview Deskew</h3>"), call_button="Preview")
            def Preview_Deskew(self, header, img_data: ImageData):
                """
                Preview the deskewing for a single timepoint

                Args:
                    header ([type]): [description]
                    img_data (ImageData): [description]
                """
                print("Previewing deskewed channel and time")
                assert img_data.size, "No image open or selected"
                assert self.time_deskew.value < LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
                assert self.chan_deskew.value < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"
                
                time = self.time_deskew.value
                channel = self.chan_deskew.value

                assert str.upper(LLSZWidget.LlszMenu.lattice.skew) in ('Y', 'X'), \
                    "Skew direction not recognised. Enter either Y or X"

                print("Deskewing for Time:", time,
                      "and Channel: ", channel)

                vol = LLSZWidget.LlszMenu.aics.dask_data

                vol_zyx= vol[time,channel,...]

                # Deskew using pyclesperanto
                deskew_final = cle.deskew_y(vol_zyx, 
                                            angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                            voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                            voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                            voxel_size_z=LLSZWidget.LlszMenu.lattice.dz).astype(np.uint16)
                deskew_final = cle.pull_zyx(deskewed)
                # TODO: Use dask
                if LLSZWidget.LlszMenu.dask:
                    print("Using CPU for deskewing")
                    # use cle library for affine transforms, but use dask and scipy
                    # deskew_final = deskew_final.compute()
                
                max_proj_deskew = np.max(deskew_final, axis=0)

                # add channel and time information to the name
                suffix_name = "_c" + str(channel) + "_t" + str(time)

                self.parent_viewer.add_image(max_proj_deskew, name="Deskew_MIP")

                # img_name="Deskewed image_c"+str(chan_deskew)+"_t"+str(time_deskew)
                self.parent_viewer.add_image(deskew_final, name="Deskewed image" + suffix_name)
                self.parent_viewer.layers[0].visible = False
                #print("Shape is ",deskew_final.shape)
                print("Deskewing complete")
                return

        @magicclass(widget_type="collapsible", name="Preview Crop")
        class Preview_Crop_Menu:

            # @click(enables ="Crop_Preview")
            @magicgui(call_button="Initialise shapes layer")
            def Initialize_Shapes_Layer(self):
                self.parent_viewer.add_shapes(shape_type='polygon', edge_width=5, edge_color='white',
                                              face_color=[1, 1, 1, 0], name="Cropping BBOX layer")
                return

            time_crop = field(int, options={"min": 0, "step": 1}, name="Time")
            chan_crop = field(int, options={"min": 0, "step": 1}, name="Channels")

            @magicgui
            def Crop_Preview(self, roi_layer: ShapesData):  # -> LayerDataTuple:
                assert roi_layer, "No coordinates found for cropping. Check if right shapes layer or initialise shapes layer and draw ROIs."
                # TODO: Add assertion to check if bbox layer or coordinates
                
                time = self.time_crop.value
                channel = self.chan_crop.value
                
                assert time < LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
                assert channel < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"
                
                print("Using channel ", channel," and time", time)
                               
                vol = LLSZWidget.LlszMenu.aics.dask_data

                vol_zyx= vol[time,channel,...]

                deskewed_shape = LLSZWidget.LlszMenu.lattice.deskew_vol_shape
                
                deskewed_volume = da.zeros(deskewed_shape)

                #Option for entering custom z values?
                z_start = 0
                z_end = deskewed_shape[0]

                crop_roi_vol_desk = crop_volume_deskew(original_volume = vol_zyx, 
                                                deskewed_volume=deskewed_volume, 
                                                roi_shape = roi_layer, 
                                                angle_in_degrees = LLSZWidget.LlszMenu.lattice.angle, 
                                                voxel_size_x =LLSZWidget.LlszMenu.lattice.dx, 
                                                voxel_size_y =LLSZWidget.LlszMenu.lattice.dy, 
                                                voxel_size_z =LLSZWidget.LlszMenu.lattice.dz, 
                                                z_start = z_start, 
                                                z_end = z_end).astype(np.uint16)

                crop_roi_vol = cle.pull_zyx(crop_roi_vol_desk)
                
                self.parent_viewer.add_image(crop_roi_vol)
                return

        @magicclass(widget_type="collapsible", name="Deskew & Save Data")
        class SaveData:

            @magicgui(time_start=dict(label="Time Start:"),
                      time_end=dict(label="Time End:", value=1),
                      ch_start=dict(label="Channel Start:"),
                      ch_end=dict(label="Channel End:", value=1),
                      save_path=dict(mode='d', label="Directory to save"))

            def Deskew_Save(self, time_start: int, time_end: int, ch_start: int, ch_end: int,
                            save_path: Path = Path(history.get_save_history()[0])):

                assert LLSZWidget.LlszMenu.open_file, "Image not initialised"
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

                #get the image data
                img_data = LLSZWidget.LlszMenu.aics.dask_data
                
                deskew_workflow = Workflow()

                #define deskewing workflow
                input_arg = "input"
                task_name = "deskewing"
                deskew_workflow.set(task_name, 
                                    cle.deskew_y, 
                                    input_arg, 
                                    angle_in_degrees = angle,
                                    voxel_size_x = dx, 
                                    voxel_size_y= dy,
                                    voxel_size_z = dz)
                print(deskew_workflow)



                #pass arguments for save tiff, callable and function arguments
                save_tiff(img_data,
                          workflow = deskew_workflow,
                          input_arg = input_arg,
                          task_name = task_name,
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
                    assert LLSZWidget.LlszMenu.open_file, "Image not initialised"
                    assert 0<= time_start <=LLSZWidget.LlszMenu.lattice.time, "Time start should be 0 or >0 or same as total time "+str(LLSZWidget.LlszMenu.lattice.time)
                    assert 0< time_end <=LLSZWidget.LlszMenu.lattice.time, "Time end should be >0 or same as total time "+str(LLSZWidget.LlszMenu.lattice.time)
                    assert 0<= ch_start <= LLSZWidget.LlszMenu.lattice.channels, "Channel start should be 0 or >0 or same as no. of channels "+str(LLSZWidget.LlszMenu.lattice.channels)
                    assert 0< ch_end <= LLSZWidget.LlszMenu.lattice.channels, "Channel end should be >0 or same as no. of channels " +str(LLSZWidget.LlszMenu.lattice.channels)
              
                    angle = LLSZWidget.LlszMenu.lattice.angle
                    dx = LLSZWidget.LlszMenu.lattice.dx
                    dy = LLSZWidget.LlszMenu.lattice.dy
                    dz = LLSZWidget.LlszMenu.lattice.dz

                    #get image data
                    img_data = LLSZWidget.LlszMenu.aics.dask_data
                    
                    #Get shape of deskewed image
                    deskewed_shape = LLSZWidget.LlszMenu.lattice.deskew_vol_shape
                    #Define a dask array with shape of deskewed image
                    deskewed_volume = da.zeros(deskewed_shape)

                    z_start = 0
                    z_end = deskewed_shape[0]
                    # save channel/s for each timepoint.
                    # TODO: Check speed -> Channel and then timepoint or vice versa, which is faster?

                    print("Cropping and saving files...")           
                    crop_deskew_workflow = Workflow()

                    #define crop and deskewing workflow
                    #Have to define a workflow every loop though!
                    input_arg = "input"
                    task_name = "crop_deskewing"
                    roi = "roi"
                    crop_deskew_workflow.set(task_name, 
                                            crop_volume_deskew,
                                            original_volume = input_arg, 
                                            deskewed_volume = deskewed_volume,
                                            roi_shape = roi,
                                            angle_in_degrees = angle,
                                            voxel_size_x = dx, 
                                            voxel_size_y= dy,
                                            voxel_size_z = dz,
                                            z_start = z_start,
                                            z_end = z_end)

                    for idx, roi_layer in enumerate(tqdm(roi_layer_list, desc="ROI:", position=0)):
                        print("Processing ROI ",idx)
                        #As the roi changes every loop, we set the roi here
                        crop_deskew_workflow.set(roi, roi_layer)
                        
                        #pass arguments for save tiff
                        save_tiff(img_data,
                                    workflow = crop_deskew_workflow,
                                    input_arg = input_arg,
                                    task_name = task_name,
                                    time_start = time_start,
                                    time_end = time_end,
                                    channel_start = ch_start,
                                    channel_end = ch_end,
                                    save_path = save_path,
                                    save_name= LLSZWidget.LlszMenu.save_name,
                                    voxel_size_x=dx,
                                    voxel_size_y=dy,
                                    voxel_size_z=dz,
                                    angle_in_degrees = angle,
                                    )

                    print("Cropping and Saving Complete -> ", save_path)
                    return
                
        @magicclass(widget_type="collapsible", name="Apply workflow to process and save")
        class ApplyWorkflowSave:
            
            # Display text
            @click(enabled=False)
            def Apply_and_Preview_Workflow(self):
                pass
            
            time_preview= field(int, options={"min": 0, "step": 1}, name="Time")
            chan_preview = field(int, options={"min": 0, "step": 1}, name="Channels")
            
            #include boolean to get task list
            #add a drop down with list of tasks
            
            #Add option to save
            
            @magicgui(get_active_workflow = dict(widget_type="Checkbox",label="Get active workflow in napari-workflow",value = False),
                      workflow_path = dict(mode='r', label="Load custom workflow (.yaml/yml)"),
                      call_button="Apply and Preview Workflow")
            def Preview_Workflow(self,
                                get_active_workflow:bool,
                                workflow_path:Path= Path.home()):
                """
                Apply a cle.workflow to the processing pipeline
                User can define a pipeline which can be inspected in napari workflow inspector
                and then execute it by ticking  the get active workflow checkbox, 
                OR
                Use a predefined workflow
                
                In both cases, if deskewing is not present as first step, it will be added on
                and rest of the task will be made followers

                Args:
                    
                """
                
                print("Previewing deskewed channel and time with workflow")
                
                if get_active_workflow:
                    #installs the workflow to napari
                    user_workflow = WorkflowManager.install(self.parent_viewer).workflow
                else:
                    user_workflow = load_workflow(workflow_path)
                assert type(user_workflow) is Workflow, "Workflow file is not a cle.worfklow. Check file! You can use workflow inspector if needed"
                
                input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(user_workflow)
                print(input_arg_first, input_arg_last, first_task_name,last_task_name )
                #get list of tasks
                task_list = list(user_workflow._tasks.keys())
                print("Workflow loaded:")
                print(user_workflow)
                
                assert self.time_preview.value < LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
                assert self.chan_preview.value < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"
                
                time = self.time_preview.value
                channel = self.chan_preview.value

                print("Deskewing for Time:", time,
                      "and Channel: ", channel)
                
                vol = LLSZWidget.LlszMenu.aics.dask_data

                vol_zyx= vol[time,channel,...]

                task_name_start = first_task_name[0]
                task_name_last = last_task_name[0]
                
                if user_workflow.get_task(task_name_start)[0] not in (cle.deskew_y,cle.deskew_x):
                    #input_file = vol_zyx
                    user_workflow.set("deskew_image",cle.deskew_y, vol_zyx,
                                      angle_in_degrees = LLSZWidget.LlszMenu.lattice.angle,
                                      voxel_size_x = LLSZWidget.LlszMenu.lattice.dx,
                                      voxel_size_y= LLSZWidget.LlszMenu.lattice.dy,
                                      voxel_size_z = LLSZWidget.LlszMenu.lattice.dz)
                    #Set input of the workflow to be from deskewing
                    user_workflow.set(input_arg_first,"deskew_image")
                else:  
                    #set the first input image to be the volume user has chosen
                    user_workflow.set(input_arg_first,vol_zyx)
                
                print("Workflow executed:")
                print(user_workflow)
                #Execute workflow
                processed_vol = user_workflow.get(task_name_last)

                # add channel and time information to the name
                suffix_name = "_c" + str(LLSZWidget.LlszMenu.chan_deskew_value) + "_t" + \
                              str(LLSZWidget.LlszMenu.time_deskew_value)
                                     
                self.parent_viewer.add_image(processed_vol, name="Workflow_processed"+ suffix_name)

                print("Workflow complete")
                return
            
        LlszMenu.Preview_Deskew.header.native.setSizePolicy(1 | 1, 0)
        
    widget = LLSZWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    widget._widget._layout.setAlignment(Qt.AlignTop)

    return widget


# hook for napari to get LLSZ Widget
@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [(plugin_wrapper, {"name": "LLSZ Widget"})]

##Testing out UI only
## Disable napari hook, remove plugin_wrapper
#ui=LLSZWidget()
#ui.show(run=True)
