import os
from pathlib import Path
from magicclass.wrappers import set_design
from magicgui import magicgui, widgets
from magicclass import magicclass, click, field, vfield, set_options
from qtpy.QtCore import Qt


import numpy as np
import dask.array as da
import pyclesperanto_prototype as cle
from tqdm import tqdm

from napari.layers import Layer
from napari.types import ImageData, ShapesData

from napari.utils import history

from .io import LatticeData,  save_tiff,save_tiff_workflow
from .llsz_core import crop_volume_deskew
from .utils import read_imagej_roi, get_first_last_image_and_task,modify_workflow_task

from napari_workflows import Workflow, WorkflowManager
from napari_workflows._io_yaml_v1 import load_workflow, save_workflow


def _workflow_widget():
    @magicclass
    class WorkflowWidget:
        @magicclass
        class WorkflowMenu:
            open_file = False
            lattice = None
            aics = None
            dask = False
            file_name = ""
            save_name = ""

            main_heading = widgets.Label(value="<h3>Napari Lattice: Workflow</h3>")
            heading1 = widgets.Label(value="Drag and drop an image file onto napari.\nChoose the corresponding Image layer by clicking 'Choose Existing Layer'.\n If choosing a czi file, no need to enter voxel sizes")
            @set_design(background_color="magenta", font_family="Consolas",visible=True)
            @set_options(pixel_size_dx={"widget_type": "FloatSpinBox", "value":0.1449922,"step": 0.000000001},
                         pixel_size_dy={"widget_type": "FloatSpinBox", "value":0.1449922, "step": 0.000000001},
                         pixel_size_dz={"widget_type": "FloatSpinBox", "value":0.3, "step": 0.000000001}
                         ) 
            def Choose_Image_Layer(self, 
                                      img_layer:Layer,
                                      pixel_size_dx: float, 
                                      pixel_size_dy: float,
                                      pixel_size_dz: float,
                                      channel_dimension_present:bool=False, 
                                      skew_dir: str="Y"):
                
                print("Using existing image layer")
                skew_dir = str.upper(skew_dir)
                assert skew_dir in ('Y', 'X'), "Skew direction not recognised. Enter either Y or X"
                if skew_dir == "X":
                    WorkflowWidget.WorkflowMenu.deskew_func = cle.deskew_x
                elif skew_dir == "Y":
                    WorkflowWidget.WorkflowMenu.deskew_func = cle.deskew_y
                #TODO: Implement deskew in X direction; pass this to lattice function below
                WorkflowWidget.WorkflowMenu.lattice = LatticeData(img_layer, 30.0, skew_dir,pixel_size_dx, pixel_size_dy,
                                                          pixel_size_dz,channel_dimension_present)
                WorkflowWidget.WorkflowMenu.dask = False  # Use GPU by default
                
                #Flag to check if file has been initialised
                WorkflowWidget.WorkflowMenu.open_file = True
               
                print("Pixel size (ZYX): ",(WorkflowWidget.WorkflowMenu.lattice.dz,WorkflowWidget.WorkflowMenu.lattice.dy,WorkflowWidget.WorkflowMenu.lattice.dx))
                print("Dimensions of image layer (ZYX): ",list(WorkflowWidget.WorkflowMenu.lattice.data.shape[-3:]))
                print("Dimensions of deskewed image (ZYX): ",WorkflowWidget.WorkflowMenu.lattice.deskew_vol_shape)
                print("Initialised")
                self["Choose_Image_Layer"].background_color = "green"
                return
                    

            # Enter custom angle if needed
            # Will only update after choosing an image
            angle = vfield(float, options={"value": 30.0}, name="Deskew Angle")
            angle_value = 30.0

            @angle.connect
            def _set_angle(self):
                try:
                    WorkflowWidget.WorkflowMenu.lattice.set_angle(self.angle)
                    WorkflowWidget.WorkflowMenu.lattice.angle_value = self.angle
                    print("Angle is set to: ", WorkflowWidget.WorkflowMenu.lattice.angle)
                except AttributeError:
                    print("Open a file first before setting angles")
                #print(WorkflowWidget.WorkflowMenu.lattice.angle)
                #print(WorkflowWidget.WorkflowMenu.lattice.angle_value)
                return

            @magicgui(labels=False, auto_call=True)
            def use_GPU(self, use_GPU: bool = True):
                """Choose to use GPU or Dask

                Args:
                    use_GPU (bool, optional): Defaults to True.
                """
                print("Use GPU set to, ", use_GPU)
                WorkflowWidget.WorkflowMenu.dask = not use_GPU
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
                assert time< WorkflowWidget.WorkflowMenu.lattice.time, "Time is out of range"
                assert channel < WorkflowWidget.WorkflowMenu.lattice.channels, "Channel is out of range"
                
                assert str.upper(WorkflowWidget.WorkflowMenu.lattice.skew) in ('Y', 'X'), \
                    "Skew direction not recognised. Enter either Y or X"

                print("Deskewing for Time:", time,"and Channel: ", channel)

                vol = WorkflowWidget.WorkflowMenu.lattice.data

                vol_zyx= vol[time,channel,...]

                # Deskew using pyclesperanto
                deskew_final = vol_zyx.map_blocks(cle.deskew_y,
                                                angle_in_degrees=WorkflowWidget.WorkflowMenu.angle_value,
                                                voxel_size_x=WorkflowWidget.WorkflowMenu.lattice.dx,
                                                voxel_size_y=WorkflowWidget.WorkflowMenu.lattice.dy,
                                                voxel_size_z=WorkflowWidget.WorkflowMenu.lattice.dz,
                                                dtype=vol.dtype,
                                                chunks=WorkflowWidget.WorkflowMenu.lattice.deskew_vol_shape)
                
                # TODO: Use dask
                if WorkflowWidget.WorkflowMenu.dask:
                    print("Using CPU for deskewing")
                    # use cle library for affine transforms, but use dask and scipy
                    # deskew_final = deskew_final.compute()
                
                max_proj_deskew = cle.maximum_z_projection(deskew_final) #np.max(deskew_final, axis=0)

                # add channel and time information to the name
                suffix_name = "_c" + str(channel) + "_t" + str(time)
                scale = (WorkflowWidget.WorkflowMenu.lattice.new_dz,WorkflowWidget.WorkflowMenu.lattice.dy,WorkflowWidget.WorkflowMenu.lattice.dx)
                self.parent_viewer.add_image(max_proj_deskew, name="Deskew_MIP",scale=scale[1:3])
                self.parent_viewer.add_image(deskew_final, name="Deskewed image" + suffix_name,scale=scale)
                self.parent_viewer.layers[0].visible = False
                #print("Shape is ",deskew_final.shape)
                print("Preview: Deskewing complete")
                return
        
        #add function for previewing cropped image
        @magicclass(widget_type="collapsible", name="Preview Crop",popup_mode="below")
        class Preview_Crop_Menu:
              
            @click(enables =["Import_ImageJ_ROI","Crop_Preview"])
            def Click_to_enable_cropping_preview(self):
                WorkflowWidget.Preview_Crop_Menu.shapes_layer = self.parent_viewer.add_shapes(shape_type='polygon', edge_width=1, edge_color='white',
                                            face_color=[1, 1, 1, 0], name="Cropping BBOX layer")
                #TO select ROIs if needed
                WorkflowWidget.Preview_Crop_Menu.shapes_layer.mode="SELECT"
                return
            
            #Import Imagej ROIs using read-roi library
            #non rectangular ROis will be converted to rectangles based on maximum bounds
            @click(enabled =False)
            def Import_ImageJ_ROI(self, path: Path = Path(history.get_open_history()[0])):
                print("Opening", path)
                roi_list = read_imagej_roi(path)
                WorkflowWidget.Preview_Crop_Menu.shapes_layer.add(roi_list,shape_type='polygon', edge_width=5, edge_color='yellow',
                                                                          face_color=[1, 1, 1, 0])
                return
            
            time_crop = field(int, options={"min": 0, "step": 1}, name="Time")
            chan_crop = field(int, options={"min": 0, "step": 1}, name="Channels")
            heading_roi = widgets.Label(value="Import or draw ROI, and then select the ROI using the cursor.")

            
            @click(enabled =False)
            def Crop_Preview(self, roi_layer: ShapesData):  # -> LayerDataTuple:
                assert roi_layer, "No coordinates found for cropping. Check if right shapes layer or initialise shapes layer and draw ROIs."
                #assert self.roi_idx.value <len(WorkflowWidget.Preview_Crop_Menu.shapes_layer.data), "ROI not present"
                #assert len(WorkflowWidget.Preview_Crop_Menu.shapes_layer.selected_data)>0, "ROI not selected"
                # TODO: Add assertion to check if bbox layer or coordinates
                time = self.time_crop.value
                channel = self.chan_crop.value

                assert time < WorkflowWidget.WorkflowMenu.lattice.time, "Time is out of range"
                assert channel < WorkflowWidget.WorkflowMenu.lattice.channels, "Channel is out of range"
                
                print("Using channel ", channel," and time", time)
                
                vol = WorkflowWidget.WorkflowMenu.lattice.data

                vol_zyx= vol[time,channel,...]

                deskewed_shape = WorkflowWidget.WorkflowMenu.lattice.deskew_vol_shape
                
                deskewed_volume = da.zeros(deskewed_shape)

                #Option for entering custom z values?
                z_start = 0
                z_end = deskewed_shape[0]

                #if only one roi selected, use the first ROI for cropping
                if len(roi_layer)==1:
                    roi_idx=0
                else:
                    roi_idx = list(WorkflowWidget.Preview_Crop_Menu.shapes_layer.selected_data)[0]
                    
                roi_choice = roi_layer[roi_idx]
                print("Previewing ROI ", roi_idx)
                
                crop_roi_vol_desk = crop_volume_deskew(original_volume = vol_zyx, 
                                                deskewed_volume=deskewed_volume, 
                                                roi_shape = roi_choice, 
                                                angle_in_degrees = WorkflowWidget.WorkflowMenu.lattice.angle, 
                                                voxel_size_x =WorkflowWidget.WorkflowMenu.lattice.dx, 
                                                voxel_size_y =WorkflowWidget.WorkflowMenu.lattice.dy, 
                                                voxel_size_z =WorkflowWidget.WorkflowMenu.lattice.dz, 
                                                z_start = z_start, 
                                                z_end = z_end).astype(vol_zyx.dtype)

                scale = (WorkflowWidget.WorkflowMenu.lattice.new_dz,WorkflowWidget.WorkflowMenu.lattice.dy,WorkflowWidget.WorkflowMenu.lattice.dx)
                self.parent_viewer.add_image(crop_roi_vol_desk,scale=scale)
                return
        
        @magicclass(widget_type="collapsible", name="Preview Workflow",popup_mode="below")
        class PreviewWorkflow:
            
            #heading2 = widgets.Label(value="<h4>Apply and Preview Workflow</h4>")
            time_preview= field(int, options={"min": 0, "step": 1}, name="Time")
            chan_preview = field(int, options={"min": 0, "step": 1}, name="Channels")
            
            #include boolean to get task list
            #add a drop down with list of tasks
            
            #Add option to save
            
            @magicgui(get_active_workflow = dict(widget_type="Checkbox",label="Get active workflow in napari-workflow",value = False),
                      workflow_path = dict(mode='r', label="Load custom workflow (.yaml/yml)"),
                      call_button="Apply and Preview Workflow")
            def Workflow_Preview(self,
                                get_active_workflow:bool,
                                workflow_path:Path= Path.home()):
                """
                Apply a napari_workflows to the processing pipeline
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
                    print("Workflow loaded from napari")
                else:
                    user_workflow = load_workflow(workflow_path)
                    print("Workflow loaded from file")
                    #WorkflowWidget.ApplyWorkflowSave.Preview_Workflow.workflow_path = workflow_path
                assert type(user_workflow) is Workflow, "Workflow file is not a napari worfklow object. Check file! You can use workflow inspector if needed"
                
                input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(user_workflow)
                print(input_arg_first, input_arg_last, first_task_name,last_task_name )
                #get list of tasks
                task_list = list(user_workflow._tasks.keys())
                print("Workflow loaded:")
                print(user_workflow)
                
                assert self.time_preview.value < WorkflowWidget.WorkflowMenu.lattice.time, "Time is out of range"
                assert self.chan_preview.value < WorkflowWidget.WorkflowMenu.lattice.channels, "Channel is out of range"
                
                time = self.time_preview.value
                channel = self.chan_preview.value

                print("Processing for Time:", time,"and Channel: ", channel)
                
                vol = WorkflowWidget.WorkflowMenu.lattice.data

                vol_zyx= vol[time,channel,...]

                task_name_start = first_task_name[0]
                task_name_last = last_task_name[0]
                #get the function associated with the first task and check if its deskewing
                if user_workflow.get_task(task_name_start)[0] not in (cle.deskew_y,cle.deskew_x):
                    #input_file = vol_zyx
                    user_workflow.set("deskew_image",cle.deskew_y, vol_zyx,
                                      angle_in_degrees = WorkflowWidget.WorkflowMenu.lattice.angle,
                                      voxel_size_x = WorkflowWidget.WorkflowMenu.lattice.dx,
                                      voxel_size_y= WorkflowWidget.WorkflowMenu.lattice.dy,
                                      voxel_size_z = WorkflowWidget.WorkflowMenu.lattice.dz)
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
                suffix_name = "_c" + str(channel) + "_t" +str(time)
                                     
                self.parent_viewer.add_image(processed_vol, name="Workflow_processed"+ suffix_name)

                print("Workflow complete")
                return
            
        @magicclass(widget_type="collapsible", name="Apply Workflow & Save",popup_mode="below")
        class WorkflowSave:

            @magicgui(time_start=dict(label="Time Start:"),
                      time_end=dict(label="Time End:", value=1),
                      ch_start=dict(label="Channel Start:"),
                      ch_end=dict(label="Channel End:", value=1),
                      Use_Cropping = dict(widget_type="Checkbox",label="Crop Data",value = False),
                      get_active_workflow = dict(widget_type="Checkbox",label="Get active workflow in napari-workflow",value = False),
                      workflow_path=dict(mode='r', label="Load custom workflow (.yaml/yml)"),
                      save_path=dict(mode='d', label="Directory to save "),
                      call_button="Apply Workflow and Save Result")            
            def Apply_Workflow_and_Save(self , 
                                        time_start: int, 
                                        time_end: int, 
                                        ch_start: int, 
                                        ch_end: int,
                                        Use_Cropping,
                                        roi_layer_list: ShapesData, 
                                        get_active_workflow:bool=False,
                                        workflow_path:Path= Path.home(),
                                        save_path: Path = Path(history.get_save_history()[0])):
                """
                Apply a user-defined analysis workflow using napari-workflows

                Args:
                    time_start (int): Start Time
                    time_end (int): End Time
                    ch_start (int): Start Channel
                    ch_end (int): End Channel
                    Use_Cropping (_type_): Use cropping based on ROIs in the shapes layer
                    roi_layer_list (ShapesData): Shapes layer to use for cropping; can be a list of shapes
                    get_active_workflow (bool, optional): Gets active workflow in napari. Defaults to False.
                    workflow_path (Path, optional): User can also choose a custom workflow defined in a yaml file.
                    save_path (Path, optional): Path to save resulting data
                """                
                assert WorkflowWidget.WorkflowMenu.open_file, "Image not initialised"
                assert 0<= time_start <=WorkflowWidget.WorkflowMenu.lattice.time, "Time start should be 0 or >0 or same as total time "+str(WorkflowWidget.WorkflowMenu.lattice.time)
                assert 0< time_end <=WorkflowWidget.WorkflowMenu.lattice.time, "Time end should be >0 or same as total time "+str(WorkflowWidget.WorkflowMenu.lattice.time)
                assert 0<= ch_start <= WorkflowWidget.WorkflowMenu.lattice.channels, "Channel start should be 0 or >0 or same as no. of channels "+str(WorkflowWidget.WorkflowMenu.lattice.channels)
                assert 0< ch_end <= WorkflowWidget.WorkflowMenu.lattice.channels, "Channel end should be >0 or same as no. of channels " +str(WorkflowWidget.WorkflowMenu.lattice.channels)

                #Get parameters
                angle = WorkflowWidget.WorkflowMenu.lattice.angle
                dx = WorkflowWidget.WorkflowMenu.lattice.dx
                dy = WorkflowWidget.WorkflowMenu.lattice.dy
                dz = WorkflowWidget.WorkflowMenu.lattice.dz

                if get_active_workflow:
                    #installs the workflow to napari
                    user_workflow = WorkflowManager.install(self.parent_viewer).workflow
                else:
                    print(workflow_path)
                    user_workflow = load_workflow(workflow_path)
                assert type(user_workflow) is Workflow, "Workflow file is not a napari worfklow object. Check file! You can use workflow inspector if needed"
                
                input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(user_workflow)
                print(input_arg_first, input_arg_last, first_task_name,last_task_name )
                #get list of tasks
                task_list = list(user_workflow._tasks.keys())
                print("Workflow loaded:")
                print(user_workflow)
                
                assert self.time_preview.value < WorkflowWidget.WorkflowMenu.lattice.time, "Time is out of range"
                assert self.chan_preview.value < WorkflowWidget.WorkflowMenu.lattice.channels, "Channel is out of range"
                
                time = self.time_preview.value
                channel = self.chan_preview.value

                print("Processing for Time:", time,"and Channel: ", channel)
                
                vol = WorkflowWidget.WorkflowMenu.lattice.data
                #convert Roi pixel coordinates to canvas coordinates
                #necessary only when scale is used for napari.viewer.add_image operations
                roi_layer_list = [x/WorkflowWidget.WorkflowMenu.lattice.dy for x in roi_layer_list]
                #vol_zyx= vol[time,channel,...]

                task_name_start = first_task_name[0]
                task_name_last = last_task_name[0]
                #if cropping, set that as first task
                if Use_Cropping:
                    deskewed_shape = WorkflowWidget.WorkflowMenu.lattice.deskew_vol_shape
                    deskewed_volume = da.zeros(deskewed_shape)
                    z_start = 0
                    z_end = deskewed_shape[0]
                    roi = "roi"
                    volume = "volume"
                    #Create workflow for cropping and deskewing
                    #volume and roi used will be set dynamically
                    user_workflow.set("crop_deskew",crop_volume_deskew,
                                      original_volume = volume,
                                      deskewed_volume = deskewed_volume,
                                      roi_shape = roi,
                                      angle_in_degrees = angle,
                                      voxel_size_x = dx,
                                      voxel_size_y= dy,
                                      voxel_size_z = dz, 
                                      z_start = z_start, 
                                      z_end = z_end)
                    #change the first task so it accepts "crop_deskew as input"
                    new_task = modify_workflow_task(old_arg=input_arg_first,task_key=task_name_start,new_arg="crop_deskew",workflow=user_workflow)
                    user_workflow.set(task_name_start,new_task)
                    
                    for idx, roi_layer in enumerate(tqdm(roi_layer_list, desc="ROI:", position=0)):
                        print("Processing ROI ",idx)
                        user_workflow.set(roi,roi_layer)
                        save_tiff_workflow(vol=vol,
                                            workflow = user_workflow,
                                            input_arg = volume,
                                            first_task = "crop_deskew",
                                            last_task = last_task_name,
                                            time_start = time_start,
                                            time_end = time_end,
                                            channel_start = ch_start,
                                            channel_end = ch_end,
                                            save_path = save_path,
                                            crop = Use_Cropping,
                                            roi_layer = roi_layer,
                                            save_name_prefix = "ROI_"+str(idx),
                                            save_name =  WorkflowWidget.WorkflowMenu.lattice.save_name,
                                            dx = dx,
                                            dy = dy,
                                            dz = dz,
                                            angle = angle)
                #IF just deskewing and its not in the tasks, add that as first task
                elif user_workflow.get_task(task_name_start)[0] not in (cle.deskew_y,cle.deskew_x):
                    input = "input"
                    #add task to the workflow
                    user_workflow.set("deskew_image",cle.deskew_y, 
                                    input_image =input,
                                    angle_in_degrees = angle,
                                    voxel_size_x = dx,
                                    voxel_size_y= dy,
                                    voxel_size_z = dz)
                    #Set input of the workflow to be from deskewing
                    #change the first task so it accepts "deskew_image" as input
                    new_task = modify_workflow_task(old_arg=input_arg_first,task_key=task_name_start,new_arg="deskew_image",workflow=user_workflow)
                    user_workflow.set(task_name_start,new_task)
                    
                    save_tiff_workflow(vol=vol,
                                            workflow = user_workflow,
                                            input_arg = input,
                                            first_task = "deskew_image",
                                            last_task = last_task_name,
                                            time_start = time_start,
                                            time_end = time_end,
                                            channel_start = ch_start,
                                            channel_end = ch_end,
                                            save_path = save_path,
                                            save_name =  WorkflowWidget.WorkflowMenu.lattice.save_name,
                                            dx = dx,
                                            dy = dy,
                                            dz = dz,
                                            angle = angle)
                ##If deskewing is already as a task, then set the first argument to input so we can modify that later
                else:


                    #we pass first argument as input
                    save_tiff_workflow(vol=vol,
                                            workflow = user_workflow,
                                            input_arg = input_arg_first,
                                            first_task = first_task_name,
                                            last_task = last_task_name,
                                            time_start = time_start,
                                            time_end = time_end,
                                            channel_start = ch_start,
                                            channel_end = ch_end,
                                            save_path = save_path,
                                            save_name =  WorkflowWidget.WorkflowMenu.lattice.save_name,
                                            dx = dx,
                                            dy = dy,
                                            dz = dz,
                                            angle = angle)

                print("Workflow complete")
                return
        #Workflow to deskew and apply workflow; 
        #Check if user wants to crop by having a crop flag        
    #Important to have this or napari won't recognize the classes and magicclass qidgets
    workflow_widget = WorkflowWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    workflow_widget._widget._layout.setAlignment(Qt.AlignTop)

    return workflow_widget   
            