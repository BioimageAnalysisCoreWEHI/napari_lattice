import os, sys, yaml
from aicsimageio import AICSImage
import numpy as np
from pathlib import Path
import dask.array as da
import pandas as pd

from magicclass.wrappers import set_design
from magicgui import magicgui, widgets
from magicclass import magicclass, field, vfield, set_options
from magicclass.utils import click
from qtpy.QtCore import Qt

from napari.layers import Layer
from napari.types import ImageData
from napari.utils import history

import pyclesperanto_prototype as cle
from .io import LatticeData
from .ui_core import _Preview, _Deskew_Save

from napari.types import ImageData, ShapesData
from .utils import read_imagej_roi
from .llsz_core import crop_volume_deskew, rl_decon
from tqdm import tqdm

from .io import LatticeData,  save_tiff, save_tiff_workflow

from .utils import read_imagej_roi, get_first_last_image_and_task,modify_workflow_task,get_all_py_files, as_type, process_custom_workflow_output
from . import config
from napari_workflows import Workflow, WorkflowManager
from napari_workflows._io_yaml_v1 import load_workflow

def _napari_lattice_widget_wrapper():
    #split widget type enables a resizable widget
    @magicclass(widget_type="split")
    class LLSZWidget:
        @magicclass(widget_type="split")
        class LlszMenu:
            open_file = False
            lattice = None
            aics = None
            dask = False
            file_name = ""
            save_name = ""

            main_heading = widgets.Label(value="<h3>Napari Lattice: Visualization & Analysis</h3>")
            heading1 = widgets.Label(value="Drag and drop an image file onto napari.\nOnce image has opened, initialize the\nplugin by clicking the button below.\nEnsure the image layer and voxel sizes are accurate in the prompt.\n If everything initalises properly, the button turns green.")
            @set_design(background_color="magenta", font_family="Consolas",visible=True,text="Initialize Plugin", max_height=75, font_size = 13) 
            @set_options(pixel_size_dx={"widget_type": "FloatSpinBox", "value":0.1449922,"step": 0.000000001},
                         pixel_size_dy={"widget_type": "FloatSpinBox", "value":0.1449922, "step": 0.000000001},
                         pixel_size_dz={"widget_type": "FloatSpinBox", "value":0.3, "step": 0.000000001}
                         )
            def Choose_Image_Layer(self,
                                      img_layer:Layer,
                                      pixel_size_dx: float = 0.1449922, 
                                      pixel_size_dy: float = 0.1449922,
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

                LLSZWidget.LlszMenu.lattice = LatticeData(img_layer, 30.0, skew_dir,pixel_size_dx, pixel_size_dy,
                                                          pixel_size_dz,channel_dimension_present)
                #LLSZWidget.LlszMenu.aics = LLSZWidget.LlszMenu.lattice.data

                LLSZWidget.LlszMenu.dask = False  # Use GPU by default
                
                #list to store psf images
                LLSZWidget.LlszMenu.lattice.psf = []
                
                LLSZWidget.LlszMenu.open_file = True
                print("Pixel size (ZYX): ",(LLSZWidget.LlszMenu.lattice.dz,LLSZWidget.LlszMenu.lattice.dy,LLSZWidget.LlszMenu.lattice.dx))
                print("Dimensions of image layer (ZYX): ",list(LLSZWidget.LlszMenu.lattice.data.shape[-3:]))
                print("Dimensions of deskewed image (ZYX): ",LLSZWidget.LlszMenu.lattice.deskew_vol_shape)
                print("Initialised")

                self["Choose_Image_Layer"].background_color = "green"
                self["Choose_Image_Layer"].text = "Plugin Initialised"
                
                #Add dimension labels
                if LLSZWidget.LlszMenu.lattice.channels == 1:
                    self.parent_viewer.dims.axis_labels = list(('Time',"Z","Y","X"))
                elif LLSZWidget.LlszMenu.lattice.channels >1:
                    self.parent_viewer.dims.axis_labels = list(('Time',"Channel","Z","Y","X"))
                elif LLSZWidget.LlszMenu.lattice.channels==1 and  LLSZWidget.LlszMenu.lattice.time>1:
                    self.parent_viewer.dims.axis_labels = list(('Time',"Z","Y","X"))

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

            #Redfishlion library for deconvolution
            deconvolution = vfield(bool, options={"enabled": False}, name="Use Deconvolution (RL) NOT ACTIVE")
            deconvolution.value = False
            @deconvolution.connect
            def _set_decon(self):
                if self.deconvolution:
                    print("Deconvolution Activated")
                    LLSZWidget.LlszMenu.deconvolution.value = True
                else:
                    print("Deconvolution Disabled")
                    LLSZWidget.LlszMenu.deconvolution.value  = False
                return
            
            @set_design(background_color="magenta", font_family="Consolas",visible=True,text="Click to select PSFs for deconvolution", max_height=75, font_size = 11)
            @set_options(header = dict(widget_type="Label",label="<h3>Enter path to the PSF images</h3>"),
                         psf_ch1_path={"widget_type": "FileEdit","label":"Channel 1/Multichannel PSF:"},
                         psf_ch2_path={"widget_type": "FileEdit","label":"Channel 2"},
                         psf_ch3_path={"widget_type": "FileEdit","label":"Channel 3"},
                         psf_ch4_path={"widget_type": "FileEdit","label":"Channel 4"}
                         )
            def deconvolution_gui(self,
                                    header,
                                    psf_ch1_path:Path,
                                    psf_ch2_path:Path,
                                    psf_ch3_path:Path,
                                    psf_ch4_path:Path):
                        #move function to ui_core; 
                        #create list with psf image arrays for each channel
                        #index corresponds to channel no
                from pathlib import WindowsPath
                assert LLSZWidget.LlszMenu.deconvolution.value==True, "Deconvolution is set to False. Tick the box to activate deconvolution."
                
                psf_paths = [psf_ch1_path,psf_ch2_path,psf_ch3_path,psf_ch4_path]
                #remove empty paths; pathlib returns current directory as "." if None or empty str specified
                psf_paths = [x for x in psf_paths if x!=WindowsPath(".")]
                
                #total no of psf images
                psf_channels = len(psf_paths)
                
                assert psf_channels>0, f"No images detected for PSF. Check the path {psf_paths}"
                
                for psf in psf_paths:
                    if os.path.exists(psf) and psf.is_file():
                            psf_aics = AICSImage(psf.__str__())
                            LLSZWidget.LlszMenu.lattice.psf.append(psf_aics.data)
                            
                            if psf_aics.dims.C>=1:
                                    psf_channels = psf_aics.dims.C

                #LLSZWidget.LlszMenu.lattice.channels =3
                if psf_channels != LLSZWidget.LlszMenu.lattice.channels:
                        print(f"PSF image has {psf_channels} channel/s, whereas image has {LLSZWidget.LlszMenu.lattice.channels}")
                
                self["deconvolution_gui"].background_color = "green"
                self["deconvolution_gui"].text = "PSFs added"
                
                return 
                
                

        @magicclass
        class Preview:          
            @magicgui(header=dict(widget_type="Label",label="<h3>Preview Deskew</h3>"),
                      time=dict(label="Time:"),
                      channel=dict(label="Channel:"),
                      call_button="Preview")
            def Preview_Deskew(self, 
                               header,
                               time:int,
                               channel:int,
                               img_data: ImageData):
                """
                Preview the deskewing for a single timepoint

                Args:
                    header ([type]): [description]
                    img_data (ImageData): [description]
                """
                _Preview(LLSZWidget,
                        self,
                        time,
                        channel,
                        img_data)
                return
        
        #Widget container to house all the widgets
        @magicclass(widget_type="tabbed", name="Functions")
        class WidgetContainer: 
            @magicclass(name="Deskew",widget_type="scrollable")
            class DeskewWidget:
               @magicgui(header=dict(widget_type="Label", label="<h3>Deskew and Save</h3>"),
                       time_start=dict(label="Time Start:"),
                       time_end=dict(label="Time End:", value=1),
                       ch_start=dict(label="Channel Start:"),
                       ch_end=dict(label="Channel End:", value=1),
                       save_path=dict(mode='d', label="Directory to save"),
                       call_button="Save")
               def Deskew_Save(self,
                               header,
                               time_start: int,
                               time_end: int,
                               ch_start: int,
                               ch_end: int,
                               save_path: Path = Path(history.get_save_history()[0])):

                   _Deskew_Save(LLSZWidget,
                               time_start,
                               time_end,
                               ch_start,
                               ch_end,
                               save_path)
                   return
               
            @magicclass(name="Crop and Deskew",widget_type="scrollable")
            class CropWidget:  
                
            #add function for previewing cropped image
                @magicclass(name="Cropping Preview",widget_type="scrollable")
                class Preview_Crop_Menu:
                    
                    @set_design(font_size=10,text="Click to activate Cropping Preview",background_color="magenta")
                    @click(enables =["Import_ImageJ_ROI","Crop_Preview"])
                    def activate_cropping(self):
                        LLSZWidget.WidgetContainer.CropWidget.Preview_Crop_Menu.shapes_layer = self.parent_viewer.add_shapes(shape_type='polygon', edge_width=1, edge_color='white',
                                                    face_color=[1, 1, 1, 0], name="Cropping BBOX layer")
                        #TO select ROIs if needed
                        LLSZWidget.WidgetContainer.CropWidget.Preview_Crop_Menu.shapes_layer.mode="SELECT"
                        self["activate_cropping"].text = "Cropping layer active"
                        self["activate_cropping"].background_color = "green"
                        return
                    
                    heading2 = widgets.Label(value="You can either import ImageJ ROI (.zip) files or manually define ROIs using the shape layer")
                    @click(enabled =False)
                    def Import_ImageJ_ROI(self, path: Path = Path(history.get_open_history()[0])):
                        print("Opening", path)
                        roi_list = read_imagej_roi(path)
                        #convert to canvas coordinates
                        roi_list = (np.array(roi_list)*LLSZWidget.LlszMenu.lattice.dy).tolist()
                        LLSZWidget.WidgetContainer.CropWidget.Preview_Crop_Menu.shapes_layer.add(roi_list,shape_type='polygon', edge_width=1, edge_color='yellow',
                                                                                face_color=[1, 1, 1, 0])
                        return
                        
                    time_crop = field(int, options={"min": 0, "step": 1}, name="Time")
                    chan_crop = field(int, options={"min": 0, "step": 1}, name="Channels")
                    heading_roi = widgets.Label(value="If there are multiple ROIs, select the ROI before clicking button below")
                    #roi_idx = field(int, options={"min": 0, "step": 1}, name="ROI number")

                    
                    @click(enabled =False)
                    def Crop_Preview(self, roi_layer: ShapesData):  # -> LayerDataTuple:
                            assert roi_layer, "No coordinates found for cropping. Check if right shapes layer or initialise shapes layer and draw ROIs."
                            #assert self.roi_idx.value <len(CropWidget.Preview_Crop_Menu.shapes_layer.data), "ROI not present"
                            #assert len(CropWidget.Preview_Crop_Menu.shapes_layer.selected_data)>0, "ROI not selected"
                            # TODO: Add assertion to check if bbox layer or coordinates
                            time = self.time_crop.value
                            channel = self.chan_crop.value

                            assert time < LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
                            assert channel < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"
                            
                            print("Using channel ", channel," and time", time)
                            
                            vol = LLSZWidget.LlszMenu.lattice.data
                            vol_zyx= vol[time,channel,...]

                            deskewed_shape = LLSZWidget.LlszMenu.lattice.deskew_vol_shape
                            #Create a dask array same shape as deskewed image
                            deskewed_volume = da.zeros(deskewed_shape)

                            #Option for entering custom z start value?
                            z_start = 0
                            z_end = deskewed_shape[0]

                            #if only one roi drawn, use the first ROI for cropping
                            if len(roi_layer)==1:
                                roi_idx=0
                            else:
                                assert len(LLSZWidget.WidgetContainer.CropWidget.Preview_Crop_Menu.shapes_layer.selected_data)>0, "Please select an ROI"
                                roi_idx = list(LLSZWidget.WidgetContainer.CropWidget.Preview_Crop_Menu.shapes_layer.selected_data)[0]

                            roi_choice = roi_layer[roi_idx] 
                            #As the original image is scaled, the coordinates are in microns, so we need to convert
                            #roi from micron to canvas/world coordinates
                            roi_choice = [x/LLSZWidget.LlszMenu.lattice.dy for x in roi_choice]
                            print("Previewing ROI ", roi_idx)
                            
                            crop_roi_vol_desk= crop_volume_deskew(original_volume = vol_zyx,
                                                                        deskewed_volume=deskewed_volume, 
                                                                        roi_shape = roi_choice, 
                                                                        angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                                                        voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                                                        voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                                                        voxel_size_z=LLSZWidget.LlszMenu.lattice.dz,
                                                                        z_start = z_start, 
                                                                        z_end = z_end).astype(vol_zyx.dtype)
                            
                            #get array back from gpu or addding cle array to napari can throw errors
                            crop_roi_vol_desk = cle.pull(crop_roi_vol_desk)
                            scale = (LLSZWidget.LlszMenu.lattice.new_dz,
                                    LLSZWidget.LlszMenu.lattice.dy,
                                    LLSZWidget.LlszMenu.lattice.dx)
                            self.parent_viewer.add_image(crop_roi_vol_desk,scale=scale)
                            return
                
                
                    @magicclass(name="Crop and Save Data",widget_type="scrollable")
                    class CropSaveData:
                        @magicgui(header=dict(widget_type="Label", label="<h3>Crop and Save Data</h3>"),
                                  time_start=dict(label="Time Start:"),
                                time_end=dict(label="Time End:", value=1),
                                ch_start=dict(label="Channel Start:"),
                                ch_end=dict(label="Channel End:", value=1),
                                save_path=dict(mode='d', label="Directory to save "))
                        def Crop_Save(self,
                                        header,
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
                                assert 0<= time_start <=LLSZWidget.LlszMenu.lattice.time, "Time start should be >0 or same as total time "+str(CropWidget.CropMenu.lattice.time)
                                assert 0< time_end <=LLSZWidget.LlszMenu.lattice.time, "Time end should be between 0 and total time "+str(CropWidget.CropMenu.lattice.time)
                                assert 0<= ch_start <= LLSZWidget.LlszMenu.lattice.channels, "Channel start should be 0 or >0 or same as no. of channels "+str(CropWidget.CropMenu.lattice.channels)
                                assert 0< ch_end <= LLSZWidget.LlszMenu.lattice.channels, "Channel end should be >0 or same as no. of channels " +str(CropWidget.CropMenu.lattice.channels)
                        
                                angle = LLSZWidget.LlszMenu.lattice.angle
                                dx = LLSZWidget.LlszMenu.lattice.dx
                                dy = LLSZWidget.LlszMenu.lattice.dy
                                dz = LLSZWidget.LlszMenu.lattice.dz
                                
                                #get image data
                                img_data = LLSZWidget.LlszMenu.lattice.data
                                #Get shape of deskewed image
                                deskewed_shape = LLSZWidget.LlszMenu.lattice.deskew_vol_shape
                                deskewed_volume = da.zeros(deskewed_shape)
                                z_start = 0
                                z_end = deskewed_shape[0]
                                
                                print("Cropping and saving files...")           

                                #necessary when scale is used for napari.viewer.add_image operations
                                roi_layer_list = [x/LLSZWidget.LlszMenu.lattice.dy for x in roi_layer_list]

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
                                        save_name_prefix  = "ROI_" + str(idx),
                                        save_path = save_path,
                                        save_name= LLSZWidget.LlszMenu.lattice.save_name,
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

            @magicclass(name="Workflow",widget_type="scrollable")
            class WorkflowWidget:  
                @magicclass(name="Preview Workflow",widget_type="scrollable")
                class PreviewWorkflow:
                    #time_preview= field(int, options={"min": 0, "step": 1}, name="Time")
                    #chan_preview = field(int, options={"min": 0, "step": 1}, name="Channels")
                    @magicgui(header=dict(widget_type="Label", label="<h3>Preview Workflow</h3>"),
                              time_preview = dict(label="Time:"),
                              chan_preview = dict(label="Channel:"),
                              get_active_workflow = dict(widget_type="Checkbox",label="Get active workflow in napari-workflow",value = False),
                            workflow_path = dict(mode='r', label="Load custom workflow (.yaml/yml)"),
                            Use_Cropping = dict(widget_type="Checkbox",label="Crop Data",value = False),
                            #custom_module=dict(widget_type="Checkbox",label="Load custom module (looks for *.py files in the workflow directory)",value = False),
                            call_button="Apply and Preview Workflow")
                    def Workflow_Preview(self,
                                        header,
                                        time_preview:int,
                                        chan_preview:int,
                                        get_active_workflow:bool,
                                        Use_Cropping:bool,
                                        roi_layer_list: ShapesData, 
                                        workflow_path:Path= Path.home()):
                        """
                            Apply napari_workflows to the processing pipeline
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
                            parent_dir = workflow_path.resolve().parents[0].__str__()+os.sep
                            print("Workflow loaded from napari")
                        else:

                            try:
                                #Automatically scan workflow file directory for *.py files. 
                                #If it findss one, load it as a module
                                import importlib
                                parent_dir = workflow_path.resolve().parents[0].__str__()+os.sep
                                sys.path.append(parent_dir)
                                custom_py_files = get_all_py_files(parent_dir)
                                if len(custom_py_files)==0: 
                                    print(f"No custom modules imported. If you'd like to use a cusotm module, place a *.py file in same folder as the workflow file {parent_dir}")
                                else:
                                    modules = map(importlib.import_module,custom_py_files)
                                    print(f"Custom modules imported {modules}")
                                user_workflow = load_workflow(workflow_path.__str__())
                            except yaml.loader.ConstructorError as e:
                                print("\033[91m While loading workflow, got the following error which may mean you need to install the corresponding module in your Python environment: \033[0m")
                                print(e)
                                
                            #user_workflow = load_workflow(workflow_path)
                            print("Workflow loaded from file")
                        
                        assert type(user_workflow) is Workflow, "Workflow loading error. Check if file is workflow or if required libraries are installed"
                        
                        input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(user_workflow)
                        print(input_arg_first, input_arg_last, first_task_name,last_task_name )
                        #get list of tasks
                        task_list = list(user_workflow._tasks.keys())
                        print("Workflow loaded:")
                        print(user_workflow)
                        #when using fields, self.time_preview.value 
                        assert time_preview < LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
                        assert chan_preview < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"

                        time = time_preview
                        channel = chan_preview

                        #to access current time and channel and pass it to workflow file
                        config.channel = channel
                        config.time = time
                        
                        print("Processing for Time:", time,"and Channel: ", channel)
                        
                        vol = LLSZWidget.LlszMenu.lattice.data

                        vol_zyx= vol[time,channel,...]

                        task_name_start = first_task_name[0]
                        try:
                            task_name_last = last_task_name[0]
                        except IndexError:
                            task_name_last = task_name_start
                        
                        #get the function associated with the first task and check if its deskewing
                        if Use_Cropping:
                            #use deskewed volume for cropping function
                            deskewed_shape = LLSZWidget.LlszMenu.lattice.deskew_vol_shape
                            deskewed_volume = da.zeros(deskewed_shape)
                            z_start = 0
                            z_end = deskewed_shape[0]
                            if user_workflow.get_task(task_name_start)[0] not in [crop_volume_deskew]:
                                #if only one roi drawn, use the first ROI for cropping
                                if len(roi_layer_list)==1:
                                    roi_idx=0
                                else: #else get the user selection
                                    assert len(LLSZWidget.WidgetContainer.CropWidget.Preview_Crop_Menu.shapes_layer.selected_data)>0, "Please select an ROI"
                                    roi_idx = list(LLSZWidget.WidgetContainer.CropWidget.Preview_Crop_Menu.shapes_layer.selected_data)[0]
                                
                                roi_choice = roi_layer_list[roi_idx]
                                #As the original image is scaled, the coordinates are in microns, so we need to convert
                                #roi to from micron to canvas/world coordinates
                                roi_choice = [x/LLSZWidget.LlszMenu.lattice.dy for x in roi_choice]
                                print("Previewing ROI ", roi_idx)
                                
                                user_workflow.set("crop_deskew_image",crop_volume_deskew,original_volume = vol_zyx, 
                                                            deskewed_volume=deskewed_volume, 
                                                            roi_shape = roi_choice, 
                                                            angle_in_degrees = LLSZWidget.LlszMenu.lattice.angle, 
                                                            voxel_size_x =LLSZWidget.LlszMenu.lattice.dx, 
                                                            voxel_size_y =LLSZWidget.LlszMenu.lattice.dy, 
                                                            voxel_size_z =LLSZWidget.LlszMenu.lattice.dz, 
                                                            z_start = z_start, 
                                                            z_end = z_end)
                                #Set input of the workflow to be from crop_deskewing
                                user_workflow.set(input_arg_first,"crop_deskew_image")
                            else:
                                user_workflow.set(input_arg_first,vol_zyx)
                                
                        elif user_workflow.get_task(task_name_start)[0] not in (cle.deskew_y,cle.deskew_x):
                            
                            user_workflow.set("deskew_image",cle.deskew_y, vol_zyx,
                                            angle_in_degrees = LLSZWidget.LlszMenu.lattice.angle,
                                            voxel_size_x = LLSZWidget.LlszMenu.lattice.dx,
                                            voxel_size_y= LLSZWidget.LlszMenu.lattice.dy,
                                            voxel_size_z = LLSZWidget.LlszMenu.lattice.dz)
                            user_workflow.set("change_bitdepth",as_type,"deskew_image",vol_zyx)
                            #Set input of the workflow to be from deskewing
                            user_workflow.set(input_arg_first,"change_bitdepth")
                        else:  
                            #set the first input image to be the volume user has chosen
                            user_workflow.set(input_arg_first,vol_zyx)
                        
                        print("Workflow executed:")
                        print(user_workflow)
                        #Execute workflow
                        processed_vol = user_workflow.get(task_name_last)
                        
                        #check if a measurement table (usually a dictionary or list)  or a tuple with different data types
                        #The function below saves the tables and adds any images to napari window
                        if type(processed_vol) in [dict,list,tuple]:
                            if(len(processed_vol)>1):
                                df = pd.DataFrame()
                                for idx,i in enumerate(processed_vol):
                                    df_temp = process_custom_workflow_output(i,parent_dir,idx,LLSZWidget,self,channel,time,preview=True)
                                    final_df = pd.concat([df,df_temp])
                                    #append dataframes from every loop and have table command outside loop?
                                widgets.Table(value=final_df).show() 

                        else:
                            #add image to napari window
                            #TODO: check if its an image napari supports?
                            process_custom_workflow_output(processed_vol,parent_dir,0,LLSZWidget,self,channel,time)

                        print("Workflow complete")
                        return
                    
                    @magicgui(header=dict(widget_type="Label", label="<h3>Apply Workflow and Save Output</h3>"),
                              time_start=dict(label="Time Start:"),
                              time_end=dict(label="Time End:", value=1),
                              ch_start=dict(label="Channel Start:"),
                              ch_end=dict(label="Channel End:", value=1),
                              Use_Cropping = dict(widget_type="Checkbox",label="Crop Data",value = False),
                              get_active_workflow = dict(widget_type="Checkbox",label="Get active workflow in napari-workflow",value = False),
                              workflow_path=dict(mode='r', label="Load custom workflow (.yaml/yml)"),
                              save_path=dict(mode='d', label="Directory to save "),
                              #custom_module=dict(widget_type="Checkbox",label="Load custom module (same dir as workflow)",value = False),
                              call_button="Apply Workflow and Save Result")            
                    def Apply_Workflow_and_Save(self , 
                                                header,
                                                time_start: int, 
                                                time_end: int, 
                                                ch_start: int, 
                                                ch_end: int,
                                                Use_Cropping,
                                                roi_layer_list: ShapesData, 
                                                get_active_workflow:bool=False,
                                                workflow_path:Path= Path.home(),
                                                #custom_module:bool=False,
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
                        assert LLSZWidget.LlszMenu.open_file, "Image not initialised"
                        assert 0<= time_start <=LLSZWidget.LlszMenu.lattice.time, "Time start should be 0 or >0 or same as total time "+str(LLSZWidget.LlszMenu.lattice.time)
                        assert 0< time_end <=LLSZWidget.LlszMenu.lattice.time, "Time end should be >0 or same as total time "+str(LLSZWidget.LlszMenu.lattice.time)
                        assert 0<= ch_start <= LLSZWidget.LlszMenu.lattice.channels, "Channel start should be 0 or >0 or same as no. of channels "+str(LLSZWidget.LlszMenu.lattice.channels)
                        assert 0< ch_end <= LLSZWidget.LlszMenu.lattice.channels, "Channel end should be >0 or same as no. of channels " +str(LLSZWidget.LlszMenu.lattice.channels)

                        #Get parameters
                        angle = LLSZWidget.LlszMenu.lattice.angle
                        dx = LLSZWidget.LlszMenu.lattice.dx
                        dy = LLSZWidget.LlszMenu.lattice.dy
                        dz = LLSZWidget.LlszMenu.lattice.dz

                        if get_active_workflow:
                            #installs the workflow to napari
                            user_workflow = WorkflowManager.install(self.parent_viewer).workflow
                            print("Workflow installed")
                        else:
                            #Automatically scan workflow file directory for *.py files. 
                            #If it findss one, load it as a module
                            import importlib
                            parent_dir = workflow_path.resolve().parents[0].__str__()+os.sep
                            sys.path.append(parent_dir)
                            custom_py_files = get_all_py_files(parent_dir)
                            if len(custom_py_files)==0: 
                                print(f"No custom modules imported. If you'd like to use a cusotm module, place a *.py file in same folder as the workflow file {parent_dir}")
                            else:
                                modules = map(importlib.import_module,custom_py_files)
                                print(f"Custom modules imported {modules}") 
                            user_workflow = load_workflow(workflow_path)

                        assert type(user_workflow) is Workflow, "Workflow file is not a napari workflow object. Check file! You can use workflow inspector if needed"

                        input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(user_workflow)
                        print(input_arg_first, input_arg_last, first_task_name,last_task_name)
                        #get list of tasks
                        task_list = list(user_workflow._tasks.keys())
                        print("Workflow loaded:")
                        print(user_workflow)

                        vol = LLSZWidget.LlszMenu.lattice.data
                        #convert Roi pixel coordinates to canvas coordinates
                        #necessary only when scale is used for napari.viewer.add_image operations
                        roi_layer_list = [x/LLSZWidget.LlszMenu.lattice.dy for x in roi_layer_list]
                        #vol_zyx= vol[time,channel,...]

                        task_name_start = first_task_name[0]
                        try:
                            task_name_last = last_task_name[0]
                        except IndexError:
                            task_name_last = task_name_start
                        #if cropping, set that as first task
                        if Use_Cropping:
                            deskewed_shape = LLSZWidget.LlszMenu.lattice.deskew_vol_shape
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
                                                    last_task = task_name_last,
                                                    time_start = time_start,
                                                    time_end = time_end,
                                                    channel_start = ch_start,
                                                    channel_end = ch_end,
                                                    save_path = save_path,
                                                    #roi_layer = roi_layer,
                                                    save_name_prefix = "ROI_"+str(idx),
                                                    save_name =  LLSZWidget.LlszMenu.lattice.save_name,
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
                                                    last_task = task_name_last,
                                                    time_start = time_start,
                                                    time_end = time_end,
                                                    channel_start = ch_start,
                                                    channel_end = ch_end,
                                                    save_path = save_path,
                                                    save_name =  LLSZWidget.LlszMenu.lattice.save_name,
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
                                                    last_task = task_name_last,
                                                    time_start = time_start,
                                                    time_end = time_end,
                                                    channel_start = ch_start,
                                                    channel_end = ch_end,
                                                    save_path = save_path,
                                                    save_name =  LLSZWidget.LlszMenu.lattice.save_name,
                                                    dx = dx,
                                                    dy = dy,
                                                    dz = dz,
                                                    angle = angle)

                        print("Workflow complete")
                        return
                        
                pass
    LLSZWidget.WidgetContainer.DeskewWidget.max_width = 100
    #max_height = 50
    #Important to have this or napari won't recognize the classes and magicclass qidgets
    widget = LLSZWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    widget._widget._layout.setAlignment(Qt.AlignTop)
   
    #widget._widget._layout.setWidgetResizable(True)
    return widget   