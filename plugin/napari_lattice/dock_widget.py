import os
import sys
import yaml
import numpy as np
from pathlib import Path
import dask.array as da
import pandas as pd
from typing import allable, Iterable, Optional, Union
from enum import Enum

from magicclass.wrappers import set_design
from magicgui import magicgui
from magicclass import magicclass, field, vfield, set_options, MagicTemplate
from magicclass.utils import click
from qtpy.QtCore import Qt

from napari.layers import Layer, Shapes
from napari.types import ImageData
from napari.utils import history
from napari import Viewer

import pyclesperanto_prototype as cle

from napari.types import ImageData, ShapesData

from tqdm import tqdm

from napari_workflows import Workflow, WorkflowManager
from napari_workflows._io_yaml_v1 import load_workflow

from lls_core import config, DeconvolutionChoice, SaveFileType, Log_Levels, DeskewDirection
from lls_core.io import LatticeData,  save_img, save_img_workflow
from lls_core.lattice_data import DeconvolutionParams
from lls_core.types import ArrayLike
from lls_core.workflow import get_first_last_image_and_task, modify_workflow_task, get_all_py_files, process_custom_workflow_output, load_custom_py_modules, import_workflow_modules, replace_first_arg
from lls_core.utils import check_dimensions, read_imagej_roi, as_type
from lls_core.llsz_core import crop_volume_deskew
from lls_core.deconvolution import read_psf, pycuda_decon, skimage_decon

from napari_lattice.ui_core import _Preview, _Deskew_Save
from napari_lattice.reader import lattice_from_napari

from copy import copy

# Enable Logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LastDimensionOptions(Enum):
    channel = "Channel"
    time = "Time"
    get_from_metadata = "Get from Metadata"

class LlszTemplate(MagicTemplate):
    @property
    def llsz_parent(self) -> "LLSZWidget":
        return self.find_ancestor(LLSZWidget)
        
    @property
    def parent_viewer(self) -> Viewer:
        viewer = super().parent_viewer
        if viewer is None:
            raise Exception("This function can only be used when inside of a Napari viewer")
        return super().parent_viewer

@magicclass(widget_type="split")
class LLSZWidget(LlszTemplate):
    open_file: bool = False
    lattice: LatticeData
    deconv: DeconvolutionParams = DeconvolutionParams()
    shapes_layer: Shapes

    @magicclass(widget_type="split")
    class LlszMenu(LlszTemplate):

        main_heading = field("<h3>Napari Lattice: Visualization & Analysis</h3>", widget_type="Label")
        heading1 = field("Drag and drop an image file onto napari.\nOnce image has opened, initialize the\nplugin by clicking the button below.\nEnsure the image layer and voxel sizes are accurate in the prompt.\n If everything initalises properly, the button turns green.", widget_type="Label")

        @set_design(background_color="magenta", font_family="Consolas", visible=True, text="Initialize Plugin", max_height=75, font_size=13)
        @set_options(pixel_size_dx={"widget_type": "FloatSpinBox", "value": 0.1449922, "step": 0.000000001},
                        pixel_size_dy={"widget_type": "FloatSpinBox",
                                    "value": 0.1449922, "step": 0.000000001},
                        pixel_size_dz={"widget_type": "FloatSpinBox",
                                    "value": 0.3, "step": 0.000000001},
                        angle={"widget_type": "FloatSpinBox",
                            "value": 30, "step": 0.1},
                        select_device={"widget_type": "ComboBox", "choices": cle.available_device_names(
                        ), "value": cle.available_device_names()[0]},
                        last_dimension_channel={"widget_type": "ComboBox", 
                                                "label": "Set Last dimension (channel/time)", "tooltip": "If the last dimension is initialised incorrectly, you can assign it as either channel/time"},
                        merge_all_channel_layers={"widget_type": "CheckBox", "value": True, "label": "Merge all napari layers as channels",
                                                "tooltip": "Use this option if the channels are in separate layers. napari-lattice requires all channels to be in same layer"},
                        skew_dir={"widget_type": "ComboBox", "choices": DeskewDirection, "value": DeskewDirection.Y,
                                "label": "Direction of skew (Y or X)", "tooltip": "Skew direction when image is acquired. Ask your microscopist for details"},
                        set_logging={"widget_type": "ComboBox", "choices": Log_Levels, "value": Log_Levels.INFO,
                                    "label": "Log Level", "tooltip": "Only use for debugging. Leave it as INFO for regular operation"}
                        )
        def Choose_Image_Layer(self,
                                img_layer: Layer,
                                pixel_size_dx: float = 0.1449922,
                                pixel_size_dy: float = 0.1449922,
                                pixel_size_dz: float = 0.3,
                                angle: float = 30,
                                select_device: str = cle.available_device_names()[
                                    0],
                                last_dimension_channel: LastDimensionOptions = LastDimensionOptions.get_from_metadata,
                                merge_all_channel_layers: bool = False,
                                skew_dir: DeskewDirection=DeskewDirection.Y,
                                set_logging: Log_Levels=Log_Levels.INFO):

            logger.setLevel(set_logging.value)
            config.log_level = set_logging.value
            logger.info(f"Logging set to {set_logging}")
            logger.info("Using existing image layer")

            # Select device for processing
            cle.select_device(select_device)

            # merge all napari image layers as one multidimensional image
            if merge_all_channel_layers:
                from napari.layers.utils.stack_utils import images_to_stack
                # get list of napari layers as a list
                layer_list = list(self.parent_viewer.layers)
                # if more than one layer
                if len(layer_list) > 1:
                    # convert the list of images into a stack
                    new_layer = images_to_stack(layer_list)
                    # select all current layers
                    self.parent_viewer.layers.select_all()
                    # remove selected layers
                    self.parent_viewer.layers.remove_selected()
                    # add the new composite image layer
                    self.parent_viewer.add_layer(new_layer)
                    img_layer = new_layer

            self.llsz_parent.lattice = lattice_from_napari(
                img=img_layer,
                last_dimension=None if last_dimension_channel == LastDimensionOptions.get_from_metadata else last_dimension_channel,
                angle=angle,
                skew=skew_dir,
                physical_pixel_sizes=(pixel_size_dx, pixel_size_dy, pixel_size_dz),
                # deconvolution = DeconvolutionParams()
            )
            # flag for ensuring a file has been opened and plugin initialised
            self.llsz_parent.open_file = True

            logger.info(
                f"Pixel size (ZYX) in microns: {self.llsz_parent.lattice.dz,self.llsz_parent.lattice.dy,self.llsz_parent.lattice.dx}")
            logger.info(
                f"Dimensions of image layer (ZYX): {list(self.llsz_parent.lattice.data.shape[-3:])}")
            logger.info(
                f"Dimensions of deskewed image (ZYX): {self.llsz_parent.lattice.deskew_vol_shape}")
            logger.info(
                f"Deskewing angle is: {self.llsz_parent.lattice.angle}")
            logger.info(
                f"Deskew Direction: {self.llsz_parent.lattice.skew}")
            # Add dimension labels correctly
            # if channel, and not time
            if self.llsz_parent.lattice.time == 0 and (last_dimension_channel or self.llsz_parent.lattice.channels > 0):
                self.parent_viewer.dims.axis_labels = ('Channel', "Z", "Y", "X")
            # if no channel, but has time
            elif self.llsz_parent.lattice.channels == 0 and self.llsz_parent.lattice.time > 0:
                self.parent_viewer.dims.axis_labels = ('Time', "Z", "Y", "X")
            # if it has channels
            elif self.llsz_parent.lattice.channels > 1:
                # If merge to stack is used, channel slider goes to the bottom
                if int(self.parent_viewer.dims.dict()["range"][0][1]) == self.llsz_parent.lattice.channels:
                    self.parent_viewer.dims.axis_labels = ('Channel', "Time", "Z", "Y", "X")
                else:
                    self.parent_viewer.dims.axis_labels = ('Time', "Channel", "Z", "Y", "X")
            # if channels initialized by aicsimagio, then channels is 1
            elif self.llsz_parent.lattice.channels == 1 and self.llsz_parent.lattice.time > 1:
                self.parent_viewer.dims.axis_labels = ('Time', "Z", "Y", "X")

            logger.info(f"Initialised")
            self["Choose_Image_Layer"].background_color = "green"
            self["Choose_Image_Layer"].text = "Plugin Initialised"

        # Pycudadecon library for deconvolution
        # options={"enabled": True},
        deconvolution = vfield(bool, name="Use Deconvolution")
        deconvolution.value = False

        @deconvolution.connect
        def _set_decon(self):
            if self.deconvolution:
                logger.info("Deconvolution Activated")
                # Enable deconvolutio by using the saved parameters
                self.llsz_parent.lattice.deconvolution = self.llsz_parent.deconv
            else:
                logger.info("Deconvolution Disabled")
                self.llsz_parent.lattice.deconvolution = None

        @set_design(background_color="magenta", font_family="Consolas", visible=True, text="Click to select PSFs for deconvolution", max_height=75, font_size=11)
        @set_options(header=dict(widget_type="Label", label="<h3>Enter path to the PSF images</h3>"),
                        psf_ch1_path={"widget_type": "FileEdit",
                                    "label": "Channel 1:"},
                        psf_ch2_path={"widget_type": "FileEdit",
                                    "label": "Channel 2"},
                        psf_ch3_path={"widget_type": "FileEdit",
                                    "label": "Channel 3"},
                        psf_ch4_path={"widget_type": "FileEdit",
                                    "label": "Channel 4"},
                        device_option={
                            "widget_type": "ComboBox", "label": "Choose processing device", "choices": DeconvolutionChoice},
                        no_iter={
                            "widget_type": "SpinBox", "label": "No of iterations (Deconvolution)", "value": 10, "min": 1, "max": 50, "step": 1}
                        )
        def deconvolution_gui(self,
                                header: str,
                                psf_ch1_path: Path,
                                psf_ch2_path: Path,
                                psf_ch3_path: Path,
                                psf_ch4_path: Path,
                                device_option: DeconvolutionChoice,
                                no_iter: int):
            """GUI for Deconvolution button"""
            # Force deconvolution to be true if we do this
            if not self.llsz_parent.lattice.deconvolution:
                raise Exception("Deconvolution is set to False. Tick the box to activate deconvolution.")
            self.llsz_parent.deconv.decon_processing = device_option
            self.llsz_parent.deconv.psf = list(read_psf([
                    psf_ch1_path,
                    psf_ch2_path,
                    psf_ch3_path,
                    psf_ch4_path,
                ],
                device_option,
                lattice_class=self.llsz_parent.lattice
            ))
            self.llsz_parent.deconv.psf_num_iter = no_iter
            self["deconvolution_gui"].background_color = "green"
            self["deconvolution_gui"].text = "PSFs added"

    @magicclass(widget_type="collapsible")
    class Preview:
        @magicgui(header=dict(widget_type="Label", label="<h3>Preview Deskew</h3>"),
                    time=dict(label="Time:", max=2**15),
                    channel=dict(label="Channel:"),
                    call_button="Preview")
        def Preview_Deskew(self,
                            header: str,
                            time: int,
                            channel: int,
                            img_data: ImageData):
            """
            Preview deskewed data for a single timepoint and channel

            """
            _Preview(LLSZWidget,
                        self,
                        time,
                        channel,
                        img_data)

    # Tabbed Widget container to house all the widgets
    @magicclass(widget_type="tabbed", name="Functions")
    class WidgetContainer(LlszTemplate):

        @magicclass(name="Deskew", widget_type="scrollable", properties={"min_width": 100})
        class DeskewWidget(LlszTemplate):

            @magicgui(header=dict(widget_type="Label", label="<h3>Deskew and Save</h3>"),
                        time_start=dict(label="Time Start:", max=2**20),
                        time_end=dict(label="Time End:", value=1, max=2**20),
                        ch_start=dict(label="Channel Start:"),
                        ch_end=dict(label="Channel End:", value=1),
                        save_as_type={
                            "label": "Save as filetype:", "choices": SaveFileType, "value": SaveFileType.h5},
                        save_path=dict(mode='d', label="Directory to save"),
                        call_button="Save")
            def Deskew_Save(self,
                            header: str,
                            time_start: int,
                            time_end: int,
                            ch_start: int,
                            ch_end: int,
                            save_as_type: str,
                            save_path: Path = Path(history.get_save_history()[0])):
                """ Widget to Deskew and Save Data"""
                _Deskew_Save(LLSZWidget,
                                time_start,
                                time_end,
                                ch_start,
                                ch_end,
                                save_as_type,
                                save_path)

        @magicclass(name="Crop and Deskew", widget_type="scrollable")
        class CropWidget(LlszTemplate):
            
            # add function for previewing cropped image
            @magicclass(name="Cropping Preview", widget_type="scrollable", properties={
                "min_width": 100,
                "shapes_layer": Shapes
            })
            class Preview_Crop_Menu(LlszTemplate):

                @set_design(font_size=10, text="Click to activate Cropping Layer", background_color="magenta")
                @click(enables=["Import_ImageJ_ROI", "Crop_Preview"])
                def activate_cropping(self):
                    self.llsz_parent.shapes_layer = self.parent_viewer.add_shapes(shape_type='polygon', edge_width=1, edge_color='white',
                                                                      face_color=[1, 1, 1, 0], name="Cropping BBOX layer")
                    # TO select ROIs if needed
                    self.llsz_parent.shapes_layer.mode = "SELECT"
                    self["activate_cropping"].text = "Cropping layer active"
                    self["activate_cropping"].background_color = "green"

                heading2 = field("You can either import ImageJ ROI (.zip) files or manually define ROIs using the shape layer", widget_type="Label")

                @click(enabled=False)
                def Import_ImageJ_ROI(self, path: Path = Path(history.get_open_history()[0])) -> None:
                    logger.info(f"Opening{path}")
                    roi_list = read_imagej_roi(str(path))
                    # convert to canvas coordinates
                    self.find_ancestor(LLSZWidget)
                    roi_list = (np.array(roi_list) * self.llsz_parent.lattice.dy).tolist()
                    self.llsz_parent.shapes_layer.add(roi_list, shape_type='polygon', edge_width=1, edge_color='yellow', face_color=[1, 1, 1, 0])

                time_crop = field(int, options={"min": 0, "step": 1, "max": 2**20}, name="Time")
                chan_crop = field(int, options={"min": 0, "step": 1}, name="Channels")
                heading_roi = field("If there are multiple ROIs, select the ROI before clicking button below", widget_type="Label")
                #roi_idx = field(int, options={"min": 0, "step": 1}, name="ROI number")

                @click(enabled=False)
                def Crop_Preview(self, roi_layer: ShapesData):

                    if not roi_layer:
                        raise Exception("No coordinates found for cropping. Check if right shapes layer or initialise shapes layer and draw ROIs.")
                    # TODO: Add assertion to check if bbox layer or coordinates

                    # Slice out the image of interest to preview
                    time = self.time_crop.value
                    channel = self.chan_crop.value
                    if time >= self.llsz_parent.lattice.time:
                        raise ValueError("Time is out of range")
                    if time >= self.llsz_parent.lattice.time:
                        raise ValueError("Channel is out of range")
                    logger.info(f"Using channel {channel} and time {time}")

                    # if only one roi drawn, use the first ROI for cropping
                    if len(self.llsz_parent.shapes_layer.selected_data) == 0:
                        raise Exception("Please select an ROI")

                    roi_idx = list(self.llsz_parent.shapes_layer.selected_data)[0]

                    # As the original image is scaled, the coordinates are in microns, so we need to convert
                    # roi from micron to canvas/world coordinates
                    roi_choice = [x/self.llsz_parent.lattice.dy for x in roi_layer[roi_idx]]
                    logger.info(f"Previewing ROI {roi_idx}")

                    # crop

                    # Set the deconvolution options
                    if self.llsz_parent.deconvolution:
                        if not self.llsz_parent.lattice.psf or not self.llsz_parent.lattice.psf_num_iter or not self.llsz_parent.lattice.decon_processing:
                            raise Exception(
                                "PSF fields should be set by this point!")
                        logger.info(
                            f"Deskewing for Time:{time} and Channel: {channel} with deconvolution")
                        decon_kwargs = dict(
                            decon_processing=self.llsz_parent.lattice.decon_processing.value,
                            psf=self.llsz_parent.lattice.psf[channel],
                            num_iter=self.llsz_parent.lattice.psf_num_iter
                        )
                    else:
                        decon_kwargs = dict()

                    crop_roi_vol_desk = cle.pull(
                        crop_volume_deskew(
                            original_volume=np.array(self.llsz_parent.lattice.data[time, channel, ...]),
                            roi_shape=roi_choice,
                            angle_in_degrees=self.llsz_parent.angle_value,
                            voxel_size_x=self.llsz_parent.lattice.dx,
                            voxel_size_y=self.llsz_parent.lattice.dy,
                            voxel_size_z=self.llsz_parent.lattice.dz,
                            deconvolution=self.llsz_parent.deconvolution,
                            # Option for entering custom z start value?
                            z_start=0,
                            z_end=self.llsz_parent.lattice.deskew_vol_shape[0],
                            skew_dir=self.llsz_parent.skew_dir,
                            **decon_kwargs
                        ).astype(self.llsz_parent.lattice.data.dtype)
                    )

                    # get array back from gpu or addding cle array to napari can throw errors
                    image = next(self.llsz_parent.lattice.process())

                    scale = (
                        self.llsz_parent.lattice.new_dz,
                        self.llsz_parent.lattice.dy,
                        self.llsz_parent.lattice.dx
                    )
                    self.parent_viewer.add_image(
                        crop_roi_vol_desk, scale=scale)

                @magicclass(name="Crop and Save Data")
                class CropSaveData(LlszTemplate):
                    @magicgui(header=dict(widget_type="Label", label="<h3>Crop and Save Data</h3>"),
                                time_start=dict(label="Time Start:"),
                                time_end=dict(label="Time End:", value=1),
                                ch_start=dict(label="Channel Start:"),
                                ch_end=dict(label="Channel End:", value=1),
                                save_as_type={
                                    "label": "Save as filetype:", "choices": SaveFileType},
                                save_path=dict(mode='d', label="Directory to save "))
                    def Crop_Save(self,
                                    header: str,
                                    time_start: int,
                                    time_end: int,
                                    ch_start: int,
                                    ch_end: int,
                                    save_as_type: SaveFileType,
                                    roi_layer_list: ShapesData,
                                    save_path: Path = Path(history.get_save_history()[0])):

                        if not roi_layer_list:
                            logger.error(
                                "No coordinates found or cropping. Initialise shapes layer and draw ROIs.")
                        else:
                            if not self.llsz_parent.open_file:
                                raise Exception("Image not initialised")

                            check_dimensions(time_start, time_end, ch_start, ch_end, self.llsz_parent.lattice.channels, self.llsz_parent.lattice.time)

                            angle = self.llsz_parent.lattice.angle
                            dx = self.llsz_parent.lattice.dx
                            dy = self.llsz_parent.lattice.dy
                            dz = self.llsz_parent.lattice.dz

                            # get image data
                            img_data = self.llsz_parent.lattice.data
                            # Get shape of deskewed image
                            deskewed_shape = self.llsz_parent.lattice.deskew_vol_shape
                            deskewed_volume = da.zeros(deskewed_shape)
                            z_start = 0
                            z_end = deskewed_shape[0]

                            logger.info("Cropping and saving files...")

                            # necessary when scale is used for napari.viewer.add_image operations
                            roi_layer_list = ShapesData([x/self.llsz_parent.lattice.dy for x in roi_layer_list])

                            for idx, roi_layer in enumerate(tqdm(roi_layer_list, desc="ROI:", position=0)):
                                # pass arguments for save tiff, callable and function arguments
                                logger.info("Processing ROI ", idx)
                                # pass parameters for the crop_volume_deskew function

                                save_img(vol=img_data,
                                            func=crop_volume_deskew,
                                            time_start=time_start,
                                            time_end=time_end,
                                            channel_start=ch_start,
                                            channel_end=ch_end,
                                            save_name_prefix="ROI_" + str(idx),
                                            save_path=save_path,
                                            save_file_type=save_as_type,
                                            save_name=self.llsz_parent.lattice.save_name,
                                            dx=dx,
                                            dy=dy,
                                            dz=dz,
                                            angle=angle,
                                            deskewed_volume=deskewed_volume,
                                            roi_shape=roi_layer,
                                            angle_in_degrees=angle,
                                            z_start=z_start,
                                            z_end=z_end,
                                            voxel_size_x=dx,
                                            voxel_size_y=dy,
                                            voxel_size_z=dz,
                                            LLSZWidget=self.llsz_parent
                                            )

                            logger.info(
                                f"Cropping and Saving Complete -> {save_path}")

        @magicclass(name="Workflow", widget_type="scrollable")
        class WorkflowWidget(LlszTemplate):

            @magicclass(name="Preview Workflow", widget_type="scrollable")
            class PreviewWorkflow(LlszTemplate):
                #time_preview= field(int, options={"min": 0, "step": 1}, name="Time")
                #chan_preview = field(int, options={"min": 0, "step": 1}, name="Channels")
                @magicgui(header=dict(widget_type="Label", label="<h3>Preview Workflow</h3>"),
                            time_preview=dict(label="Time:", max=2**20),
                            chan_preview=dict(label="Channel:"),
                            get_active_workflow=dict(
                                widget_type="Checkbox", label="Get active workflow in napari-workflow", value=False),
                            workflow_path=dict(
                                mode='r', label="Load custom workflow (.yaml/yml)"),
                            Use_Cropping=dict(
                                widget_type="Checkbox", label="Crop Data", value=False),
                            #custom_module=dict(widget_type="Checkbox",label="Load custom module (looks for *.py files in the workflow directory)",value = False),
                            call_button="Apply and Preview Workflow")
                def Workflow_Preview(self,
                                        header: str,
                                        time_preview: int,
                                        chan_preview: int,
                                        get_active_workflow: bool,
                                        Use_Cropping: bool,
                                        roi_layer_list: ShapesData,
                                        workflow_path: Path = Path.home()):
                    """
                    Apply napari_workflows to the processing pipeline
                    User can define a pipeline which can be inspected in napari workflow inspector
                    and then execute it by ticking  the get active workflow checkbox, 
                    OR
                    Use a predefined workflow

                    In both cases, if deskewing is not present as first step, it will be added on
                    and rest of the task will be made followers
                    """
                    logger.info("Previewing deskewed channel and time with workflow")
                    user_workflow = get_workflow(self.parent_viewer if get_active_workflow else workflow_path)

                    # when using fields, self.time_preview.value
                    if time_preview >= self.llsz_parent.lattice.time:
                        raise ValueError("Time is out of range")
                    if chan_preview >= self.llsz_parent.lattice.channels:
                        raise ValueError("Channel is out of range")

                    time = time_preview
                    channel = chan_preview

                    # to access current time and channel and pass it to workflow file
                    config.channel = channel
                    config.time = time

                    logger.info(f"Processing for Time: {time} and Channel: {channel}")

                    logger.info("Workflow to be executed:")
                    logger.info(user_workflow)
                    # Execute workflow
                    processed_vol = user_workflow.get(task_name_last)

                    # check if a measurement table (usually a dictionary or list)  or a tuple with different data types
                    # The function below saves the tables and adds any images to napari window
                    if type(processed_vol) in [dict, list, tuple]:
                        if (len(processed_vol) > 1):
                            df = pd.DataFrame()
                            for idx, i in enumerate(processed_vol):
                                df_temp = process_custom_workflow_output(
                                    i, parent_dir, idx, LLSZWidget, self, channel, time, preview=True)
                                final_df = pd.concat([df, df_temp])
                                # append dataframes from every loop and have table command outside loop?
                            # TODO: Figure out why table is not displaying
                            from napari_spreadsheet import _widget
                            table_viewer = _widget.TableViewerWidget(
                                show=True)
                            table_viewer.add_spreadsheet(final_df)
                            # widgets.Table(value=final_df).show()

                    else:
                        # add image to napari window
                        # TODO: check if its an image napari supports?
                        process_custom_workflow_output(
                            processed_vol, parent_dir, 0, LLSZWidget, self, channel, time)

                    print("Workflow complete")

                @magicgui(header=dict(widget_type="Label", label="<h3>Apply Workflow and Save Output</h3>"),
                            time_start=dict(label="Time Start:", max=2**20),
                            time_end=dict(label="Time End:",
                                        value=1, max=2**20),
                            ch_start=dict(label="Channel Start:"),
                            ch_end=dict(label="Channel End:", value=1),
                            Use_Cropping=dict(
                                widget_type="Checkbox", label="Crop Data", value=False),
                            get_active_workflow=dict(
                                widget_type="Checkbox", label="Get active workflow in napari-workflow", value=False),
                            workflow_path=dict(
                                mode='r', label="Load custom workflow (.yaml/yml)"),
                            save_as_type={
                                "label": "Save as filetype:", "choices": SaveFileType},
                            save_path=dict(
                                mode='d', label="Directory to save "),
                            #custom_module=dict(widget_type="Checkbox",label="Load custom module (same dir as workflow)",value = False),
                            call_button="Apply Workflow and Save Result")
                def Apply_Workflow_and_Save(self,
                                            header: str,
                                            time_start: int,
                                            time_end: int,
                                            ch_start: int,
                                            ch_end: int,
                                            Use_Cropping: bool,
                                            roi_layer_list: ShapesData,
                                            get_active_workflow: bool = False,
                                            workflow_path: Path = Path.home(),
                                            save_as_type: str = SaveFileType.tiff,
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
                    if not self.llsz_parent.open_file:
                        raise Exception("Image not initialised")

                    check_dimensions(time_start, time_end, ch_start, ch_end,
                                        self.llsz_parent.lattice.channels, self.llsz_parent.lattice.time)

                    # Get parameters
                    angle = self.llsz_parent.lattice.angle
                    dx = self.llsz_parent.lattice.dx
                    dy = self.llsz_parent.lattice.dy
                    dz = self.llsz_parent.lattice.dz

                    user_workflow = get_workflow(self.parent_viewer if get_active_workflow else workflow_path)
                        
                    input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(
                        user_workflow)
                    logger.info(f"{input_arg_first=}, {input_arg_last=}, {first_task_name=}, {last_task_name=}")
                    logger.info(f"Workflow loaded: {user_workflow}")

                    vol = self.llsz_parent.lattice.data
                    task_name_start = first_task_name[0]

                    try:
                        task_name_last = last_task_name[0]
                    except IndexError:
                        task_name_last = task_name_start

                    # variables to hold task name, initialize it as None
                    # if gpu, set otf_path, otherwise use psf
                    psf = None
                    otf_path = None

                    if self.llsz_parent.lattice.decon_processing == DeconvolutionChoice.cuda_gpu:
                        #otf_path = "otf_path"
                        psf_arg = "psf"
                        psf = self.llsz_parent.lattice.psf
                    else:
                        psf_arg = "psf"
                        psf = self.llsz_parent.lattice.psf
                    # if cropping, set that as first task

                    if Use_Cropping:
                        # convert Roi pixel coordinates to canvas coordinates
                        # necessary only when scale is used for napari.viewer.add_image operations
                        roi_layer_list = [x/self.llsz_parent.lattice.dy for x in roi_layer_list]

                        deskewed_shape = self.llsz_parent.lattice.deskew_vol_shape
                        deskewed_volume = da.zeros(deskewed_shape)
                        z_start = 0
                        z_end = deskewed_shape[0]
                        roi = "roi"
                        volume = "volume"
                        # Check if decon ticked, if so set as first and crop as second?

                        # Create workflow for cropping and deskewing
                        # volume and roi used will be set dynamically
                        user_workflow.set("crop_deskew_image", crop_volume_deskew,
                                            original_volume=volume,
                                            deskewed_volume=deskewed_volume,
                                            roi_shape=roi,
                                            angle_in_degrees=angle,
                                            voxel_size_x=dx,
                                            voxel_size_y=dy,
                                            voxel_size_z=dz,
                                            z_start=z_start,
                                            z_end=z_end,
                                            deconvolution=self.llsz_parent.deconvolution.value,
                                            decon_processing=self.llsz_parent.lattice.decon_processing,
                                            psf=psf_arg,
                                            skew_dir=self.llsz_parent.skew_dir)

                        # change the first task so it accepts "crop_deskew as input"
                        new_task = modify_workflow_task(
                            old_arg=input_arg_first, task_key=task_name_start, new_arg="crop_deskew_image", workflow=user_workflow)
                        user_workflow.set(task_name_start, new_task)

                        for idx, roi_layer in enumerate(tqdm(roi_layer_list, desc="ROI:", position=0)):
                            print("Processing ROI ", idx)
                            user_workflow.set(roi, roi_layer)
                            save_img_workflow(vol=vol,
                                                workflow=user_workflow,
                                                input_arg=volume,
                                                first_task="crop_deskew_image",
                                                last_task=task_name_last,
                                                time_start=time_start,
                                                time_end=time_end,
                                                channel_start=ch_start,
                                                channel_end=ch_end,
                                                save_file_type=save_as_type,
                                                save_path=save_path,
                                                #roi_layer = roi_layer,
                                                save_name_prefix="ROI_" + \
                                                str(idx),
                                                save_name=self.llsz_parent.lattice.save_name,
                                                dx=dx,
                                                dy=dy,
                                                dz=dz,
                                                angle=angle,
                                                deconvolution=self.llsz_parent.deconvolution.value,
                                                decon_processing=self.llsz_parent.lattice.decon_processing,
                                                otf_path=otf_path,
                                                psf_arg=psf_arg,
                                                psf=psf)

                    # IF just deskewing and its not in the tasks, add that as first task
                    elif user_workflow.get_task(task_name_start)[0] not in (cle.deskew_y, cle.deskew_x):
                        input = "input"
                        # add task to the workflow
                        user_workflow.set("deskew_image",
                                            self.llsz_parent.deskew_func,
                                            input_image=input,
                                            angle_in_degrees=angle,
                                            voxel_size_x=dx,
                                            voxel_size_y=dy,
                                            voxel_size_z=dz,
                                            linear_interpolation=True)
                        # Set input of the workflow to be from deskewing
                        # change workflow task starts from is "deskew_image" and
                        new_task = modify_workflow_task(
                            old_arg=input_arg_first, task_key=task_name_start, new_arg="deskew_image", workflow=user_workflow)
                        user_workflow.set(task_name_start, new_task)

                        # if deconvolution checked, add it to start of workflow (add upstream of deskewing)
                        if self.llsz_parent.deconvolution:
                            psf = "psf"
                            otf_path = "otf_path"
                            input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(
                                user_workflow)

                            if self.llsz_parent.lattice.decon_processing == DeconvolutionChoice.cuda_gpu:
                                user_workflow.set("deconvolution",
                                                    pycuda_decon,
                                                    image=input,
                                                    psf=psf_arg,
                                                    dzdata=self.llsz_parent.lattice.dz,
                                                    dxdata=self.llsz_parent.lattice.dx,
                                                    dzpsf=self.llsz_parent.lattice.dz,
                                                    dxpsf=self.llsz_parent.lattice.dx,
                                                    num_iter=self.llsz_parent.lattice.psf_num_iter)
                                # user_workflow.set(input_arg_first,"deconvolution")
                            else:
                                user_workflow.set("deconvolution",
                                                    skimage_decon,
                                                    vol_zyx=input,
                                                    psf=psf_arg,
                                                    num_iter=self.llsz_parent.lattice.psf_num_iter,
                                                    clip=False,
                                                    filter_epsilon=0,
                                                    boundary='nearest')
                            # modify the user workflow so "deconvolution" is accepted
                            new_task = modify_workflow_task(
                                old_arg=input_arg_first, task_key=task_name_start, new_arg="deconvolution", workflow=user_workflow)
                            user_workflow.set(task_name_start, new_task)
                            input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(
                                user_workflow)
                            task_name_start = first_task_name[0]

                        save_img_workflow(vol=vol,
                                            workflow=user_workflow,
                                            input_arg=input,
                                            first_task=task_name_start,
                                            last_task=task_name_last,
                                            time_start=time_start,
                                            time_end=time_end,
                                            channel_start=ch_start,
                                            channel_end=ch_end,
                                            save_file_type=save_as_type,
                                            save_path=save_path,
                                            save_name=self.llsz_parent.lattice.save_name,
                                            dx=dx,
                                            dy=dy,
                                            dz=dz,
                                            angle=angle,
                                            deconvolution=self.llsz_parent.deconvolution,
                                            decon_processing=self.llsz_parent.lattice.decon_processing,
                                            otf_path=otf_path,
                                            psf_arg=psf_arg,
                                            psf=psf)

                    # If deskewing is already as a task, then set the first argument to input so we can modify that later
                    else:
                        # if deskewing is already first task, then check if deconvolution needed
                        # if deconvolution checked, add it to start of workflow (add upstream of deskewing)
                        if self.llsz_parent.deconvolution:
                            psf = "psf"
                            otf_path = "otf_path"
                            input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(
                                user_workflow)

                            if self.llsz_parent.lattice.decon_processing == DeconvolutionChoice.cuda_gpu:
                                user_workflow.set("deconvolution",
                                                    pycuda_decon,
                                                    image=input,
                                                    psf=psf_arg,
                                                    dzdata=self.llsz_parent.lattice.dz,
                                                    dxdata=self.llsz_parent.lattice.dx,
                                                    dzpsf=self.llsz_parent.lattice.dz,
                                                    dxpsf=self.llsz_parent.lattice.dx,
                                                    num_iter=self.llsz_parent.lattice.psf_num_iter)
                                # user_workflow.set(input_arg_first,"deconvolution")
                            else:
                                user_workflow.set("deconvolution",
                                                    skimage_decon,
                                                    vol_zyx=input,
                                                    psf=psf_arg,
                                                    num_iter=self.llsz_parent.lattice.psf_num_iter,
                                                    clip=False,
                                                    filter_epsilon=0,
                                                    boundary='nearest')
                            # modify the user workflow so "deconvolution" is accepted
                            new_task = modify_workflow_task(
                                old_arg=input_arg_first, task_key=task_name_start, new_arg="deconvolution", workflow=user_workflow)
                            user_workflow.set(task_name_start, new_task)
                            input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(
                                user_workflow)
                            task_name_start = first_task_name[0]

                        # we pass first argument as input
                        save_img_workflow(vol=vol,
                                            workflow=user_workflow,
                                            input_arg=input_arg_first,
                                            first_task=task_name_start,
                                            last_task=task_name_last,
                                            time_start=time_start,
                                            time_end=time_end,
                                            channel_start=ch_start,
                                            channel_end=ch_end,
                                            save_file_type=save_as_type,
                                            save_path=save_path,
                                            save_name=self.llsz_parent.lattice.save_name,
                                            dx=dx,
                                            dy=dy,
                                            dz=dz,
                                            angle=angle,
                                            deconvolution=self.llsz_parent.deconvolution,
                                            decon_processing=self.llsz_parent.lattice.decon_processing,
                                            otf_path=otf_path,
                                            psf_arg=psf_arg,
                                            psf=psf)

                    print("Workflow complete")
                    return


def _napari_lattice_widget_wrapper() -> LLSZWidget:
    # split widget type enables a resizable widget
    #max_height = 50
    # Important to have this or napari won't recognize the classes and magicclass qidgets
    widget = LLSZWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    widget._widget._layout.setAlignment(Qt.AlignTop)

    # widget._widget._layout.setWidgetResizable(True)
    return widget

def get_workflow(source: Union[Path, Viewer]) -> Workflow:
    """
    Gets a user defined workflow object, either from a viewer or from a file

    Args:
        source: Either the path to a workflow file, or a Napari viewer from which to extract the workflow
    """
    if isinstance(source, Viewer):
        # installs the workflow to napari
        user_workflow = WorkflowManager.install(source).workflow
        logger.info("Workflow installed")
    else:
        import_workflow_modules(source)
        user_workflow = load_workflow(str(source))

    if not isinstance(user_workflow, Workflow):
        raise Exception("Workflow file is not a napari workflow object. Check file! You can use workflow inspector if needed")

    logger.info(f"Workflow loaded: {user_workflow}")
    return user_workflow
