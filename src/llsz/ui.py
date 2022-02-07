#UI for reading files, deskewing and cropping

import os
from pathlib import Path
import aicsimageio
from magicclass.wrappers import set_design
from magicgui import magicgui
from magicclass import magicclass, click, field, vfield

import numpy as np
from napari.types import ImageData,ShapesData, LayerDataTuple
from napari_plugin_engine import napari_hook_implementation
from napari.utils import progress, history
#from scipy.ndimage.measurements import label
from tqdm import tqdm

from llsz.array_processing import get_deskew_arr
from llsz.transformations import apply_deskew_transformation
from llsz.io import LatticeData,LatticeData_czi
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.types import PhysicalPixelSizes
from llsz.crop_utils import crop_deskew_roi
from llsz.utils import suppress_stdout_stderr
from napari_time_slicer import time_slicer
from napari_tools_menu import register_function
import pyclesperanto_prototype as cle
from qtpy.QtCore import Qt

from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLayout, QLabel, QTabWidget, QScrollArea
from qtpy.QtWidgets import QFileDialog
from qtpy import uic
from magicgui.widgets import create_widget
from napari.layers import Labels, Image

import os


class LLSZWidget_designer(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.UI_FILE = os.path.join(os.path.dirname(__file__), 'dock_widget.ui')
        uic.loadUi(self.UI_FILE, self)

        # allocate member attributes
        self.skew_dir = ''
        self.img_data = None
        self.dask = False  # Use GPU by default
        self.lattice = None

        self.time_deskew = 0
        self.chan_deskew = 0

        # connect buttons
        self.btn_open_czi_file.clicked.connect(self._open_a_czi_File)
        self.spinbox_deskew_angle.valueChanged.connect(self._set_angle)
        self.checkbox_use_gpu.stateChanged.connect(self._toggle_GPU)

        # create input dropdowns
        # self.combobox_img_data.activated.connect(lambda: self.populate_dropdown(self.combobox_img_data, Image))

    def populate_dropdown(self, dd_widget, layertype):
        """
        Populate entries in a dropdown widget from available layers in the viewer
        """

        layers = self.viewer.layers
        layers = [layer.name for layer in self.viewer.layers if isinstance(layer, layertype)]
        dd_widget.addItems(layers)

        return


    def Preview_Deskew(self):
        """
        Preview the deskewing for a single timepoint.
        """

        self.time_deskew = self.spinbox_time_deskew.value
        self.chan_deskew = self.spinbox_channels_deskew.value

        # create function-internal variable reference
        t_deskew = self.time_deskew
        c_deskew = self.chan_deskew

        print("Previewing deskewed channel and time")
        assert img_data.size, "No image open or selected"
        assert self.time_deskew.value < self.lattice.time, "Time is out of range"
        assert self.chan_deskew.value < self.lattice.channels, "Channel is out of range"
        #stack=self.aics

        assert str.upper(self.lattice.skew) in ('Y','X'), "Skew direction not recognised. Enter either Y or X"

        print("Deskewing for Time:", t_deskew, "and Channel", c_deskew )

        #get user-specified 3D volume
        print(img_data.shape)
        if(len(img_data.shape)==3):
            raw_vol = img_data
        else:
            raw_vol = img_data[time_deskew, chan_deskew,:,:,:]

        #Deskew using pyclesperanto
        deskew_final = cle.deskew_y(raw_vol,
                                    angle_in_degrees=self.lattice.angle,
                                    voxel_size_x=self.lattice.dx,
                                    voxel_size_y=self.lattice.dy,
                                    voxel_size_z=self.lattice.dz)

        #TODO: Use dask
        if self.dask:
            print("Using CPU for deskewing")
            #use cle library for affine transforms, but use dask and scipy
            #deskew_final = deskew_final.compute()
        max_proj_deskew = np.max(deskew_final,axis=0)

        #add channel and time information to the name
        suffix_name = "_c" + str(c_deskew) + "_t" + str(t_deskew)

        self.viewer.add_image(max_proj_deskew, name="Deskew_MIP")

        #img_name="Deskewed image_c"+str(chan_deskew)+"_t"+str(time_deskew)
        self.viewer.add_image(deskew_final,name="Deskewed image"+suffix_name)
        self.viewer.layers[0].visible = False
        print("Deskewing complete")
        return
        #return (deskew_full, {"name":"Uncropped data"})
        #(deskew_final, {"name":img_name})

    def _toggle_GPU(self):
        if self.checkbox_use_gpu.isChecked():
            self.dask = True
        else:
            self.dask = False


    def _set_angle(self):

        if self.lattice is not None:
            self.lattice.set_angle(self.spinbox_deskew_angle.value)
            print("Angle is: ",self.lattice.angle)
        else:
            print("Open a file first before setting angles")
        return


    def _open_a_czi_File(self):

        filename, _ = QFileDialog.getOpenFileName(self, 'Open file',
                                                  filter='*.czi')

        if filename is None:
            return

        self.lattice = LatticeData_czi(filename, 30.0, "Y")
        self.aics = self.lattice.data

        self.file_name = os.path.splitext(os.path.basename(filename))[0]
        self.save_name = os.path.splitext(os.path.basename(filename))[0]

        self.viewer.add_image(self.aics.dask_data)
        self.populate_dropdown(self.combobox_img_data, Image)

    def Choose_Existing_Layer(self,
                              img_data: ImageData,
                              pixel_size_dx: float,
                              pixel_size_dy: float,
                              pixel_size_dz: float,
                              skew_dir: str):
        print("Using existing image layer")
        skew_dir = str.upper(skew_dir)
        assert skew_dir in ('Y','X'), "Skew direction not recognised. Enter either Y or X"
        self.lattice=LatticeData(img_data, 30.0, skew_dir,pixel_size_dx,pixel_size_dy,pixel_size_dz)
        self.aics = self.lattice.data
        self["Choose_Existing_Layer"].background_color = "green"
        lattice = self.lattice





def plugin_wrapper():
    @magicclass(widget_type="scrollable", name ="LLSZ analysis")
    class LLSZWidget:
        @magicclass(widget_type="frame")#, close_on_run=False
        class llsz_menu:

            open_file = False

            @set_design(background_color="orange", font_family="Consolas",visible=True)
            @click(hides="Choose_Existing_Layer")
            def Open_a_czi_File(self, path:Path = Path(history.get_open_history()[0])):
                print("Opening", path)
                #update the napari settings to use the opened file path as last opened path
                history.update_open_history(path.__str__())
                self.lattice=LatticeData_czi(path, 30.0, "Y")
                self.aics = self.lattice.data
                self.file_name = os.path.splitext(os.path.basename(path))[0]
                self.save_name=os.path.splitext(os.path.basename(path))[0]
                self.parent_viewer.add_image(self.aics.dask_data)
                self["Open_a_czi_File"].background_color = "green"
                self.dask = False #Use GPU by default
                self.open_file = True #if open button used
                lattice = self.lattice

            #Display text
            @click(enabled=False)
            def OR(self):
                pass

            @set_design(background_color="magenta", font_family="Consolas",visible=True)
            @click(hides="Open_a_czi_File")
            def Choose_Existing_Layer(self, img_data:ImageData,pixel_size_dx:float,pixel_size_dy:float,pixel_size_dz:float,skew_dir:str):
                print("Using existing image layer")
                skew_dir = str.upper(skew_dir)
                assert skew_dir in ('Y','X'), "Skew direction not recognised. Enter either Y or X"
                self.lattice=LatticeData(img_data, 30.0, skew_dir,pixel_size_dx,pixel_size_dy,pixel_size_dz)
                self.aics = self.lattice.data
                self["Choose_Existing_Layer"].background_color = "green"
                self.dask = False #Use GPU by default
                self.open_file = True #if open button used
                lattice = self.lattice

            #Enter custom angle if needed
            #Will only update after choosing an image
            angle = vfield(float,options={"value": 30.0},name = "Deskew Angle")
            @angle.connect
            def _set_angle(self):
                try:
                    self.lattice.set_angle(self.angle)
                    print("Angle is: ",self.lattice.angle)
                except AttributeError:
                    print("Open a file first before setting angles")
                return

            @magicgui(labels = False,auto_call=True)
            def Use_GPU(self, Use_GPU:bool = True):
                """Choose to use GPU or Dask

                Args:
                    Use_GPU (bool, optional): Defaults to True.
                """
                print("Use GPU set to, ", Use_GPU)
                self.dask = not(Use_GPU)
                return


            time_deskew = field(int, options={"min": 0,  "step": 1},name = "Time")
            chan_deskew = field(int, options={"min": 0,  "step": 1},name = "Channels")

            @magicgui(header=dict(widget_type="Label",label="<h3>Preview Deskew</h3>"), call_button = "Preview")
            def Preview_Deskew(self, header, img_data:ImageData):
                """
                Preview the deskewing for a single timepoint

                Args:
                    header ([type]): [description]
                    img_data (ImageData): [description]
                """
                print("Previewing deskewed channel and time")
                assert img_data.size, "No image open or selected"
                assert self.time_deskew.value < self.lattice.time, "Time is out of range"
                assert self.chan_deskew.value < self.lattice.channels, "Channel is out of range"
                time_deskew = self.time_deskew.value
                chan_deskew = self.chan_deskew.value
                #stack=self.aics
                angle=self.lattice.angle

                assert str.upper(self.lattice.skew) in ('Y','X'), "Skew direction not recognised. Enter either Y or X"

                print("Deskewing for Time:",time_deskew, "and Channel", chan_deskew )

                #get user-specified 3D volume
                print(img_data.shape)
                if(len(img_data.shape)==3):
                    raw_vol = img_data
                else:
                    raw_vol = img_data[time_deskew, chan_deskew,:,:,:]

                #Deskew using pyclesperanto
                deskew_final = cle.deskew_y(raw_vol, angle_in_degrees=angle,voxel_size_x=self.lattice.dx,voxel_size_y=self.lattice.dy,voxel_size_z=self.lattice.dz)

                #TODO: Use dask
                if self.dask:
                    print("Using CPU for deskewing")
                    #use cle library for affine transforms, but use dask and scipy
                    #deskew_final = deskew_final.compute()
                max_proj_deskew=np.max(deskew_final,axis=0)

                #add channel and time information to the name
                suffix_name = "_c"+str(chan_deskew)+"_t"+str(time_deskew)

                self.parent_viewer.add_image(max_proj_deskew,name="Deskew_MIP")

                #img_name="Deskewed image_c"+str(chan_deskew)+"_t"+str(time_deskew)
                self.parent_viewer.add_image(deskew_final,name="Deskewed image"+suffix_name)
                self.parent_viewer.layers[0].visible = False
                print("Deskewing complete")
                return
                #return (deskew_full, {"name":"Uncropped data"})
                #(deskew_final, {"name":img_name})

        @magicclass(widget_type="collapsible", name="Preview Crop")
        class llsz_menu3:

            #@click(enables ="Crop_Preview")
            @magicgui(call_button="Initialise shapes layer")
            def Initialize_Shapes_Layer(self):
                self.parent_viewer.add_shapes(shape_type='polygon', edge_width=5,edge_color='white',face_color=[1,1,1,0],name="Cropping BBOX layer")
                return

            time_crop = field(int, options={"min": 0,  "step": 1},name = "Time")
            chan_crop = field(int, options={"min": 0,  "step": 1},name = "Channels")
            @magicgui
            def Crop_Preview(self,roi_layer:ShapesData):# -> LayerDataTuple:
                if not roi_layer:
                    print("No coordinates found or cropping. Initialise shapes layer and draw ROIs.")
                else:
                    #TODO: Add assertion to check if bbox layer or coordinates
                    print("Using channel and time", self.chan_crop.value,self.time_crop.value)
                    #if passing roi layer as layer, use roi.data
                    #rotate around deskew_vol_shape
                    #going back from shape of deskewed volume to original for cropping
                    crop_roi_vol=crop_deskew_roi(roi_layer[0],self.lattice.deskew_vol_shape,self.aics.dask_data,self.lattice.angle,self.lattice.dy,self.lattice.dz,
                                                 self.lattice.deskew_z_start,self.lattice.deskew_z_end,self.time_crop.value,self.chan_crop.value,self.lattice.skew,reverse=True)
                    translate_x=int(roi_layer[0][0][0])
                    translate_y=int(roi_layer[0][0][1])
                    #crop_img_layer= (crop_roi_vol , #{'translate' : [0,translate_x,translate_y] })
                    self.parent_viewer.add_image(crop_roi_vol,translate = (0,translate_x,translate_y))
                return

        @magicclass(widget_type="collapsible", name="Save Data")
        class llsz_menu4:

            @magicgui(time_start = dict(label="Time Start:"),
                       time_end = dict(label="Time End:", value =1 ),
                       ch_start = dict(label="Channel Start:"),
                       ch_end = dict(label="Channel End:", value =1 ),
                       save_path = dict(mode ='d',label="Directory to save "))
            def Deskew_Save(self, time_start:int, time_end:int, ch_start:int, ch_end:int, save_path:Path = Path(history.get_save_history()[0])):
                assert self.open_file, "Image not initialised"
                assert time_start>=0, "Time start should be >0"
                assert time_end < self.lattice.time and time_end >0, "Check time entry "
                assert ch_start >= 0, "Channel start should be >0"
                assert ch_end <= self.lattice.channels and ch_end >= 0 , "Channel end should be less than "+str(self.lattice.channels)

                time_range = range(time_start, time_end)
                channel_range = range(ch_start, ch_end)

                angle=self.lattice.angle
                shear_factor = self.lattice.shear_factor
                scaling_factor = self.lattice.scaling_factor

                #Convert path to string
                save_path = save_path.__str__()

                #save channel/s for each timepoint.
                #TODO: Check speed -> Channel and then timepoint or vice versa, which is faster?
                for time_point in tqdm(time_range, desc = "Time", position=0):
                    images_array=[]
                    for ch in tqdm(channel_range, desc = "Channels", position=1,leave = False):

                        deskew_img=get_deskew_arr(self.aics, self.lattice.deskew_shape, self.lattice.deskew_vol_shape, time= time_point, channel=ch, scene=0, skew_dir=self.lattice.skew)
                        #Perform deskewing on the skewed dask array
                        deskew_full=apply_deskew_transformation(deskew_img,angle,shear_factor,scaling_factor,self.lattice.deskew_translate_y,reverse=False,dask=self.dask)
                        #Crop the z slices to get only the deskewed array and not the empty area
                        deskew_final=deskew_full[self.lattice.deskew_z_start:self.lattice.deskew_z_end].astype('uint16')
                        images_array.append(deskew_final)

                    images_array=np.array(images_array) #convert to array of arrays
                    #images_array is in the format CZYX, but when saving for imagej using tifffile, it should be TZCYXS; may need to add a flag for this
                    #images_array=np.swapaxes(images_array,0,1)
                    final_name=save_path+os.sep+"C"+str(ch)+"T"+str(time_point)+"_"+self.save_name+".ome.tif"
                    aics_image_pixel_sizes = PhysicalPixelSizes(self.lattice.dz,self.lattice.dy,self.lattice.dx)
                    OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes = aics_image_pixel_sizes)

                history.update_save_history(final_name)
                print("Deskewing and Saving Complete -> ",save_path)
                return

        @magicclass(widget_type="collapsible", name="Crop and Save Data")
        class llsz_menu5:

            @magicgui(time_start = dict(label="Time Start:"),
                       time_end = dict(label="Time End:", value =1 ),
                       ch_start = dict(label="Channel Start:"),
                       ch_end = dict(label="Channel End:", value =1 ),
                       save_path = dict(mode ='d',label="Directory to save "))
            def Crop_Save(self, time_start:int, time_end:int, ch_start:int, ch_end:int, roi_layer_list:ShapesData, save_path:Path = Path(history.get_save_history()[0])):
                if not roi_layer_list:
                    print("No coordinates found or cropping. Initialise shapes layer and draw ROIs.")
                else:
                    assert self.open_file, "Image not initialised"
                    assert time_start>=0, "Time start should be >0"
                    assert time_end < self.lattice.time and time_end >0, "Check time entry "
                    assert ch_start >= 0, "Channel start should be >0"
                    assert ch_end <= self.lattice.channels and ch_end >= 0 , "Channel end should be less than "+str(self.lattice.channels)

                    time_range = range(time_start, time_end)
                    channel_range = range(ch_start, ch_end)
                    angle=self.lattice.angle
                    #Convert path to string
                    save_path = save_path.__str__()
                    #save channel/s for each timepoint.
                    #TODO: Check speed -> Channel and then timepoint or vice versa, which is faster?
                    print("Cropping and saving files...")
                    for idx,roi_layer in enumerate(tqdm(roi_layer_list, desc = "ROI:", position=0)):
                        for time_point in tqdm(time_range, desc = "Time:", position=1,leave = False):
                            images_array=[]
                            for ch in tqdm(channel_range, desc = "Channels:", position=2,leave = False):
                                #suppress any printing to console
                                with suppress_stdout_stderr():
                                    crop_roi_vol=crop_deskew_roi(roi_layer,self.lattice.deskew_vol_shape,self.aics.dask_data,angle,self.lattice.dy,self.lattice.dz,
                                                                    self.lattice.deskew_z_start,self.lattice.deskew_z_end,time_point,ch,self.lattice.skew,reverse=True)
                                images_array.append(crop_roi_vol)
                            images_array=np.array(images_array) #convert to array of arrays
                            #images_array is in the format TCZYX, but when saving for imagej, it should be TZCYXS; may need to add a flag for this
                            # TODO: Add flag for saving in imagej format and options for other file formats
                            #images_array=np.swapaxes(images_array,0,1)
                            final_name=save_path+os.sep+"ROI_"+str(idx)+"_C"+str(ch)+"T"+str(time_point)+"_"+self.save_name+".ome.tif"
                            #create aicsimageio physical pixel size variable using PhysicalPixelSizes class
                            aics_image_pixel_sizes = PhysicalPixelSizes(self.lattice.dz,self.lattice.dy,self.lattice.dx)
                            OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes = aics_image_pixel_sizes )
                    history.update_save_history(final_name)
                    print("Cropping and Saving Complete -> ",save_path)
                    return

    widget = LLSZWidget()
    # aligning collapsible widgets at the top instead of having them centered vertically
    widget._widget._layout.setAlignment(Qt.AlignTop)

    return widget






#hook for napari to get LLSZ Widget
@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [(plugin_wrapper, {"name" : "LLSZ Widget"} ), LLSZWidget_designer]



#Testing out UI only
#Disable napari hook and enable the following two lines to just test the UI
#ui=LLSZWidget()
#ui.show(run=True)
