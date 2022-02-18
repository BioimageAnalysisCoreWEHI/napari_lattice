#Opening and saving files

import aicsimageio
import aicspylibczi
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.types import PhysicalPixelSizes
from pathlib import Path

import pyclesperanto_prototype as cle
import sys
import dask
from dask.distributed import Client
from dask.cache import Cache
from llsz.utils import etree_to_dict
from llsz.utils import get_deskewed_shape
from llsz.llsz_core import crop_volume_deskew

import os
import numpy as np
import dask 

from napari.types import ImageData

from tqdm import tqdm

from napari_workflows import Workflow

#add options for configuring dask scheduler

def read_img(img_path):
    """Return AICSimage object from image path

    Args:
        img_path ([type]): [description]

    Returns:
        AICImage: [description]
    """    
    #Error handling for czi file
    try:
        #stack=aicsimageio.imread_dask(img_location)
        stack= aicsimageio.AICSImage(img_path) #using AICSImage will read data as STCZYX
        #stack_meta=aicsimageio.imread_dask(img_location)
    except Exception as e:
        print("Error: A ", sys.exc_info()[0], "has occurred. See below for details.")
        if "method or operation is not implemented" in (str(e)):
            print("If it is a CZI file, try resaving it as a czi again using Zen software. This could be due to the file being in an uncompressed format from the microscope.")
        raise
    
    #Dask setup
    #Setting up dask scheduler
    client = Client(processes=False)  # start distributed scheduler locally.  Launch dashboard
    #memory_limit='30GB',
    home_dir = os.path.expanduser('~')
    dask.config.set({"temporary_directory":home_dir,"optimization.fuse.active": False,
                    'array.slicing.split_large_chunks': False})
    cache = Cache(2e9)
    cache.register()
    print("If using dask, dask Client can be viewed at:",client.dashboard_link)

    #Check metadata to verify postprocessing or any modifications by Zen
    if os.path.splitext(img_path)[1][1:].strip().lower() == "czi":
        check_metadata(img_path)
    return stack

def convert_imgdata_aics(img_data:ImageData):
    """Return AICSimage object from napari ImageData type

    Args:
        img_data ([type]): [description]

    Returns:
        AICImage: [description]
    """    
    #Error handling for czi file
    try:
        #stack=aicsimageio.imread_dask(img_location)
        stack= aicsimageio.AICSImage(img_data) #using AICSImage will read data as STCZYX
        #stack_meta=aicsimageio.imread_dask(img_location)
    except Exception as e:
        print("Error: A ", sys.exc_info()[0], "has occurred. See below for details.")
        raise
    
    #Dask setup
    #Setting up dask scheduler
    client = Client(processes=False)  # start distributed scheduler locally.  Launch dashboard
    #memory_limit='30GB',
    dask.config.set({"temporary_directory":"C:\\Dask_temp\\","optimization.fuse.active": False,
                    'array.slicing.split_large_chunks': False})
    cache = Cache(2e9)
    cache.register()
    print("If using dask, dask Client can be viewed at:",client.dashboard_link)

    #Check metadata to verify postprocessing or any modifications by Zen
    return stack


#will flesh this out once Zeiss lattice has more relevant metadata in the czi file
def check_metadata(img_path):
    print("Checking CZI metadata")
    metadatadict_czi = etree_to_dict(aicspylibczi.CziFile(img_path).meta)
    metadatadict_czi = metadatadict_czi["ImageDocument"]["Metadata"]
    acquisition_mode_setup=metadatadict_czi["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"]["HelperSetups"]["AcquisitionModeSetup"]["Detectors"]
    print(acquisition_mode_setup["Detector"]["ImageOrientation"])
    print("Image Orientation: If any post processing has been applied, it will appear here.\n \
      For example, Zen 3.2 flipped the image so the coverslip was towards the bottom of Z stack. So, value will be 'Flip'")
    return

#TODO: write save function for deskew and for crop

def save_tiff(vol,
              workflow:Workflow,
              input_arg:str,
              task_name:str,
              time_start:int,
              time_end:int,
              channel_start:int,
              channel_end:int,
              save_path:Path,
              save_name_prefix:str = "",
              save_name:str = "img",
              dx:float = 1,
              dy:float = 1,
              dz:float = 1):
    """
    Applies a workflow to the image and saves the output
    Use of workflows ensures its agnostic to the processing operation

    Args:
        vol (_type_): Volume to process
        workflow (Workflow): Clesperanto workflow
        input_arg (str): name for input image
        task_name (str): name of the task that should be executed in the workflow
        time_start (int): _description_
        time_start (int): _description_
        time_end (int): _description_
        channel_start (int): _description_
        channel_end (int): _description_
        save_path (Path): _description_
        save_name_prefix (str, optional): Add a prefix to name. For example, if processng ROIs, add ROI_1_. Defaults to "".
        save_name (str, optional): name of file being saved. Defaults to "img".
        dx (float, optional): _description_. Defaults to 1.
        dy (float, optional): _description_. Defaults to 1.
        dz (float, optional): _description_. Defaults to 1.
    """              
    
    save_path = save_path.__str__()
    
    time_range = range(time_start, time_end)
    channel_range = range(channel_start, channel_end)
    
    #convert voxel sixes to an aicsimage physicalpixelsizes object for metadata
    aics_image_pixel_sizes = PhysicalPixelSizes(dz,dy,dx)

    for time_point in tqdm(time_range, desc="Time", position=0):
        images_array = []      
        for ch in tqdm(channel_range, desc="Channels", position=1,leave=False):

            if len(vol.shape) == 3:
                raw_vol = vol
            else:
                raw_vol = vol[time_point, ch, :, :, :]
            
            #Use workflows to execute
            #Set the input argument
            workflow.set(input_arg, raw_vol)

            #Execute workflow
            processed_vol = workflow.get(task_name)

            images_array.append(processed_vol)
            
        images_array = np.array(images_array)
        final_name = save_path + os.sep +save_name_prefix+ "C" + str(ch) + "T" + str(
                        time_point) + "_" + save_name + ".ome.tif"
    
        OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes=aics_image_pixel_sizes)
    
    return

#Defining classes for initialising files. 
#Converts it to an aicsimageio object and makes it easy to access metadata

#TODO: Merge both file initialising classes
#Class for initializing lattice czi files
class LatticeData_czi():
    def __init__(self,img,angle,skew) -> None:
        self.angle = angle
        self.skew = skew
    
        #Read in image path or imageData and return an AICS objec. Allows standardised access to metadata
        #if isinstance(img,(np.ndarray,dask.array.core.Array)):
        #    self.data = convert_imgdata_aics(img)
        #else:
        self.data = read_img(img)
        
        self.dims = self.data.dims
        self.time = self.data.dims.T
        self.channels = self.data.dims.C

        self.dz,self.dy,self.dx = self.data.physical_pixel_sizes

        self.deskew_vol_shape = get_deskewed_shape(self.data.dask_data, self.angle,self.dx,self.dy,self.dz)
        pass 

    def get_angle(self):
        return self.angle

    def set_angle(self, angle:float):
        self.angle = angle

    def set_skew(self, skew:str):
        self.skew = skew        

#Class for initializing non czi files
#TODO: Add option to read tiff files (images from Janelia lattice can be specified by changing angle and skew during initialisation)
class LatticeData():
    def __init__(self,img,angle,skew,dx,dy,dz) -> None:
        self.angle = angle
        self.skew = skew
    
        #Read in image path or imageData and return an AICS objec. Allows standardised access to metadata
        #if isinstance(img,(np.ndarray,dask.array.core.Array)):
        self.data = convert_imgdata_aics(img)
        #else:
        #    self.data = read_img(img)
        #set metadata based on user input if aicsimageio cannot find it
        if None in self.data.physical_pixel_sizes:
            self.dx = dx
            self.dy = dy
            self.dz = dz
        else:
            self.dz,self.dy,self.dx = self.data.physical_pixel_sizes
        
        self.dims = self.data.dims
        self.time = self.data.dims.T
        self.channels = self.data.dims.C

        #process the file to get parameters for deskewing
        self.deskew_vol_shape = get_deskewed_shape(self.data, self.angle,self.dx,self.dy,self.dz)
        pass 

    def get_angle(self):
        return self.angle

    def set_angle(self, angle:float):
        self.angle = angle

    def set_skew(self, skew:str):
        self.skew = skew 
