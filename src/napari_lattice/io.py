#Opening and saving files
from multiprocessing.dummy import Array
import aicsimageio
from aicsimageio.writers import OmeTiffWriter
from aicsimageio.types import PhysicalPixelSizes
from pathlib import Path

import pyclesperanto_prototype as cle
import sys
import dask
import dask.array as da
import xarray 
from napari.layers import image, Layer
import dask.array as da

from dask.distributed import Client
from dask.cache import Cache

from .utils import etree_to_dict
from .utils import get_deskewed_shape, dask_expand_dims,modify_workflow_task
from .llsz_core import crop_volume_deskew

import os
import numpy as np
from napari.types import ImageData
from napari_workflows import Workflow
from tqdm import tqdm


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
    metadatadict_czi = etree_to_dict(aicsimageio.AICSImage(img_path).metadata)
    metadatadict_czi = metadatadict_czi["ImageDocument"]["Metadata"]
    acquisition_mode_setup=metadatadict_czi["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"]["HelperSetups"]["AcquisitionModeSetup"]["Detectors"]
    print(acquisition_mode_setup["Detector"]["ImageOrientation"])
    print("Image Orientation: If any post processing has been applied, it will appear here.\n \
      For example, Zen 3.2 flipped the image so the coverslip was towards the bottom of Z stack. So, value will be 'Flip'")
    return

#TODO: write save function for deskew and for crop

def save_tiff(vol,
              func:callable,
              time_start:int,
              time_end:int,
              channel_start:int,
              channel_end:int,
              save_path:Path,
              save_name_prefix:str = "",
              save_name:str = "img",
              dx:float = 1,
              dy:float = 1,
              dz:float = 1,
              angle:float = None,
              *args,**kwargs):
    """
    Applies a function as described in callable
    Args:
        vol (_type_): Volume to process
        func (callable): _description_
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
        angle_in_degrees(float, optional) = Deskewing angle in degrees, used to calculate new z
    """              
    
    save_path = save_path.__str__()
    
    time_range = range(time_start, time_end)
    channel_range = range(channel_start, channel_end)
    
    #Calculate new_pixel size in z
    #convert voxel sixes to an aicsimage physicalpixelsizes object for metadata
    if angle>0:
        import math
        new_dz = math.sin(angle * math.pi / 180.0) * dz
        aics_image_pixel_sizes = PhysicalPixelSizes(new_dz,dy,dx)
    else:
        aics_image_pixel_sizes = PhysicalPixelSizes(dz,dy,dx)

    if func is crop_volume_deskew:
        #create folder for each ROI
        save_name_prefix = save_name_prefix + "_"
        save_path = save_path+os.sep+save_name_prefix+os.sep
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for time_point in tqdm(time_range, desc="Time", position=0):
        images_array = []      
        for ch in tqdm(channel_range, desc="Channels", position=1,leave=False):
            try:
                if len(vol.shape) == 3:
                    raw_vol = vol
                elif len(vol.shape) == 4:
                    raw_vol = vol[time_point, :, :, :]
                elif len(vol.shape) == 5:
                    raw_vol = vol[time_point, ch, :, :, :]
            except IndexError:
                print("Check shape of volume. Expected volume with shape 3,4 or 5. Got ",vol.shape) 
            
            image_type = raw_vol.dtype
            
            #Apply function to a volume
            if func is cle.deskew_y:
                #print(raw_vol.shape)
                #import dask.array as da
                #if raw_vol not da.core.Array:
                    #raw_vol=da.from_array(raw_vol)
                #process_vol = raw_vol.map_blocks(func,input_image = raw_vol, dtype=image_type,*args,**kwargs)
                processed_vol = func(input_image = raw_vol, *args,**kwargs).astype(image_type)
            elif func is crop_volume_deskew:
                processed_vol = func(original_volume = raw_vol, *args,**kwargs).astype(image_type)
            else:
                processed_vol = func( *args,**kwargs).astype(image_type)
            
            images_array.append(processed_vol)
            
        images_array = np.array(images_array)

        final_name = save_path + os.sep +save_name_prefix+ "C" + str(ch) + "T" + str(
                        time_point) + "_" +save_name+ ".ome.tif"
    
        OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes=aics_image_pixel_sizes)
    images_array = None
    return

def save_tiff_workflow(vol,
              workflow:Workflow,
              input_arg:str,
              first_task:str,
              last_task:str,
              time_start:int,
              time_end:int,
              channel_start:int,
              channel_end:int,
              save_path:Path,
              save_name_prefix:str = "",
              save_name:str = "img",
              dx:float = 1,
              dy:float = 1,
              dz:float = 1,
              angle:float = None):
    """
    Applies a workflow to the image and saves the output
    Use of workflows ensures its agnostic to the processing operation
    Args:
        vol (_type_): Volume to process
        workflow (Workflow): napari workflow
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
        angle(float, optional) = Deskewing angle in degrees, used to calculate new z
    """              
    
    save_path = save_path.__str__()
    
    time_range = range(time_start, time_end)
    channel_range = range(channel_start, channel_end)
    
    #Calculate new_pixel size in z
    #convert voxel sixes to an aicsimage physicalpixelsizes object for metadata
    if angle:
        import math
        new_dz = math.sin(angle * math.pi / 180.0) * dz
        aics_image_pixel_sizes = PhysicalPixelSizes(new_dz,dy,dx)
    else:     
        aics_image_pixel_sizes = PhysicalPixelSizes(dz,dy,dx)

    for time_point in tqdm(time_range, desc="Time", position=0):
        images_array = []      
        for ch in tqdm(channel_range, desc="Channels", position=1,leave=False):

            if len(vol.shape) == 3:
                raw_vol = vol
            else:
                raw_vol = vol[time_point, ch, :, :, :]
            
            #Set input to the workflow to be volume from each time point and channel
            workflow.set(input_arg,raw_vol)
            #execute workflow
            processed_vol = workflow.get(last_task)
            images_array.append(processed_vol)    
        
        images_array = np.array(images_array)
        final_name = save_path + os.sep +save_name_prefix+ "C" + str(ch) + "T" + str(
                        time_point) + "_" + save_name + ".ome.tif"
    
        OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes=aics_image_pixel_sizes)
    
    return


#Class for initializing non czi files
#TODO: Add option to read tiff files (images from Janelia lattice can be specified by changing angle and skew during initialisation)
class LatticeData():
    def __init__(self,img,angle,skew,dx,dy,dz,channel_dimension) -> None:
        self.angle = angle
        self.skew = skew
        #if image layer
        print(type(img))
        if type(img) is image.Image: #napari layer image
            #check if its an aicsimageio object and has voxel size info            
            if 'aicsimage' in img.metadata.keys() and img.metadata['aicsimage'].physical_pixel_sizes != (None,None,None):
                img_data_aics = img.metadata['aicsimage']
                self.data = img_data_aics.dask_data
                self.dims = img_data_aics.dims
                self.time = img_data_aics.dims.T
                self.channels = img_data_aics.dims.C
                self.dz,self.dy,self.dx = img_data_aics.physical_pixel_sizes
            else:
                print("Cannot read voxel size from metadata")
                if 'aicsimage' in img.metadata.keys():
                    img_data_aics = img.metadata['aicsimage']
                    self.data = img_data_aics.dask_data
                    self.dims = img_data_aics.dims
                    self.time = img_data_aics.dims.T
                    self.channels = img_data_aics.dims.C
                else: 
                    #if no aicsimageio key in metadata
                    #get the data and convert it into an aicsimage object
                    img_data_aics = aicsimageio.AICSImage(img.data)
                    self.data = img_data_aics.dask_data
                
                #read metadata for pixel sizes
                if None in img_data_aics.physical_pixel_sizes or img_data_aics.physical_pixel_sizes== False:
                    self.dx = dx
                    self.dy = dy
                    self.dz = dz
                else:
                    self.dz,self.dy,self.dx = img.data.physical_pixel_sizes
                #if not channel_dimension:
                #if xarray, access data using .data method
                    #if type(img.data) in [xarray.core.dataarray.DataArray,np.ndarray]:
                        #img = img.data
                #img = dask_expand_dims(img,axis=1) ##if no channel dimension specified, then expand axis at index 1
            #if no path returned by source.path, get image name with colon and spaces removed
            if img.source.path is None:
                self.save_name = img.name.replace(":","").strip() #remove colon (:) and any leading spaces
                self.save_name = '_'.join(self.save_name.split()) #replace any group of spaces with "_"
                
            else:
                self.save_name = img.name

        elif type(img) in [np.ndarray,da.core.Array]:
            img_data_aics = aicsimageio.AICSImage(img.data)
            self.data = img_data_aics.dask_data
            self.dx = dx
            self.dy = dy
            self.dz = dz
            
        elif type(img) is aicsimageio.aics_image.AICSImage:
            
            if img.physical_pixel_sizes != (None,None,None):
                self.data = img.dask_data
                self.dims = img.dims
                self.time = img.dims.T
                self.channels = img.dims.C
                self.dz,self.dy,self.dx = img.physical_pixel_sizes
                
            else:
                self.data = img.dask_data
                self.dims = img.dims
                self.time = img.dims.T
                self.channels = img.dims.C
                self.dz,self.dy,self.dx = dz,dy,dx
                
        else:
            raise Exception("Has to be an image layer or array, got type: ",type(img))    
            
            #set new z voxel size
        if self.skew == "Y":
            import math
            self.new_dz = math.sin(self.angle * math.pi / 180.0) * self.dz
                
        #process the file to get shape of final deskewed image
        self.deskew_vol_shape = get_deskewed_shape(self.data, self.angle,self.dx,self.dy,self.dz)

        pass 

    def get_angle(self):
        return self.angle

    def set_angle(self, angle:float):
        self.angle = angle

    def set_skew(self, skew:str):
        self.skew = skew 