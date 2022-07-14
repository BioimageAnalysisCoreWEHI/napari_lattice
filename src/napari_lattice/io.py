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
from napari.layers import image, Layer
import dask.array as da
import pandas as pd

from dask.distributed import Client
from dask.cache import Cache

from .utils import etree_to_dict
from .utils import get_deskewed_shape,_process_custom_workflow_output_batch
from .llsz_core import crop_volume_deskew
from . import config

import os
import numpy as np
from napari.types import ImageData
from napari_workflows import Workflow
from tqdm import tqdm
from tifffile import imsave


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
        #aics_image_pixel_sizes = PhysicalPixelSizes(new_dz,dy,dx)
    else:
        #aics_image_pixel_sizes = PhysicalPixelSizes(dz,dy,dx)
        new_dz = dz

    if func is crop_volume_deskew:
        #create folder for each ROI; disabled as each roi is saved as hyperstack
        save_name_prefix = save_name_prefix + "_"
        #save_path = save_path+os.sep+save_name_prefix+os.sep
        #if not os.path.exists(save_path):
            #os.makedirs(save_path)
        im_final=[]
    
    #loop is ordered so image is saved in order TCZYX for ometiffwriter
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
                #process_vol = raw_vol.map_blocks(func,input_image = raw_vol, dtype=image_type,*args,**kwargs)
                processed_vol = func(input_image = raw_vol, *args,**kwargs).astype(image_type)
            elif func is crop_volume_deskew:
                processed_vol = func(original_volume = raw_vol, *args,**kwargs).astype(image_type)
            else:
                processed_vol = func( *args,**kwargs).astype(image_type)
            
            images_array.append(processed_vol)
        
        images_array = np.array(images_array)   
        #For functions other than cropping save each timepoint
        if func != crop_volume_deskew: 
            final_name = save_path + os.sep +save_name_prefix+ "C" + str(ch) + "T" + str(
                            time_point) + "_" +save_name+ ".tif"
            #OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes=aics_image_pixel_sizes)
            imsave(final_name,images_array, bigtiff=True, resolution=(1./dx, 1./dy),
               metadata={'spacing': new_dz, 'unit': 'um', 'axes': 'TZCYX'})#imagej=True
        elif func is crop_volume_deskew:
            im_final.append(images_array)
            
    #if using cropping, save whole stack instead of individual timepoints
    if func is crop_volume_deskew:
        im_final = np.array(im_final)
        final_name = save_path + os.sep +save_name_prefix+ "_" +save_name+ ".tif"
        #OmeTiffWriter.save(im_final, final_name, physical_pixel_sizes=aics_image_pixel_sizes)
        im_final = np.swapaxes(im_final,1,2)
        #imagej=True; ImageJ hyperstack axes must be in TZCYXS order
        
        imsave(final_name,im_final, bigtiff=True, resolution=(1./dx, 1./dy),
               metadata={'spacing': new_dz, 'unit': 'um', 'axes': 'TZCYX'})#imagej=True
        im_final = None
   
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
    
    #adding +1 at the end so the last channel and time is included
    time_range = range(time_start, time_end+1)
    channel_range = range(channel_start, channel_end+1)
    
    #Calculate new_pixel size in z
    #convert voxel sixes to an aicsimage physicalpixelsizes object for metadata
    if angle:
        import math
        new_dz = math.sin(angle * math.pi / 180.0) * dz
        #aics_image_pixel_sizes = PhysicalPixelSizes(new_dz,dy,dx)
    else:     
        #aics_image_pixel_sizes = PhysicalPixelSizes(dz,dy,dx)
        new_dz = dz

    
    for time_point in tqdm(time_range, desc="Time", position=0):
        images_array = []
        data_table = []     
        for ch in tqdm(channel_range, desc="Channels", position=1,leave=False):

            if len(vol.shape) == 3:
                raw_vol = vol
            else:
                raw_vol = vol[time_point, ch, :, :, :]
            
            #to access current time and channel, create a file config.py in same dir as workflow or in home directory
            #add "channel = 0" and "time=0" in the file and save
            #https://docs.python.org/3/faq/programming.html?highlight=global#how-do-i-share-global-variables-across-modules
            
            config.channel = ch
            config.time = time_point

            
            #Set input to the workflow to be volume from each time point and channel
            workflow.set(input_arg,raw_vol)
            #execute workflow
            processed_vol = workflow.get(last_task)

            images_array.append(processed_vol)    
        
        images_array = np.array(images_array)
        
        #check if output from workflow a list of dicts, list and/or images
        no_elements = len(processed_vol)
        if type(processed_vol) not in [np.ndarray,cle._tier0._pycl.OCLArray, da.core.Array]:
            array_element_type = [type(images_array[0,i]) for i in range(no_elements)]
        else:
            array_element_type = type(processed_vol)
            
        #check if output from workflow a list of dicts, list and/or images
        
        if any([i in [dict,list,tuple] for i in array_element_type]):
            if (len(processed_vol)>1) and (type(processed_vol) in [tuple]):
                _process_custom_workflow_output_batch(raw_vol,
                                                      no_elements,
                                                        array_element_type,
                                                        channel_range,
                                                        images_array,
                                                        save_path,
                                                        time_point,
                                                        ch,
                                                        save_name_prefix,
                                                        save_name,
                                                        dx,
                                                        dy,
                                                        new_dz)
                #return list, concatenate every iteration and create a bigger dataframe
            #check if list and it it contains dict or images
            elif (len(processed_vol)>1) and (type(processed_vol) in [list]) and any([type(i) in [dict,np.ndarray,cle._tier0._pycl.OCLArray, da.core.Array] for i in processed_vol]):
                _process_custom_workflow_output_batch(raw_vol,
                                                        no_elements,
                                                        array_element_type,
                                                        channel_range,
                                                        images_array,
                                                        save_path,
                                                        time_point,
                                                        ch,
                                                        save_name_prefix,
                                                        save_name,
                                                        dx,
                                                        dy,
                                                        new_dz)         
            #if a single dict or   list of dicts
            elif type(images_array) in [dict] or type(images_array[0]) in [dict]:
                #convert to pandas dataframe
                for j in channel_range:
                    images_array[j].update({"Channel/Time":"C"+str(j)+"T"+str(time_point)})
                output_dict_pd = [pd.DataFrame(i) for i in images_array]
                output_dict_pd = pd.concat(output_dict_pd)
                #set index to the channel/time
                output_dict_pd = output_dict_pd.set_index("Channel/Time")            
                dict_save_path = os.path.join(save_path,"C" + str(ch) + "T" + str(time_point) + "_measurement.csv")
                output_dict_pd.to_csv(dict_save_path, index=False)
            
            #if a single list or list of lists
            elif type(images_array) in [list] or type(images_array[0]) in [list]:
                row_idx=[]
                for j in channel_range:
                    row_idx.append("C"+str(j)+"T"+str(time_point))
                    
                output_list_pd = pd.DataFrame(np.vstack(images_array),index=row_idx)
                #Save path
                list_save_path = os.path.join(save_path,"C" + str(ch) + "T" + str(time_point) + "_measurement.csv")
                output_list_pd.to_csv(list_save_path, index=False)
        
                
        #processing as an iamge    
        else:
            
            final_name = save_path + os.sep +save_name_prefix+ "C" + str(ch) + "T" + str(
                            time_point) + "_" + save_name + ".tif"
            #OmeTiffWriter.save(images_array, final_name, physical_pixel_sizes=aics_image_pixel_sizes)
            #images from above are returned as czyx, so swap 
            #print(images_array.shape)
            images_array = np.swapaxes(images_array,0,1).astype(raw_vol.dtype)
            #imagej=True; ImageJ hyperstack axes must be in TZCYXS order
            imsave(final_name,images_array, bigtiff=True, imagej=True, resolution=(1./dx,1./dy),
               metadata={'spacing': new_dz, 'unit': 'um', 'axes': 'ZCYX'})#imagej=True
            #images_array = None
    
    return

#class for initilazing lattice data and setting metadata
#TODO: handle scenes
class LatticeData():
    def __init__(self,img,angle,skew,dx,dy,dz,channel_dimension) -> None:
        self.angle = angle
        self.skew = skew
        #if image layer
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