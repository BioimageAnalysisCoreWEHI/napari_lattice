#Opening and saving files
#Add option to save max projections??

import aicsimageio
import aicspylibczi
from aicsimageio.writers import OmeTiffWriter
import sys
import dask
from dask.distributed import Client
from dask.cache import Cache
from llsz.utils import etree_to_dict, get_scale_factor,get_shear_factor
from llsz.llsz_core import process_czi

from tqdm.notebook import trange, tqdm

#add options for configuring dask scheduler

def read_czi(img_path):
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
            print("If it is a CZI file, resave it as a czi again using Zen software. This is due to the file being in an uncompressed format from the microscope.")
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
    check_metadata(img_path)
    return stack

#will flesh this out once lattice has more metadata in the czi file
def check_metadata(img_path):
    print("Checking CZI metadata")
    metadatadict_czi = etree_to_dict(aicspylibczi.CziFile(img_path).meta)
    metadatadict_czi = metadatadict_czi["ImageDocument"]["Metadata"]
    acquisition_mode_setup=metadatadict_czi["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"]["HelperSetups"]["AcquisitionModeSetup"]["Detectors"]
    print(acquisition_mode_setup["Detector"]["ImageOrientation"])
    print("Image Orientation: If any post processing has been applied, it will appear here.\
      For example, Zen 3.2 flipped the image so the coverslip was towards the bottom of Z stack. So, value will be 'Flip'")
    return

#TODO: Add option to read tiff files (images from Janelia lattice can be specified by changing angle and skew during initialisation)
class LatticeData():
    def __init__(self,path,angle,skew) -> None:
        self.angle = angle
        self.skew = skew

        #Read in a AICS image and get data and metadata
        self.data = read_czi(path)
        
        self.dims = self.data.dims
        self.time = self.data.dims.T
        self.channels = self.data.dims.C

        self.dz,self.dy,self.dx = self.data.physical_pixel_sizes
        self.shear_factor = get_shear_factor(self.angle)
        self.scaling_factor = get_scale_factor(self.angle, self.dy , self.dz)

        #process the file to get parameters for deskewing
        self.deskew_shape, self.deskew_vol_shape, self.deskew_translate_y, self.deskew_z_start, self.deskew_z_end = process_czi(self.data, self.angle, self.skew)
        pass 
    
    def get_angle(self):
        return self.angle

    def set_angle(self, angle:float):
        self.angle = angle

    def set_skew(self, skew:str):
        self.skew = skew        
        
#read each time point and save?

"""
deskew_size=tuple((nz,deskewed_y,nx))
deskew_chunk_size=tuple((nz,deskewed_y,nx))



save_dir="Z://Pradeep//Lightsheet//Niall_test//"
save_name=os.path.splitext(os.path.basename(img_location))[0]
with Profiler() as prof1, ResourceProfiler(dt=0.25) as rprof1,CacheProfiler() as cprof1:
    #for time_point in tqdm(range(0,time)):
    for time_point in trange(time-1, desc="Time"):
        images_array=[] 
        for ch in trange(channels, desc="Channels"):
            img=stack.get_image_dask_data("ZYX",T=time_point,C=ch,S=scenes)
            #chunk size essential for next step, otherwise takes too long
            deskew_img=da.zeros(deskew_size,dtype=stack_meta.dtype,chunks=deskew_chunk_size)
            deskew_img[:,:ny,:]=img
            #result=lattice_process(deskew_img)# for stack12 in tqdm(deskew_img)]
            deskew_all=deskew_rotate_zeiss(deskew_img,angle,shear_factor=deskew_factor,reverse=False,dask=False)
            deskew_final=deskew_all[z_start:z_end]/65535.0
            deskew_final=np.flip(deskew_final,axis=(0))
            deskew_final=img_as_uint(deskew_final)
            #Stack into a large dask array
            #deskewed_stack= da.stack(result, axis=0)#.reshape((367, 2, 501, deskewed_dim, nx)).rechunk((1,1,501, deskewed_dim, nx))
            images_array.append(deskew_final)
        ch_img=np.array(images_array) #convert to array of arrays
        #currently CZYX
        #when saving imagej=True format, it should be TZCYXS
        #swap for imagej=TRue saving
        ch_img=np.swapaxes(ch_img,0,1)

        processed_time_save=save_dir+"//T"+str(time_point)+"_"+save_name+".ome.tif"

        tifffile.imwrite(processed_time_save,ch_img,metadata={"axes":"ZCYX",'spacing': dy, 'unit': 'um',},dtype='uint16')#make 16-bit: 
        #tifffile.imwrite(processed_time_save,ch_img,metadata={"axes":"CZYX",'spacing': dy, 'unit': 'um',},dtype='uint16',imagej=True)#make 16-bit: 
        #print("Time "+str(time_point)+" saved")
    print("Complete")

"""
    