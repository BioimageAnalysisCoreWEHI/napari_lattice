import os
import numpy as np
from pathlib import Path

from aicsimageio import AICSImage
from napari.types import ImageData

import pyclesperanto_prototype as cle
from .io import save_img
from .utils import pad_image_nearest_multiple, check_dimensions

from napari_lattice.llsz_core import pycuda_decon, skimage_decon

#Enable Logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
def _Preview(LLSZWidget,
            self_class,
            time:int,
            channel:int,
            img_data: ImageData):
    
    print("Previewing deskewed channel and time")
    assert img_data.size, "No image open or selected"
    assert time< LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
    assert channel < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"
    
    assert str.upper(LLSZWidget.LlszMenu.lattice.skew) in ('Y', 'X'), \
        "Skew direction not recognised. Enter either Y or X"


    vol = LLSZWidget.LlszMenu.lattice.data

    
    vol_zyx= np.array(vol[time,channel,:,:,:])

    #apply deconvolution if checked
    if LLSZWidget.LlszMenu.deconvolution.value:
        print(f"Deskewing for Time:{time} and Channel: {channel} with deconvolution")
        psf = LLSZWidget.LlszMenu.lattice.psf[channel]
        if LLSZWidget.LlszMenu.lattice.decon_processing == "cuda_gpu":
            decon_data = pycuda_decon(image = vol_zyx,
                                      psf = psf,
                                      dzdata=LLSZWidget.LlszMenu.lattice.dz,
                                      dxdata=LLSZWidget.LlszMenu.lattice.dx,
                                      dzpsf=LLSZWidget.LlszMenu.lattice.dz,
                                      dxpsf=LLSZWidget.LlszMenu.lattice.dx)
            #pycuda_decon(image,otf_path,dzdata,dxdata,dzpsf,dxpsf)
        else:
            decon_data = skimage_decon(vol_zyx=vol_zyx, psf=psf, num_iter=10, clip=False, filter_epsilon=0, boundary='nearest')

        
        deskew_final = cle.deskew_y(decon_data, 
                                angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                voxel_size_z=LLSZWidget.LlszMenu.lattice.dz,
                                linear_interpolation=True).astype(vol.dtype)
    else:
        print("Deskewing for Time:", time,"and Channel: ", channel)
        deskew_final = cle.deskew_y(vol_zyx, 
                                angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                voxel_size_z=LLSZWidget.LlszMenu.lattice.dz,
                                linear_interpolation=True).astype(vol.dtype)

    # if getting an error LogicError: clSetKernelArg failed:    #INVALID_ARG_SIZE - when processing arg#13 (1-based)
    # make sure array is pulled from GPU
    
    deskew_final = cle.pull(deskew_final)
    # TODO: Use dask
    if LLSZWidget.LlszMenu.dask:
        print("Using CPU for deskewing")
        # use cle library for affine transforms, but use dask and scipy
        # deskew_final = deskew_final.compute()
    
    max_proj_deskew = cle.maximum_z_projection(deskew_final) #np.max(deskew_final, axis=0)

    # add channel and time information to the name
    suffix_name = "_c" + str(channel) + "_t" + str(time)
    scale = (LLSZWidget.LlszMenu.lattice.new_dz,LLSZWidget.LlszMenu.lattice.dy,LLSZWidget.LlszMenu.lattice.dx)
    #TODO:adding img of difff scales change dim slider
    self_class.parent_viewer.add_image(deskew_final, name="Deskewed image" + suffix_name,scale=scale)
    self_class.parent_viewer.add_image(max_proj_deskew, name="Deskew_MIP",scale=scale[1:3])
    self_class.parent_viewer.layers[0].visible = False
    #print("Shape is ",deskew_final.shape)
    print("Preview: Deskewing complete")
    return
    
def _Deskew_Save(LLSZWidget,
                 time_start: int,
                 time_end: int,
                 ch_start: int,
                 ch_end: int,
                 save_as_type: str,
                 save_path: Path):

                assert LLSZWidget.LlszMenu.open_file, "Image not initialised"
                check_dimensions(time_start,time_end,ch_start,ch_end,LLSZWidget.LlszMenu.lattice.channels,LLSZWidget.LlszMenu.lattice.time)
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
                save_img(vol = img_data,
                          func = LLSZWidget.LlszMenu.deskew_func,
                          time_start = time_start,
                          time_end = time_end,
                          channel_start = ch_start,
                          channel_end = ch_end,
                          save_file_type = save_as_type,
                          save_path = save_path,
                          save_name= LLSZWidget.LlszMenu.lattice.save_name,
                          dx = dx,
                          dy = dy,
                          dz = dz,
                          angle = angle,
                          angle_in_degrees = angle,
                          voxel_size_x=dx,
                          voxel_size_y=dy,
                          voxel_size_z=dz,
                          linear_interpolation=True,
                          LLSZWidget = LLSZWidget)
                
                print("Deskewing and Saving Complete -> ", save_path)
                return    


def _read_psf(psf_ch1_path:Path,
              psf_ch2_path:Path,
              psf_ch3_path:Path,
              psf_ch4_path:Path,
              use_gpu_decon:str="cpu",
              LLSZWidget = None,
              lattice = None,
              terminal:bool=False,
              #lattice = None
              ):
    
    #add flag for terminal
    if terminal:
        decon_value = True
        decon_option = lattice.decon_processing
        lattice_class = lattice
    else:
        decon_value = LLSZWidget.LlszMenu.deconvolution.value
        lattice_class = LLSZWidget.LlszMenu.lattice
        decon_option = use_gpu_decon

    #if terminal, use decon_value instead of  LLSZWidget.LLszMenu.deconvolution.value
    #use lattice.decon_processing instead of LLSZWidget.LlszMenu.lattice.decon_processing
    #use lattice.psf instead of LLSZWidget.LlszMenu.lattice.psf
    assert decon_value==True, "Deconvolution is set to False. Tick the box to activate deconvolution."
    psf_paths = [psf_ch1_path,psf_ch2_path,psf_ch3_path,psf_ch4_path]

    #remove empty paths; pathlib returns current directory as "." if None or empty str specified
    #When running batch processing,empty directory will be an empty string
    
    import platform
    from pathlib import PureWindowsPath, PosixPath
    
    if platform.system() =="Linux":
        psf_paths = [Path(x) for x in psf_paths if x!=PosixPath(".") and x!=""]
    elif platform.system()=="Windows":
        psf_paths = [Path(x) for x in psf_paths if x!=PureWindowsPath(".") and x!=""]
        
    logging.debug(f"PSF paths are {psf_paths}")
    #total no of psf images
    psf_channels = len(psf_paths)
    assert psf_channels>0, f"No images detected for PSF. Check the psf paths -> {psf_paths}"


    #Use CUDA for deconvolution
    if decon_option == "cuda_gpu":
        import importlib
        pycudadecon_import = importlib.util.find_spec("pycudadecon")
        assert pycudadecon_import, f"Pycudadecon not detected. Please install using: conda install -c conda-forge pycudadecon"
        otf_names = ["ch1","ch2","ch3","ch4"]
        channels =[488,561,640,123]
        #get temp directory to save generated otf
        import tempfile
        temp_dir = tempfile.gettempdir()+os.sep
     
    for idx,psf in enumerate(psf_paths):
        if os.path.exists(psf) and psf.is_file():
            if psf.suffix == ".czi":
                from aicspylibczi import CziFile
                psf_czi = CziFile(psf.__str__())
                psf_aics = psf_czi.read_image()
                #make sure shape is 3D
                psf_aics = psf_aics[0][0]#np.expand_dims(psf_aics[0],axis=0)
                #if len(psf_aics[0])>=1:
                    #psf_channels = len(psf_aics[0])
                assert len(psf_aics.shape) == 3, f"PSF should be a 3D image (shape of 3), but got {psf_aics.shape}"
                #pad psf to multiple of 16 for decon
                psf_aics = pad_image_nearest_multiple(img=psf_aics,nearest_multiple=16)
                lattice_class.psf.append(psf_aics)              
            else:
                psf_aics = AICSImage(psf.__str__())
                psf_aics_data = psf_aics.data[0][0]
                psf_aics_data = pad_image_nearest_multiple(img=psf_aics_data,nearest_multiple=16)
                lattice_class.psf.append(psf_aics_data)
                if psf_aics.dims.C>=1:
                        psf_channels = psf_aics.dims.C
    
    
    #LLSZWidget.LlszMenu.lattice.channels =3
    if psf_channels != lattice_class.channels:
            print(f"PSF image has {psf_channels} channel/s, whereas image has {lattice_class.channels}")

    return
