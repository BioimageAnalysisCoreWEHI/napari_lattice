import os
import numpy as np
from pathlib import Path

from magicclass.wrappers import set_design
from magicgui import magicgui, widgets
from magicclass import magicclass, vfield, set_options
from qtpy.QtCore import Qt

from napari.layers import Layer
from napari.types import ImageData
from napari.utils import history

import pyclesperanto_prototype as cle
from .io import LatticeData,  save_img

from napari_lattice.llsz_core import rl_decon,cuda_decon

from aicsimageio import AICSImage


    
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

    vol_zyx= vol[time,channel,:,:,:]

    # Deskew using pyclesperanto
    #Apply deconvolution if needed
    
    
    
    if LLSZWidget.LlszMenu.deconvolution.value:
        print(f"Deskewing for Time:{time} and Channel: {channel} with deconvolution")
        psf = LLSZWidget.LlszMenu.lattice.psf[channel]
        if LLSZWidget.LlszMenu.lattice.decon_processing == "cuda_gpu":
            decon_data = cuda_decon(image = vol_zyx,psf = psf,niter = 10)
        else:
            decon_data = rl_decon(image = vol_zyx,psf = psf,niter = 10,method = LLSZWidget.LlszMenu.lattice.decon_processing)
        
        deskew_final = cle.deskew_y(decon_data, 
                                angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                voxel_size_z=LLSZWidget.LlszMenu.lattice.dz).astype(vol.dtype)
    else:
        print("Deskewing for Time:", time,"and Channel: ", channel)
        deskew_final = cle.deskew_y(vol_zyx, 
                                angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                voxel_size_z=LLSZWidget.LlszMenu.lattice.dz).astype(vol.dtype)

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
                assert 0<= time_start <=LLSZWidget.LlszMenu.lattice.time, "Time start should be 0 or same as total time: "+str(LLSZWidget.LlszMenu.lattice.time)
                assert 0<= time_end <=LLSZWidget.LlszMenu.lattice.time, "Time end should be >0 or same as total time: "+str(LLSZWidget.LlszMenu.lattice.time)
                assert 0<= ch_start <= LLSZWidget.LlszMenu.lattice.channels, "Channel start should be 0 or same as no. of channels: "+str(LLSZWidget.LlszMenu.lattice.channels)
                assert 0<= ch_end <= LLSZWidget.LlszMenu.lattice.channels, "Channel end should be >0 or same as no. of channels: " +str(LLSZWidget.LlszMenu.lattice.channels)
              
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
                          func = cle.deskew_y,
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
                          LLSZWidget = LLSZWidget)
                
                print("Deskewing and Saving Complete -> ", save_path)
                return    


def _read_psf(psf_ch1_path:Path=Path(""),
              psf_ch2_path:Path=Path(""),
              psf_ch3_path:Path=Path(""),
              psf_ch4_path:Path=Path(""),
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
    from pathlib import PureWindowsPath, PosixPath
    assert decon_value==True, "Deconvolution is set to False. Tick the box to activate deconvolution."

    #Use CUDA for deconvolution
    if decon_option == "cuda_gpu":
        import importlib
        cucim_import = importlib.util.find_spec("cucim")
        cupy_import = importlib.util.find_spec("cupy")
        assert cucim_import and cupy_import, f"Please install cucim and cupy. Otherwise, please select another option"
    

    
    psf_paths = [psf_ch1_path,psf_ch2_path,psf_ch3_path,psf_ch4_path]
    #remove empty paths; pathlib returns current directory as "." if None or empty str specified
    psf_paths = [x for x in psf_paths if x!=PureWindowsPath(".") and x!=PosixPath(".")]

    #total no of psf images
    psf_channels = len(psf_paths)

    assert psf_channels>0, f"No images detected for PSF. Check the path {psf_paths}"

    for psf in psf_paths:
        print(psf)
        if os.path.exists(psf) and psf.is_file():
            if os.path.splitext(psf.__str__())[1] == ".czi":
                from aicspylibczi import CziFile
                psf_czi = CziFile(psf.__str__())
                psf_aics = psf_czi.read_image()
                if len(psf_aics[0])>=1:
                    psf_channels = len(psf_aics[0])
                #make sure shape is 3D
                psf_aics = psf_aics[0][0]#np.expand_dims(psf_aics[0],axis=0)
                assert len(psf_aics.shape) == 3, f"PSF should be a 3D image (shape of 3), but got {psf_aics.shape}"
                lattice_class.psf.append(psf_aics)
                
            else:
                psf_aics = AICSImage(psf.__str__())
                lattice_class.psf.append(psf_aics.data)
                
                if psf_aics.dims.C>=1:
                        psf_channels = psf_aics.dims.C
    #LLSZWidget.LlszMenu.lattice.channels =3
    if psf_channels != lattice_class.channels:
            print(f"PSF image has {psf_channels} channel/s, whereas image has {lattice_class.channels}")

    return