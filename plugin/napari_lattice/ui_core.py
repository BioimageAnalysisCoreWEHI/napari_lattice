from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

import pyclesperanto_prototype as cle
from lls_core.utils import check_dimensions

from lls_core.llsz_core import pycuda_decon, skimage_decon
from lls_core import config, DeskewDirection, DeconvolutionChoice
from lls_core .io import save_img

if TYPE_CHECKING:
    from napari.types import ImageData

# Enable Logging
import logging
logger = logging.getLogger(__name__)
# inherit log level from config
logger.setLevel(config.log_level)

def _Preview(LLSZWidget,
             self_class,
             time: int,
             channel: int,
             img_data: ImageData):

    logger.info("Previewing deskewed channel and time")
    assert img_data.size, "No image open or selected"
    assert time < LLSZWidget.LlszMenu.lattice.time, "Time is out of range"
    assert channel < LLSZWidget.LlszMenu.lattice.channels, "Channel is out of range"
    assert LLSZWidget.LlszMenu.lattice.skew in DeskewDirection, f"Skew direction not recognised. Got {LLSZWidget.LlszMenu.lattice.skew}"

    vol = LLSZWidget.LlszMenu.lattice.data
    vol_zyx = np.array(vol[time, channel, :, :, :])

    # apply deconvolution if checked
    if LLSZWidget.LlszMenu.deconvolution.value:
        print(
            f"Deskewing for Time:{time} and Channel: {channel} with deconvolution")
        psf = LLSZWidget.LlszMenu.lattice.psf[channel]
        if LLSZWidget.LlszMenu.lattice.decon_processing == DeconvolutionChoice.cuda_gpu:
            decon_data = pycuda_decon(image=vol_zyx,
                                      psf=psf,
                                      dzdata=LLSZWidget.LlszMenu.lattice.dz,
                                      dxdata=LLSZWidget.LlszMenu.lattice.dx,
                                      dzpsf=LLSZWidget.LlszMenu.lattice.dz,
                                      dxpsf=LLSZWidget.LlszMenu.lattice.dx)
            # pycuda_decon(image,otf_path,dzdata,dxdata,dzpsf,dxpsf)
        else:
            decon_data = skimage_decon(
                vol_zyx=vol_zyx, psf=psf, num_iter=10, clip=False, filter_epsilon=0, boundary='nearest')

        deskew_final = LLSZWidget.LlszMenu.deskew_func(decon_data,
                                                       angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                                       voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                                       voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                                       voxel_size_z=LLSZWidget.LlszMenu.lattice.dz,
                                                       linear_interpolation=True).astype(vol.dtype)
    else:
        logger.info(f"Deskewing for Time:{time} and Channel: {channel}")
        deskew_final = LLSZWidget.LlszMenu.deskew_func(vol_zyx,
                                                       angle_in_degrees=LLSZWidget.LlszMenu.angle_value,
                                                       voxel_size_x=LLSZWidget.LlszMenu.lattice.dx,
                                                       voxel_size_y=LLSZWidget.LlszMenu.lattice.dy,
                                                       voxel_size_z=LLSZWidget.LlszMenu.lattice.dz,
                                                       linear_interpolation=True).astype(vol.dtype)

    # if getting an error LogicError: clSetKernelArg failed:    #INVALID_ARG_SIZE - when processing arg#13 (1-based)
    # make sure array is pulled from GPU

    deskew_final = cle.pull(deskew_final)
    # TODO: Use dask
    # if LLSZWidget.LlszMenu.dask:
    #logger.info(f"Using CPU for deskewing")
    # use cle library for affine transforms, but use dask and scipy
    # deskew_final = deskew_final.compute()

    max_proj_deskew = cle.maximum_z_projection(deskew_final)

    # add channel and time information to the name
    suffix_name = "_c" + str(channel) + "_t" + str(time)
    scale = (LLSZWidget.LlszMenu.lattice.new_dz,
             LLSZWidget.LlszMenu.lattice.dy, LLSZWidget.LlszMenu.lattice.dx)
    # TODO:adding img of difff scales change dim slider
    self_class.parent_viewer.add_image(
        deskew_final, name="Deskewed image" + suffix_name, scale=scale)
    self_class.parent_viewer.add_image(
        max_proj_deskew, name="Deskew_MIP", scale=scale[1:3])
    self_class.parent_viewer.layers[0].visible = False

    logger.info(f"Preview: Deskewing complete")
    return


def _Deskew_Save(LLSZWidget,
                 time_start: int,
                 time_end: int,
                 ch_start: int,
                 ch_end: int,
                 save_as_type,
                 save_path: Path):

    assert LLSZWidget.LlszMenu.open_file, "Image not initialised"
    check_dimensions(time_start, time_end, ch_start, ch_end,
                     LLSZWidget.LlszMenu.lattice.channels, LLSZWidget.LlszMenu.lattice.time)
    #time_range = range(time_start, time_end)
    #channel_range = range(ch_start, ch_end)
    angle = LLSZWidget.LlszMenu.lattice.angle
    dx = LLSZWidget.LlszMenu.lattice.dx
    dy = LLSZWidget.LlszMenu.lattice.dy
    dz = LLSZWidget.LlszMenu.lattice.dz

    # Convert path to string
    #save_path = save_path.__str__()

    # get the image data as dask array
    img_data = LLSZWidget.LlszMenu.lattice.data

    # pass arguments for save tiff, callable and function arguments
    save_img(vol=img_data,
             func=LLSZWidget.LlszMenu.deskew_func,
             time_start=time_start,
             time_end=time_end,
             channel_start=ch_start,
             channel_end=ch_end,
             save_file_type=save_as_type,
             save_path=save_path,
             save_name=LLSZWidget.LlszMenu.lattice.save_name,
             dx=dx,
             dy=dy,
             dz=dz,
             angle=angle,
             angle_in_degrees=angle,
             voxel_size_x=dx,
             voxel_size_y=dy,
             voxel_size_z=dz,
             linear_interpolation=True,
             LLSZWidget=LLSZWidget)

    print("Deskewing and Saving Complete -> ", save_path)
    return

