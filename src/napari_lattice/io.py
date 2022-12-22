# Opening and saving files
import aicsimageio

from pathlib import Path

import pyclesperanto_prototype as cle
import sys
import dask
import dask.array as da
import resource_backed_dask_array
from napari.layers import image
import dask.array as da
import pandas as pd

from dask.distributed import Client
from dask.cache import Cache

from napari_lattice.utils import etree_to_dict
from napari_lattice.utils import get_deskewed_shape
from napari_lattice.llsz_core import crop_volume_deskew, skimage_decon, pycuda_decon

from napari_lattice import config


import os
import numpy as np
from napari.types import ImageData
from napari_workflows import Workflow
from tqdm import tqdm
from tifffile import imwrite, TiffWriter

import npy2bdv
from . import DeskewDirection, DeconvolutionChoice, SaveFileType

# Enable Logging
import logging
logger = logging.getLogger(__name__)


def convert_imgdata_aics(img_data: ImageData):
    """Return AICSimage object from napari ImageData type

    Args:
        img_data ([type]): [description]

    Returns:
        AICImage: [description]
    """
    # Error handling for czi file
    try:
        # stack=aicsimageio.imread_dask(img_location)
        # using AICSImage will read data as STCZYX
        stack = aicsimageio.AICSImage(img_data)
        # stack_meta=aicsimageio.imread_dask(img_location)
    except Exception as e:
        print("Error: A ", sys.exc_info()[
              0], "has occurred. See below for details.")
        raise

    # Dask setup
    # Setting up dask scheduler
    # start distributed scheduler locally.  Launch dashboard
    client = Client(processes=False)
    # memory_limit='30GB',
    dask.config.set({"temporary_directory": "C:\\Dask_temp\\", "optimization.fuse.active": False,
                    'array.slicing.split_large_chunks': False})
    cache = Cache(2e9)
    cache.register()
    print("If using dask, dask Client can be viewed at:", client.dashboard_link)

    # Check metadata to verify postprocessing or any modifications by Zen
    return stack


# will flesh this out once Zeiss lattice has more relevant metadata in the czi file
def check_metadata(img_path):
    print("Checking CZI metadata")
    metadatadict_czi = etree_to_dict(aicsimageio.AICSImage(img_path).metadata)
    metadatadict_czi = metadatadict_czi["ImageDocument"]["Metadata"]
    acquisition_mode_setup = metadatadict_czi["Experiment"]["ExperimentBlocks"][
        "AcquisitionBlock"]["HelperSetups"]["AcquisitionModeSetup"]["Detectors"]
    print(acquisition_mode_setup["Detector"]["ImageOrientation"])
    print("Image Orientation: If any post processing has been applied, it will appear here.\n \
      For example, Zen 3.2 flipped the image so the coverslip was towards the bottom of Z stack. So, value will be 'Flip'")
    return

# TODO: write save function for deskew and for crop


def save_img(vol,
             func: callable,
             time_start: int,
             time_end: int,
             channel_start: int,
             channel_end: int,
             save_file_type,
             save_path: Path,
             save_name_prefix: str = "",
             save_name: str = "img",
             dx: float = 1,
             dy: float = 1,
             dz: float = 1,
             angle: float = None,
             LLSZWidget=None,
             terminal: bool = False,
             lattice=None,
             *args, **kwargs):
    """
    Applies a function as described in callable
    Args:
        vol (_type_): Volume to process
        func (callable): _description_
        time_start (int): _description_
        time_end (int): _description_
        channel_start (int): _description_
        channel_end (int): _description_
        save_file_type: either 'tiff' or SaveFileType.h5
        save_path (Path): _description_
        save_name_prefix (str, optional): Add a prefix to name. For example, if processng ROIs, add ROI_1_. Defaults to "".
        save_name (str, optional): name of file being saved. Defaults to "img".
        dx (float, optional): _description_. Defaults to 1.
        dy (float, optional): _description_. Defaults to 1.
        dz (float, optional): _description_. Defaults to 1.
        angle(float, optional) = Deskewing angle in degrees, used to calculate new z
        LLSZWidget(class,optional) = LLSZWidget class
    """

    save_path = save_path.__str__()

    # replace any : with _ and remove spaces in case it hasn't been processed/skipped
    save_name = save_name.replace(":", "_").replace(" ", "")

    time_range = range(time_start, time_end+1)

    channel_range = range(channel_start, channel_end+1)

    # Calculate new_pixel size in z after deskewing
    if angle > 0:
        import math
        new_dz = math.sin(angle * math.pi / 180.0) * dz
    else:
        new_dz = dz

    if func is crop_volume_deskew:
        # create folder for each ROI; disabled as each roi is saved as hyperstack
        save_name_prefix = save_name_prefix + "_"
        #save_path = save_path+os.sep+save_name_prefix+os.sep
        # if not os.path.exists(save_path):
        # os.makedirs(save_path)
        im_final = []

    # setup bdvwriter
    if save_file_type == SaveFileType.h5:
        if func is crop_volume_deskew:
            save_path_h5 = save_path + os.sep + save_name_prefix + "_" + save_name + ".h5"
        else:
            save_path_h5 = save_path + os.sep + save_name + ".h5"

        bdv_writer = npy2bdv.BdvWriter(save_path_h5,
                                       compression='gzip',
                                       nchannels=len(channel_range),
                                       subsamp=(
                                           (1, 1, 1), (1, 2, 2), (2, 4, 4)),
                                       overwrite=False)

        # bdv_writer = npy2bdv.BdvWriter(save_path_h5, compression=None, nchannels=len(channel_range)) #~30% faster, but up to 10x bigger filesize
    else:
        pass

    if terminal:
        if lattice.decon_processing:
            decon_value = True
            decon_option = lattice.decon_processing  # decon_processing holds the choice
            lattice_class = lattice
            logging.debug(f"Decon option {decon_option}")

    else:
        try:
            decon_value = LLSZWidget.LlszMenu.deconvolution.value
            lattice_class = LLSZWidget.LlszMenu.lattice
            decon_option = LLSZWidget.LlszMenu.lattice.decon_processing
        except:
            decon_value = 0
            lattice_class = 0
            decon_option = 0

    # loop is ordered so image is saved in order TCZYX for ometiffwriter
    for loop_time_idx, time_point in enumerate(tqdm(time_range, desc="Time", position=0)):
        images_array = []
        for loop_ch_idx, ch in enumerate(tqdm(channel_range, desc="Channels", position=1, leave=False)):
            try:
                if len(vol.shape) == 3:
                    raw_vol = vol
                elif len(vol.shape) == 4:
                    raw_vol = vol[time_point, :, :, :]
                elif len(vol.shape) == 5:
                    raw_vol = vol[time_point, ch, :, :, :]
            except IndexError:
                assert ch <= channel_end, f"Channel out of range. Got {ch}, but image has channels {channel_end+1}"
                assert time_point <= channel_end, f"Channel out of range. Got {ch}, but image has channels {channel_end+1}"
                assert len(vol.shape) in [
                    3, 4, 5], f"Check shape of volume. Expected volume with shape 3,4 or 5. Got {vol.shape} with shape {len(vol.shape)}"
                print(f"Using time points {time_point} and channel {ch}")
                exit()

            #raw_vol = np.array(raw_vol)
            image_type = raw_vol.dtype
            print(decon_value)
            print(decon_option)
            print(func)
            # Add a check for last timepoint, in case acquisition incomplete
            if time_point == time_end:
                orig_shape = raw_vol.shape
                raw_vol = raw_vol.compute()
                if raw_vol.shape != orig_shape:
                    print(
                        f"Time {time_point}, channel {ch} is incomplete. Actual shape {orig_shape}, got {raw_vol.shape}")
                    z_diff, y_diff, x_diff = np.subtract(
                        orig_shape, raw_vol.shape)
                    print(f"Padding with{z_diff,y_diff,x_diff}")
                    raw_vol = np.pad(
                        raw_vol, ((0, z_diff), (0, y_diff), (0, x_diff)))
                    assert raw_vol.shape == orig_shape, f"Shape of last timepoint still doesn't match. Got {raw_vol.shape}"

            # If deconvolution is checked
            if decon_value and func != crop_volume_deskew:
                # Use CUDA or skimage for deconvolution based on user choice
                if decon_option == DeconvolutionChoice.cuda_gpu:
                    raw_vol = pycuda_decon(image=raw_vol,
                                           #otf_path = LLSZWidget.LlszMenu.lattice.otf_path[ch],
                                           psf=lattice_class.psf[ch],
                                           dzdata=lattice_class.dz,
                                           dxdata=lattice_class.dx,
                                           dzpsf=lattice_class.dz,
                                           dxpsf=lattice_class.dx,
                                           num_iter=lattice_class.psf_num_iter)
                else:
                    raw_vol = skimage_decon(vol_zyx=raw_vol,
                                            psf=lattice_class.psf[ch],
                                            num_iter=lattice_class.psf_num_iter,
                                            clip=False, filter_epsilon=0, boundary='nearest')

            # The following will apply the user-passed function to the input image
            if func is crop_volume_deskew and decon_value == True:
                processed_vol = func(original_volume=raw_vol,
                                     deconvolution=decon_value,
                                     decon_processing=decon_option,
                                     psf=lattice_class.psf[ch],
                                     num_iter=lattice_class.psf_num_iter,
                                     *args, **kwargs).astype(image_type)

            elif func is cle.deskew_y or func is cle.deskew_x:
                processed_vol = func(input_image=raw_vol,
                                     *args, **kwargs).astype(image_type)

            elif func is crop_volume_deskew:
                processed_vol = func(
                    original_volume=raw_vol, *args, **kwargs).astype(image_type)
            else:
                # if its not deskew or crop/deskew, apply the user-passed function and any specific parameters
                processed_vol = func(*args, **kwargs).astype(image_type)

            processed_vol = cle.pull_zyx(processed_vol)

            if save_file_type == SaveFileType.h5:
                # convert opencl array to dask array
                #pvol = da.asarray(processed_vol)
                # channel and time index are based on loop iteration
                bdv_writer.append_view(processed_vol,
                                       time=loop_time_idx,
                                       channel=loop_ch_idx,
                                       voxel_size_xyz=(dx, dy, new_dz),
                                       voxel_units='um')

                #print("\nAppending volume to h5\n")
            else:
                images_array.append(processed_vol)

        # if function is not for cropping, then dataset can be quite large, so save each channel and timepoint separately
        # otherwise, append it into im_final

        if func != crop_volume_deskew and save_file_type == SaveFileType.tiff:
            final_name = save_path + os.sep + save_name_prefix + "C" + str(ch) + "T" + str(
                time_point) + "_" + save_name+".tif"
            images_array = np.array(images_array)
            images_array = np.expand_dims(images_array, axis=0)
            images_array = np.swapaxes(images_array, 1, 2)
            imwrite(final_name,
                    images_array,
                    bigtiff=True,
                    resolution=(1./dx, 1./dy, "MICROMETER"),
                    metadata={'spacing': new_dz, 'unit': 'um', 'axes': 'TZCYX'}, imagej=True)
        elif save_file_type == SaveFileType.tiff:
            # convert list of arrays into a numpy array and then append to im_final
            im_final.append(np.array(images_array))

    # close the h5 writer or if its tiff, save images
    if save_file_type == SaveFileType.h5:
        bdv_writer.write_xml()
        bdv_writer.close()

    elif func is crop_volume_deskew and save_file_type == SaveFileType.tiff:

        im_final = np.array(im_final)
        final_name = save_path + os.sep + save_name_prefix + "_" + save_name + ".tif"

        im_final = np.swapaxes(im_final, 1, 2)
        # imagej=True; ImageJ hyperstack axes must be in TZCYXS order

        imwrite(final_name,
                im_final,
                # specify resolution unit for consistent metadata)
                resolution=(1./dx, 1./dy, "MICROMETER"),
                metadata={'spacing': new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                imagej=True)
        im_final = None

    return


def save_img_workflow(vol,
                      workflow: Workflow,
                      input_arg: str,
                      first_task: str,
                      last_task: str,
                      time_start: int,
                      time_end: int,
                      channel_start: int,
                      channel_end: int,
                      save_file_type,
                      save_path: Path,
                      save_name_prefix: str = "",
                      save_name: str = "img",
                      dx: float = 1,
                      dy: float = 1,
                      dz: float = 1,
                      angle: float = None,
                      deconvolution: bool = False,
                      decon_processing: str = None,
                      psf_arg=None,
                      psf=None,
                      otf_path=None):
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

    # TODO: Implement h5 saving

    save_path = save_path.__str__()

    # replace any : with _ and remove spaces in case it hasn't been processed/skipped
    save_name = save_name.replace(":", "_").replace(" ", "")

    # adding +1 at the end so the last channel and time is included

    time_range = range(time_start, time_end + 1)
    channel_range = range(channel_start, channel_end + 1)

    # Calculate new_pixel size in z
    # convert voxel sixes to an aicsimage physicalpixelsizes object for metadata
    if angle:
        import math
        new_dz = math.sin(angle * math.pi / 180.0) * dz
        #aics_image_pixel_sizes = PhysicalPixelSizes(new_dz,dy,dx)
    else:
        #aics_image_pixel_sizes = PhysicalPixelSizes(dz,dy,dx)
        new_dz = dz

    # get list of all functions in the workflow
    workflow_functions = [i[0] for i in workflow._tasks.values()]

    # iterate through time and channels and apply workflow
    # TODO: add error handling so the image writers will "close",if an error causes the program to exit
    # try except?
    for loop_time_idx, time_point in enumerate(tqdm(time_range, desc="Time", position=0)):
        output_array = []
        data_table = []
        for loop_ch_idx, ch in enumerate(tqdm(channel_range, desc="Channels", position=1, leave=False)):

            if len(vol.shape) == 3:
                raw_vol = vol
            else:
                raw_vol = vol[time_point, ch, :, :, :]

            # TODO: disable if support for resourc backed dask array is added
            # if type(raw_vol) in [resource_backed_dask_array]:
            # raw_vol = raw_vol.compute() #convert to numpy array as resource backed dask array not su

            # to access current time and channel, create a file config.py in same dir as workflow or in home directory
            # add "channel = 0" and "time=0" in the file and save
            # https://docs.python.org/3/faq/programming.html?highlight=global#how-do-i-share-global-variables-across-modules

            config.channel = ch
            config.time = time_point

            # if deconvolution, need to define psf and choose the channel appropriate one
            if deconvolution:
                workflow.set(psf_arg, psf[ch])
                # if decon_processing == "cuda_gpu":
                # workflow.set("psf",psf[ch])
                # else:
                # workflow.set("psf",psf[ch])

            # Add a check for last timepoint, in case acquisition incomplete or an incomplete frame
            if time_point == time_end:
                orig_shape = raw_vol.shape
                raw_vol = raw_vol.compute()
                if raw_vol.shape != orig_shape:
                    print(
                        f"Time {time_point}, channel {ch} is incomplete. Actual shape {orig_shape}, got {raw_vol.shape}")
                    z_diff, y_diff, x_diff = np.subtract(
                        orig_shape, raw_vol.shape)
                    print(f"Padding with{z_diff,y_diff,x_diff}")
                    raw_vol = np.pad(
                        raw_vol, ((0, z_diff), (0, y_diff), (0, x_diff)))
                    assert raw_vol.shape == orig_shape, f"Shape of last timepoint still doesn't match. Got {raw_vol.shape}"

            # Set input to the workflow to be volume from each time point and channel
            workflow.set(input_arg, raw_vol)
            # execute workflow
            processed_vol = workflow.get(last_task)

            output_array.append(processed_vol)

        output_array = np.array(output_array)

        # if workflow returns multiple objects (images, dictionaries, lsits etc..), each object can be accessed by
        # output_array[:,index_of_object]

        # use data from first timepoint to get the output type from workflow
        # check if multiple objects in the workflow output, if so, get the index for each item
        # currently, images, lists and dictionaries are supported
        if loop_time_idx == 0:

            # get no of elements

            no_elements = len(processed_vol)
            # initialize lsits to hold indexes for each datatype
            list_element_index = []  # store indices of lists
            dict_element_index = []  # store indices of dicts
            # store indices of images (numpy array, dask array and pyclesperanto array)
            image_element_index = []

            # single output and is just dictionary
            if type(processed_vol) in [dict]:
                dict_element_index = [0]
            # if image
            elif type(processed_vol) in [np.ndarray, cle._tier0._pycl.OCLArray, da.core.Array, resource_backed_dask_array.ResourceBackedDaskArray]:
                image_element_index = [0]
                no_elements = 1
            # multiple elements
            # list with values returns no_elements>1 so make sure its actually a list with different objects
            # test this with different workflows
            elif no_elements > 1 and type(processed_vol[0]) not in [np.int16, np.int32, np.float16, np.float32, np.float64, int, float]:
                array_element_type = [type(output_array[0, i])
                                      for i in range(no_elements)]
                image_element_index = [idx for idx, data_type in enumerate(
                    array_element_type) if data_type in [np.ndarray, cle._tier0._pycl.OCLArray, da.core.Array]]
                dict_element_index = [idx for idx, data_type in enumerate(
                    array_element_type) if data_type in [dict]]
                list_element_index = [idx for idx, data_type in enumerate(
                    array_element_type) if data_type in [list]]
            elif type(processed_vol) is list:
                list_element_index = [0]

            # setup required image writers
            if len(image_element_index) > 0:
                # pass list of images and index to function
                writer_list = []
                # create an image writer for each image
                for element in range(len(image_element_index)):
                    final_save_path = save_path + os.sep + save_name_prefix + "_" + \
                        str(element)+"_" + save_name + \
                        "." + save_file_type.value
                    # setup writer based on user choice of filetype
                    if save_file_type == SaveFileType.h5:
                        bdv_writer = npy2bdv.BdvWriter(final_save_path,
                                                       compression='gzip',
                                                       nchannels=len(
                                                           channel_range),
                                                       subsamp=(
                                                           (1, 1, 1), (1, 2, 2), (2, 4, 4)),
                                                       overwrite=True)  # overwrite set to True; is this good practice?
                        writer_list.append(bdv_writer)
                    else:
                        # imagej =true throws an error
                        writer_list.append(TiffWriter(
                            final_save_path, bigtiff=True))

        # handle image saving: either h5 or tiff saving
        if len(image_element_index) > 0:
            # writer_idx is for each writers, image_idx will be the index of images
            for writer_idx, image_idx in enumerate(image_element_index):
                # access the image
                # print(len(time_range))
                if len(channel_range) == 1:
                    if (len(time_range)) == 1:  # if only one timepoint
                        im_final = np.stack(
                            output_array[image_idx, ...]).astype(raw_vol.dtype)
                    else:
                        im_final = np.stack(
                            output_array[0, image_idx]).astype(raw_vol.dtype)
                else:
                    im_final = np.stack(
                        output_array[:, image_idx]).astype(raw_vol.dtype)
                if save_file_type == SaveFileType.h5:
                    for ch_idx in channel_range:
                        # write h5 images as 3D stacks
                        assert len(
                            im_final.shape) >= 3, f"Image shape should be >=3, got {im_final.shape}"
                        # print(im_final.shape)
                        if len(im_final.shape) == 3:
                            im_channel = im_final
                        elif len(im_final.shape) > 3:
                            im_channel = im_final[ch_idx, ...]

                        writer_list[writer_idx].append_view(im_channel,
                                                            time=loop_time_idx,
                                                            channel=loop_ch_idx,
                                                            voxel_size_xyz=(
                                                                dx, dy, new_dz),
                                                            voxel_units='um')
                else:  # default to tif
                    # Use below with imagej=True
                    # if len(im_final.shape) ==4: #if only one image with no channel, then dimension will 1,z,y,x, so swap 0 and 1
                    #    im_final = np.swapaxes(im_final,0,1).astype(raw_vol.dtype) #was 1,2,but when stacking images, dimension is CZYX
                    #    im_final = im_final[np.newaxis,...] #adding extra dimension for T
                    # elif len(im_final.shape)>4:
                    #    im_final = np.swapaxes(im_final,1,2).astype(raw_vol.dtype) #if image with multiple channels, , it will be 1,c,z,y,x
                    # imagej=True; ImageJ hyperstack axes must be in TZCYXS order
                    #images_array = np.swapaxes(images_array,0,1).astype(raw_vol.dtype)
                    writer_list[writer_idx].write(im_final,
                                                  resolution=(
                                                      1./dx, 1./dy, "MICROMETER"),
                                                  metadata={'spacing': new_dz, 'unit': 'um', 'axes': 'TZCYX', 'PhysicalSizeX': dx,
                                                            'PhysicalSizeXUnit': 'µm', 'PhysicalSizeY': dy, 'PhysicalSizeYUnit': 'µm'})
                    im_final = None

        # handle dict saving
        # convert to pandas dataframe; update columns with channel and time
        if len(dict_element_index) > 0:
            # Iterate through the dict  output from workflow and add columns for Channel and timepoint
            for element in dict_element_index:
                for j in channel_range:
                    output_array[j, element].update({"Channel": "C"+str(j)})
                    output_array[j, element].update(
                        {"Time": "T"+str(time_point)})

                # convert to pandas dataframe
                output_dict_pd = [pd.DataFrame(i)
                                  for i in output_array[:, element]]

                output_dict_pd = pd.concat(output_dict_pd)
                # set index to the channel/time
                output_dict_pd = output_dict_pd.set_index(["Time", "Channel"])

                # Save path
                dict_save_path = os.path.join(
                    save_path, "Measurement_"+save_name_prefix)
                if not (os.path.exists(dict_save_path)):
                    os.mkdir(dict_save_path)

                #dict_save_path = os.path.join(dict_save_path,"C" + str(ch) + "T" + str(time_point)+"_"+str(element) + "_measurement.csv")
                dict_save_path = os.path.join(
                    dict_save_path, "Summary_measurement_"+save_name_prefix+"_"+str(element)+"_.csv")
                # Opens csv and appends it if file already exists; not efficient.
                if os.path.exists(dict_save_path):
                    output_dict_pd_existing = pd.read_csv(
                        dict_save_path, index_col=["Time", "Channel"])
                    output_dict_summary = pd.concat(
                        (output_dict_pd_existing, output_dict_pd))
                    output_dict_summary.to_csv(dict_save_path)
                else:
                    output_dict_pd.to_csv(dict_save_path)

        if len(list_element_index) > 0:
            row_idx = []
            for element in dict_element_index:
                for j in channel_range:
                    row_idx.append("C"+str(j)+"T"+str(time_point))
                output_list_pd = pd.DataFrame(
                    np.vstack(output_array[:, element]), index=row_idx)
                # Save path
                list_save_path = os.path.join(
                    save_path, "Measurement_"+save_name_prefix)
                if not (os.path.exists(list_save_path)):
                    os.mkdir(list_save_path)
                list_save_path = os.path.join(list_save_path, "C" + str(ch) + "T" + str(
                    time_point)+"_"+save_name_prefix+"_"+str(element) + "_measurement.csv")
                output_list_pd.to_csv(list_save_path)

    if len(image_element_index) > 0:
        for writer_idx in range(len(image_element_index)):
            if save_file_type == SaveFileType.h5:
                # write h5 metadata
                writer_list[writer_idx].write_xml()
            # close the writers (applies for both tiff and h5)
            writer_list[writer_idx].close()

    return

# class for initializing lattice data and setting metadata
# TODO: handle scenes


class LatticeData():
    def __init__(self, img, angle, skew, dx, dy, dz, last_dimension) -> None:
        self.angle = angle
        self.skew = skew

        # if image layer

        if type(img) is image.Image:  # napari layer image
            # check if its an aicsimageio object and has voxel size info
            if 'aicsimage' in img.metadata.keys() and img.metadata['aicsimage'].physical_pixel_sizes != (None, None, None):
                img_data_aics = img.metadata['aicsimage']
                self.data = img_data_aics.dask_data
                self.dims = img_data_aics.dims
                self.time = img_data_aics.dims.T
                self.channels = img_data_aics.dims.C
                self.dz, self.dy, self.dx = img_data_aics.physical_pixel_sizes
            else:
                print("Cannot read voxel size from metadata")
                if 'aicsimage' in img.metadata.keys():
                    img_data_aics = img.metadata['aicsimage']
                    self.data = img_data_aics.dask_data
                    self.dims = img_data_aics.dims
                    # if aicsimageio tiffreader assigns last dim as time when it should be channel, user can override this
                    if last_dimension:
                        if len(img.data.shape) == 4:
                            if last_dimension.lower() == "channel":
                                self.channels = img.data.shape[0]
                                self.time = 0
                            elif last_dimension.lower() == "time":
                                self.time = img.data.shape[0]
                                self.channels = 0
                        elif len(img.data.shape) == 5:
                            if last_dimension.lower() == "channel":
                                self.channels = img.data.shape[0]
                                self.time = img.data.shape[1]
                            elif last_dimension.lower() == "time":
                                self.time = img.data.shape[0]
                                self.channels = img.data.shape[1]
                    else:
                        self.time = img_data_aics.dims.T
                        self.channels = img_data_aics.dims.C
                else:
                    # if no aicsimageio key in metadata
                    # get the data and convert it into an aicsimage object
                    img_data_aics = aicsimageio.AICSImage(img.data)
                    self.data = img_data_aics.dask_data
                    # if user has specified ch
                    if last_dimension:
                        if len(img.data.shape) == 4:
                            if last_dimension.lower() == "channel":
                                self.channels = img.data.shape[0]
                                self.time = 0
                            elif last_dimension.lower() == "time":
                                self.time = img.data.shape[0]
                                self.channels = 0
                        elif len(img.data.shape) == 5:
                            if last_dimension.lower() == "channel":
                                self.channels = img.data.shape[0]
                                self.time = img.data.shape[1]
                            elif last_dimension.lower() == "time":
                                self.time = img.data.shape[0]
                                self.channels = img.data.shape[1]
                    else:
                        if last_dimension:
                            if len(img.data.shape) == 4:
                                if last_dimension.lower() == "channel":
                                    self.channels = img.data.shape[0]
                                    self.time = 0
                                elif last_dimension.lower() == "time":
                                    self.time = img.data.shape[0]
                                    self.channels = 0
                            elif len(img.data.shape) == 5:
                                if last_dimension.lower() == "channel":
                                    self.channels = img.data.shape[0]
                                    self.time = img.data.shape[1]
                                elif last_dimension.lower() == "time":
                                    self.time = img.data.shape[0]
                                    self.channels = img.data.shape[1]
                        else:

                            self.time = img.data.shape[0]
                            self.channels = img.data.shape[1]

                # read metadata for pixel sizes
                if None in img_data_aics.physical_pixel_sizes or img_data_aics.physical_pixel_sizes == False:
                    self.dx = dx
                    self.dy = dy
                    self.dz = dz
                else:
                    self.dz, self.dy, self.dx = img.data.physical_pixel_sizes
                # if not last_dimension:
                # if xarray, access data using .data method
                    # if type(img.data) in [xarray.core.dataarray.DataArray,np.ndarray]:
                    #img = img.data
                # img = dask_expand_dims(img,axis=1) ##if no channel dimension specified, then expand axis at index 1
            # if no path returned by source.path, get image name with colon and spaces removed
            # if last axes of "aicsimage data" shape is not equal to time, then swap channel and time
            if self.data.shape[0] != self.time or self.data.shape[1] != self.channels:
                self.data = np.swapaxes(self.data, 0, 1)

            if img.source.path is None:
                # remove colon (:) and any leading spaces
                self.save_name = img.name.replace(":", "").strip()
                # replace any group of spaces with "_"
                self.save_name = '_'.join(self.save_name.split())

            else:
                file_name_noext = os.path.basename(img.source.path)
                file_name = os.path.splitext(file_name_noext)[0]
                # remove colon (:) and any leading spaces
                self.save_name = file_name.replace(":", "").strip()
                # replace any group of spaces with "_"
                self.save_name = '_'.join(self.save_name.split())

        elif type(img) in [np.ndarray, da.core.Array, resource_backed_dask_array.ResourceBackedDaskArray]:
            img_data_aics = aicsimageio.AICSImage(img.data)
            self.data = img_data_aics.dask_data
            self.dx = dx
            self.dy = dy
            self.dz = dz

        elif type(img) is aicsimageio.aics_image.AICSImage:

            if img.physical_pixel_sizes != (None, None, None):
                self.data = img.dask_data
                self.dims = img.dims
                self.time = img.dims.T
                self.channels = img.dims.C
                self.dz, self.dy, self.dx = img.physical_pixel_sizes

            else:
                self.data = img.dask_data
                self.dims = img.dims
                self.time = img.dims.T
                self.channels = img.dims.C
                self.dz, self.dy, self.dx = dz, dy, dx

        else:
            raise Exception(
                "Has to be an image layer or array, got type: ", type(img))

        # set new z voxel size
        if self.skew == DeskewDirection.Y or self.skew == DeskewDirection.X:
            import math
            self.new_dz = math.sin(self.angle * math.pi / 180.0) * self.dz

        # process the file to get shape of final deskewed image
        self.deskew_vol_shape, self.deskew_affine_transform = get_deskewed_shape(
            self.data, self.angle, self.dx, self.dy, self.dz)
        print(f"Channels: {self.channels}, Time: {self.time}")
        print("If channel and time need to be swapped, you can enforce this by choosing 'Last dimension is channel' when initialising the plugin")

        pass

    def get_angle(self):
        return self.angle

    def set_angle(self, angle: float):
        self.angle = angle

    def set_skew(self, skew):
        self.skew = skew
