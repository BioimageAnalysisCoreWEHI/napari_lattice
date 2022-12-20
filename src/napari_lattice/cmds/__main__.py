# lattice_processing.py
# Run processing on command line instead of napari.
# Example for deskewing files in a folder
# python lattice_processing.py --input /home/pradeep/to_deskew --output /home/pradeep/output_save/ --processing deskew

import argparse
import os
import glob
import sys
import re
from napari_lattice.io import LatticeData, save_img, save_img_workflow
from napari_lattice.utils import read_imagej_roi, get_all_py_files, get_first_last_image_and_task, modify_workflow_task, check_dimensions
from napari_lattice.llsz_core import crop_volume_deskew
from aicsimageio import AICSImage
import pyclesperanto_prototype as cle
from tqdm import tqdm
import dask.array as da
from napari_workflows import Workflow, WorkflowManager
from napari_workflows._io_yaml_v1 import load_workflow
from pathlib import Path
import yaml
from .. import config

from .._dock_widget import _napari_lattice_widget_wrapper

from ..ui_core import _read_psf
from .. import DeskewDirection, DeconvolutionChoice, SaveFileType
from enum import Enum


# define parser class so as to print help message
class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


class ProcessingOptions(Enum):
    deskew = "deskew"
    crop = "crop"
    workflow = "workflow"
    workflow_crop = "workflow_crop"


def args_parse():
    """ Parse input arguments"""
    parser = argparse.ArgumentParser(description="Lattice Data Analysis")
    parser.add_argument('--input', type=str, nargs=1, help="Enter input file",
                        required=False)  # only false if using config file
    parser.add_argument('--output', type=str, nargs=1, help="Enter save folder",
                        required=False)  # only false if using config file
    parser.add_argument('--skew_direction', type=DeskewDirection, nargs=1,
                        help="Enter the direction of skew (default is Y)",
                        action="store",
                        choices=("Y", "X"),
                        default=DeskewDirection.Y)
    parser.add_argument('--deskew_angle', type=float, nargs=1,
                        help="Enter the deskew angle (default is 30)",
                        default=30.0)
    parser.add_argument('--processing', type=ProcessingOptions, nargs=1,
                        help="Enter the processing option: deskew, crop, workflow or workflow_crop", required=False,
                        action="store",
                        choices=(ProcessingOptions.deskew, ProcessingOptions.crop,
                                 ProcessingOptions.workflow, ProcessingOptions.workflow_crop),
                        default=None)
    parser.add_argument('--deconvolution', type=DeconvolutionChoice, nargs=1,
                        help="Specify the device to use for deconvolution. Options are cpu or cuda_gpu",
                        action="store")
    parser.add_argument('--deconvolution_num_iter', type=int, nargs=1,
                        help="Enter the number of iterations to run Richardson-Lucy deconvolution (default is 10)")
    parser.add_argument('--deconvolution_psf', type=str, nargs="+",
                        help="Enter paths to psf file/s separated by commas or you can enter each path with double quotes")  # use + for nargs for flexible no of args
    parser.add_argument('--roi_file', type=str, nargs=1,
                        help="Enter the path to the ROI file for performing cropping (only valid for -processing where crop or workflow_crop is specified")
    parser.add_argument('--voxel_sizes', type=tuple, nargs=1,
                        help="Enter the voxel sizes as (dz,dy,dx). Make sure they are in brackets",
                        default=(0.3, 0.1499219272808386, 0.1499219272808386))
    parser.add_argument('--file_extension', type=str, nargs=1,
                        help="If choosing a folder, enter the extension of the files (make sure you enter it with the dot at the start, i.e., .czi or .tif), else .czi and .tif files will be used")
    parser.add_argument('--time_range', type=int, nargs=2,
                        help="Enter time range to extract, default will be entire timeseries if no range is specified. For example, 0 9 will extract first 10 timepoints",
                        default=(None, None))
    parser.add_argument('--channel_range', type=int, nargs=2,
                        help="Enter channel range to extract, default will be all channels if no range is specified. For example, 0 1 will extract first two channels.",
                        default=(None, None))
    parser.add_argument('--workflow_path', type=str, nargs=1,
                        help="Enter path to the workflow file '.yml", default=[None])
    parser.add_argument('--output_file_type', type=SaveFileType, nargs=1,
                        help="Save as either tiff or h5, defaults to h5",
                        action="store",
                        choices=(SaveFileType.tiff, SaveFileType.h5),
                        default=[SaveFileType.h5]),
    parser.add_argument('--channel', type=bool, nargs=1,
                        help="If input is a tiff file and there are channel dimensions but no time dimensions, choose as True",
                        default=False)
    parser.add_argument('--config', type=str, nargs=1,
                        help="Location of config file, all other arguments will be ignored and overwriten by those in the yaml file", default=None)
    parser.add_argument('--roi_number', type=int, nargs=1,
                        help="Process an individual ROI, loop all if unset", default=None)
    parser.add_argument('--set_logging', type=str, nargs=1,
                        help="Set logging level [INFO,DEBUG]", default=["INFO"])

    args = parser.parse_args()
    return args


def main():
    args = args_parse()

    # Enable Logging
    import logging
    logger = logging.getLogger(__name__)

    # Setting empty strings for psf paths
    psf_ch1_path = ""
    psf_ch2_path = ""
    psf_ch3_path = ""
    psf_ch4_path = ""

    # IF using a config file, set a lot of parameters here
    # the rest are scattered throughout the code when needed
    # could be worth bringing everything up top
    print(args.config)

    if args.config:
        try:
            with open(args.config[0], 'r') as con:
                try:
                    processing_parameters = yaml.safe_load(con)

                except Exception as exc:
                    print(exc)

        except FileNotFoundError as exc:
            exit(f"Config yml file {args.config[0]} not found, please specify")

        if not processing_parameters:
            logging.error(f"Config file not loaded properly")
        # this is how I propose setting the command line variables
        # If they're in the config, get from there. If not
        # Look in command line. If not there, then exit
        if 'input' in processing_parameters:
            input_path = processing_parameters['input']
        elif args.input is not None:
            input_path = args.input[0]
        else:
            exit("Input not set")
        print("Processing file %s" % input_path)

        if 'output' in processing_parameters:
            output_path = processing_parameters['output']
        elif args.output is not None:
            output_path = args.output[0] + os.sep
        else:
            exit("Output not set")

        # If requried setting not in config file = use defaults from argparse
        # get method for a dictionary will get a value if specified, if not, will use value from args as default value

        dz, dy, dx = processing_parameters.get('voxel_sizes', args.voxel_sizes)
        channel_dimension = processing_parameters.get('channel', args.channel)
        skew_dir = processing_parameters.get(
            'skew_direction', DeskewDirection.Y)
        deskew_angle = processing_parameters.get('deskew_angle', 30.0)
        processing = ProcessingOptions[processing_parameters.get(
            'processing', None).lower()]
        time_start, time_end = processing_parameters.get(
            'time_range', (None, None))
        channel_start, channel_end = processing_parameters.get(
            'channel_range', (None, None))
        output_file_type = SaveFileType[processing_parameters.get(
            'output_file_type', args.output_file_type).lower()]

        # to allow for either/or CLI/config file Todo for rest of parameters?
        if 'roi_number' in processing_parameters:
            roi_to_process = processing_parameters.get('roi_number')
        elif args.roi_number is not None:
            roi_to_process = args.roi_number[0]
        else:
            roi_to_process = None

        log_level = processing_parameters.get('--set_logging', "INFO")
        workflow_path = processing_parameters.get('workflow_path', None)
        if workflow_path is not None:
            workflow_path = Path(workflow_path)

        logging.basicConfig(level=log_level.upper())
        logging.info(f"Logging set to {log_level.upper()}")

        if not processing:
            logging.error("Processing option not set.")
            exit()

        deconvolution = processing_parameters.get('deconvolution', None)
        if deconvolution is not None:
            deconvolution = DeconvolutionChoice[deconvolution.lower()]
            psf_arg = "psf"
            deconvolution_num_iter = processing_parameters.get(
                'deconvolution_num_iter', 10)
            psf_paths = processing_parameters.get('deconvolution_psf')
            logging.debug(psf_paths)
            if not psf_paths:
                logging.error("PSF paths not set option not set.")
                exit()
            else:
                psf_ch1_path = psf_paths[0].replace(",", "").strip()
                psf_ch2_path = psf_paths[1].replace(",", "").strip()
                psf_ch3_path = psf_paths[2].replace(",", "").strip()
                psf_ch4_path = psf_paths[3].replace(",", "").strip()

        else:
            deconvolution = False

        if processing == ProcessingOptions.crop or processing == ProcessingOptions.workflow_crop:
            if 'roi_file' in processing_parameters:
                roi_file = processing_parameters.get('roi_file', False)
            elif args.roi_file is not None:
                roi_file = args.roi_file[0]
            else:
                exit("Specify roi file")
            assert os.path.exists(roi_file), "Cannot find " + roi_file
            print("Processing using roi file %s" % roi_file)

        assert os.path.exists(input_path), "Cannot find input " + input_path
        assert os.path.exists(output_path), "Cannot find output " + output_path

        file_extension = processing_parameters.get(
            'file_extension', [".czi", ".tif", ".tiff"])

    # setting (some - see above) parameters from CLI
    else:  # if not using config file
        input_path = args.input[0]
        output_path = args.output[0] + os.sep
        dz, dy, dx = args.voxel_sizes
        deskew_angle = args.deskew_angle
        channel_dimension = args.channel
        time_start, time_end = args.time_range
        channel_start, channel_end = args.channel_range
        skew_dir = args.skew_direction
        processing = args.processing[0]
        output_file_type = args.output_file_type[0]
        roi_to_process = args.roi_number
        workflow_path = args.workflow_path[0]

        log_level = args.set_logging[0]
        logging.basicConfig(level=log_level.upper())
        logging.info(f"Logging set to {log_level.upper()}")

        if roi_to_process:
            logging.info(f"Processing ROI {roi_to_process}")

        if not processing:
            logging.error("Processing option not set.")
            exit()

        # deconvolution
        if args.deconvolution:
            deconvolution = args.deconvolution[0]
            psf_arg = "psf"
            if args.deconvolution_psf:
                psf_paths = re.split(';|,', args.deconvolution_psf[0])
                logging.debug(psf_paths)
                try:
                    psf_ch1_path = psf_paths[0]
                    psf_ch2_path = psf_paths[1]
                    psf_ch3_path = psf_paths[2]
                    psf_ch4_path = psf_paths[3]
                except IndexError as e:
                    pass
            else:
                logging.error("PSF paths not set option not set.")
                exit()
            # num of iter default is 10 if nothing specified
            if args.deconvolution_num_iter:
                deconvolution_num_iter = args.deconvolution_num_iter
            else:
                deconvolution_num_iter = 10
        else:
            deconvolution = False

        # output file type to save
        if not output_file_type:
            output_file_type = SaveFileType.h5

        # Get
        if processing == ProcessingOptions.crop or processing == ProcessingOptions.workflow_crop:
            assert args.roi_file, "Specify roi_file (ImageJ/FIJI ROI Zip file)"
            roi_file = args.roi_file[0]
            if os.path.isfile(roi_file):  # if file make sure it is a zip file or roi file
                roi_file_extension = os.path.splitext(roi_file)[1]
                assert roi_file_extension == ".zip" or roi_file_extension == "roi", "ROI file is not a zip or .roi file"

        # Check if input and output paths exist
        assert os.path.exists(input_path), "Cannot find input " + input_path
        assert os.path.exists(output_path), "Cannot find output " + output_path

        if not args.file_extension:
            file_extension = [".czi", ".tif", ".tiff"]
        else:
            file_extension = args.file_extension

    # Initialise list of images and ROIs
    img_list = []
    roi_list = []
    logging.debug(f"Deconvolution is set to {deconvolution} and option is ")

    logging.debug(f"Output file type is {output_file_type}")
    # If input_path a directory, get a list of images
    if os.path.isdir(input_path):
        for file_type in file_extension:
            img_list.extend(glob.glob(input_path + os.sep + '*' + file_type))
        print("List of images: ", img_list)
    elif os.path.isfile(input_path) and (os.path.splitext(input_path))[1] in file_extension:
        # if a single file, just add filename to the image list
        img_list.append(input_path)
    else:
        sys.exit("Do not recognise " + input_path + " as directory or file")

    # If cropping, get list of roi files with matching image names
    if processing == ProcessingOptions.crop or processing == ProcessingOptions.workflow_crop:
        if os.path.isdir(roi_file):
            for img in img_list:
                img_name = os.path.basename(os.path.splitext(img)[0])
                roi_temp = roi_file + os.sep + img_name + ".zip"

                if os.path.exists(roi_temp):
                    roi_list.append(roi_temp)
                else:
                    sys.exit("Cannot find ROI file for " + img)

            print("List of ROIs: ", roi_list)
        elif os.path.isfile(roi_file):
            roi_list.append(roi_file)
        assert len(roi_list) == len(
            img_list), "Image and ROI lists do not match"
    else:
        # add list of empty strings so that it can run through for loop
        no_files = len(img_list)
        roi_list = [""] * no_files

    # loop through the list of images and rois
    for img, roi_path in zip(img_list, roi_list):
        print("Processing Image " + img)
        if processing == ProcessingOptions.crop or processing == ProcessingOptions.workflow_crop:
            print("Processing ROI " + roi_path)
        aics_img = AICSImage(img)

        # check if scene valid; if not it iterates through all scenes
        len_scenes = len(aics_img.scenes)
        for scene in range(len_scenes):
            aics_img.set_scene(scene)
            test = aics_img.get_image_dask_data("YX", T=0, C=0, Z=0)
            try:
                test_max = test.max().compute()
                if test_max:
                    print(f"Scene {scene} is valid")
                    break
            except Exception as e:
                print(f"Scene {scene} not valid")

        # Initialize Lattice class
        lattice = LatticeData(aics_img, deskew_angle,
                              skew_dir, dx, dy, dz, channel_dimension)

        # Chance deskew function absed on skew direction
        if lattice.skew == DeskewDirection.Y:
            lattice.deskew_func = cle.deskew_y
            lattice.skew_dir = DeskewDirection.Y

        elif lattice.skew == DeskewDirection.X:
            lattice.deskew_func = cle.deskew_x
            lattice.skew_dir = DeskewDirection.X

        if time_start is None or time_end is None:
            time_start, time_end = 0, lattice.time - 1

        if channel_start is None or channel_end is None:
            channel_start, channel_end = 0, lattice.channels - 1

        # Verify dimensions
        check_dimensions(time_start, time_end, channel_start,
                         channel_end, lattice.channels, lattice.time)
        print("dimensions verified")
        # If deconvolution, set the parameters in the lattice class
        if deconvolution:
            lattice.decon_processing = deconvolution
            lattice.psf_num_iter = deconvolution_num_iter
            logging.debug(f"Num of iterations decon, {lattice.psf_num_iter}")
            logging.info("DECONVOLUTIONING!")
            lattice.psf = []
            lattice.otf_path = []
            # todo this should maybe go somewhere else?
            _read_psf(psf_ch1_path,
                      psf_ch2_path,
                      psf_ch3_path,
                      psf_ch4_path,
                      decon_option=lattice.decon_processing,
                      lattice_class=lattice)

        else:
            lattice.decon_processing = None

        # Override pixel values by reading metadata if file is czi
        if os.path.splitext(img)[1] == ".czi":
            dz, dy, dx = lattice.dz, lattice.dy, lattice.dx
            logging.info(f"Pixel values from metadata (zyx): {dz},{dy},{dx}")

        # Setup workflows based on user input
        if processing == ProcessingOptions.workflow or processing == ProcessingOptions.workflow_crop:
            # load workflow from path
            # if args.config and 'workflow_path' in processing_parameters:
            #workflow_path = Path(processing_parameters['workflow_path'])
            # else:
            #workflow_path = Path(workflow_path)

            # load custom modules (*.py) in same directory as workflow file
            import importlib
            parent_dir = workflow_path.resolve().parents[0].__str__() + os.sep

            sys.path.append(parent_dir)
            custom_py_files = get_all_py_files(parent_dir)

            if len(custom_py_files) > 0:
                modules = map(importlib.import_module, custom_py_files)
                logging.info(f"Custom modules imported {modules}")

            # workflow has to be reloaded for each image and reinitialised
            user_workflow = load_workflow(workflow_path.__str__())
            assert type(
                user_workflow) is Workflow, "Workflow file is not a napari workflow object. Check file!"

            input_arg_first, input_arg_last, first_task_name, last_task_name = get_first_last_image_and_task(
                user_workflow)
            logging.debug(input_arg_first, input_arg_last,
                          first_task_name, last_task_name)

            # get list of tasks
            task_list = list(user_workflow._tasks.keys())

            print("Workflow loaded:")
            logging.info(user_workflow)

            task_name_start = first_task_name[0]
            try:
                task_name_last = last_task_name[0]
            except IndexError:
                task_name_last = task_name_start

            # if workflow involves cropping, assign first task as crop_volume_deskew
            if processing == ProcessingOptions.workflow_crop:
                deskewed_shape = lattice.deskew_vol_shape
                deskewed_volume = da.zeros(deskewed_shape)
                z_start = 0
                z_end = deskewed_shape[0]
                roi = "roi"
                volume = "volume"
                # Create workflow for cropping and deskewing
                # volume and roi used will be set dynamically
                user_workflow.set("crop_deskew", crop_volume_deskew,
                                  original_volume=volume,
                                  deskewed_volume=deskewed_volume,
                                  roi_shape=roi,
                                  angle_in_degrees=deskew_angle,
                                  voxel_size_x=dx,
                                  voxel_size_y=dy,
                                  voxel_size_z=dz,
                                  z_start=z_start,
                                  z_end=z_end,
                                  skew_dir=lattice.skew)
                # change the first task so it accepts "crop_deskew as input"
                new_task = modify_workflow_task(
                    old_arg=input_arg_first, task_key=task_name_start, new_arg="crop_deskew", workflow=user_workflow)
                user_workflow.set(task_name_start, new_task)

            elif processing == ProcessingOptions.workflow:
                # Verify if deskewing function is in workflow; if not, add as first task
                if user_workflow.get_task(task_name_start)[0] not in (cle.deskew_y, cle.deskew_x):
                    custom_workflow = True
                    input = "input"

                    # add task to the workflow
                    user_workflow.set("deskew_image", lattice.deskew_func,
                                      input_image=input,
                                      angle_in_degrees=deskew_angle,
                                      voxel_size_x=dx,
                                      voxel_size_y=dy,
                                      voxel_size_z=dz,
                                      linear_interpolation=True)
                    # Set input of the workflow to be from deskewing
                    # change the first task so it accepts "deskew_image" as input
                    new_task = modify_workflow_task(old_arg=input_arg_first, task_key=task_name_start,
                                                    new_arg="deskew_image", workflow=user_workflow)
                    user_workflow.set(task_name_start, new_task)
                else:
                    custom_workflow = False

        img_data = lattice.data

        save_name = os.path.splitext(os.path.basename(img))[0]

        # Create save directory for each image
        save_path = output_path + os.sep + \
            os.path.basename(os.path.splitext(img)[0]) + os.sep

        if not os.path.exists(save_path):
            try:
                os.mkdir(save_path)
            except FileExistsError:
                # this is sometimes caused when running parallel jobs
                # can safely be ignored (I hope)
                pass

        logging.info(f"Saving at {save_path}")

        # Deskewing only

    if processing == ProcessingOptions.deskew:
        # deconvolution
        if lattice.decon_processing:
            save_img(vol=img_data,
                     func=lattice.deskew_func,
                     time_start=time_start,
                     time_end=time_end,
                     channel_start=channel_start,
                     channel_end=channel_end,
                     save_path=save_path,
                     save_name=save_name,
                     save_file_type=output_file_type,
                     dx=dx,
                     dy=dy,
                     dz=dz,
                     angle=deskew_angle,
                     terminal=True,
                     lattice=lattice,
                     angle_in_degrees=deskew_angle,
                     voxel_size_x=dx,
                     voxel_size_y=dy,
                     voxel_size_z=dz,
                     linear_interpolation=True
                     )

        else:
            save_img(vol=img_data,
                     func=lattice.deskew_func,
                     time_start=time_start,
                     time_end=time_end,
                     channel_start=channel_start,
                     channel_end=channel_end,
                     save_path=save_path,
                     save_name=save_name,
                     save_file_type=output_file_type,
                     dx=dx,
                     dy=dy,
                     dz=dz,
                     angle=deskew_angle,
                     angle_in_degrees=deskew_angle,
                     voxel_size_x=dx,
                     voxel_size_y=dy,
                     voxel_size_z=dz
                     )

    # Crop and deskew
    elif processing == ProcessingOptions.crop or processing == ProcessingOptions.workflow_crop:
        print(roi_path)
        roi_img = read_imagej_roi(roi_path)

        deskewed_shape = lattice.deskew_vol_shape
        deskewed_volume = da.zeros(deskewed_shape)

        # Can modify for entering custom z values
        z_start = 0
        z_end = deskewed_shape[0]

        # if roi number is specified, roi_img will be a list containing only one roi.
        if roi_to_process is not None:
            # just do one ROI
            assert roi_to_process < len(
                roi_img), f"ROI specified is {roi_to_process}, which is less than total ROIs ({len(roi_img)})"
            logging.info(f"Processing single ROI: {roi_to_process}")
            # If only one ROI, single loop
            roi_img = [roi_img[roi_to_process]]

        # loop through rois in the roi list
        for idx, roi_layer in enumerate(tqdm(roi_img, desc="ROI:", position=0)):

            if roi_to_process is not None:
                roi_label = str(roi_to_process)
            else:
                roi_label = str(idx)

            print("Processing ROI " + str(idx) +
                  " of " + str(len(roi_img)))
            deskewed_shape = lattice.deskew_vol_shape
            deskewed_volume = da.zeros(deskewed_shape)

            # Can modify for entering custom z values
            z_start = 0
            z_end = deskewed_shape[0]

            if processing == ProcessingOptions.crop:
                # deconvolution
                if lattice.decon_processing:
                    save_img(img_data,
                             func=crop_volume_deskew,
                             time_start=time_start,
                             time_end=time_end,
                             channel_start=channel_start,
                             channel_end=channel_end,
                             save_name_prefix="ROI_" + roi_label + "_",
                             save_path=save_path,
                             save_name=save_name,
                             save_file_type=output_file_type,
                             dx=dx,
                             dy=dy,
                             dz=dz,
                             angle=deskew_angle,
                             terminal=True,
                             lattice=lattice,
                             deskewed_volume=deskewed_volume,
                             roi_shape=roi_layer,
                             angle_in_degrees=deskew_angle,
                             z_start=z_start,
                             z_end=z_end,
                             voxel_size_x=dx,
                             voxel_size_y=dy,
                             voxel_size_z=dz,
                             )
                else:
                    print("SHOULD BE DOING THIS")
                    print(save_path)
                    print(save_name)
                    save_img(img_data,
                             func=crop_volume_deskew,
                             time_start=time_start,
                             time_end=time_end,
                             channel_start=channel_start,
                             channel_end=channel_end,
                             save_name_prefix="ROI_" + roi_label + "_",
                             save_path=save_path,
                             save_name=save_name,
                             save_file_type=output_file_type,
                             dx=dx,
                             dy=dy,
                             dz=dz,
                             angle=deskew_angle,
                             deskewed_volume=deskewed_volume,
                             roi_shape=roi_layer,
                             angle_in_degrees=deskew_angle,
                             z_start=z_start,
                             z_end=z_end,
                             voxel_size_x=dx,
                             voxel_size_y=dy,
                             voxel_size_z=dz,
                             )

            elif processing == ProcessingOptions.workflow_crop:
                # deconvolution
                user_workflow.set(roi, roi_layer)

                if lattice.decon_processing:
                    save_img_workflow(vol=img_data,
                                      workflow=user_workflow,
                                      input_arg=volume,
                                      first_task="crop_deskew",
                                      last_task=task_name_last,
                                      time_start=time_start,
                                      time_end=time_end,
                                      channel_start=channel_start,
                                      channel_end=channel_end,
                                      save_path=save_path,
                                      save_name_prefix="ROI_" + roi_label,
                                      save_name=save_name,
                                      save_file_type=output_file_type,
                                      dx=dx,
                                      dy=dy,
                                      dz=dz,
                                      angle=deskew_angle,
                                      deconvolution=True,
                                      decon_processing=lattice.decon_processing,
                                      psf=lattice.psf,
                                      psf_arg=psf_arg)
                else:
                    save_img_workflow(vol=img_data,
                                      workflow=user_workflow,
                                      input_arg=volume,
                                      first_task="crop_deskew",
                                      last_task=task_name_last,
                                      time_start=time_start,
                                      time_end=time_end,
                                      channel_start=channel_start,
                                      channel_end=channel_end,
                                      save_path=save_path,
                                      save_file_type=output_file_type,
                                      save_name_prefix="ROI_" + roi_label,
                                      save_name=save_name,
                                      dx=dx,
                                      dy=dy,
                                      dz=dz,
                                      angle=deskew_angle,
                                      deconvolution=False)

    elif processing == ProcessingOptions.workflow:
        # if deskew_image task set above manually
        if custom_workflow:
            if lattice.decon_processing:
                save_img_workflow(vol=img_data,
                                  workflow=user_workflow,
                                  input_arg=input,
                                  first_task="deskew_image",
                                  last_task=task_name_last,
                                  time_start=time_start,
                                  time_end=time_end,
                                  channel_start=channel_start,
                                  channel_end=channel_end,
                                  save_path=save_path,
                                  save_name=save_name,
                                  save_file_type=output_file_type,
                                  dx=dx,
                                  dy=dy,
                                  dz=dz,
                                  angle=deskew_angle,
                                  deconvolution=True,
                                  decon_processing=lattice.decon_processing,
                                  psf=lattice.psf,
                                  psf_arg=psf_arg)
            else:
                save_img_workflow(vol=img_data,
                                  workflow=user_workflow,
                                  input_arg=input,
                                  first_task="deskew_image",
                                  last_task=task_name_last,
                                  time_start=time_start,
                                  time_end=time_end,
                                  channel_start=channel_start,
                                  channel_end=channel_end,
                                  save_path=save_path,
                                  save_name=save_name,
                                  save_file_type=output_file_type,
                                  dx=dx,
                                  dy=dy,
                                  dz=dz,
                                  angle=deskew_angle)
        else:
            if lattice.decon_processing:
                save_img_workflow(vol=img_data,
                                  workflow=user_workflow,
                                  input_arg=input_arg_first,
                                  first_task=first_task_name,
                                  last_task=task_name_last,
                                  time_start=time_start,
                                  time_end=time_end,
                                  channel_start=channel_start,
                                  channel_end=channel_end,
                                  save_path=save_path,
                                  save_name=save_name,
                                  save_file_type=output_file_type,
                                  dx=dx,
                                  dy=dy,
                                  dz=dz,
                                  angle=deskew_angle,
                                  deconvolution=True,
                                  decon_processing=lattice.decon_processing,
                                  psf=lattice.psf,
                                  psf_arg=psf_arg)
            else:
                save_img_workflow(vol=img_data,
                                  workflow=user_workflow,
                                  input_arg=input_arg_first,
                                  first_task=first_task_name,
                                  last_task=task_name_last,
                                  time_start=time_start,
                                  time_end=time_end,
                                  channel_start=channel_start,
                                  channel_end=channel_end,
                                  save_path=save_path,
                                  save_name=save_name,
                                  save_file_type=output_file_type,
                                  dx=dx,
                                  dy=dy,
                                  dz=dz,
                                  angle=deskew_angle)


if __name__ == '__main__':
    main()
