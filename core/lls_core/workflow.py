"""
Functions related to manipulating Napari Workflows
"""
from __future__ import annotations

from os import path
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
from lls_core.llsz_core import crop_volume_deskew
from lls_core.types import ArrayLike
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_workflows import Workflow

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_first_last_image_and_task(user_workflow: Workflow) -> Tuple[str, str, str, str]:
    """
    Get images and tasks for first and last entry
    Returns:
        Tuple of (name of first input image, name of last input image, name of first task, name of last task)
    """

    # get image with no preprocessing step (first image)
    input_arg_first = user_workflow.roots()[0]
    # get last image
    input_arg_last = user_workflow.leafs()[0]
    # get name of preceding image as that is the input to last task
    img_source = user_workflow.sources_of(input_arg_last)[0]
    first_task_name = []
    last_task_name = []

    # loop through workflow keys and get key that has
    for key in user_workflow._tasks.keys():
        for task in user_workflow._tasks[key]:
            if task == input_arg_first:
                first_task_name.append(key)
            elif task == img_source:
                last_task_name.append(key)

    return input_arg_first, img_source, first_task_name[0], last_task_name[0] if len(last_task_name) > 0 else first_task_name[0]


def modify_workflow_task(old_arg: str, task_key: str, new_arg: str, workflow: Workflow) -> tuple:
    """
    Modify items in a workflow task
    Workflow is not modified, only a new task with updated arg is returned
    Args:
        old_arg: The argument in the workflow that needs to be modified
        new_arg: New argument
        task_key: Name of the task within the workflow
        workflow: Workflow

    Returns:
        tuple: Modified task with name task_key
    """
    task = workflow._tasks[task_key]
    # convert tuple to list for modification
    task_list = list(task)
    try:
        item_index = task_list.index(old_arg)
    except ValueError:
        raise Exception(f"{old_arg} not found in workflow file")

    task_list[item_index] = new_arg
    modified_task = tuple(task_list)
    return modified_task

def replace_first_arg(workflow: Workflow, new_arg: str, old_arg: Optional[str] = None):
    """
    Replaces an argument in the first task of a workflow with some other value
    Args:
        old_arg: If provided, the name of an argument to replace, otherwise it defaults to the
            first workflow arg
    """
    img_arg_first, _, first_task_name, _ = get_first_last_image_and_task(workflow)
    if old_arg is None:
        old_arg = img_arg_first
    workflow.set(
        first_task_name,
        modify_workflow_task(
            old_arg=old_arg,
            task_key=first_task_name,
            new_arg=new_arg,
            workflow=workflow
        )
    )

def load_custom_py_modules(custom_py_files):
    import sys
    from importlib import import_module, reload
    test_first_module_import = import_module(custom_py_files[0])
    if test_first_module_import not in sys.modules:
        modules = map(import_module, custom_py_files)
    else:
        modules = map(reload, custom_py_files)
    return modules
    

# TODO: CHANGE so user can select modules? Safer
def get_all_py_files(directory: str) -> list[str]:
    """get all py files within directory and return as a list of filenames
    Args:
        directory: Directory with .py files
    """
    import glob
    from os.path import basename, dirname, isfile, join
    
    modules = glob.glob(join(dirname(directory), "*.py"))
    all = [basename(f)[:-3] for f in modules if isfile(f)
           and not f.endswith('__init__.py')]
    print(f"Files found are: {all}")

    return all

def import_script(script: Path):
    """
    Imports a Python script given its path
    """
    import importlib.util
    import sys
    module_name = script.stem
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise Exception(f"Failed to import {script}!")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

def import_workflow_modules(workflow: Path):
    """
    Imports all the Python files that might be used in a given custom workflow

    Args:
        workflow: Path to the workflow YAML file
    """
    counter = 0
    for script in workflow.parent.glob("*.py"):
        if script.stem == "__init__":
            # Skip __init__.py
            continue
        import_script(script)
        counter += 1

    if counter == 0:
        logger.warn(f"No custom modules imported. If you'd like to use a custom module, place a *.py file in same folder as the workflow file {workflow.parent}")
    else:
        logger.info(f"{counter} custom modules imported")


def process_custom_workflow_output(workflow_output: Union[dict, list, ArrayLike],
                                   save_dir: Optional[str]=None,
                                   idx: Optional[str]=None,
                                   LLSZWidget=None,
                                   widget_class=None,
                                   channel=0,
                                   time=0,
                                   preview: bool = True):
    """Check the output from a custom workflow; 
    saves tables and images separately

    Args:
        workflow_output (_type_): _description_
        save_dir (_type_): _description_
        idx (_type_): _description_
        LLSZWidget (_type_): _description_
        widget_class (_type_): _description_
        channel (_type_): _description_
        time (_type_): _description_
    """
    if isinstance(workflow_output, (dict, list)):
        # create function for tthis dataframe bit
        df = pd.DataFrame(workflow_output)
        if preview:
            save_path = path.join(
                save_dir, "lattice_measurement_"+str(idx)+".csv")
            print(f"Detected a dictionary as output, saving preview at", save_path)
            df.to_csv(save_path, index=False)
            return df

        else:
            return df
    elif isinstance(workflow_output, (np.ndarray, cle._tier0._pycl.OCLArray, da.core.Array)):
        if preview:
            suffix_name = str(idx)+"_c" + str(channel) + "_t" + str(time)
            scale = (LLSZWidget.LlszMenu.lattice.new_dz,
                     LLSZWidget.LlszMenu.lattice.dy, LLSZWidget.LlszMenu.lattice.dx)
            widget_class.parent_viewer.add_image(
                workflow_output, name="Workflow_preview_" + suffix_name, scale=scale)
        else:
            return workflow_output

def augment_workflow(
    workflow: Workflow,
    lattice: LatticeData,
    times: range,
    channels: range
    ) -> Iterable[Workflow]:
        """
        Yields copies of the input workflow, modified with the addition of deskewing and optionally,
        cropping and deconvolution
        """
        user_workflow = copy(workflow)   
        _, _, first_task_name, _ = get_first_last_image_and_task(workflow)

        for loop_time_idx, time_point in enumerate(times):
            output_array = []
            data_table = []
            for loop_ch_idx, ch in enumerate(channels):

        if crop:
            yield from make_crop_workflows(
                user_workflow=user_workflow,
                roi_layer_list=roi_layer_list,
                lattice=lattice,
                deconvolution=deconvolution
            )

                # save_img_workflow(vol=vol,
                #                     workflow=user_workflow,
                #                     input_arg=volume,
                #                     first_task="crop_deskew_image",
                #                     last_task=task_name_last,
                #                     time_start=time_start,
                #                     time_end=time_end,
                #                     channel_start=ch_start,
                #                     channel_end=ch_end,
                #                     save_file_type=save_as_type,
                #                     save_path=save_path,
                #                     #roi_layer = roi_layer,
                #                     save_name_prefix="ROI_" + \
                #                     str(idx),
                #                     save_name=self.llsz_parent.lattice.save_name,
                #                     dx=dx,
                #                     dy=dy,
                #                     dz=dz,
                #                     angle=angle,
                #                     deconvolution=self.llsz_parent.deconvolution.value,
                #                     decon_processing=self.llsz_parent.lattice.decon_processing,
                #                     otf_path=otf_path,
                #                     psf_arg=psf_arg,
                #                     psf=psf)
        else:
            INPUT_ARG = "input"

            # IF just deskewing and its not in the tasks, add that as first task
            if user_workflow.get_task(first_task_name)[0] not in (cle.deskew_y, cle.deskew_x):
                # add task to the workflow
                user_workflow.set(
                    "deskew_image",
                    lattice.deskew_func,
                    input_image=INPUT_ARG,
                    angle_in_degrees=lattice.angle,
                    voxel_size_x=lattice.dx,
                    voxel_size_y=lattice.dy,
                    voxel_size_z=lattice.dz,
                    linear_interpolation=True
                )
                # Set input of the workflow to be from deskewing
                # change workflow task starts from is "deskew_image" and
                replace_first_arg(user_workflow, new_arg="deskew_image")

            # if deconvolution checked, add it to start of workflow (add upstream of deskewing)
            if deconvolution:
                PSF_ARG = "psf"

                if lattice.decon_processing == DeconvolutionChoice.cuda_gpu:
                    user_workflow.set(
                        "deconvolution",
                        pycuda_decon,
                        image=INPUT_ARG,
                        psf=PSF_ARG,
                        dzdata=lattice.dz,
                        dxdata=lattice.dx,
                        dzpsf=lattice.dz,
                        dxpsf=lattice.dx,
                        num_iter=lattice.psf_num_iter
                    )
                    # user_workflow.set(input_arg_first,"deconvolution")
                else:
                    user_workflow.set(
                        "deconvolution",
                        skimage_decon,
                        vol_zyx=INPUT_ARG,
                        psf=PSF_ARG,
                        num_iter=lattice.psf_num_iter,
                        clip=False,
                        filter_epsilon=0,
                        boundary='nearest'
                    )
                # modify the user workflow so "deconvolution" is accepted
                replace_first_arg(user_workflow, new_arg="deconvolution")

                yield workflow

                # save_img_workflow(vol=vol,
                #                     workflow=user_workflow,
                #                     input_arg=INPUT_ARG,
                #                     first_task=task_name_start,
                #                     last_task=task_name_last,
                #                     time_start=time_start,
                #                     time_end=time_end,
                #                     channel_start=ch_start,
                #                     channel_end=ch_end,
                #                     save_file_type=save_as_type,
                #                     save_path=save_path,
                #                     save_name=self.llsz_parent.lattice.save_name,
                #                     dx=dx,
                #                     dy=dy,
                #                     dz=dz,
                #                     angle=angle,
                #                     deconvolution=self.llsz_parent.deconvolution,
                #                     decon_processing=self.llsz_parent.lattice.decon_processing,
                #                     otf_path=OTF_PATH_ARG,
                #                     psf_arg=psf_arg,
                #                     psf=PSF_ARG)


def make_crop_workflows(
    user_workflow: Workflow,
    roi_layer_list: ShapesData,
    lattice: LatticeData,
    deconvolution: bool
) -> Iterable[Workflow]:
    """
    Yields a copy of `user_workflow` for each region of interest, with deskewing, cropping and deconvolution steps added on to the start
    """

    # Convert Roi pixel coordinates to canvas coordinates
    # necessary only when scale is used for napari.viewer.add_image operations
    
    # Here we generate a workflow for each ROI
    for idx, roi_layer in enumerate(tqdm([x/lattice for x in roi_layer_list], desc="ROI:", position=0)):
        # Check if decon ticked, if so set as first and crop as second?

        # Create workflow for cropping and deskewing
        # volume and roi used will be set dynamically
        current_workflow = copy(user_workflow)
        current_workflow.set(
            "crop_deskew_image",
            crop_volume_deskew,
            original_volume="volume",
            roi_shape="roi",
            angle_in_degrees=lattice.angle,
            voxel_size_x=lattice.dx,
            voxel_size_y=lattice.dy,
            voxel_size_z=lattice.dy,
            z_start=0,
            z_end=lattice.deskew_vol_shape[0],
            deconvolution=deconvolution,
            decon_processing=lattice.decon_processing,
            psf="psf",
            skew_dir=lattice.skew_dir
        )

        # change the first task so it accepts "crop_deskew as input"
        replace_first_arg(current_workflow, new_arg="crop_deskew_image")

        logging.info(f"Processing ROI {idx}")
        current_workflow.set("roi", roi_layer)

        yield current_workflow
