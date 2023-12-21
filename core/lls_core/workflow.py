"""
Functions related to manipulating Napari Workflows
"""
from __future__ import annotations

from os import path
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
import pyclesperanto_prototype as cle
from lls_core.types import ArrayLike
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_workflows import Workflow

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_workflow_inputs(workflow: Workflow) -> Tuple[str, int, str]:
    """
    Yields tuples of (task_name, argument_index, input_argument) corresponding to the workflow's inputs,
    namely the arguments that are unfilled.
    Note that the index returned is the index in the overall task tuple, which includes the task name
    """
    for root_arg in workflow.roots():
        for taskname, (task_func, *args) in workflow._tasks.items():
            if root_arg in args:
                return taskname, args.index(root_arg) + 1, root_arg
    raise Exception("No inputs could be calculated")

def update_workflow(workflow: Workflow, task_name: str, task_index: int, new_value: Any):
    task = list(workflow.get_task(task_name))
    task[task_index] = new_value
    workflow.set_task(task_name, tuple(task))

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
    Replies one argument in a workflow with another
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

def _import_workflow_modules(workflow: Path) -> None:
    """
    Imports all the Python files that might be used in a given custom workflow

    Args:
        workflow: Path to the workflow YAML file
    """
    if not workflow.exists():
        raise Exception("Workflow doesn't exist!")
    if not workflow.is_file():
        raise Exception("Workflow must be a file!")

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

def workflow_from_path(workflow: Path) -> Workflow:
    """
    Imports the dependency modules for a workflow, and loads it from disk
    """
    from napari_workflows._io_yaml_v1 import load_workflow
    _import_workflow_modules(workflow)
    return load_workflow(str(workflow))

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
