"""
Functions related to manipulating Napari Workflows
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_workflows import Workflow

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_workflow_inputs(workflow: Workflow) -> Optional[Tuple[str, int, str]]:
    """
    Yields tuples of (task_name, argument_index, input_argument) corresponding to the workflow's inputs,
    namely the arguments that are unfilled.
    Note that the index returned is the index in the overall task tuple, which includes the task name
    """
    for root_arg in workflow.roots():
        for taskname, (task_func, *args) in workflow._tasks.items():
            if root_arg in args:
                return taskname, args.index(root_arg) + 1, root_arg
    return None

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
