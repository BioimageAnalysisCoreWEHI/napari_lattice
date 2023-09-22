from typing import Callable
import os
from copy import copy
from numpy.typing import NDArray

from napari_workflows import Workflow
import tempfile

from tests.utils import invoke
from pathlib import Path
from .params import config_types
from .utils import invoke


def test_napari_workflow(workflow: Workflow, test_image: NDArray):
    """
    Test napari workflow to see if it works before we run it using napari_lattice
    This is without deskewing
    """
    workflow = copy(workflow)
    # Set input image to be the "raw" image
    workflow.set("input", test_image)
    labeling = workflow.get("labeling")
    assert labeling[2, 2, 2] == 1

@config_types
def test_workflow_cli(workflow_config_cli: dict, save_func: Callable, cli_param: str):
    """
    Test workflow processing via CLI
    This will apply deskewing before processing the workflow
    """
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        save_func(workflow_config_cli, fp)
        fp.flush()

        # Deskew, apply workflow and save as h5
        invoke([
            "process",
            cli_param, fp.name
        ])

    # checks if h5 file written
    save_dir = Path(workflow_config_cli["save_dir"])
    saved_files = list(save_dir.glob("*.h5"))
    assert len(saved_files) > 0
    assert len(list(save_dir.glob("*.xml"))) > 0

    import npy2bdv
    for h5_img in saved_files:
        h5_file = npy2bdv.npy2bdv.BdvEditor(str(h5_img))
        label_img = h5_file.read_view(time=0, channel=0)
        assert label_img.shape == (3, 14, 5)
        assert label_img[1, 6, 2] == 1
