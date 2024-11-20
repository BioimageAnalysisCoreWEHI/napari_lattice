from typing import Callable
from copy import copy
from numpy.typing import NDArray

from napari_workflows import Workflow
import tempfile

from pandas import DataFrame
from lls_core.cropping import Roi
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData

from tests.utils import invoke
from pathlib import Path
from .params import config_types
from .utils import invoke, valid_image_path


def test_napari_workflow(image_workflow: Workflow, test_image: NDArray):
    """
    Test napari workflow to see if it works before we run it using napari_lattice
    This is without deskewing
    """
    workflow = copy(image_workflow)
    # Set input image to be the "raw" image
    workflow.set("deskewed_image", test_image)
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

def test_image_workflow(lls7_t1_ch1: Path, image_workflow: Workflow):
    # Test that a regular workflow that returns an image directly works
    with tempfile.TemporaryDirectory() as tmpdir:
        for output in LatticeData(
            input_image = lls7_t1_ch1,
            workflow = image_workflow,
            save_dir = tmpdir
        ).process_workflow().process():
            assert isinstance(output.data, Path)
            assert valid_image_path(output.data)

def test_table_workflow(lls7_t1_ch1: Path, table_workflow: Workflow):
    # Test a complex workflow that returns a tuple of images and data
    with tempfile.TemporaryDirectory() as tmpdir:
        params = LatticeData(
            input_image = lls7_t1_ch1,
            workflow = table_workflow,
            save_dir = tmpdir
        )
        for output in params.process_workflow().process():
            data = output.data
            assert isinstance(data, (DataFrame, Path))
            if isinstance(data, DataFrame):
                nrow, ncol = data.shape
                assert nrow == params.nslices
                assert ncol > 0
                # Check that time and channel are included
                assert data.iloc[0, 0] == "T0"
                assert data.iloc[0, 1] == "C0"
            else:
                assert valid_image_path(data)

def test_argument_order(rbc_tiny: Path):
    # Tests that only the first unfilled argument is passed an array
    with tempfile.TemporaryDirectory() as tmpdir:
        params = LatticeData(
            input_image = rbc_tiny,
            workflow = "core/tests/workflows/argument_order/test_workflow.yml",
            save_dir = tmpdir
        )
        for output in params.process_workflow().process():
            pass

def test_sum_preview(rbc_tiny: Path):
    import numpy as np
    # Tests that we can sum the preview result. This is required for the plugin
    with tempfile.TemporaryDirectory() as tmpdir:
        params = LatticeData(
            input_image = rbc_tiny,
            workflow = "core/tests/workflows/binarisation/workflow.yml",
            save_dir = tmpdir
        )
        previews = list(params.process_workflow().roi_previews())
        assert len(previews) == 1, "There should be 1 preview when cropping is disabled"
        assert previews[0].ndim == 3, "A preview should be a 3D image"

def test_crop_workflow(lls7_t1_ch1: Path):
    # Tests that crop workflows only process each ROI lazily

    with tempfile.TemporaryDirectory() as tmpdir:
        params = LatticeData(
            input_image = lls7_t1_ch1,
            workflow = "core/tests/workflows/binarisation/workflow.yml",
            save_dir = tmpdir,
            crop=CropParams(
                roi_list=[
                    Roi((174.0, 24.0), (174.0, 88.0), (262.0, 88.0), (262.0, 24.0)),
                    Roi((174.0, 24.0), (174.0, 88.0), (262.0, 88.0), (262.0, 24.0)),
                ]
            )
        )
        next(iter(params.process_workflow().save()))
        for file in Path(tmpdir).iterdir():
            # Only the first ROI should have been processed
            assert "ROI_0" in file.name
