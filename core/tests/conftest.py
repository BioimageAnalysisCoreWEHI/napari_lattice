from importlib_resources import as_file
from typer.testing import CliRunner
import pytest
from skimage.io import imsave
import numpy as np
from pathlib import Path
import pyclesperanto_prototype as cle
import tempfile
from numpy.typing import NDArray
from copy import copy
from lls_core.sample import resources

from napari_workflows import Workflow
from napari_workflows._io_yaml_v1 import save_workflow

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()

@pytest.fixture
def lls7_t1_ch1():
    with as_file(resources / "LLS7_t1_ch1.czi") as image_path:
        yield image_path

@pytest.fixture
def rbc_tiny():
    with as_file(resources / "RBC_tiny.czi") as image_path:
        yield image_path

@pytest.fixture
def multi_channel_time():
    with as_file(resources / "multich_multi_time.tif") as image_path:
        yield image_path

@pytest.fixture(params=[
    "LLS7_t1_ch1.czi",
    "LLS7_t1_ch3.czi",
    "LLS7_t2_ch1.czi",
    "LLS7_t2_ch3.czi",
])
def minimal_image_path(request: pytest.FixtureRequest):
    """
    Fixture function that yields a minimal set of test images as file paths
    """
    with as_file(resources / request.param) as image_path:
        yield image_path

@pytest.fixture(params=[
    "RBC_tiny.czi",
    "RBC_lattice.tif",
    "LLS7_t1_ch1.czi",
    "LLS7_t1_ch3.czi",
    "LLS7_t2_ch1.czi",
    "LLS7_t2_ch3.czi",
    "multich_multi_time.tif"
])
def image_path(request: pytest.FixtureRequest):
    """
    Fixture function that yields test images as file paths
    """
    with as_file(resources / request.param) as image_path:
        yield image_path

@pytest.fixture
def image_workflow() -> Workflow:
    # Simple segmentation workflow that returns an image
    image_seg_workflow = Workflow()
    image_seg_workflow.set("gaussian", cle.gaussian_blur, "deskewed_image", sigma_x=1, sigma_y=1, sigma_z=1)
    image_seg_workflow.set("binarisation", cle.threshold, "gaussian", constant=0.5)
    image_seg_workflow.set("labeling", cle.connected_components_labeling_box, "binarisation")
    return image_seg_workflow

@pytest.fixture
def table_workflow(image_workflow: Workflow) -> Workflow:
    # Complex workflow that returns a tuple of (image, dict, dict with multiple values, list, int)
    ret = copy(image_workflow)
    ret.set("result", lambda x: (
        x,
        {
            "foo": 1,
            "bar": 2
        },
        {'multi1': [1, 2, 3], 'multi2': ['a', 'b', 'c']},
        ["foo", "bar"],
        1
    ), "labeling")
    return ret

@pytest.fixture
def test_image() -> NDArray[np.float64]:
    raw = np.zeros((5, 5, 5))
    raw[2, 2, 2] = 10
    return raw

@pytest.fixture
def workflow_config(image_workflow: Workflow, test_image: NDArray):
    # Create a config file
    yield {
        "input_image": test_image,
        "workflow": image_workflow,
    }

@pytest.fixture
def workflow_config_cli(image_workflow: Workflow, test_image: NDArray):
    with tempfile.TemporaryDirectory() as tempdir_str:
        tempdir = Path(tempdir_str)
        input = tempdir / "raw.tiff"
        output = tempdir / "output"
        output.mkdir(parents=True)
        workflow_path = tempdir / "workflow.json"
        save_workflow(str(workflow_path), image_workflow) 
        
        # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
        imsave(input, test_image)
        assert input.exists()

        # Create a config file
        yield {
            key: str(val)
            for key, val in 
            {
                "input_image": input,
                "save_dir": output,
                "workflow": workflow_path,
            }.items()
        }
