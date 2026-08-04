from importlib_resources import as_file
from typer.testing import CliRunner
import pytest
from skimage.io import imsave
import numpy as np
from pathlib import Path
import pyclesperanto as cle
import tempfile
from numpy.typing import NDArray
from copy import copy
from types import SimpleNamespace
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
    image_seg_workflow.set("binarisation", cle.greater_constant, "gaussian", scalar=0.5)
    image_seg_workflow.set("labeling", cle.connected_component_labeling, "binarisation", connectivity="box")
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
        
        # Save the test_image (5x5x5 zeros with a value of 10 at (2,2,2)) to disk
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

# --- generated CZI fixtures --------------------------------------------------
#
# pylibCZIrw's writer never emits the <Scenes> element that bioio's `scene_name()`
# requires, so `BioImage.scenes` raises UnsupportedMetadataError on every file it
# produces - with or without `write_metadata`, with or without an explicit `scene=`.
# `czi_metadata` only reads three attributes off the BioImage, so the tests stand a
# stub in for those. That substitutes metadata the writer cannot produce; the geometry
# under test still comes from pylibCZIrw.

class _CziStubImage:
    """Stand-in for a `BioImage` over a generated CZI."""

    def __init__(self, metadata, n_scenes: int, scene_index: int):
        self.scenes = tuple(f"Scene:{i}" for i in range(n_scenes))
        self.current_scene_index = scene_index
        self.reader = SimpleNamespace(metadata=metadata)


@pytest.fixture
def czi_stub_image():
    """Factory: `czi_stub_image(path, n_scenes=1, scene_index=0)`."""
    from xml.etree import ElementTree
    from pylibCZIrw import czi as pyczi

    def make(path, n_scenes: int = 1, scene_index: int = 0) -> _CziStubImage:
        with pyczi.open_czi(str(path)) as czi:
            metadata = ElementTree.fromstring(czi.raw_metadata)
        return _CziStubImage(metadata, n_scenes, scene_index)

    return make


@pytest.fixture(scope="session")
def drift_czi(tmp_path_factory):
    """
    A CZI whose subblocks are narrower than the canvas, because each timepoint records
    a different stage offset. This is the file shape that crashed the old reader:
    aicspylibczi reports the 20-wide subblock while pylibCZIrw and bioio report the
    25-wide canvas, and reshaping one into the other raises.

    Yields `(path, planes, offsets)`; `planes[(t, z)]` is the array as written.
    """
    from pylibCZIrw import czi as pyczi

    path = tmp_path_factory.mktemp("czi") / "drift.czi"
    rng = np.random.default_rng(0)
    offsets = {0: 5, 1: 0, 2: 2}   # x position per timepoint; canvas is 5 + 20 wide
    planes = {}
    with pyczi.create_czi(str(path)) as writer:
        for t in range(3):
            for z in range(4):
                plane = rng.integers(1, 500, size=(12, 20), dtype=np.uint16)
                planes[(t, z)] = plane
                writer.write(
                    plane, location=(offsets[t], 0), plane={"T": t, "Z": z, "C": 0}
                )
    return path, planes, offsets


@pytest.fixture(scope="session")
def noncontiguous_scene_czi(tmp_path_factory):
    """
    Two scenes whose CZI scene keys are 1 and 2 - neither zero-based nor a range.

    A plate acquisition can hold exactly this. bioio maps its own 0..N-1 scene index
    through `sorted(scenes_bounding_rectangle_no_pyramid)`; reading the BioIO index
    straight into pylibCZIrw instead picks the wrong scene (or none at all).

    Yields `(path, planes)`; `planes[(czi_scene, z)]` is the array as written.
    """
    from pylibCZIrw import czi as pyczi

    path = tmp_path_factory.mktemp("czi") / "noncontiguous_scene.czi"
    rng = np.random.default_rng(2)
    planes = {}
    with pyczi.create_czi(str(path)) as writer:
        for scene in (1, 2):
            for z in range(3):
                plane = rng.integers(1, 500, size=(10, 14), dtype=np.uint16)
                planes[(scene, z)] = plane
                writer.write(
                    plane,
                    location=((scene - 1) * 20, 0),
                    plane={"Z": z, "C": 0},
                    scene=scene,
                )
    return path, planes


@pytest.fixture
def czi_read_calls(monkeypatch):
    """
    Records the plane dict of every `pylibCZIrw` read for the duration of the test.

    Wraps the library's reader rather than patching `lls_core.czi_reader`, so the test
    pins the observable property - one read per plane - and not how the module happens
    to be written.
    """
    from contextlib import contextmanager
    from pylibCZIrw import czi as pyczi

    calls: list = []
    real_open_czi = pyczi.open_czi

    class _CountingReader:
        def __init__(self, inner):
            self._inner = inner

        def read(self, *args, **kwargs):
            calls.append(kwargs.get("plane"))
            return self._inner.read(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    @contextmanager
    def counting_open_czi(*args, **kwargs):
        with real_open_czi(*args, **kwargs) as reader:
            yield _CountingReader(reader)

    monkeypatch.setattr(pyczi, "open_czi", counting_open_czi)
    return calls
