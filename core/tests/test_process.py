from typing import Any, List, Optional
import pytest
from lls_core.models import LatticeData
from lls_core.models.crop import CropParams
from lls_core.sample import resources
from importlib_resources import as_file
import tempfile
from pathlib import Path
from napari_workflows import Workflow
from pytest import FixtureRequest

from .params import parameterized

root = Path(__file__).parent / "data" 

def open_psf(name: str):
    with as_file(resources / "psfs" / "zeiss_simulated" / name) as path:
        return path

@parameterized
def test_process(minimal_image_path: str, args: dict):
    # Processes a minimal set of images, with multiple parameter combinations
    for slice in LatticeData.parse_obj({
        "input_image": minimal_image_path,
        **args
    }).process().slices:
        assert slice.data.ndim == 3

def test_process_all(image_path: str):
    # Processes all input images, but without parameter combinations
    for slice in LatticeData.parse_obj({
        "input_image": image_path
    }).process().slices:
        assert slice.data.ndim == 3

@parameterized
def test_save(minimal_image_path: str, args: dict):
    with tempfile.TemporaryDirectory() as tempdir:
        LatticeData.parse_obj({
            "input_image": minimal_image_path,
            "save_dir": tempdir,
            **args
        }).process().save_image()
        results = list(Path(tempdir).iterdir())
        assert len(results) > 0

def test_process_deconv_crop():
    for slice in LatticeData.parse_obj({
        "input_image": root / "raw.tif",
        "deconvolution": {
            "psf": [root / "psf.tif"],
        },
        "crop": CropParams(roi_list = [[[0, 0], [0, 110], [95, 0], [95, 110]]])
    }).process().slices:
        assert slice.data.ndim == 3

def test_process_time_range(multi_channel_time: Path):
    from lls_core.models.output import SaveFileType
    with tempfile.TemporaryDirectory() as outdir:
        LatticeData.parse_obj({
            "input_image": multi_channel_time,
            # Channels 2 & 3
            "channel_range": range(1, 3),
            # Time point 2
            "time_range": range(1, 2),
            "save_dir": outdir,
            "save_type": SaveFileType.h5
        }).save()

@pytest.mark.parametrize(["background"], [(1, ), ("auto",), ("second_last",)])
@parameterized
def test_process_deconvolution(args: dict, background: Any):
    for slice in LatticeData.parse_obj({
        "input_image": root / "raw.tif",
        "deconvolution": {
            "psf": [root / "psf.tif"],
            "background": background
        },
        **args
    }).process().slices:
        assert slice.data.ndim == 3

@parameterized
@pytest.mark.parametrize(["workflow_name"], [("image_workflow", ), ("table_workflow", )])
def test_process_workflow(args: dict, request: FixtureRequest, workflow_name: str):
    from pandas import DataFrame

    workflow: Workflow = request.getfixturevalue(workflow_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        for roi, output in LatticeData.parse_obj({
            "input_image": root / "raw.tif",
            "workflow": workflow,
            "save_dir": tmpdir,
            **args
        }).process_workflow().process():
            assert roi is None or isinstance(roi, int)
            assert isinstance(output, (Path, DataFrame))

@pytest.mark.parametrize(["roi_subset"], [
    [None],
    [[0]],
    [[0, 1]],
])
@parameterized
def test_process_crop_roi_file(args: dict, roi_subset: Optional[List[int]]):
    # Test cropping with a roi zip file, selecting different subsets from that file
    with as_file(resources / "RBC_tiny.czi") as lattice_path:
        rois = root / "crop" / "two_rois.zip"
        for slice in LatticeData.parse_obj({
            "input_image": lattice_path,
            "crop": {
                "roi_list": [rois],
                "roi_subset": roi_subset
            },
            **args
        }).process().slices:
            assert slice.data.ndim == 3

@pytest.mark.parametrize(["roi"], [
    [[[
        (174.0, 24.0),
        (174.0, 88.0),
        (262.0, 88.0),
        (262.0, 24.0)
    ]]],
    [[[
        (174.13, 24.2),
        (173.98, 87.87),
        (262.21, 88.3),
        (261.99, 23.79)
    ]]],
])
@parameterized
def test_process_crop_roi_manual(args: dict, roi: List):
    # Test manually provided ROIs, both with integer and float values
    with as_file(resources / "RBC_tiny.czi") as lattice_path:
        for slice in LatticeData.parse_obj({
            "input_image": lattice_path,
            "crop": {
                "roi_list": roi
            },
            **args
        }).process().slices:
            assert slice.data.ndim == 3
