from typing import Any, List, Optional
import pytest
from lls_core.models import LatticeData
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
def test_process(image_path: str, args: dict):
    for slice in LatticeData.parse_obj({
        "input_image": image_path,
        **args
    }).process().slices:
        assert slice.data.ndim == 3

@parameterized
def test_save(image_path: str, args: dict):
    with tempfile.TemporaryDirectory() as tempdir:
        LatticeData.parse_obj({
            "input_image": image_path,
            "save_dir": tempdir,
            **args
        }).process().save_image()
        results = list(Path(tempdir).iterdir())
        assert len(results) > 0

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
def test_process_crop(args: dict, roi_subset: Optional[List[int]]):
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
