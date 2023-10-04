from typing import Any, List, Optional
import pytest
from lls_core.models import LatticeData
from lls_core.sample import resources
from importlib_resources import as_file
import tempfile
from pathlib import Path
from napari_workflows import Workflow

from .params import inputs, parameterized

root = Path(__file__).parent / "data" 

def open_psf(name: str):
    with as_file(resources / "psfs" / "zeiss_simulated" / name) as path:
        return path
@inputs
@parameterized
def test_process(path: str, args: dict):
    with as_file(resources / path) as lattice_path:
        for slice in LatticeData.parse_obj({
            "input_image": lattice_path,
            **args
        }).process().slices:
            assert slice.data.ndim == 3

@inputs
@parameterized
def test_save(path: str, args: dict):
    with as_file(resources / path) as lattice_path, tempfile.TemporaryDirectory() as tempdir:
        LatticeData.parse_obj({
            "input_image": lattice_path,
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
def test_process_workflow(args: dict, workflow: Workflow):
    for slice in LatticeData.parse_obj({
        "input_image": root / "raw.tif",
        "workflow": workflow,
        **args
    }).process().slices:
        assert slice.data.ndim == 3

@pytest.mark.parametrize(["roi_subset"], [
    [None],
    [[0]],
    [[0, 1]],
])
@parameterized
def test_process_crop(args: dict, roi_subset: Optional[List[int]], workflow: Workflow):
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
