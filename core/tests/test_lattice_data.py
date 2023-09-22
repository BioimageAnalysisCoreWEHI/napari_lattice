from typing import Any
import pytest
from lls_core.models import LatticeData
from lls_core.sample import resources
from importlib_resources import as_file
import tempfile
from pathlib import Path
from napari_workflows import Workflow

from .params import inputs, parameterized

def open_psf(name: str):
    with as_file(resources / "psfs" / "zeiss_simulated" / name) as path:
        return path
@inputs
@parameterized
def test_process(path: str, args: dict):
    with as_file(resources / path) as lattice_path:
        for slice in LatticeData.parse_obj({
            "image": lattice_path,
            **args
        }).process().slices:
            assert slice.data.ndim == 3

@inputs
@parameterized
def test_save(path: str, args: dict):
    with as_file(resources / path) as lattice_path, tempfile.TemporaryDirectory() as tempdir:
        LatticeData.parse_obj({
            "image": lattice_path,
            "save_dir": tempdir,
            **args
        }).process().save_image()
        results = list(Path(tempdir).iterdir())
        assert len(results) > 0

@pytest.mark.parametrize(["background"], [(1, ), ("auto",), ("second_last",)])
@parameterized
def test_process_deconvolution(args: dict, background: Any):
    root = Path(__file__).parent / "data" 
    for slice in LatticeData.parse_obj({
        "image": root / "raw.tif",
        "deconvolution": {
            "psf": [root / "psf.tif"],
            "background": background
        },
        **args
    }).process().slices:
        assert slice.data.ndim == 3

@parameterized
def test_process_workflow(args: dict, workflow: Workflow):
    root = Path(__file__).parent / "data" 
    for slice in LatticeData.parse_obj({
        "image": root / "raw.tif",
        "workflow": workflow,
        **args
    }).process().slices:
        assert slice.data.ndim == 3
