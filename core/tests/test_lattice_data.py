from copy import copy
from typing import Any
import pytest
from lls_core.models import LatticeData
from lls_core.models.output import SaveFileType
from lls_core.sample import resources
from importlib_resources import as_file
from importlib_resources.abc import Traversable
import tempfile
from pathlib import Path

inputs = pytest.mark.parametrize(
    ["path"], [
    ("RBC_tiny.czi", ),
    ("RBC_lattice.tif", ),
    ("LLS7_t1_ch1.czi", ),
    ("LLS7_t1_ch3.czi", ),
    ("LLS7_t2_ch1.czi", ),
    ("LLS7_t2_ch3.czi", ),
    ("multich_multi_time.tif", ),
])

def open_psf(name: str):
    with as_file(resources / "psfs" / "zeiss_simulated" / name) as path:
        return path

parameterized = pytest.mark.parametrize("args", [
    {"skew": "X"},
    {"skew": "Y"},
    {"angle": 30},
    {"angle": 90},
    {"physical_pixel_sizes": (1, 1, 1)},
    {"save_type": SaveFileType.h5},
    {"save_type": SaveFileType.tiff},

    # Cropping enabled
    {"crop": {"roi_list": []}},
])

@inputs
@parameterized
def test_process(path: str, args: dict):
    args = copy(args)
    with as_file(resources / path) as lattice_path:
        args["image"] = lattice_path
        for slice in LatticeData.parse_obj(args).process().slices:
            assert slice.data.ndim == 3

@pytest.mark.parametrize(["background"], [(1, ), ("auto",), ("second_last",)])
@parameterized
def test_process_deconvolution(args: dict, background: Any):
    root = Path(__file__).parent / "data" 
    args = copy(args)
    args["image"] = root / "raw.tif"
    args["deconvolution"] = {
        "psf": [root / "psf.tif"],
        "background": background
    }
    for slice in LatticeData.parse_obj(args).process().slices:
        assert slice.data.ndim == 3


@inputs
@parameterized
def test_save(path: str, args: dict):
    args = copy(args)
    with as_file(resources / path) as lattice_path, tempfile.TemporaryDirectory() as tempdir:
        args["image"] = lattice_path
        args["save_dir"] = tempdir
        LatticeData.parse_obj(args).process().save_image()
        results = list(Path(tempdir).iterdir())
        assert len(results) > 0
