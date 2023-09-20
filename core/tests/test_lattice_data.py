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
    ["path", "channels"], [
    ("RBC_tiny.czi", 1),
    ("RBC_lattice.tif", 1),
    ("LLS7_t1_ch1.czi", 1),
    ("LLS7_t1_ch3.czi", 1),
    ("LLS7_t2_ch1.czi", 1),
    ("LLS7_t2_ch3.czi", 1),
    ("multich_multi_time.tif", 3),
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
def test_process(path: str, channels:int, args: dict):
    with as_file(resources / path) as lattice_path:
        args["image"] = lattice_path
        if "deconvolution" in args:
            args["deconvolution"]["psf"] = args["deconvolution"]["psf"] * channels
        for slice in LatticeData.parse_obj(args).process().slices:
            assert slice.data.ndim == 3

@pytest.mark.parametrize(["background"], [(1, ), ("auto",), ("second_last",)])
@parameterized
def test_process_deconvolution(args: dict, background: Any):
    root = Path(__file__).parent / "data" 
    args["image"] = root / "raw.tif"
    args["deconvolution"] = {
        "psf": [root / "psf.tif"],
        "background": background
    }
    for slice in LatticeData.parse_obj(args).process().slices:
        assert slice.data.ndim == 3


@inputs
@parameterized
def test_save(path: str, channels: int, args: dict):
    with as_file(resources / path) as lattice_path, tempfile.TemporaryDirectory() as tempdir:
        args["image"] = lattice_path
        args["save_dir"] = tempdir
        LatticeData.parse_obj(args).process().save_image()
        results = list(Path(tempdir).iterdir())
        assert len(results) > 0
