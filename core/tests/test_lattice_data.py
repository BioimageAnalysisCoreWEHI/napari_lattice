import pytest
from lls_core.models import LatticeData
from lls_core.models.output import SaveFileType
from lls_core.sample import resources
from importlib_resources import as_file
from importlib_resources.abc import Traversable
import tempfile
from pathlib import Path
from lls_core import DeconvolutionChoice
from itertools import product

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
    {"rois": []},
    {"psf": [open_psf("488.czi")]},
    {"psf": [open_psf("640.tif")]},
    {"background": 1},
    {"background": "auto"},
    {"background": "second_last"},
    {"save_type": SaveFileType.h5},
    {"save_type": SaveFileType.tiff},
])

@inputs
@parameterized
def test_process(path: str, args: dict):
    with as_file(resources / path) as lattice_path:
        args["image"] = lattice_path
        for slice in LatticeData.parse_obj(args).process().slices:
            assert slice.data.ndim == 3

@inputs
@parameterized
def test_save(path: str, args: dict):
    with as_file(resources / path) as lattice_path, tempfile.TemporaryDirectory() as tempdir:
        args["image"] = lattice_path
        args["save_dir"] = tempdir
        LatticeData.parse_obj(args).process().save_image()
        results = list(Path(tempdir).iterdir())
        assert len(results) > 0
