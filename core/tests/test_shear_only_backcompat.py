"""Back-compat guard: with coverslip_rotation=True (the default), the
non-crop pipeline must produce output byte-identical to stock pyclesperanto
deskew -- cle.pull_zyx(cle.deskew_y(...)) for Y-skew and cle.deskew_x(...) for
X-skew. Anisotropic voxels (dy != dx) keep the X path meaningfully distinct.
"""
import numpy as np
import pytest
import pyclesperanto_prototype as cle
from xarray import DataArray
import tempfile
from lls_core import DeskewDirection
from lls_core.models.lattice_data import LatticeData


@pytest.mark.parametrize("skew,skew_dir,ref_func", [
    ("Y", DeskewDirection.Y, cle.deskew_y),
    ("X", DeskewDirection.X, cle.deskew_x),
])
def test_non_crop_default_byte_identical(skew, skew_dir, ref_func):
    raw = np.zeros((24, 70, 80), dtype=np.float32)
    raw[3:7, 15:45, 20:60] = 100
    with tempfile.TemporaryDirectory() as d:
        lat = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(2.0, 1.04, 0.9), angle=45,
            skew=skew_dir,
            coverslip_rotation=True,  # True = stock deskew (Zeiss LLS default)
            save_name="t", save_dir=d)
        out = np.asarray(next(iter(lat.process().slices)).data)
    # Mirror the call (lattice_data._process_non_crop):
    # cle.pull_zyx(deskew_{y,x}(..., voxel_size_x/y/z=..., angle_in_degrees=...)).
    ref = np.asarray(cle.pull_zyx(ref_func(
        raw, angle_in_degrees=45, voxel_size_x=0.9, voxel_size_y=1.04, voxel_size_z=2.0)))
    np.testing.assert_array_equal(out, ref)  # byte-identical, toggle on (stock deskew)
