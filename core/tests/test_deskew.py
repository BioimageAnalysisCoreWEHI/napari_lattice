#filename and function name should start with "test_" when using pytest
import pyclesperanto_prototype as cle 
import numpy as np 
from lls_core.models.lattice_data import LatticeData
from xarray import DataArray
import tempfile

def test_deskew():

    raw = np.zeros((5,5,5))
    raw[2,0,0] = 10
    
    deskewed = cle.deskew_y(raw,angle_in_degrees=60)
    
    #np.argwhere(deskewed>0)
    assert deskewed.shape == (4,8,5)
    assert deskewed[2,2,0] == 0.5662433505058289

def test_lattice_data_deskew():
    raw = DataArray(np.zeros((5, 5, 5)), dims=["X", "Y", "Z"])
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=raw,
            physical_pixel_sizes = (1, 1, 1),
            save_name="test",
            save_dir=tmpdir
        )
        assert lattice.derived.deskew_vol_shape == (2, 9, 5)
