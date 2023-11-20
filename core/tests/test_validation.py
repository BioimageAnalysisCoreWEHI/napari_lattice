from pathlib import Path
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData

def test_default_save_dir(rbc_tiny: Path):
    # Test that the save dir is inferred to be the input dir
    params = LatticeData(input_image=rbc_tiny)
    assert params.save_dir == rbc_tiny.parent

def test_auto_z_range(rbc_tiny: Path):
    # Tests that the Z range is automatically set, and it is set
    # based on the size of the deskewed volume
    params = LatticeData(input_image=rbc_tiny, crop=CropParams(
        roi_list=[]
    ))
    assert params.crop.z_range == (0, 59)
