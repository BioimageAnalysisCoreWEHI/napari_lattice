from pathlib import Path
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData
from lls_core.models.deskew import DeskewParams
from lls_core.models.output import OutputParams
import pytest
from pydantic.v1 import ValidationError
import tempfile
from unittest.mock import patch, PropertyMock

def test_default_save_dir(rbc_tiny: Path):
    # Test that the save dir is inferred to be the input dir
    params = LatticeData(input_image=rbc_tiny)
    assert params.save_dir == rbc_tiny.parent

def test_auto_z_range(rbc_tiny: Path):
    # Tests that the Z range is automatically set, and it is set
    # based on the size of the deskewed volume
    params = LatticeData(input_image=rbc_tiny, crop=CropParams(
        roi_list=[[[0, 0], [0, 1], [1, 0], [1, 1]]]
    ))
    assert params.crop.z_range == (0, 59)

def test_reject_crop():
    # Tests that the parameters fail validation if cropping is specified without an ROI
    with pytest.raises(ValidationError):
        CropParams(
            roi_list=[]
        )

def test_pixel_tuple_order(rbc_tiny: Path):
    # Tests that a tuple of Z, Y, X is appropriately assigned in the right order
    deskew = DeskewParams(
        input_image=rbc_tiny,
        physical_pixel_sizes=(1., 2., 3.)
    )

    assert deskew.physical_pixel_sizes.X == 3.
    assert deskew.physical_pixel_sizes.Y == 2.
    assert deskew.physical_pixel_sizes.Z == 1.

def test_allow_trailing_slash():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = OutputParams(
            save_dir=f"{tmpdir}/"
        )
        assert str(output.save_dir) == tmpdir

def test_infer_czi_pixel_sizes(rbc_tiny: Path):
    mock = PropertyMock()
    with patch("aicsimageio.AICSImage.physical_pixel_sizes", new=mock):
        DeskewParams(input_image=rbc_tiny)
        # The AICSImage should be queried for the pixel sizes
        assert mock.called
