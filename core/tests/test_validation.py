from lls_core.models.lattice_data import LatticeData
from lls_core.sample import resources
from importlib_resources import as_file


def test_default_save_dir():
    # Test that the save dir is inferred to be the input dir
    with as_file(resources / "RBC_tiny.czi") as path:
        params = LatticeData(input_image=path)
        assert params.save_dir == path.parent
