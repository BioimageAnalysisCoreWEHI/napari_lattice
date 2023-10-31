# Tests for napari_lattice using arguments and saving output files as h5, as well as tiff

from aicsimageio.aics_image import AICSImage
from npy2bdv import BdvEditor
import numpy as np
from pathlib import Path
import tempfile
from tests.utils import invoke
import yaml

def create_image(path: Path):
    # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
    raw = np.zeros((5, 5, 5))
    raw[2, 4, 2] = 10
    # Save image as a tif filw in home directory
    AICSImage(raw).save(path)
    assert path.exists()


def create_data(dir: Path) -> Path:
    # Creates and returns a YAML config file
    input_file = dir / 'raw.tiff'
    config_location = dir / "config_deskew.yaml"

    # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
    raw = np.zeros((5, 5, 5))
    raw[2, 4, 2] = 10
    # Save image as a tif filw in home directory
    AICSImage(raw).save(input_file)
    assert input_file.exists()

    config: dict[str, str] = {
        "input_image": str(input_file),
        "save_dir": str(dir),
        "save_type": "h5"
    }

    with config_location.open("w") as fp:
        yaml.safe_dump(config, fp)

    return config_location

def assert_tiff(output_dir: Path):
    """Checks that a valid TIFF was generated in the directory"""
    results = list(output_dir.glob("*.tif"))
    assert len(results) > 0
    for result in results:
        AICSImage(result).get_image_data()

def assert_h5(output_dir: Path):
    """Checks that a valid H5 was generated"""
    h5s = list(output_dir.glob("*.h5"))
    assert len(h5s) > 0
    assert len(list(output_dir.glob("*.xml"))) == len(h5s)
    for h5 in h5s:
        BdvEditor(str(h5)).read_view()

def test_batch_deskew_h5():
    """Write image to disk and then execute napari_lattice from terminal
       Checks if an deskewed output file is created for both tif and h5
    """
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        input_file = out_dir / 'raw.tiff'
        create_image(input_file)
        # Batch deskew and save as h5
        invoke([
            str(input_file),
            "--save-dir", str(out_dir),
            "--save-type", "h5"
        ])

        assert_h5(out_dir)

def test_batch_deskew_tiff():
    # tiff file deskew
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        input_file = out_dir / 'raw.tiff'
        create_image(input_file)
        invoke([
            str(input_file),
            "--save-dir", str(out_dir),
            "--save-type", "tiff"
        ])

        assert_tiff(out_dir)

def test_yaml_deskew():
    """Write image to disk and then execute napari_lattice from terminal
       Checks if an deskewed output file is created for both tif and h5
    """
    with tempfile.TemporaryDirectory() as test_dir:
        test_dir = Path(test_dir)
        config_location = create_data(test_dir)
        # Batch deskew and save as h5
        invoke(["--yaml-config", str(config_location)], )
        assert_h5(test_dir)