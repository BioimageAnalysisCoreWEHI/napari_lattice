# Tests for napari_lattice using the config file and saving ouput as h5
# Thanks to DrLachie for cool function to write the config file

from skimage.io import imread, imsave
import tempfile
import numpy as np
from pathlib import Path
from lls_core.cmds.__main__ import app
from tests.utils import invoke

def write_config_file(config_settings: dict, output_file_location: Path):
    # Write config file for napari_lattice
    with output_file_location.open('w') as f:
        for key, val in config_settings.items():
            if val is not None:
                if type(val) is str:
                    print('%s: "%s"' % (key, val), file=f)

                if type(val) is int:
                    print('%s: %i' % (key, val), file=f)

                if type(val) is list:
                    print("%s:" % key, file=f)
                    for x in val:
                        if type(x) is int:
                            print(" - %i" % x, file=f)
                        else:
                            print(' - "%s"' % x, file=f)

    print("Config found written to %s" % output_file_location)

def create_data(dir: Path) -> Path:
    input_file = dir / 'raw.tiff'
    config_location = dir / "config_deskew.yaml"

    # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
    raw = np.zeros((5, 5, 5))
    raw[2, 4, 2] = 10
    # Save image as a tif filw in home directory
    imsave(input_file, raw)
    assert input_file.exists()

    config: dict[str, str] = {
        "input_image": str(input_file),
        "save_dir": str(dir),
        "save_type": "h5"
    }

    write_config_file(config, config_location)
    assert config_location.exists()

    return config_location


def test_yaml_deskew():
    """Write image to disk and then execute napari_lattice from terminal
       Checks if an deskewed output file is created for both tif and h5
    """
    with tempfile.TemporaryDirectory() as test_dir:
        test_dir = Path(test_dir)
        config_location = create_data(test_dir)
        # Batch deskew and save as h5
        invoke(["--yaml-config", str(config_location)], )

        # checks if h5 files written
        assert (test_dir / "raw.h5").exists()
        # assert (test_dir / "raw" / "raw.xml").exists()
