# Tests for napari_lattice using arguments and saving output files as h5, as well as tiff

from skimage.io import imsave
import numpy as np
from pathlib import Path
import tempfile
from tests.utils import invoke

def create_image(path: Path):
    # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
    raw = np.zeros((5, 5, 5))
    raw[2, 4, 2] = 10
    # Save image as a tif filw in home directory
    imsave(str(path), raw)
    assert path.exists()


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
            "process",
            str(input_file),
            "--save-dir", str(out_dir),
            "--save-type", "h5"
        ])

        # checks if h5 files written
        assert (out_dir / "raw.h5").exists()
        # assert (out_dir / "raw.xml").exists()


def test_batch_deskew_tiff():
    # tiff file deskew
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        input_file = out_dir / 'raw.tiff'
        create_image(input_file)
        invoke([
            "process",
            str(input_file),
            "--save-dir", str(out_dir),
            "--save-type", "tiff"
        ])

        # checks if tiff written
        assert len(list(out_dir.glob("*.tif"))) == 1
