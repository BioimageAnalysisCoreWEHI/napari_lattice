# Tests for napari_lattice using arguments and saving output files as h5, as well as tiff

import subprocess
from skimage.io import imsave
import numpy as np
from pathlib import Path
import tempfile

home_dir = Path.home()
img_dir = home_dir / "raw.tiff"


def create_image():
    # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
    raw = np.zeros((5, 5, 5))
    raw[2, 4, 2] = 10
    # Save image as a tif filw in home directory
    imsave(img_dir, raw)
    assert img_dir.exists()


def test_batch_deskew_h5():
    """Write image to disk and then execute napari_lattice from terminal
       Checks if an deskewed output file is created for both tif and h5
    """
    create_image()
    assert img_dir.exists()
    # Batch deskew and save as h5
    with tempfile.TemporaryDirectory() as out_dir:
        result = subprocess.run([
            "napari_lattice",
            "--input", img_dir,
            "--output", out_dir,
            "--processing", "deskew",
            "--output_file_type", "h5"
        ])
    assert result.returncode == 0

    # checks if h5 files written
    assert (home_dir / "raw" / "raw.h5").exists()
    assert (home_dir / "raw" / "raw.xml").exists()


def test_batch_deskew_tiff():
    # tiff file deskew
    with tempfile.TemporaryDirectory() as out_dir:
        result = subprocess.run([
                "napari_lattice",
                "--input", img_dir,
                "--output", out_dir,
                "--processing", "deskew",
                "--output_file_type", "tiff"
        ])
        assert result.returncode == 0

    # checks if tiff written
    assert (home_dir / "raw" / "C0T0_raw.tif").exists()


# verify output file by opening and checking if pixel value and coordinate
