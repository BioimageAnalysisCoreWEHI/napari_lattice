# Tests for napari_lattice using arguments and saving output files as h5, as well as tiff

import subprocess
from skimage.io import imsave
import numpy as np
from pathlib import Path
import tempfile
from lls_core.cmds.__main__ import main as run_cli

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
        run_cli([
            "--input", str(input_file),
            "--output", str(out_dir),
            "--processing", "deskew",
            "--output_file_type", "h5"
        ])

        # checks if h5 files written
        assert (out_dir / "raw" / "raw.h5").exists()
        assert (out_dir / "raw" / "raw.xml").exists()


def test_batch_deskew_tiff():
    # tiff file deskew
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        input_file = out_dir / 'raw.tiff'
        create_image(input_file)
        run_cli([
            "--input", str(input_file),
            "--output", str(out_dir),
            "--processing", "deskew",
            "--output_file_type", "tiff"
        ])

        # checks if tiff written
        assert (out_dir / "raw" / "C0T0_raw.tif").exists()


        # verify output file by opening and checking if pixel value and coordinate
