# Tests for napari_lattice using arguments and saving output files as h5, as well as tiff

import subprocess
from skimage.io import imread, imsave
import os
import numpy as np
from pathlib import Path

home_dir = str(Path.home())+os.sep
img_dir = home_dir+"raw.tiff"


def create_image():
    # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
    raw = np.zeros((5, 5, 5))
    raw[2, 4, 2] = 10
    # Save image as a tif filw in home directory
    imsave(img_dir, raw)
    assert os.path.exists(img_dir)


def test_batch_deskew_h5():
    """Write image to disk and then execute napari_lattice from terminal
       Checks if an deskewed output file is created for both tif and h5
    """
    create_image()
    assert os.path.exists(img_dir)
    # Batch deskew and save as h5
    cmd = f"napari_lattice --input '{img_dir}' --output '{home_dir}' --processing deskew --output_file_type h5"
    deskew_process_h5 = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True)

    # Read data from stdout and stderr. waits for process to terminate, or it will execute next line immmediately
    _ = deskew_process_h5.communicate()

    # checks if h5 files written
    assert os.path.exists(home_dir+"raw/raw.h5")
    assert os.path.exists(home_dir+"raw/raw.xml")


def test_batch_deskew_tiff():
    # tiff file deskew
    cmd = f"napari_lattice --input {img_dir} --output {home_dir} --processing deskew --output_file_type tiff"
    deskew_process_tiff = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True)
    _ = deskew_process_tiff.communicate()

    # checks if tiff written
    assert os.path.exists(home_dir+"raw/C0T0_raw.tif")


# verify output file by opening and checking if pixel value and coordinate
