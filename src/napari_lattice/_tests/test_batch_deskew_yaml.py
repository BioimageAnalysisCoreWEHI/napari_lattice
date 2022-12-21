# Tests for napari_lattice using the config file and saving ouput as h5
# Thanks to DrLachie for cool function to write the config file

import subprocess
from skimage.io import imread, imsave
import os
import numpy as np
from pathlib import Path
import platform

# For testing in Windows
if platform.system() == "Windows":
    home_dir = str(Path.home())
    home_dir = home_dir.replace("\\", "\\\\")
    img_dir = home_dir + "\\\\raw.tiff"
    config_location = home_dir + "\\\\config_deskew.yaml"
else:
    home_dir = str(Path.home())
    img_dir = os.path.join(home_dir, "raw.tiff")
    config_location = os.path.join(home_dir, "config_deskew.yaml")


def write_config_file(config_settings, output_file_location):
    # Write config file for napari_lattice
    with open(output_file_location, 'w') as f:
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


def create_data():
    # Create a zero array of shape 5x5x5 with a value of 10 at (2,4,2)
    raw = np.zeros((5, 5, 5))
    raw[2, 4, 2] = 10
    # Save image as a tif filw in home directory
    imsave(img_dir, raw)
    assert os.path.exists(img_dir)

    config = {
        "input": img_dir,
        "output": home_dir,
        "processing": "deskew",
        "output_file_type": "h5"}

    write_config_file(config, config_location)
    assert os.path.exists(config_location)


def test_yaml_deskew():
    """Write image to disk and then execute napari_lattice from terminal
       Checks if an deskewed output file is created for both tif and h5
    """
    create_data()
    # Batch deskew and save as h5
    cmd = f"napari_lattice --config {config_location}"
    deskew_process_yaml = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True)

    # Read data from stdout and stderr. waits for process to terminate, or it will execute next line immmediately
    _ = deskew_process_yaml.communicate()

    # checks if h5 files written
    assert os.path.exists(os.path.join(home_dir, "raw", "raw.h5"))
    assert os.path.exists(os.path.join(home_dir, "raw", "raw.xml"))
