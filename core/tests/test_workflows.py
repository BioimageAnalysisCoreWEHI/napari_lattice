import subprocess
from skimage.io import imread, imsave
import os
import numpy as np
from pathlib import Path
import platform
import pyclesperanto_prototype as cle

from napari_workflows import Workflow
from napari_workflows._io_yaml_v1 import load_workflow, save_workflow

# For testing in Windows
if platform.system() == "Windows":
    home_dir = str(Path.home())
    home_dir = home_dir.replace("\\", "\\\\")
    img_dir = home_dir + "\\\\raw.tiff"
    workflow_location = home_dir + "\\\\deskew_segment.yaml"
    config_location = home_dir + "\\\\config_deskew.yaml"
else:
    home_dir = str(Path.home())
    img_dir = os.path.join(home_dir, "raw.tiff")
    workflow_location = os.path.join(home_dir, "deskew_segment.yaml")
    config_location = os.path.join(home_dir, "config_deskew.yaml")


def write_config_file(config_settings, output_file_location):
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
    raw[2, 2, 2] = 10
    # Save image as a tif filw in home directory
    imsave(img_dir, raw)
    assert os.path.exists(img_dir)

    # Create a config file
    config = {
        "input": img_dir,
        "output": home_dir,
        "processing": "workflow",
        "workflow_path": workflow_location,
        "output_file_type": "h5"}

    write_config_file(config, config_location)
    assert os.path.exists(config_location)


def create_workflow():
    # Zeiss lattice
    voxel_size_x_in_microns = 0.14499219272808386
    voxel_size_y_in_microns = 0.14499219272808386
    voxel_size_z_in_microns = 0.3
    deskewing_angle_in_degrees = 30.0

    # Instantiate segmentation workflow
    image_seg_workflow = Workflow()

    image_seg_workflow.set("gaussian", cle.gaussian_blur,
                           "input", sigma_x=1, sigma_y=1, sigma_z=1)

    image_seg_workflow.set("binarisation", cle.threshold,
                           "gaussian", constant=0.5)

    image_seg_workflow.set(
        "labeling", cle.connected_components_labeling_box, "binarisation")

    save_workflow(workflow_location, image_seg_workflow)

    assert os.path.exists(workflow_location)


def test_napari_workflow():
    """Test napari workflow to see if it works before we run it using napari_lattice
       This is without deskewing
    """
    create_data()
    create_workflow()

    image_seg_workflow = load_workflow(workflow_location)

    # Open the saved image from above
    raw = imread(img_dir)
    # Set input image to be the "raw" image
    image_seg_workflow.set("input", raw)
    labeling = image_seg_workflow.get("labeling")
    assert (labeling[2, 2, 2] == 1)


def test_workflow_lattice():
    """Test workflow by loading into napari lattice
       This will apply deskewing before processing the workflow
    """
    # Deskew, apply workflow and save as h5
    cmd = f"napari_lattice --config {config_location}"
    deskew_process_yaml = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True)

    # Read data from stdout and stderr. waits for process to terminate, or it will execute next line immmediately
    _ = deskew_process_yaml.communicate()

    # checks if h5 file written
    h5_img = os.path.join(home_dir, "raw", "_0_raw.h5")
    assert os.path.exists(h5_img)

    import npy2bdv
    h5_file = npy2bdv.npy2bdv.BdvEditor(h5_img)

    label_img = h5_file.read_view(time=0, channel=0)

    assert (label_img.shape == (3, 14, 5))
    assert (label_img[1, 6, 2] == 1)
