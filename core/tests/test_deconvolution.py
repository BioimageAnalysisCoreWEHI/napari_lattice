# Using similar template as Talley Lamberts from pydcudadecon
# https://github.com/tlambert03/pycudadecon/blob/main/tests/test_decon.py
# Github runner has no GPU, so cannot test deconvolution. Should run locally.

import numpy.testing as npt
from skimage.io import imread
import pyclesperanto_prototype as cle

from os.path import dirname
import os

import pytest


test_data_dir = os.path.join(dirname(__file__), "data")
# data directory containing raw, psf and deconvolved data
ATOL = 0.015
RTOL = 0.15

try:
    gpu_devices = cle.available_device_names(dev_type="gpu")
except:
    gpu_devices = []

# if no GPU devices, skip test; currently does not check if its non NVIDIA devices, so it can throw an error if a non-NVIDIA Gpu is used

try:
    import pycudadecon._libwrap
    cuda_decon_available = True
except (FileNotFoundError, ModuleNotFoundError):
    cuda_decon_available = False

@pytest.mark.skipif(condition=len(gpu_devices) < 1, reason="GPU not detected, so deconvolution with pycudadecon skipped.")
@pytest.mark.skipif(condition=not cuda_decon_available, reason="cudadecon library is not installed")
def test_deconvolution_pycudadecon():
    from lls_core.llsz_core import pycuda_decon

    data = imread(test_data_dir+"/raw.tif")
    psf = imread(test_data_dir+"/psf.tif")
    decon_saved = imread(test_data_dir+"/deconvolved.tif")
    deconvolved = pycuda_decon(image=data, psf=psf, num_iter=10,background="auto")
    npt.assert_allclose(deconvolved, decon_saved, atol=ATOL)  # , verbose=True)
    
# Test for opencl deconvolution
# def test_deconvolution_opencl():
#    pass
