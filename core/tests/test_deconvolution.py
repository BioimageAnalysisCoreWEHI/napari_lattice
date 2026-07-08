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

def test_skimage_decon_volume_smaller_than_psf():
    # Regression test: when a cropped volume is smaller than half the PSF along
    # an axis, dask's map_overlap used to raise
    # "The overlapping depth N is larger than your array M".
    # This surfaced non-deterministically in CI (only Python 3.11), because the
    # crop bounds round to exactly 45 z-slices on some numpy/BLAS builds while
    # the PSF's z half-depth is 46. The overlap depth must be clamped per axis.
    import numpy as np
    from lls_core.deconvolution import skimage_decon

    psf = np.ones((93, 78, 78), dtype=np.float32)
    psf /= psf.sum()
    # z=45 is smaller than psf.shape[0] // 2 == 46
    vol = np.random.rand(45, 95, 110).astype(np.float32)

    out = skimage_decon(
        vol, psf, num_iter=1, clip=False, filter_epsilon=0, boundary="nearest"
    ).compute()

    assert out.shape == vol.shape


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
