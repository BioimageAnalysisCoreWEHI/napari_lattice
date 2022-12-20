# Using similar template as Talley Lamberts from pydcudadecon
# https://github.com/tlambert03/pycudadecon/blob/main/tests/test_decon.py

import pyclesperanto_prototype as cle
import numpy as np
import numpy.testing as npt

from skimage.io import imread
from napari_lattice.llsz_core import pycuda_decon

# data directory containing raw, psf and deconvolved data
test_data_dir = "./data/"
ATOL = 0.015
RTOL = 0.15


def test_deconvolution():

    data = imread(test_data_dir+"raw.tif")
    psf = imread(test_data_dir+"psf.tif")
    decon_saved = imread(test_data_dir+"deconvolved.tif")

    deconvolved = pycuda_decon(image=data, psf=psf, num_iter=10)
    npt.assert_allclose(deconvolved, decon_saved, atol=ATOL)  # , verbose=True)
