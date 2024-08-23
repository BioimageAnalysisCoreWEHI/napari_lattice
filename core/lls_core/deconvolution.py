from __future__ import annotations

from pathlib import Path

from strenum import StrEnum

import logging
import importlib.util
from typing import Collection, Iterable,Union,Literal, Optional, TYPE_CHECKING
from aicsimageio.aics_image import AICSImage
from skimage.io import imread
from aicspylibczi import CziFile
from numpy.typing import NDArray
import os
import numpy as np
from dask.array.core import Array as DaskArray

from lls_core.utils import array_to_dask, pad_image_nearest_multiple
from lls_core.types import ArrayLike, is_arraylike

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData

class DeconvolutionChoice(StrEnum):
    """
    Deconvolution algorithm
    """
    cuda_gpu = "cuda_gpu"
    opencl_gpu = "opencl_gpu"
    cpu = "cpu"

logger = logging.getLogger(__name__)

def read_psf(psf_paths: Collection[Path],
              decon_option: DeconvolutionChoice,
              lattice_class: LatticeData) -> Iterable[NDArray]:
    """Read PSF files and return a list of PSF arrays appended to lattice_class.psf

    Args:
        decon_option (enum): Enum option from DeconvolutionChoice
        lattice_class: lattice class object, either LLSZWidget.LlszMenu.lattice or lattice class from batch processing
    """

    # remove empty paths; pathlib returns current directory as "." if None or empty str specified
    # When running batch processing,empty directory will be an empty string

    logging.debug(f"PSF paths are {psf_paths}")
    # total no of psf images
    psf_channels = len(psf_paths)
    assert psf_channels > 0, f"No images detected for PSF. Check the psf paths -> {psf_paths}"

    # Use CUDA for deconvolution
    if decon_option == DeconvolutionChoice.cuda_gpu:
        pycudadecon_import = importlib.util.find_spec("pycudadecon")
        assert pycudadecon_import, f"Pycudadecon not detected. Please install using: conda install -c conda-forge pycudadecon"
        otf_names = ["ch1", "ch2", "ch3", "ch4"]
        channels = [488, 561, 640, 123]
        # get temp directory to save generated otf
        import tempfile
        temp_dir = tempfile.gettempdir()+os.sep

    for psf in psf_paths:
        if psf.exists() and psf.is_file():
            if psf.suffix == ".czi":
                psf_czi = CziFile(psf.__str__())
                psf_aics = psf_czi.read_image()
                # make sure shape is 3D
                psf_aics = psf_aics[0][0]  # np.expand_dims(psf_aics[0],axis=0)
                # if len(psf_aics[0])>=1:
                #psf_channels = len(psf_aics[0])
                assert len(
                    psf_aics.shape) == 3, f"PSF should be a 3D image (shape of 3), but got {psf_aics.shape}"
                # pad psf to multiple of 16 for decon
                yield pad_image_nearest_multiple(img=psf_aics, nearest_multiple=16)
            else:
                #Use skimage to read tiff
                if psf.suffix in [".tif", ".tiff"]:
                   psf_aics_data = imread(str(psf))
                   if len(psf_aics_data.shape) != 3:
                       raise ValueError(f"PSF should be a 3D image (shape of 3), but got {psf_aics.shape}")
                else:
                    #Use AICSImageIO
                    psf_aics = AICSImage(str(psf))
                    psf_aics_data = psf_aics.data[0][0]
                    psf_aics_data = pad_image_nearest_multiple(
                        img=psf_aics_data, nearest_multiple=16)           
                    if psf_aics.dims.C != lattice_class.channels:
                        logger.warn(
                            f"PSF image has {psf_channels} channel/s, whereas image has {lattice_class.channels}")

                yield psf_aics_data

# Ideally we want to use OpenCL, but in the case of deconvolution most CUDA based
# libraries are better designed.. Atleast until  RL deconvolution is available in pyclesperant
# Talley Lamberts pycudadecon is a great library and highly optimised.
def pycuda_decon(
    image: ArrayLike,
    otf_path: Optional[str]=None,
    dzdata: float=0.3,
    dxdata: float=0.1449922,
    dzpsf: float=0.3,
    dxpsf: float=0.1449922,
    psf: Optional[ArrayLike]=None,
    num_iter: int = 10,
    cropping: bool = False,
    background: Union[float,Literal["auto","second_last"]] = 0 
):
    """Perform deconvolution using pycudadecon
    pycudadecon can return cropped images, so we pad the images with dimensions that are a multiple of 64

    Args:
        image : _description_
        otf_path : (path to the generated otf file, if available. Otherwise psf needs to be provided)
        dzdata : (pixel size in z in microns)
        dxdata : (pixel size in xy  in microns)
        dzpsf : (pixel size of original psf file in z  microns)
        dxpsf : (pixel size of original psf file in xy microns)
        psf (tiff): option to provide psf instead of the otfpath, this can be used when calling decon function
        num_iter (int): number of iterations
        cropping (bool): option to specify if cropping option is enabled
        background (float,"auto","second_last"): background value to use for deconvolution. Can specify a value. If using auto, defaults to pycudadecon settings which is median of last slice
        'second_last' means median of second last slice. This option is because last slice can sometimes be incomplete
    Returns:
        np.array: _description_
    """
    image = np.squeeze(image)
    assert image.ndim == 3, f"Image needs to be 3D. Got {image.ndim}"

    #Option for median of second last slices to be used
    #Similar to pycudadecon
    if isinstance(background, str) and background == "second_last":
        background = np.median(image[-2])
    
    # if dask array, convert to numpy array
    if isinstance(image, DaskArray):
        image = np.array(image)

    orig_img_shape = image.shape

    # if cropping, add extra padding to avoid edge artefact
    # not doing this with with large images as images can be quite large with all the padding leading to memory issues
    if cropping:
        # pad image y dimensionswith half of psf shape
        z_psf_pad, y_psf_pad, x_psf_pad = np.array(psf.shape) // 2
        image = np.pad(
            image,
            (
                (z_psf_pad, z_psf_pad),
                (y_psf_pad, y_psf_pad),
                (x_psf_pad, x_psf_pad),
            ),
            mode="reflect",
        )

    # pad image to a multiple of 64
    image = pad_image_nearest_multiple(img=image, nearest_multiple=64)

    if is_arraylike(psf):
        from pycudadecon import RLContext, TemporaryOTF, rl_decon

        psf = np.squeeze(psf)  # remove unit dimensions
        assert psf.ndim == 3, f"PSF needs to be 3D. Got {psf.ndim}"
        # Temporary OTF generation; RLContext ensures memory cleanup (runs rl_init and rl_cleanup)
        with TemporaryOTF(psf) as otf:
            with RLContext(
                rawdata_shape=image.shape,
                otfpath=otf.path,
                dzdata=dzdata,
                dxdata=dxdata,
                dzpsf=dzpsf,
                dxpsf=dxpsf,
            ) as ctx:
                decon_res = rl_decon(
                    im=image, output_shape=ctx.out_shape, n_iters=num_iter,background=background
                )

    else:
        from pycudadecon import rl_decon, rl_init, rl_cleanup

        rl_init(
            rawdata_shape=image.shape,
            otfpath=otf_path,
            dzdata=dzdata,
            dxdata=dxdata,
            dzpsf=dzpsf,
            dxpsf=dxpsf,
        )
        decon_res = rl_decon(image)
        rl_cleanup()

    if cropping:
        # remove psf padding if cropping was chosen
        decon_res = decon_res[
            z_psf_pad:-z_psf_pad, y_psf_pad:-y_psf_pad, x_psf_pad:-x_psf_pad
        ]

    # remove padding; get shape difference and use this shape difference to remove padding
    shape_diff = np.array(decon_res.shape) - np.array(orig_img_shape)
    # if above is negative,
    if shape_diff[0] == 0:
        shape_diff[0] = -orig_img_shape[0]
    if shape_diff[1] == 0:
        shape_diff[1] = -orig_img_shape[1]
    if shape_diff[2] == 0:
        shape_diff[2] = -orig_img_shape[2]

    decon_res = decon_res[: -shape_diff[0], : -shape_diff[1], : -shape_diff[2]]
    # print(orig_img_shape)
    # make sure the decon image and the original image shapes are the same
    assert (
        decon_res.shape == orig_img_shape
    ), f"Deconvolved {decon_res.shape} and original image shape {orig_img_shape} do not match."
    return decon_res


def skimage_decon(
    vol_zyx, psf, num_iter: int, clip: bool, filter_epsilon, boundary: str
):
    """Deconvolution using scikit image

    Args:
        vol_zyx (_type_): _description_
        psf (_type_): _description_
        num_iter (_type_): _description_
        clip (_type_): _description_
        filter_epsilon (_type_): _description_
        boundary (_type_): _description_

    Returns:
        _type_: _description_
    """
    from skimage.restoration import richardson_lucy as rl_decon_skimage

    depth = tuple(np.array(psf.shape) // 2)
    vol_zyx = array_to_dask(vol_zyx)
    decon_data = vol_zyx.map_overlap(
        rl_decon_skimage,
        psf=psf,
        num_iter=num_iter,
        clip=clip,
        filter_epsilon=filter_epsilon,
        boundary=boundary,
        depth=depth,
        trim=True,
    )
    return decon_data


def pyopencl_decon(vol_zyx, psf, num_iter: int, clip: bool, filter_epsilon):
    """Deconvolution using RedFishLion library for pyopencl based deconvolution
    This is slower than pycudadecon

    Args:
        vol_zyx (_type_): _description_
        psf (_type_): _description_
        num_iter (_type_): _description_
        clip (_type_): _description_
        filter_epsilon (_type_): _description_
        boundary (_type_): _description_

    Returns:
        _type_: _description_
    """
    # CROP PSF; Crop psfs when reading and initialising psfs

    return
