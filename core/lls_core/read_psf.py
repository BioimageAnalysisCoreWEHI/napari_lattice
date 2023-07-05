from pathlib import Path, PosixPath
from lls_core import DeconvolutionChoice
from lls_core.lattice_data import LatticeData
import logging

logger = logging.getLogger(__name__)

def _read_psf(psf_ch1_path: Path,
              psf_ch2_path: Path,
              psf_ch3_path: Path,
              psf_ch4_path: Path,
              decon_option: DeconvolutionChoice,
              lattice_class: LatticeData):
    """Read PSF files and return a list of PSF arrays appended to lattice_class.psf

    Args:
        decon_option (enum): Enum option from DeconvolutionChoice
        lattice_class: lattice class object, either LLSZWidget.LlszMenu.lattice or lattice class from batch processing
    """
    # get the psf paths into a list
    psf_paths = [psf_ch1_path, psf_ch2_path, psf_ch3_path, psf_ch4_path]

    # remove empty paths; pathlib returns current directory as "." if None or empty str specified
    # When running batch processing,empty directory will be an empty string

    import platform
    from pathlib import PureWindowsPath, PosixPath

    if platform.system() == "Linux":
        psf_paths = [Path(x) for x in psf_paths if x !=
                     PosixPath(".") and x != ""]
    elif platform.system() == "Windows":
        psf_paths = [Path(x) for x in psf_paths if x !=
                     PureWindowsPath(".") and x != ""]

    logging.debug(f"PSF paths are {psf_paths}")
    # total no of psf images
    psf_channels = len(psf_paths)
    assert psf_channels > 0, f"No images detected for PSF. Check the psf paths -> {psf_paths}"

    # Use CUDA for deconvolution
    if decon_option == DeconvolutionChoice.cuda_gpu:
        import importlib
        pycudadecon_import = importlib.util.find_spec("pycudadecon")
        assert pycudadecon_import, f"Pycudadecon not detected. Please install using: conda install -c conda-forge pycudadecon"
        otf_names = ["ch1", "ch2", "ch3", "ch4"]
        channels = [488, 561, 640, 123]
        # get temp directory to save generated otf
        import tempfile
        temp_dir = tempfile.gettempdir()+os.sep

    for idx, psf in enumerate(psf_paths):
        if os.path.exists(psf) and psf.is_file():
            if psf.suffix == ".czi":
                from aicspylibczi import CziFile
                psf_czi = CziFile(psf.__str__())
                psf_aics = psf_czi.read_image()
                # make sure shape is 3D
                psf_aics = psf_aics[0][0]  # np.expand_dims(psf_aics[0],axis=0)
                # if len(psf_aics[0])>=1:
                #psf_channels = len(psf_aics[0])
                assert len(
                    psf_aics.shape) == 3, f"PSF should be a 3D image (shape of 3), but got {psf_aics.shape}"
                # pad psf to multiple of 16 for decon
                psf_aics = pad_image_nearest_multiple(
                    img=psf_aics, nearest_multiple=16)
                lattice_class.psf.append(psf_aics)
            else:
                psf_aics = AICSImage(psf.__str__())
                psf_aics_data = psf_aics.data[0][0]
                psf_aics_data = pad_image_nearest_multiple(
                    img=psf_aics_data, nearest_multiple=16)
                lattice_class.psf.append(psf_aics_data)
                if psf_aics.dims.C >= 1:
                    psf_channels = psf_aics.dims.C

    #LLSZWidget.LlszMenu.lattice.channels =3
    if psf_channels != lattice_class.channels:
        logger.warn(
            f"PSF image has {psf_channels} channel/s, whereas image has {lattice_class.channels}")