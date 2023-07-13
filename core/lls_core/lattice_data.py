from __future__ import annotations
# class for initializing lattice data and setting metadata
# TODO: handle scenes
from dataclasses import dataclass
from aicsimageio.aics_image import AICSImage
from aicsimageio.dimensions import Dimensions
from lls_core import DeskewDirection, DeconvolutionChoice
from numpy.typing import NDArray

from typing import List, Optional, TYPE_CHECKING, Tuple, TypeVar

from aicsimageio.types import ArrayLike
import pyclesperanto_prototype as cle

from lls_core.utils import get_deskewed_shape

if TYPE_CHECKING:
    import pyclesperanto_prototype as cle

T = TypeVar("T")
def raise_if_none(obj: Optional[T], message: str) -> T:
    if obj is None:
        raise TypeError(message)
    return obj

@dataclass
class LatticeData:
    """
    Holds data and metadata for a given image in a consistent format
    """
    #: 3-5D array
    data: ArrayLike
    dims: Dimensions
    #: Pixel size in microns
    dx: float
    dy: float
    dz: float
    #: Geometry of the light path
    skew: DeskewDirection
    angle: float

    #: The filename of this data when it is saved
    save_name: str
    decon_processing: Optional[DeconvolutionChoice]

    # Dimensions of the deskewed output
    deskew_vol_shape: Tuple[int]
    deskew_affine_transform: cle.AffineTransform3D

    # PSF data that should be refactored into another class eventually
    psf: List[NDArray]
    psf_num_iter: int
    otf_path: List

    #: Number of time points
    time: int = 0
    #: Number of channels
    channels: int = 0

    # TODO: add defaults here, rather than in the CLI
    def __init__(self, img: AICSImage, angle: float, skew: DeskewDirection, save_name: str, dx: Optional[float] = None, dy: Optional[float] = None, dz: Optional[float] = None):
        # Note: The reason we copy all of these fields rather than just storing the AICSImage is because that class is mostly immutable and so not suitable
        self.angle = angle
        self.skew = skew
        self.data = img.dask_data
        self.dims = img.dims
        self.time = img.dims.T
        self.channels = img.dims.C
        self.dx = raise_if_none(dx or img.physical_pixel_sizes.X, "dx cannot be None")
        self.dy = raise_if_none(dy or img.physical_pixel_sizes.Y, "dy cannot be None")
        self.dz = raise_if_none(dz or img.physical_pixel_sizes.Z, "dz cannot be None")
        self.save_name = save_name

        # set new z voxel size
        if skew == DeskewDirection.Y or skew == DeskewDirection.X:
            import math
            dz = math.sin(angle * math.pi / 180.0) * dz

        # process the file to get shape of final deskewed image
        self.deskew_vol_shape, self.deskew_affine_transform = get_deskewed_shape(self.data, angle, self.dx, self.dy, self.dz)
        print(f"Channels: {self.channels}, Time: {self.time}")
        print("If channel and time need to be swapped, you can enforce this by choosing 'Last dimension is channel' when initialising the plugin")

    # Hack to ensure that .skew_dir behaves identically to .skew
    @property
    def skew_dir(self) -> DeskewDirection:
        return self.skew

    @skew_dir.setter
    def skew_dir(self, value: DeskewDirection):
        self.skew = value

    @property
    def deskew_func(self):
        # Chance deskew function absed on skew direction
        if self.skew == DeskewDirection.Y:
            return cle.deskew_y
        elif self.skew == DeskewDirection.X:
            return cle.deskew_x
        else:
            raise ValueError()

    def get_angle(self) -> float:
        return self.angle

    def set_angle(self, angle: float) -> None:
        self.angle = angle

    def set_skew(self, skew: DeskewDirection) -> None:
        self.skew = skew