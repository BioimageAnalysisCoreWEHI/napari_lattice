from __future__ import annotations
# class for initializing lattice data and setting metadata
# TODO: handle scenes
from dataclasses import dataclass, field
from aicsimageio.aics_image import AICSImage
from aicsimageio.dimensions import Dimensions
from numpy.typing import NDArray
from dataclasses import dataclass
import math
import numpy as np

from typing import Any, List, Literal, Optional, TYPE_CHECKING, Tuple, TypeVar

from aicsimageio.types import ArrayLike, PhysicalPixelSizes
import pyclesperanto_prototype as cle

from lls_core import DeskewDirection, DeconvolutionChoice
from lls_core.utils import get_deskewed_shape

if TYPE_CHECKING:
    import pyclesperanto_prototype as cle

T = TypeVar("T")
def raise_if_none(obj: Optional[T], message: str) -> T:
    if obj is None:
        raise TypeError(message)
    return obj

@dataclass
class DefinedPixelSizes:
    """
    Like PhysicalPixelSizes, but it's a dataclass, and
    none of its fields are None
    """
    X: float = 0.14
    Y: float = 0.14
    Z: float = 0.3

@dataclass
class LatticeData:
    """
    Holds data and metadata for a given image in a consistent format
    """
    #: 3-5D array
    data: ArrayLike
    dims: Dimensions

    #: The filename of this data when it is saved
    save_name: str

    #: Geometry of the light path
    skew: DeskewDirection = DeskewDirection.Y
    angle: float = 30.0

    decon_processing: Optional[DeconvolutionChoice] = None

    #: Pixel size in microns
    physical_pixel_sizes: DefinedPixelSizes = field(default_factory=DefinedPixelSizes)

    new_dz: Optional[float] = None

    # Dimensions of the deskewed output
    deskew_vol_shape: Optional[Tuple[int, ...]] = None
    deskew_affine_transform: Optional[cle.AffineTransform3D] = None

    # PSF data that should be refactored into another class eventually
    psf: Optional[List[NDArray]] = None
    psf_num_iter: Optional[int] = None
    otf_path: Optional[List] = None

    #: Number of time points
    time: int = 0
    #: Number of channels
    channels: int = 0

    # TODO: add defaults here, rather than in the CLI
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

    @property
    def dx(self) -> float:
        return self.physical_pixel_sizes.X

    @dx.setter
    def dx(self, value: float):
        self.physical_pixel_sizes.X = value

    @property
    def dy(self) -> float:
        return self.physical_pixel_sizes.Y

    @dy.setter
    def dy(self, value: float):
        self.physical_pixel_sizes.Y = value

    @property
    def dz(self) -> float:
        return self.physical_pixel_sizes.Z

    @dz.setter
    def dz(self, value: float):
        self.physical_pixel_sizes.Z = value

    def get_angle(self) -> float:
        return self.angle

    def set_angle(self, angle: float) -> None:
        self.angle = angle

    def set_skew(self, skew: DeskewDirection) -> None:
        self.skew = skew

    def __post_init__(self):
        # set new z voxel size
        if self.skew == DeskewDirection.Y or self.skew == DeskewDirection.X:
            self.new_dz = math.sin(self.angle * math.pi / 180.0) * self.dz

        # process the file to get shape of final deskewed image
        self.deskew_vol_shape, self.deskew_affine_transform = get_deskewed_shape(self.data, self.angle, self.dx, self.dy, self.dz)
        print(f"Channels: {self.channels}, Time: {self.time}")
        print("If channel and time need to be swapped, you can enforce this by choosing 'Last dimension is channel' when initialising the plugin")

def lattice_from_aics(img: AICSImage, physical_pixel_sizes: PhysicalPixelSizes = PhysicalPixelSizes(None, None, None), **kwargs: Any) -> LatticeData:
    # Note: The reason we copy all of these fields rather than just storing the AICSImage is because that class is mostly immutable and so not suitable

    pixel_sizes = DefinedPixelSizes(
        X = physical_pixel_sizes[0] or img.physical_pixel_sizes.X or LatticeData.physical_pixel_sizes.X,
        Y = physical_pixel_sizes[1] or img.physical_pixel_sizes.Y or LatticeData.physical_pixel_sizes.Y, 
        Z = physical_pixel_sizes[2] or img.physical_pixel_sizes.Z or LatticeData.physical_pixel_sizes.Z 
    )

    return LatticeData(
        data = img.dask_data,
        dims = img.dims,
        time = img.dims.T,
        channels = img.dims.C,
        physical_pixel_sizes = pixel_sizes,
        **kwargs
    )

def img_from_array(arr: ArrayLike, last_dimension: Optional[Literal["channel", "time"]] = None, **kwargs: Any) -> AICSImage:
    """
    Creates an AICSImage from an array without metadata

    Args:
        arr (ArrayLike): An array
        last_dimension: How to handle the dimension order
        kwargs: Additional arguments to pass to the AICSImage constructor
    """    
    dim_order: str

    if len(arr.shape) < 3 or len(arr.shape) > 5:
        raise ValueError("Array dimensions must be in the range [3, 5]")

    # if aicsimageio tiffreader assigns last dim as time when it should be channel, user can override this
    if len(arr.shape) == 3:
        dim_order="ZYX"
    else:
        if last_dimension not in ["channel", "time"]:
            raise ValueError("last_dimension must be either channel or time")
        if len(arr.shape) == 4:
            if last_dimension == "channel":
                dim_order = "CZYX"
            elif last_dimension == "time":
                dim_order = "TZYX"
        elif len(arr.shape) == 5:
            if last_dimension == "channel":
                dim_order = "CTZYX"
            elif last_dimension == "time":
                dim_order = "TCZYX"
        else:
            raise ValueError()

    img = AICSImage(image=arr, dim_order=dim_order, **kwargs)

    # if last axes of "aicsimage data" shape is not equal to time, then swap channel and time
    if img.data.shape[0] != img.dims.T or img.data.shape[1] != img.dims.C:
        arr = np.swapaxes(arr, 0, 1)
    return AICSImage(image=arr, dim_order=dim_order, **kwargs)


def lattice_fom_array(arr: ArrayLike, last_dimension: Optional[Literal["channel", "time"]] = None, **kwargs: Any) -> LatticeData:
    """
    Creates a `LatticeData` from an array

    Args:
        arr: Array to use as the data source
        last_dimension: See img_from_array
    """   
    aics = img_from_array(arr, last_dimension)
    return lattice_from_aics(aics, **kwargs)