# class for initializing lattice data and setting metadata
# TODO: handle scenes
from dataclasses import dataclass
from aicsimageio.aics_image import AICSImage
from aicsimageio.dimensions import Dimensions
from lls_core import DeskewDirection, DeconvolutionChoice

from typing import Optional

from aicsimageio.types import ArrayLike

from lls_core.utils import get_deskewed_shape

@dataclass
class LatticeData:
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

    #: Number of time points
    time: int = 0
    #: Number of channels
    channels: int = 0

    # TODO: add defaults
    def __init__(self, img: AICSImage, angle: float, skew: DeskewDirection, dx: float, dy: float, dz: float, save_name: str):
        self.angle = angle
        self.skew = skew

        if img.physical_pixel_sizes != (None, None, None):
            self.data = img.dask_data
            self.dims = img.dims
            self.time = img.dims.T
            self.channels = img.dims.C
            self.dz = img.physical_pixel_sizes.Z or dz
            self.dy = img.physical_pixel_sizes.X or dx
            self.dz = img.physical_pixel_sizes.Y or dy

        else:
            self.data = img.dask_data
            self.dims = img.dims
            self.time = img.dims.T
            self.channels = img.dims.C

        # set new z voxel size
        if skew == DeskewDirection.Y or skew == DeskewDirection.X:
            import math
            dz = math.sin(angle * math.pi / 180.0) * dz

        # process the file to get shape of final deskewed image
        self.deskew_vol_shape, self.deskew_affine_transform = get_deskewed_shape(self.data, angle, dx, dy, dz)
        print(f"Channels: {self.channels}, Time: {self.time}")
        print("If channel and time need to be swapped, you can enforce this by choosing 'Last dimension is channel' when initialising the plugin")

    def get_angle(self) -> float:
        return self.angle

    def set_angle(self, angle: float) -> None:
        self.angle = angle

    def set_skew(self, skew: DeskewDirection) -> None:
        self.skew = skew