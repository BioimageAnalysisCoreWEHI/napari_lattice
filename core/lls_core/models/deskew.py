from __future__ import annotations
# class for initializing lattice data and setting metadata
# TODO: handle scenes
from pydantic import Field, NonNegativeFloat, validator, root_validator

from typing import Any, Tuple
from typing_extensions import Self, TYPE_CHECKING

import pyclesperanto_prototype as cle

from lls_core import DeskewDirection
from xarray import DataArray

from lls_core.models.utils import FieldAccessMixin, enum_choices
from lls_core.types import image_like_to_image
from lls_core.utils import get_deskewed_shape

if TYPE_CHECKING:
    from aicsimageio.types import PhysicalPixelSizes

# DeskewDirection = Literal["X", "Y"]

class DefinedPixelSizes(FieldAccessMixin):
    """
    Like PhysicalPixelSizes, but it's a dataclass, and
    none of its fields are None
    """
    X: NonNegativeFloat = 0.1499219272808386
    Y: NonNegativeFloat = 0.1499219272808386
    Z: NonNegativeFloat = 0.3

    @classmethod
    def from_physical(cls, pixels: PhysicalPixelSizes) -> Self:
        from lls_core.utils import raise_if_none

        return DefinedPixelSizes(
            X=raise_if_none(pixels.X, "All pixels must be defined"),
            Y=raise_if_none(pixels.Y, "All pixels must be defined"),
            Z=raise_if_none(pixels.Z, "All pixels must be defined"),
        )


class DeskewParams(FieldAccessMixin):
    image: DataArray = Field(
        description="A 3-5D array containing the image data."
    )
    skew: DeskewDirection = Field(
        default=DeskewDirection.Y,
        description=f"Axis along which to deskew the image. Choices: {enum_choices(DeskewDirection)}."
    )
    angle: float = Field(
        default=30.0,
        description="Angle of deskewing, in degrees."
    )
    physical_pixel_sizes: DefinedPixelSizes = Field(
    default_factory=DefinedPixelSizes,
    description="Pixel size of the microscope, in microns."
)
    deskew_vol_shape: Tuple[int, ...] = Field(
        init_var=False,
        default=None,
        description="Dimensions of the deskewed output. This is set automatically based on other input parameters, and doesn't need to be provided by the user."
    )

    deskew_affine_transform: cle.AffineTransform3D = Field(init_var=False, default=None, description="Deskewing transformation function. This is set automatically based on other input parameters, and doesn't need to be provided by the user.")

    @property
    def dims(self):
        return self.image.dims

    @validator("skew", pre=True)
    def convert_skew(cls, v: Any):
        # Allow skew to be provided as a string
        if isinstance(v, str):
            return DeskewDirection[v]
        return v

    @validator("physical_pixel_sizes", pre=True)
    def convert_pixels(cls, v: Any):
        # Allow the pixel sizes to be specified as a tuple
        if isinstance(v, tuple) and len(v) == 3:
            return DefinedPixelSizes(X=v[0], Y=v[1], Z=v[2])
        return v

    # convert_image = validator("image", pre=True, allow_reuse=True)(image_like_to_image)

    @validator("image", pre=True)
    def reshaping(cls, v: Any):
        # This allows a user to pass in any array-like object and have it
        # converted and reshaped appropriately
        array = image_like_to_image(v)
        if not set(array.dims).issuperset({"X", "Y", "Z"}):
            raise ValueError("The input array must at least have XYZ coordinates")
        if "T" not in array.dims:
            array = array.expand_dims("T")
        if "C" not in array.dims:
            array = array.expand_dims("C")
        return array.transpose("T", "C", "Z", "Y", "X")

    def get_3d_slice(self) -> DataArray:
        return self.image.isel(C=0, T=0)

    @root_validator(pre=False)
    def set_deskew(cls, values: dict) -> dict:
        """
        Sets the default deskew shape values if the user has not provided them
        """
        # process the file to get shape of final deskewed image
        if "image" not in values:
            return values
        data: DataArray = cls.reshaping(values["image"])
        if values.get('deskew_vol_shape') is None:
            if values.get('deskew_affine_transform') is None:
                # If neither has been set, calculate them ourselves
                values["deskew_vol_shape"], values["deskew_affine_transform"] = get_deskewed_shape(data.isel(C=0, T=0).to_numpy(), values["angle"], values["physical_pixel_sizes"].X, values["physical_pixel_sizes"].Y, values["physical_pixel_sizes"].Z, values["skew"])
            else:
                raise ValueError("deskew_vol_shape and deskew_affine_transform must be either both specified or neither specified")
        return values
