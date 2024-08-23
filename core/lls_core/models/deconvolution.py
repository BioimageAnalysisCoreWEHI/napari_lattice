
from pydantic import Field, NonNegativeInt, validator

from typing import Any, List, Literal, Union
from typing_extensions import TypedDict

from xarray import DataArray

from lls_core import DeconvolutionChoice
from lls_core.models.utils import enum_choices, FieldAccessModel

from lls_core.types import image_like_to_image, ImageLike

Background = Union[float, Literal["auto", "second_last"]]
class DeconvolutionParams(FieldAccessModel):
    """
    Parameters for the optional deconvolution step
    """
    decon_processing: DeconvolutionChoice = Field(
        default=DeconvolutionChoice.cpu,
        description=f"Hardware to use to perform the deconvolution. Choices: {enum_choices(DeconvolutionChoice)}"
    )
    psf: List[DataArray] = Field(
        default=[],
        description="List of Point Spread Functions to use for deconvolution. Each of which should be a 3D array."
    )
    psf_num_iter: NonNegativeInt = Field(
        default=10,
        description="Number of iterations to perform in deconvolution"
    )
    background: Background = Field(
        default=0,
        description='Background value to subtract for deconvolution. Only used when decon_processing is set to GPU. This can either be a literal number, "auto" which uses the median of the last slice, or "second_last" which uses the median of the last slice.'
    )

    @validator("decon_processing", pre=True)
    def convert_decon(cls, v: Any):
        if isinstance(v, str):
            return DeconvolutionChoice[v]
        return v

    @validator("psf", pre=True, each_item=True, allow_reuse=True)
    def convert_image(cls, v):
        img = image_like_to_image(v)
        # Ensure the PSF is 3D
        if "C" in img.dims:
            img = img.isel(C=0)
        if "T" in img.dims:
            img = img.isel(T=0)
        if len(img.dims) != 3:
            raise ValueError("PSF is not a 3D array!")
        return img
