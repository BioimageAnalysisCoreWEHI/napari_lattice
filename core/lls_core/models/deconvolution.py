
from pydantic import Field, NonNegativeInt, validator

from typing import Any, List, Literal, Optional, Union, TYPE_CHECKING
from typing_extensions import Self, TypedDict, Unpack

from xarray import DataArray

from lls_core import DeconvolutionChoice
from lls_core.models.utils import enum_choices, FieldAccessMixin

from lls_core.types import image_like_to_image, ImageLike

Background = Union[float, Literal["auto", "second_last"]]
class MakeKwargs(TypedDict, total=False):
    decon_processing: Union[DeconvolutionChoice, str]
    psf: List[ImageLike]
    psf_num_iter: int
    background: str

class DeconvolutionParams(FieldAccessMixin, arbitrary_types_allowed=True):
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

    convert_image = validator("psf", pre=True, each_item=True, allow_reuse=True)(image_like_to_image)

    # @classmethod
    # def make(
    #     cls,
    #     **kwargs: Unpack[MakeKwargs]
    # ) -> Self:
    #     dest= { }
    #     decon_processing = kwargs.pop("decon_processing", None)
    #     if isinstance(decon_processing, DeconvolutionChoice):
    #         dest["decon_processing"] = decon_processing
    #     elif decon_processing is not None:
    #         dest["decon_processing"] = DeconvolutionChoice(decon_processing)

    #     DeconvolutionParams(**dest)
    #     dest
    #     1 + "a"

    #     DeconvolutionParams(**kwargs)

    #     background_processed: Background
    #     try:
    #         background_processed = float(background)
    #     except Exception:
    #         if background == "auto" or background == "second_last":
    #             background_processed = background
    #         else:
    #             raise Exception("Invalid background option")

    #     return DeconvolutionParams(
    #         decon_processing=decon_processing,
    #         background=background_processed,
    #         psf = [image_like_to_image(it) for it in psf],
    #         psf_num_iter=psf_num_iter
    #     )
