
from pydantic import BaseModel, Field, NonNegativeInt
from pydantic_numpy import NDArray

from typing import List, Literal, Union


from lls_core import DeconvolutionChoice
from lls_core.models.utils import enum_choices

decon_processing: DeconvolutionChoice = Field(
    default=DeconvolutionChoice.cpu,
    description=f"Hardware to use to perform the deconvolution. Choices: {enum_choices(DeconvolutionChoice)}")
psf: List[NDArray] = Field(
    default=[],
    description="List of Point Spread Functions to use for deconvolution. Each of which should be a 3D array."
)
psf_num_iter: NonNegativeInt = Field(
    default=10,
    description="Number of iterations to perform in deconvolution"
)
Background = Union[float, Literal["auto", "second_last"]]
background: Background = Field(
    default=0,
    description='Background value to subtract for deconvolution. Only used when decon_processing is set to GPU. This can either be a literal number, "auto" which uses the median of the last slice, or "second_last" which uses the median of the last slice.'
)
class DeconvolutionParams(BaseModel, arbitrary_types_allowed=True):
    """
    Parameters for the optional deconvolution step
    """
    decon_processing: DeconvolutionChoice = decon_processing
    psf: List[NDArray] = psf
    psf_num_iter: NonNegativeInt = psf_num_iter
    background: Union[float, Literal["auto", "second_last"]] = background
