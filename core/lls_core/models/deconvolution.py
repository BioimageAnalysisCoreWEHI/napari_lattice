
from pathlib import Path

from pydantic.v1 import Field, NonNegativeInt, root_validator, validator

from typing_extensions import Any, List, Literal, Union

from xarray import DataArray

from lls_core.models.utils import enum_choices, FieldAccessModel
from lls_core.deconvolution import DeconvolutionChoice
from lls_core.types import image_like_to_image
from pydantic import Field, NonNegativeInt, field_validator

Background = Union[float, Literal["auto", "second_last"]]
class DeconvolutionParams(FieldAccessModel):
    """
    Parameters for the optional deconvolution step
    """
    decon_processing: DeconvolutionChoice = Field(
        default=DeconvolutionChoice.cpu,
        description=f"Hardware to use to perform the deconvolution. Choices: `{enum_choices(DeconvolutionChoice)}`. Can be provided as `str`."
    )
    psf: List[DataArray] = Field(
        default=[],
        description="List of Point Spread Functions to use for deconvolution. Each of which should be a 3D array. Each PSF can also be provided as a `str` path, in which case they will be loaded from disk as images."
    )
    psf_paths: List[Path] = Field(
        default=[],
        cli_hide=True,
        description="Internal: the filesystem paths the PSFs were loaded from, if any. "
                    "The validated `psf` field holds arrays, which cannot be serialised "
                    "back to paths, so output metadata records these instead. "
                    "Not a user-facing parameter."
    )
    decon_num_iter: NonNegativeInt = Field(
        default=10,
        description="Number of iterations to perform in deconvolution"
    )
    background: Background = Field(
        default=0,
        description='Background value to subtract for deconvolution. Only used when `decon_processing` is set to `GPU`. This can either be a literal number, "auto" which uses the median of the last slice, or "second_last" which uses the median of the last slice.'
    )

    @field_validator("decon_processing", mode="before")
    @classmethod
    @root_validator(pre=True)
    def capture_psf_paths(cls, values: dict) -> dict:
        "Record the PSF paths before `convert_image` replaces them with arrays."
        from lls_core.types import is_pathlike

        given = values.get("psf")
        if given is None:
            return values
        if is_pathlike(given):
            given = [given]
        paths = [str(item) for item in given if is_pathlike(item)]
        if paths:
            values["psf_paths"] = paths
        return values

    @validator("decon_processing", pre=True)
    def convert_decon(cls, v: Any):
        if isinstance(v, str):
            return DeconvolutionChoice[v]
        return v

    @field_validator("psf", mode="before")
    @classmethod
    def convert_image(cls, v):
        # each_item=True doesn't exist in Pydantic v2, so apply the per-item
        # conversion manually. If v isn't list-like, leave it for the normal
        # list-type validation to produce the right error.
        if not isinstance(v, (list, tuple)):
            return v

        def _convert(item):
            img = image_like_to_image(item)
            # Ensure the PSF is 3D
            if "C" in img.dims:
                img = img.isel(C=0)
            if "T" in img.dims:
                img = img.isel(T=0)
            if len(img.dims) != 3:
                raise ValueError("PSF is not a 3D array!")
            return img

        return [_convert(item) for item in v]
