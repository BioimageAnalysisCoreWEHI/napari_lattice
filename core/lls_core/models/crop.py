from typing import List, Tuple, Any
from pydantic import Field, NonNegativeInt, validator
from xarray import DataArray
from lls_core.models.utils import FieldAccessMixin


class CropParams(FieldAccessMixin):
    """
    Parameters for the optional cropping step
    """
    roi_list: List[DataArray] = Field(
        description="List of regions of interest, each of which must be an NxD array, where N is the number of vertices and D the coordinates of each vertex.",
        default = []
    )
    z_range: Tuple[NonNegativeInt, NonNegativeInt] = Field(
        default=None,
        description="The range of Z slices to take. All Z slices before the first index or after the last index will be cropped out."
    )

    @validator("roi_list", each_item=True)
    def read_roi(cls, v: Any):
        from lls_core.types import is_pathlike
        from lls_core.cropping import read_roi_array
        # Converts the ROI from a path to an array
        if is_pathlike(v):
            return read_roi_array(v)
        return v
