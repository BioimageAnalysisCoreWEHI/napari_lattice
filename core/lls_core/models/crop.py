from typing import List, Tuple
from pydantic import Field, NonNegativeInt
from xarray import DataArray
from lls_core.models.utils import FieldAccessMixin


class CropParams(FieldAccessMixin, arbitrary_types_allowed=True):
    """
    Parameters for the optional cropping step
    """
    roi_list: List[DataArray] = Field(
        description="List of regions of interest, each of which must be an NxD array, where N is the number of vertices and D the coordinates of each vertex."
    )
    z_range: Tuple[NonNegativeInt, NonNegativeInt] = Field(
        default=None,
        description="The range of Z slices to take. All Z slices before the first index or after the last index will be cropped out."
    )
