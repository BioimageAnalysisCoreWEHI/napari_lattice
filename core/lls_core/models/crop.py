
from typing import Tuple
from pydantic import BaseModel, Field, NonNegativeInt
from napari.types import ShapesData


z_range: Tuple[NonNegativeInt, NonNegativeInt] = Field(
    default=None,
    description="The range of Z slices to take. All Z slices before the first index or after the last index will be cropped out."
)
class CropParams(BaseModel, arbitrary_types_allowed=True):
    """
    Parameters for the optional cropping step
    """
    roi_list: ShapesData = Field(
        description="List of regions of interest, each of which must be an NxD array, where N is the number of vertices and D the coordinates of each vertex."
    )
    z_range: Tuple[NonNegativeInt, NonNegativeInt] = z_range
