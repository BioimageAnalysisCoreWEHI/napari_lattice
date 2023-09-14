from typing import List, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Self
from pydantic import Field, NonNegativeInt
from napari.types import ShapesData
from xarray import DataArray
from lls_core.models.utils import FieldAccessMixin
from lls_core.types import image_like_to_image, ImageLike

if TYPE_CHECKING:
    from lls_core.types import ArrayLike
    from aicsimageio import AICSImage



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
    # @classmethod
    # def make(
    #     cls,
    #     roi_list: List[Union[ArrayLike, ImageLike]],
    #     z_range: Tuple[NonNegativeInt, NonNegativeInt]
    # ) -> Self:
    #     return CropParams(
    #         roi_list=[image_like_to_image(it) for it in roi_list],
    #         z_range=z_range
    #     )

    # @classmethod
    # def from_img_metadata(
    #     cls,
    #     img: DataArray,
    #     roi_list: List[Union[ArrayLike, ImageLike]],
    #     z_range: Optional[Tuple[NonNegativeInt, NonNegativeInt]]
    # ) -> Self:
    #     if z_range is None:
    #         z_range = (0, img.sizes["Z"])

    #     return CropParams.make(
    #         roi_list=roi_list,
    #         z_range=z_range
    #     )
