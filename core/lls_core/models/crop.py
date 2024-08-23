from typing_extensions import Any, Iterable, List, Tuple
from pydantic.v1 import Field, NonNegativeInt, validator
from lls_core.models.utils import FieldAccessModel
from lls_core.cropping import Roi

class CropParams(FieldAccessModel):
    """
    Parameters for the optional cropping step.
    Note that cropping is performed in the space of the deskewed shape.
    This is to support the workflow of performing a preview deskew and using that
    to calculate the cropping coordinates.
    """
    roi_list: List[Roi] = Field(
        description="List of regions of interest, each of which must be an `N × D` array, where N is the number of vertices and D the coordinates of each vertex. This can alternatively be provided as a `str` or `Path`, or a list of those, in which case each they are interpreted as paths to ImageJ ROI zip files that are read from disk.",
        cli_description="List of regions of interest, each of which must be the file path to ImageJ ROI file.",
        default = []
    )
    roi_subset: List[int] = Field(
        description="A subset of all the ROIs to process. Each list item should be an index into the ROI list indicating an ROI to include. This allows you to process only a subset of the regions from a ROI file specified using the `roi_list` parameter. If `None`, it is assumed that you want to process all ROIs.",
        default=None
    )
    z_range: Tuple[NonNegativeInt, NonNegativeInt] = Field(
        default=None,
        description="The range of Z slices to take as a tuple of the form `(first, last)`. All Z slices before the first index or after the last index will be cropped out.",
        cli_description="An array with two items, indicating the index of the first and last Z slice to include."
    )

    @property
    def selected_rois(self) -> Iterable[Roi]:
        "Returns the relevant ROIs that should be processed"
        for i in self.roi_subset:
            yield self.roi_list[i]

    @validator("roi_list", pre=True)
    def read_roi(cls, v: Any) -> List[Roi]:
        from lls_core.types import is_pathlike
        from lls_core.cropping import read_imagej_roi
        from numpy import ndarray
        # Allow a single path
        if is_pathlike(v):
            v = [v]

        rois: List[Roi] = []
        for item in v:
            if is_pathlike(item):
                rois += read_imagej_roi(item)
            elif isinstance(item, ndarray):
                rois.append(Roi.from_array(item))
            elif isinstance(item, Roi):
                rois.append(item)
            else:
                # Try converting an iterable to ROI
                try:
                    rois.append(Roi(*item))
                except:
                    raise ValueError(f"{item} cannot be intepreted as an ROI")

        if len(rois) == 0:
            raise ValueError("At least one region of interest must be specified if cropping is enabled")

        return rois

    @validator("roi_subset", pre=True, always=True)
    def default_roi_range(cls, v: Any, values: dict):
        # If the roi range isn't provided, assume all rois should be processed
        if v is None and "roi_list" in values:
            return list(range(len(values["roi_list"])))
        return v
