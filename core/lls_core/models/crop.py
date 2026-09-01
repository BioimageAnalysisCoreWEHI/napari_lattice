from typing_extensions import Any, Iterable, List, Tuple
from lls_core.models.utils import FieldAccessModel
from lls_core.cropping import Roi, RoiUnits
from pydantic import Field, NonNegativeInt, ValidationInfo, field_validator, model_validator

class CropParams(FieldAccessModel):
    """
    Parameters for the optional cropping step.
    Note that cropping is performed in the space of the deskewed shape.
    This is to support the workflow of performing a preview deskew and using that
    to calculate the cropping coordinates.

    `roi_list` is always in deskewed-image pixels. A file given in microns is
    converted on the way in, once the pixel size is known - see `roi_units`.
    """
    roi_list: List[Roi] = Field(
        description="List of regions of interest, each of which must be an `N × D` array, where N is the number of vertices and D the coordinates of each vertex. This can alternatively be provided as a `str` or `Path`, or a list of those, in which case they are interpreted as paths to ImageJ ROI (.roi/.zip) or napari shapes (.csv) files that are read from disk.",
        cli_description="List of regions of interest, each of which must be the file path to an ImageJ ROI (.roi/.zip) or napari shapes (.csv) file.",
        default = []
    )
    roi_units: RoiUnits = Field(
        default=RoiUnits.Auto,
        description="The units the `roi_list` coordinates are in. 'Auto' takes it from the file type: ImageJ ROIs are pixels, and a napari shapes CSV saved from the plugin's crop layer is microns, because that layer is unscaled while the image layer carries the pixel size. Set it explicitly for a CSV written by anything else.",
        cli_description="Units of the ROI coordinates. 'Auto' (default) reads .roi/.zip as Pixels and .csv as Microns.",
    )
    roi_subset: List[int] = Field(
        description="A subset of all the ROIs to process. Each list item should be an index into the ROI list indicating an ROI to include. This allows you to process only a subset of the regions from a ROI file specified using the `roi_list` parameter. If `None`, it is assumed that you want to process all ROIs.",
        default=None,
        validate_default=True
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

    @model_validator(mode="before")
    @classmethod
    def resolve_roi_units(cls, values: dict) -> dict:
        """
        Settle `Auto` into a real unit while the source paths are still visible -
        the `roi_list` validator replaces them with coordinates.
        """
        from lls_core.cropping import units_for_path
        from lls_core.types import is_pathlike

        if values.get("roi_units", RoiUnits.Auto) != RoiUnits.Auto:
            return values

        given = values.get("roi_list") or []
        if is_pathlike(given):
            given = [given]
        implied = {units_for_path(item) for item in given if is_pathlike(item)}
        if len(implied) > 1:
            raise ValueError(
                "roi_list mixes file types that imply different units "
                f"({sorted(implied)}); set roi_units explicitly"
            )
        # No files (coordinates passed directly) means pixels, the API's unit.
        values["roi_units"] = implied.pop() if implied else RoiUnits.Pixels
        return values

    @field_validator("roi_list", mode="before")
    @classmethod
    def read_roi(cls, v: Any) -> List[Roi]:
        from lls_core.types import is_pathlike
        from lls_core.cropping import read_rois
        from numpy import ndarray
        # Allow a single path
        if is_pathlike(v):
            v = [v]

        rois: List[Roi] = []
        for item in v:
            if is_pathlike(item):
                rois += read_rois(item)
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

    @field_validator("roi_subset", mode="before")
    @classmethod
    def parse_roi_subset(cls, v: Any):
        # Accept comma-separated string ("2,5,7"), or a list with comma-separated
        # strings (CLI). Convert everything to int type indices
        # Bad input raises Value Error
        if v is None:
            return v
        if isinstance(v, str):
            v = [v]
        result: List[int] = []
        for item in v:
            if isinstance(item, str):
                for piece in item.split(","):
                    piece = piece.strip()
                    if piece:
                        result.append(int(piece))
            else:
                result.append(int(item))
        return result
    
    @field_validator("roi_subset", mode="before")
    @classmethod
    def default_roi_range(cls, v: Any, info: ValidationInfo):
        # If the roi range isn't provided, assume all rois should be processed
        if v is None and "roi_list" in info.data:
            return list(range(len(info.data["roi_list"])))
        return v
