from typing_extensions import Any, Dict, Iterable, List, Optional, Tuple, Union
from pydantic.v1 import Field, NonNegativeInt, root_validator, validator
from lls_core.models.utils import FieldAccessModel
from lls_core.cropping import Roi, RoiUnits
from pathlib import Path

class CropParams(FieldAccessModel):
    """
    Parameters for the optional cropping step.
    Note that cropping is performed in the space of the deskewed shape.
    This is to support the workflow of performing a preview deskew and using that
    to calculate the cropping coordinates.

    `roi_list` is always in deskewed-image pixels. A file given in microns is
    converted on the way in, once the pixel size is known - see `roi_units`.
    """
    trackmate_file: Path = Field(
        description="File path to the Trackmate file for tracked ROIs. Accepts .xml or .csv files.",
        cli_description= "File path to the Trackmate file for tracked ROIs. Accepts .xml or .csv files.",
        default = None
    )
    trackmate_window_size: dict[str,float] = Field(
        description="cropping window size for TrackMate ROIs, defines the `N × N` square that defines the crop size in μm",
        cli_description="cropping window size for TrackMate ROIs in μm",
        default = None
    )
    trackmate_window_size_default: float = Field(
        description="default cropping window size for TrackMate ROIs, defines the `N × N` square that defines the crop size in μm",
        cli_description="default cropping window size for TrackMate ROIs in μm",
        default = 30.0
    )
    roi_list: List[Roi] = Field(
        description="List of regions of interest, each of which must be an `N × D` array, where N is the number of vertices and D the coordinates of each vertex. This can alternatively be provided as a `str` or `Path`, or a list of those, in which case they are interpreted as paths to ImageJ ROI (.roi/.zip) or napari shapes (.csv) files that are read from disk.",
        cli_description="Either a list of regions of interest, each a file path to an ImageJ ROI (.roi/.zip) or napari shapes (.csv) file, or a singular trackmate tracking file, either a (.xml) or (.csv) file.",
        default = []
    )
    roi_units: RoiUnits = Field(
        default=RoiUnits.Auto,
        description="The units the `roi_list` coordinates are in. 'Auto' takes it from the file type: ImageJ ROIs are pixels, and a napari shapes CSV saved from the plugin's crop layer is microns, because that layer is unscaled while the image layer carries the pixel size. Set it explicitly for a CSV written by anything else.",
        cli_description="Units of the ROI coordinates. 'Auto' (default) reads .roi/.zip as Pixels and .csv as Microns.",
    )
    roi_subset: List[int] = Field(
        description="A subset of all the ROIs/Tracks to process. Each list item should be an index into the ROI list or Tracks in a Trackmate file indicating an ROI/Track to include. This allows you to process only a subset of the regions from a ROI/Tracking file specified using the `roi_list` parameter. If `None`, it is assumed that you want to process all ROIs/Tracks.",
        default=None
    )
    z_range: Tuple[NonNegativeInt, NonNegativeInt] = Field(
        default=None,
        description="The range of Z slices to take as a tuple of the form `(first, last)`. All Z slices before the first index or after the last index will be cropped out.",
        cli_description="An array with two items, indicating the index of the first and last Z slice to include."
    )
    roi_by_time:Dict[int,Dict[int, Roi]] = Field(
        default=None,
        description="One region of interest per timepoint, for a crop that follows a tracked object rather than staying still. ",
    )

    @property
    def selected_rois(self) -> Iterable[Roi]:
        "Returns the relevant ROIs that should be processed"
        for i in self.roi_subset:
            yield self.roi_list[i]

    def window_for_track(self,track: int|str) -> float:
        """
        returns window size for a given track
        """
        if not self.trackmate_window_size:
            return self.trackmate_window_size_default
        if str(track) in self.trackmate_window_size.keys():
            return self.trackmate_window_size[str(track)]
        return self.trackmate_window_size_default
    
    def roi_for_time(self, time: int, roi_index: int) -> Roi:
        """
        The crop window at `time`. Only a track-driven crop moves, every 
        other crop uses the same crop window at every timepoint.
        """
        if self.roi_by_time is None:
            return self.roi_list[roi_index]
        try:
            return self.roi_by_time[roi_index][time]
        except KeyError:
            raise ValueError(
                f"The track has no position at timepoint {time}; it covers timepoints "
                f"{min(self.roi_by_time[roi_index])}-{max(self.roi_by_time[roi_index])}. Restrict the time "
                "range to the timepoints the track covers."
            ) from None
        

    @root_validator(pre=True)
    def set_window_sizes(cls,values:dict)->dict:
        v = values.get("trackmate_window_size")
        if v is None:
            return values
        if isinstance(v,str):
            #if input is a string assume a single default value provided
            try:
                x = float(v)
            except ValueError:
                raise ValueError("default track window size must be float")
            values["trackmate_window_size_default"] = x
            return values
        if isinstance(v,tuple):
            if len(v) == 1:
                #if input is a tuple of length 1 assume a single default value provided
                try:
                    x = float(v[0])
                except ValueError:
                    raise ValueError("default track window size must be float")
                values["trackmate_window_size_default"] = x
                return values
            elif len(v) > 2:
                #if input is greater than 2 raise value error
                raise ValueError("track window size recieves too many inputs")
            default, tracks = v
            default = 30 if not default else default
            values["trackmate_window_size_default"] = float(default)
            values["trackmate_window_size"] = tracks
            return values
        return values   

    @root_validator(pre=True)
    def set_track_path(cls,values:dict):
        from lls_core.types import is_pathlike
        from lls_core.trackmate_io import is_trackmate_file
        #if no path has been given not using trackmate
        if not values.get("trackmate_file"):
            if values.get("roi_list") and is_pathlike(values.get("roi_list")[0]) and is_trackmate_file(values.get("roi_list")[0]):
                path = Path(values.get("roi_list")[0])
                if not path.exists(): 
                    raise FileNotFoundError(f"TrackMate File not found: {path}")
                if len(values.get("roi_list")) > 1:
                    """if multiple roi files inputted when using trackmate file raise error"""
                    raise ValueError("Tracking Based ROI cropping does not support multiple input files")
                values["trackmate_file"] = path
        return values 
        
    @root_validator()
    def set_roi_by_time(cls,values: dict) -> dict:
        """
        set roi by time when using trackmate tracking
        """
        from lls_core.cropping import track_to_rois
        from lls_core.trackmate_io import load_trackmate_tracks
        path = Path(values.get("trackmate_file")) if values.get("trackmate_file") else None
        #if no path has been given not using trackmate
        if path is None or values.get("roi_by_time"):
            return values
        if not path.exists(): 
            raise FileNotFoundError(f"TrackMate File not found: {path}")
        #load tracks from file
        tracks = load_trackmate_tracks(path)
        #initialise dict
        values["roi_by_time"] = {}
        for id in values.get("roi_subset"):
            track = tracks[str(id)]["trackData"]
            window_sizes = values.get("trackmate_window_size")
            size = values.get("trackmate_window_size_default")
            if window_sizes and str(id) in window_sizes.keys():
                size = window_sizes[str(id)]
            values["roi_by_time"][id]=track_to_rois(track, size)
        return values

        

    @root_validator(pre=True)
    def resolve_roi_units(cls, values: dict) -> dict:
        """
        Settle `Auto` into a real unit while the source paths are still visible -
        the `roi_list` validator replaces them with coordinates.
        """
        from lls_core.cropping import units_for_path
        from lls_core.types import is_pathlike
        from lls_core.trackmate_io import is_trackmate_file
        
        if values.get("roi_units", RoiUnits.Auto) != RoiUnits.Auto:
            return values

        is_trackmate = values.get("roi_list") and is_pathlike(values.get("roi_list")[0]) and is_trackmate_file(values.get("roi_list")[0])
        given = values.get("roi_list") or []
        if is_pathlike(given):
            given = [given]
        implied = {units_for_path(item) for item in given if is_pathlike(item)}
        if len(implied) > 1:
            raise ValueError(
                "roi_list mixes file types that imply different units "
                f"({sorted(implied)}); set roi_units explicitly"
            )
        if implied:
            values["roi_units"] = implied.pop()
        elif values.get("roi_by_time") or values.get("trackmate_file") or is_trackmate:
            # Track ROIs come from TrackMate, which uses microns.
            values["roi_units"] = RoiUnits.Microns
        else:
            # No files (coordinates passed directly) means pixels, the API's unit.
            values["roi_units"] = RoiUnits.Pixels
        return values

    @validator("roi_list", pre=True)
    def read_roi(cls, v: Any,values) -> List[Roi]:
        from lls_core.types import is_pathlike
        from lls_core.cropping import read_rois
        from numpy import ndarray
        from lls_core.trackmate_io import is_trackmate_file,tracks_exist
        #catch trackmate file
        if values.get("trackmate_file"):
            if not tracks_exist(values.get("trackmate_file")):
                raise ValueError("At least one region of interest must be specified if cropping is enabled")
            return [Roi((0,0),(0,0),(0,0),(0,0))]
        else:
            # Allow a single path
            if is_pathlike(v):
                v = [v]
            rois: List[Roi] = []
            for item in v:
                if is_pathlike(item):
                    if is_trackmate_file(item):
                        """if trackmate file provided while using constant ROIs raise error"""
                        raise ValueError("Tracking Based ROI cropping incompatible with predefined ROIs")
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
        if not rois or len(rois) < 1:
            raise ValueError("At least one region of interest must be specified if cropping is enabled")
        
        return rois

    @validator("roi_subset", pre=True)
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
    

    @validator("roi_subset", pre=True, always=True)
    def default_roi_range(cls, v: Any, values: dict):
        # If the roi/track range isn't provided, assume all rois/tracks should be processed
        from lls_core.trackmate_io import load_trackmate_tracks
        path = Path(values.get("trackmate_file")) if values.get("trackmate_file") else None
        if v is not None:
            return v
        #if no path has been given not using trackmate
        if path is None:
            if "roi_list" in values:
                return list(range(len(values["roi_list"])))
        elif not path.exists(): 
            raise FileNotFoundError(f"TrackMate File not found: {path}")
        elif "trackmate_file" in values:
            return list(load_trackmate_tracks(path).keys())
        return v
