from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple, Tuple, List

if TYPE_CHECKING:
    from lls_core.types import PathLike
    from typing_extensions import Self
    from numpy.typing import NDArray

RoiCoord = Tuple[float, float]

class Roi(NamedTuple):
    top_left: RoiCoord
    top_right: RoiCoord
    bottom_left: RoiCoord
    bottom_right: RoiCoord

    @classmethod
    def from_array(cls, array: NDArray) -> Self:
        import numpy as np
        return Roi(*np.reshape(array, (-1, 2)).tolist())

def read_roi_array(roi: PathLike) -> NDArray:
    from read_roi import read_roi_file
    from numpy import array
    return array(read_roi_file(str(roi)))

def read_imagej_roi(roi_path: PathLike) -> List[Roi]:
    """Read an ImageJ ROI zip file so it loaded into napari shapes layer
        If non rectangular ROI, will convert into a rectangle based on extreme points
    Args:
        roi_zip_path (zip file): ImageJ ROI zip file

    Returns:
        list: List of ROIs
    """
    from pathlib import Path
    from os import fspath
    from read_roi import read_roi_file, read_roi_zip

    roi_path = Path(fspath(roi_path))

    # handle reading single roi or collection of rois in zip file
    if roi_path.suffix == ".zip":
        ij_roi = read_roi_zip(roi_path)
    elif roi_path.suffix == ".roi":
        ij_roi = read_roi_file(str(roi_path))
    else:
        raise Exception("ImageJ ROI file needs to be a zip/roi file")

    if ij_roi is None:
        raise Exception("Failed reading ROI file")

    # initialise list of rois
    roi_list = []

    # Read through each roi and create a list so that it matches the organisation of the shapes from napari shapes layer
    for value in ij_roi.values():
        if value['type'] in ('oval', 'rectangle'):
            width = int(value['width'])
            height = int(value['height'])
            left = int(value['left'])
            top = int(value['top'])
            roi = Roi((top, left), (top, left+width), (top+height, left+width), (top+height, left))
            roi_list.append(roi)
        elif value['type'] in ('polygon', 'freehand'):
            left = min(int(it) for it in value['x'])
            top = min(int(it) for it in value['y'])
            right = max(int(it) for it in value['x'])
            bottom = max(int(it) for it in value['y'])
            roi = Roi((top, left), (top, right), (bottom, right), (bottom, left))
            roi_list.append(roi)
        else:
            print(f"Cannot read ROI {value}. Recognised as type {value['type']}")

    return roi_list
