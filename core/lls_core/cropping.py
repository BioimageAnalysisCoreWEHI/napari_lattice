from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lls_core.types import PathLike
    from numpy.typing import NDArray

def read_roi_array(roi: PathLike) -> NDArray:
    from read_roi import read_roi_file
    from numpy import array
    return array(read_roi_file(roi))

def read_imagej_roi(roi_path: PathLike):
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
        ij_roi = read_roi_file(roi_path)
    else:
        raise Exception("ImageJ ROI file needs to be a zip/roi file")

    if ij_roi is None:
        raise Exception("Failed reading ROI file")

    # initialise list of rois
    roi_list = []

    # Read through each roi and create a list so that it matches the organisation of the shapes from napari shapes layer
    for value in ij_roi.values():
        if value['type'] in ('oval', 'rectangle'):
            width = value['width']
            height = value['height']
            left = value['left']
            top = value['top']
            roi = [[top, left], [top, left+width], [top+height, left+width], [top+height, left]]
            roi_list.append(roi)
        elif value['type'] in ('polygon', 'freehand'):
            left = min(value['x'])
            top = min(value['y'])
            right = max(value['x'])
            bottom = max(value['y'])
            roi = [[top, left], [top, right], [bottom, right], [bottom, left]]
            roi_list.append(roi)
        else:
            print(f"Cannot read ROI {value}. Recognised as type {value['type']}")

    return roi_list
