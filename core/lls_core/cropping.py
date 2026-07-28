from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple, Tuple, List

from strenum import StrEnum

if TYPE_CHECKING:
    from lls_core.types import PathLike
    from typing_extensions import Self
    from numpy.typing import NDArray

RoiCoord = Tuple[float, float]


class RoiUnits(StrEnum):
    """
    The units an ROI file's coordinates are in.

    ROI files carry no unit, and the two are off by a factor of 1/dy (~6.9x at a
    0.145 um pixel), so it has to be declared. `Auto` takes it from the file type,
    which is right for the two formats we write and read; override it for a CSV
    from elsewhere. `CropParams.roi_list` is always pixels - microns are converted
    on the way in.
    """
    Auto = "Auto"
    Pixels = "Pixels"
    Microns = "Microns"

    @classmethod
    def _missing_(cls, value: object) -> "RoiUnits | None":
        # Accept the spellings people actually type: any case, singular or plural.
        if isinstance(value, str):
            return _ROI_UNIT_ALIASES.get(value.strip().lower())
        return None


_ROI_UNIT_ALIASES = {
    "auto": RoiUnits.Auto,
    "pixel": RoiUnits.Pixels,
    "pixels": RoiUnits.Pixels,
    "micron": RoiUnits.Microns,
    "microns": RoiUnits.Microns,
}


def units_for_path(roi_path: PathLike) -> RoiUnits:
    """
    The units an ROI file is expected to be in, from its type.

    ImageJ writes pixels. A napari shapes CSV holds whatever that layer's data
    coordinates were; saved from the plugin's crop layer - which is unscaled while
    the image layer carries the pixel size - those are canvas microns.
    """
    from pathlib import Path
    from os import fspath

    if Path(fspath(roi_path)).suffix.lower() == ".csv":
        return RoiUnits.Microns
    return RoiUnits.Pixels

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

def read_napari_csv(roi_path: PathLike) -> List[Roi]:
    """
    Read a shapes layer saved by napari (File > Save Selected Layer, .csv).

    One row per vertex, grouped by the `index` column:

        index,shape-type,vertex-index,axis-0,axis-1
        0,polygon,0,100.0,100.0

    Non-rectangular shapes become their bounding rectangle, as for ImageJ ROIs. Only
    the last two axes are used, matching the plugin's own shape-to-ROI conversion, so
    3D shapes are accepted and their leading axes ignored.
    """
    import csv
    from collections import OrderedDict
    from os import fspath

    shapes: "OrderedDict[str, List[RoiCoord]]" = OrderedDict()
    with open(fspath(roi_path), newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        axes = [name for name in columns if name.startswith("axis-")]
        if "index" not in columns or len(axes) < 2:
            raise Exception(
                f"{roi_path} is not a napari shapes CSV: expected an 'index' column and "
                f"at least two 'axis-N' columns, found {columns}"
            )
        for row in reader:
            shapes.setdefault(row["index"], []).append(
                (float(row[axes[-2]]), float(row[axes[-1]]))
            )

    roi_list = []
    for vertices in shapes.values():
        top = min(y for y, _ in vertices)
        bottom = max(y for y, _ in vertices)
        left = min(x for _, x in vertices)
        right = max(x for _, x in vertices)
        roi_list.append(Roi((top, left), (top, right), (bottom, right), (bottom, left)))

    if not roi_list:
        raise Exception(f"No shapes found in {roi_path}")
    return roi_list


def read_rois(roi_path: PathLike) -> List[Roi]:
    """
    Read ROIs from an ImageJ .roi/.zip or a napari shapes .csv.

    Coordinates are returned as they are stored; see `RoiUnits` for why the caller
    must know whether they are pixels or microns.
    """
    from pathlib import Path
    from os import fspath

    if Path(fspath(roi_path)).suffix.lower() == ".csv":
        return read_napari_csv(roi_path)
    return read_imagej_roi(roi_path)


def scale_rois(rois: List[Roi], factor: float) -> List[Roi]:
    """Multiply every ROI coordinate by `factor`, e.g. to convert microns to pixels."""
    return [
        Roi(*[(y * factor, x * factor) for y, x in roi])
        for roi in rois
    ]


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
