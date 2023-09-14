from typing import Union
from typing_extensions import TypeGuard, Any, TypeAlias
from dask.array.core import Array as DaskArray
# from numpy.typing import NDArray
from pyopencl.array import Array as OCLArray
import numpy as np
from pydantic_numpy import NDArray, NDArrayFp32, NumpyModel
from xarray import DataArray
from aicsimageio import AICSImage
from os import PathLike, fspath

ArrayLike: TypeAlias = Union[DaskArray, NDArray, OCLArray, DataArray]

def is_arraylike(arr: Any) -> TypeGuard[ArrayLike]:
    return isinstance(arr, (DaskArray, np.ndarray, OCLArray, DataArray))

ImageLike: TypeAlias = Union[PathLike, AICSImage, ArrayLike]
def image_like_to_image(img: ImageLike) -> DataArray:
    """
    Converts an image in one of many formats to a DataArray
    """
    if isinstance(img, PathLike):
        img = AICSImage(fspath(img))
    if isinstance(img, AICSImage):
        return img.xarray_dask_data
    else:
        return DataArray(img)
