from typing import Union
from typing_extensions import TypeGuard, Any, TypeAlias
from dask.array.core import Array as DaskArray
# from numpy.typing import NDArray
from pyopencl.array import Array as OCLArray
import numpy as np
from numpy.typing import NDArray
from xarray import DataArray
from bioio import BioImage
from os import fspath, PathLike as OriginalPathLike

# This is a superset of os.PathLike
PathLike: TypeAlias = Union[str, bytes, OriginalPathLike]
def is_pathlike(x: Any) -> TypeGuard[PathLike]:
    return isinstance(x, (str, bytes, OriginalPathLike))

ArrayLike: TypeAlias = Union[DaskArray, NDArray, OCLArray, DataArray]

def is_arraylike(arr: Any) -> TypeGuard[ArrayLike]:
    return isinstance(arr, (DaskArray, np.ndarray, OCLArray, DataArray))

ImageLike: TypeAlias = Union[PathLike, BioImage, ArrayLike]
def image_like_to_image(img: ImageLike) -> DataArray:
    """
    Converts an image in one of many formats to a DataArray
    """
    # First try treating it as a path
    path = None
    try:
        path = fspath(img)
        img = BioImage(path)
    except TypeError:
        pass
    if isinstance(img, BioImage):
        # CZIs read far faster through czi_reader; identical pixels either way.
        from lls_core.czi_reader import czi_path_of, czi_xarray
        czi_path = path if path is not None else czi_path_of(img)
        if czi_path is not None:
            fast = czi_xarray(czi_path, img)
            if fast is not None:
                return fast
        return img.xarray_dask_data
    else:
        for required_key in ("shape", "dtype", "ndim", "__array__", "__array_ufunc__"):
            if not hasattr(img, required_key):
                raise ValueError(f"The provided object {img} is not array like!")
        return DataArray(img)
