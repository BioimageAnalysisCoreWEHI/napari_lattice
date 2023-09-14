from typing import Union
from typing_extensions import TypeGuard, Any, TypeAlias
from dask.array.core import Array as DaskArray
# from numpy.typing import NDArray
from pyopencl.array import Array as OCLArray
import numpy as np
import pydantic_numpy.dtype as pnd
from pydantic_numpy import NDArray, NDArrayFp32, NumpyModel
from xarray import DataArray

ArrayLike: TypeAlias = Union[DaskArray, NDArray, OCLArray, DataArray]

def is_arraylike(arr: Any) -> TypeGuard[ArrayLike]:
    return isinstance(arr, (DaskArray, np.ndarray, OCLArray, DataArray))
