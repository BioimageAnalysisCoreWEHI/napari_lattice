from typing import Union
from typing_extensions import TypeGuard, Any, TypeAlias
from dask.array.core import Array as DaskArray
from numpy.typing import NDArray
from pyopencl.array import Array as OCLArray
import numpy as np

ArrayLike: TypeAlias = Union[DaskArray, NDArray, OCLArray]

def is_arraylike(arr: Any) -> TypeGuard[ArrayLike]:
    return isinstance(arr, (DaskArray, np.ndarray, OCLArray))
