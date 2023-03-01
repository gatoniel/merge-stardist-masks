from typing import TypeVar

import numpy as np
import numpy.typing as npt
from numba import njit
from typeguard import typechecked

T = TypeVar("T", bound=np.generic)


@typechecked
@njit
def numba_sum(
    array: npt.NDArray[T],
) -> T:
    return np.sum(array)


x = np.arange(4)
s = numba_sum.py_func(x)
s = numba_sum(x)
