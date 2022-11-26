"""Test case to bring up numba typeguard error."""
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from numba import njit

T = TypeVar("T", bound=np.generic)


@njit
def numba_sum_local(
    array: npt.NDArray[T],
) -> T:
    """Simple function to be wrapped by @njit."""
    return np.sum(array)
