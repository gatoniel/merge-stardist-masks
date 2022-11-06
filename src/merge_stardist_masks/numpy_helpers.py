"""Convenience functions to search and fill numpy arrays."""
from __future__ import annotations

from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt
from numba import njit

T = TypeVar("T", bound=np.generic)


def start_cycling_through_array(
    array: npt.NDArray[T], threshold: Union[float, int]
) -> npt.NDArray[np.int64]:
    """Return sorted indices of array values of given threshold."""
    sum_thresh = np.sum(array > threshold)
    sort = np.flip(np.argsort(array, axis=None))[:sum_thresh]
    return sort


@njit
def cycle_through_array(
    array: npt.NDArray[T],
    sort: npt.NDArray[np.int64],
    new_threshold: Union[float, int] = 0,
) -> Tuple[int, npt.NDArray[np.int64], bool]:
    """Find next item in array above given threshold based on preselected indices."""
    sub_array = np.take_along_axis(
        array,
        sort,
        axis=None,
    )
    for i in range(len(sort)):
        if sub_array[i] > new_threshold:
            return sort[i], sort[i + 1 :], True
    return sort[i], sort[i + 1 :], False
