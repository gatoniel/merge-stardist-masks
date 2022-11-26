"""Test suite for the numba typeguard error."""
import numpy as np

from .numba_typeguard import numba_sum_local
from merge_stardist_masks.numba_typeguard import numba_sum


def test_numba_sum_njit() -> None:
    """Test wrapped function."""
    x = np.arange(4)
    s = numba_sum(x)
    assert s == np.sum(x)


def test_numba_sum() -> None:
    """Test py_func of wrapped function."""
    x = np.arange(4)
    s = numba_sum.py_func(x)
    assert s == np.sum(x)


def test_numba_sum_local_njit() -> None:
    """Test of locally imported does not throw error."""
    x = np.arange(4)
    s = numba_sum_local(x)
    assert s == np.sum(x)


def test_numba_sum_local() -> None:
    """Python function of locally imported function."""
    x = np.arange(4)
    s = numba_sum_local.py_func(x)
    assert s == np.sum(x)
