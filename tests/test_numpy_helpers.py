"""Test the numpy convenience functions."""
import numpy as np

import merge_stardist_masks.numpy_helpers as nh


def test_cycling_through_array() -> None:
    """Test the @njit decorated function to cycle through next elements of arrays."""
    for func in [nh.cycle_through_array, nh.cycle_through_array.py_func]:
        x = np.arange(9).reshape((3, 3))

        sort = nh.start_cycling_through_array(x, 3)
        np.testing.assert_array_equal(  # type: ignore [no-untyped-call]
            sort, np.array([8, 7, 6, 5, 4])
        )

        ind, sort, status = func(x, sort, 0)

        assert ind == 8
        assert status
        np.testing.assert_array_equal(  # type: ignore [no-untyped-call]
            sort, np.array([7, 6, 5, 4])
        )

        x[2, :] = -1

        ind, sort, status = func(x, sort, 0)

        assert ind == 5
        assert status
        np.testing.assert_array_equal(  # type: ignore [no-untyped-call]
            sort, np.array([4])
        )

        x[...] = -1
        ind, sort, status = func(x, sort, 0)
        assert not status
