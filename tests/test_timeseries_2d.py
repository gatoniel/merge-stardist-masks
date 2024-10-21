"""Test the timeseries_2d module."""

import numpy as np

import merge_stardist_masks.timeseries_2d as t2d

b = (slice(None),) * 3
ss_grid = (slice(0, None, 2),) * 3


def test_subsample() -> None:
    """Test subsampling with b and ss_grid tuples."""
    x = np.empty((4, 12, 12), dtype=int)
    y = t2d.subsample(x, 0, b, ss_grid)

    assert y.shape == (6, 6)


def test_star_dist_timeseries() -> None:
    """Shape of distances is subsampled and all timepoints should have same dists."""
    x = np.zeros((3, 16, 16), dtype=np.uint8)
    x[:, 4:8, 4:8] = 1

    y = t2d.star_dist_timeseries(x, 8, "cpp", (2, 2))

    assert y.shape == (8, 8, 24)
    np.testing.assert_equal(y[..., :8], y[..., 8:16])
    np.testing.assert_equal(y[..., :8], y[..., -8:])


def test_edt_prob_timeseries() -> None:
    """Test shape and whether the same output is in all timepoints."""
    x = np.zeros((3, 16, 16), dtype=np.uint8)
    x[:, 4:8, 4:8] = 1

    y = t2d.edt_prob_timeseries(x, b, ss_grid)

    assert y.shape == (8, 8, 3)
    np.testing.assert_equal(y[..., 0], y[..., 1])
    np.testing.assert_equal(y[..., 0], y[..., 2])


def test_touching_pixels_2d_timeseries() -> None:
    """Test shape and whether the same output is in all timepoints."""
    x = np.zeros((3, 16, 16), dtype=np.uint8)
    x[:, 4:8, 4:8] = 1
    x[:, 4:8, 8:12] = 2

    y = t2d.touching_pixels_2d_timeseries(x, b, ss_grid)
    print(t2d.subsample(x, 0, b, ss_grid))
    print(x[0, ...])
    print(y[..., -1])

    assert y.shape == (8, 8, 3)
    np.testing.assert_equal(y[..., 0], y[..., 1])
    np.testing.assert_equal(y[..., 0], y[..., 2])


def test_bordering_gaussian_weights_timeseries() -> None:
    """Test shape and whether the same output is in all timepoints."""
    x = np.zeros((3, 16, 16), dtype=np.uint8)
    x[:, 4:8, 4:8] = 1
    x[:, 4:8, 8:12] = 2

    y_ = t2d.touching_pixels_2d_timeseries(x, b, ss_grid)
    print(t2d.subsample(x, 0, b, ss_grid))
    print(x[0, ...])
    print(y_[..., -1])

    y = t2d.bordering_gaussian_weights_timeseries(y_, x, 2, b, ss_grid)

    assert y.shape == (8, 8, 3)
    np.testing.assert_equal(y[..., 0], y[..., 1])
    np.testing.assert_equal(y[..., 0], y[..., 2])
