"""Test the functions to determine the touching pixels of neighboring objects."""

from unittest import mock

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation  # type: ignore [import-untyped]
from scipy.ndimage import generate_binary_structure

import merge_stardist_masks.touching_pixels as tp


def slow_correct_touching_pixels(
    lbl: npt.NDArray[np.int_],
) -> npt.NDArray[np.bool_]:
    """Slow but correct way to determine the touching pixels in 2d/3d label images."""
    struct = generate_binary_structure(lbl.ndim, lbl.ndim)
    mask: npt.NDArray[np.bool_] = lbl > 0
    lbl_ids = np.unique(lbl[mask])
    expanded = np.zeros(lbl.shape, dtype=bool)
    for lbl_id in lbl_ids:
        mask_ = lbl == lbl_id
        dilation = binary_dilation(mask_, struct)
        expanded_ = np.logical_xor(mask_, dilation)
        expanded = np.logical_or(expanded, expanded_)
    touching_pixels: npt.NDArray[np.bool_] = np.logical_and(mask, expanded)
    return touching_pixels


def test_touching_pixels_2d() -> None:
    """Test whether output of touching_pixels_2d is correct."""
    lbl = np.zeros((4, 4), dtype=int)
    lbl[:2, :2] = 1
    lbl[:2, 2:] = 2
    lbl[2:, 2:] = 3
    lbl[0, 3] = 0

    correct_touching_pixels = slow_correct_touching_pixels(lbl)
    fast_touching_pixels = tp.touching_pixels_2d(lbl)

    np.testing.assert_equal(correct_touching_pixels, fast_touching_pixels)

    with mock.patch("merge_stardist_masks.touching_pixels.numba.types.bool_", np.bool_):
        py_func_fast_touching_pixels = tp.touching_pixels_2d.py_func(lbl)
    np.testing.assert_equal(correct_touching_pixels, py_func_fast_touching_pixels)

    mask: npt.NDArray[np.bool_] = lbl > 0
    bordering = np.zeros_like(mask)
    tp.touching_pixels_2d_helper.py_func(lbl, mask, bordering)
    np.testing.assert_equal(correct_touching_pixels, bordering)


def test_touching_pixels_3d() -> None:
    """Test whether output of touching_pixels_3d is correct."""
    lbl = np.zeros((4, 4, 4), dtype=int)
    lbl[:2, :2, :2] = 1
    lbl[:2, :2, 2:] = 2
    lbl[:2, 2:, 2:] = 3
    lbl[2:, 1:3, 1:3] = 4

    correct_touching_pixels = slow_correct_touching_pixels(lbl)
    fast_touching_pixels = tp.touching_pixels_3d(lbl)

    np.testing.assert_equal(correct_touching_pixels, fast_touching_pixels)

    with mock.patch("merge_stardist_masks.touching_pixels.numba.types.bool_", np.bool_):
        py_func_fast_touching_pixels = tp.touching_pixels_3d.py_func(lbl)
    np.testing.assert_equal(correct_touching_pixels, py_func_fast_touching_pixels)


def test_determine_neighbor_2d() -> None:
    """Test determine_neighbor_2d functions."""
    lbl = np.zeros((3, 3), dtype=int)
    lbl[1, 1] = 1
    lbl[2, 2] = 2
    mask: npt.NDArray[np.bool_] = lbl > 0
    bordering = np.zeros_like(mask)
    tp.determine_neighbor_2d.py_func(1, 1, 1, 1, lbl, mask, bordering)
    tp.determine_neighbor_2d.py_func(1, 0, 1, 1, lbl, mask, bordering)
    np.testing.assert_equal(bordering, mask)


def test_determine_neighbor_3d() -> None:
    """Test determine_neighbor_3d functions."""
    lbl = np.zeros((3, 3, 3), dtype=int)
    lbl[1, 1, 1] = 1
    lbl[2, 2, 2] = 2
    mask: npt.NDArray[np.bool_] = lbl > 0
    bordering = np.zeros_like(mask)
    tp.determine_neighbor_3d.py_func(1, 1, 1, 1, 1, 1, lbl, mask, bordering)
    tp.determine_neighbor_3d.py_func(1, 0, 1, 1, 1, 1, lbl, mask, bordering)
    np.testing.assert_equal(bordering, mask)


def test_determine_neighbors_2d() -> None:
    """Test determine_neighbors_2d functions."""
    lbl = np.zeros((3, 3), dtype=int)
    lbl[1, 1] = 1
    lbl[2, 2] = 2
    mask: npt.NDArray[np.bool_] = lbl > 0
    bordering = np.zeros_like(mask)
    offsets = np.array([[1, 1], [0, 1]])
    tp.determine_neighbors_2d.py_func(1, 1, offsets, lbl, mask, bordering)
    tp.determine_neighbors_2d.py_func(0, 1, offsets, lbl, mask, bordering)
    np.testing.assert_equal(bordering, mask)


def test_determine_neighbors_3d() -> None:
    """Test determine_neighbors_3d functions."""
    lbl = np.zeros((3, 3, 3), dtype=int)
    lbl[1, 1, 1] = 1
    lbl[2, 2, 2] = 2
    mask: npt.NDArray[np.bool_] = lbl > 0
    bordering = np.zeros_like(mask)
    offsets = np.array([[1, 1, 1], [0, 1, 1]])
    tp.determine_neighbors_3d.py_func(1, 1, 1, offsets, lbl, mask, bordering)
    tp.determine_neighbors_3d.py_func(0, 1, 1, offsets, lbl, mask, bordering)
    np.testing.assert_equal(bordering, mask)


def test_bordering_gaussian_weights() -> None:
    """Check for output > 0 and == 0 outside of objects."""
    lbl = np.zeros((4, 4, 4), dtype=int)
    lbl[:2, :2, :2] = 1
    lbl[:2, :2, 2:] = 2
    lbl[:2, 2:, 2:] = 3
    lbl[2:, 1:3, 1:3] = 4

    fast_touching_pixels = tp.touching_pixels_3d(lbl)
    bordering_weight = tp.bordering_gaussian_weights(fast_touching_pixels, lbl)

    assert np.all(bordering_weight >= 0)
    assert np.all(bordering_weight[lbl == 0] == 0)
