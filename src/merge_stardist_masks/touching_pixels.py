"""Functions to calculate optimized prob maps and distance weights."""
from __future__ import annotations

import numba  # type: ignore [import]
import numpy as np
import numpy.typing as npt
from edt import edt  # type: ignore [import]
from numba import njit


@njit  # type: ignore [misc]
def determine_neighbor_2d(
    y: int,
    off_y: int,
    x: int,
    off_x: int,
    lbl: npt.NDArray[np.int_],
    mask: npt.NDArray[np.bool_],
    bordering: npt.NDArray[np.bool_],
) -> None:
    """Utility function that is called several times in the below function."""
    y_ = y + off_y
    x_ = x + off_x
    if mask[y_, x_] and lbl[y, x] != lbl[y_, x_]:
        bordering[y, x] = True
        bordering[y_, x_] = True


@njit  # type: ignore [misc]
def determine_neighbors_2d(
    y: int,
    x: int,
    offsets: npt.NDArray[np.int_],
    lbl: npt.NDArray[np.int_],
    mask: npt.NDArray[np.bool_],
    bordering: npt.NDArray[np.bool_],
) -> None:
    """Utility function that checks for different neighbors."""
    if mask[y, x]:
        for i in range(len(offsets)):
            off_y, off_x = offsets[i, :]
            determine_neighbor_2d(y, off_y, x, off_x, lbl, mask, bordering)


@njit  # type: ignore [misc]
def touching_pixels_2d_helper(
    lbl: npt.NDArray[np.int_],
    mask: npt.NDArray[np.bool_],
    bordering: npt.NDArray[np.bool_],
) -> None:
    """Helper to calculate the pixels of objects that touch other objects in 2D."""
    all_offsets = np.array([(1, -1), (0, 1), (1, 1), (1, 0)])
    x0_offsets = np.array([(0, 1), (1, 1), (1, 0)])

    for y in range(lbl.shape[0] - 1):
        for x in range(1, lbl.shape[1] - 1):
            determine_neighbors_2d(y, x, all_offsets, lbl, mask, bordering)
        x = 0
        determine_neighbors_2d(y, x, x0_offsets, lbl, mask, bordering)

        x = lbl.shape[1] - 1
        if mask[y, x]:
            off_y = 1
            off_x = 0
            determine_neighbor_2d(y, off_y, x, off_x, lbl, mask, bordering)

    y = lbl.shape[0] - 1
    off_y = 0
    off_x = 1
    for x in range(0, lbl.shape[1] - 1):
        if mask[y, x]:
            determine_neighbor_2d(y, off_y, x, off_x, lbl, mask, bordering)


@njit  # type: ignore [misc]
def touching_pixels_2d(lbl: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
    """Calculate the pixels of objects that touch other objects in 2D."""
    bordering = np.zeros(lbl.shape, dtype=numba.types.bool_)
    touching_pixels_2d_helper(lbl, lbl > 0, bordering)
    return bordering


@njit  # type: ignore [misc]
def touching_pixels_3d(lbl: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
    """Calculate the pixels of objects that touch other objects in 3D."""
    all_offsets = np.array(
        [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, -1),
            (1, -1, 0),
            (1, -1, -1),
            (1, 1, -1),
            (1, -1, 1),
            (0, 1, -1),
        ]
    )
    x0_offsets = np.array(
        [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (1, -1, 0),
            (1, -1, 1),
        ]
    )
    x1_offsets = np.array(
        [
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, -1),
            (1, -1, 0),
            (1, -1, -1),
            (1, 1, -1),
            (0, 1, -1),
        ]
    )
    y0_offsets = np.array(
        [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, -1),
            (1, 1, -1),
            (0, 1, -1),
        ]
    )
    y0x0_offsets = np.array(
        [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
    )
    y0x1_offsets = np.array(
        [
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 0, -1),
            (1, 1, -1),
            (0, 1, -1),
        ]
    )
    y1_offsets = np.array(
        [
            (1, 0, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 0, -1),
            (1, -1, 0),
            (1, -1, -1),
            (1, -1, 1),
        ]
    )
    y1x0_offsets = np.array(
        [
            (1, 0, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, -1, 0),
            (1, -1, 1),
        ]
    )
    y1x1_offsets = np.array(
        [
            (1, 0, 0),
            (1, 0, -1),
            (1, -1, 0),
            (1, -1, -1),
        ]
    )
    bordering = np.zeros(lbl.shape, dtype=numba.types.bool_)
    # bordering = np.zeros(lbl.shape, dtype=bool)
    mask: npt.NDArray[np.bool_] = lbl > 0

    z_max = lbl.shape[0] - 1
    y_max = lbl.shape[1] - 1
    x_max = lbl.shape[2] - 1

    for z in range(z_max):
        # Y=0
        y = 0
        # Y0X0_OFFSETS
        x = 0
        determine_neighbors_3d(z, y, x, y0x0_offsets, lbl, mask, bordering)

        # Y0_OFFSETS
        for x in range(1, x_max):
            determine_neighbors_3d(z, y, x, y0_offsets, lbl, mask, bordering)

        # Y0X1_OFFSETS
        x = x_max
        determine_neighbors_3d(z, y, x, y0x1_offsets, lbl, mask, bordering)

        for y in range(1, y_max):
            # X0_OFFSETS
            x = 0
            determine_neighbors_3d(z, y, x, x0_offsets, lbl, mask, bordering)

            # ALL_OFFSETS
            for x in range(1, x_max):
                determine_neighbors_3d(z, y, x, all_offsets, lbl, mask, bordering)

            # X1_OFFSTES
            x = lbl.shape[2] - 1
            determine_neighbors_3d(z, y, x, x1_offsets, lbl, mask, bordering)

        # Y=Y_MAX
        y = y_max
        # Y1X0_OFFSETS
        x = 0
        determine_neighbors_3d(z, y, x, y1x0_offsets, lbl, mask, bordering)

        # Y1_OFFSETS
        for x in range(1, x_max):
            determine_neighbors_3d(z, y, x, y1_offsets, lbl, mask, bordering)

        # Y1X1_OFFSETS
        x = x_max
        determine_neighbors_3d(z, y, x, y1x1_offsets, lbl, mask, bordering)

    touching_pixels_2d_helper(lbl[z_max, ...], mask[z_max, ...], bordering[z_max, ...])
    return bordering


@njit  # type: ignore [misc]
def determine_neighbor_3d(
    z: int,
    off_z: int,
    y: int,
    off_y: int,
    x: int,
    off_x: int,
    lbl: npt.NDArray[np.int_],
    mask: npt.NDArray[np.bool_],
    bordering: npt.NDArray[np.bool_],
) -> None:
    """Utility function that is called several times in the below function."""
    z_ = z + off_z
    y_ = y + off_y
    x_ = x + off_x
    if mask[z_, y_, x_] and lbl[z, y, x] != lbl[z_, y_, x_]:
        bordering[z, y, x] = True
        bordering[z_, y_, x_] = True


@njit  # type: ignore [misc]
def determine_neighbors_3d(
    z: int,
    y: int,
    x: int,
    offsets: npt.NDArray[np.int_],
    lbl: npt.NDArray[np.int_],
    mask: npt.NDArray[np.bool_],
    bordering: npt.NDArray[np.bool_],
) -> None:
    """Helper to iterate over different offsets."""
    if mask[z, y, x]:
        for i in range(len(offsets)):
            off_z, off_y, off_x = offsets[i, :]
            determine_neighbor_3d(z, off_z, y, off_y, x, off_x, lbl, mask, bordering)


def bordering_gaussian_weights(
    border_pixels: npt.NDArray[np.bool_], lbl: npt.NDArray[np.int_], sigma: int = 2
) -> npt.NDArray[np.single]:
    """Gaussian of edt from border_pixels only for pixels with lbl > 0."""
    bordering_edt = edt(np.logical_not(border_pixels))
    bordering_weight = np.zeros_like(lbl, dtype=float)
    _mask: npt.NDArray[np.bool_] = lbl > 0
    bordering_weight[_mask] = np.exp(-np.square(bordering_edt[_mask]) / 2 / sigma**2)

    return bordering_weight
