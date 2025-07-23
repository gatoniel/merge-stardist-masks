"""Separate the worker and initializer to modify global state with loky.

See https://github.com/joblib/loky/issues/206
"""

from __future__ import annotations

import atexit
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from stardist.rays3d import Rays_Base  # type: ignore [import-untyped]

from .naive_fusion import my_polyhedron_to_label
from .naive_fusion import paint_in_without_overlaps
from .naive_fusion import SlicePointReturn


T = TypeVar("T", bound=np.generic)


new_probs: npt.NDArray[np.double]
points: npt.NDArray[np.int_]
lbl: npt.NDArray[np.intc]
dists: npt.NDArray[np.double]


def _load_shared_memory(
    shape: Tuple[int, ...], dtype: npt.DTypeLike, name: str
) -> npt.NDArray[T]:
    shm = SharedMemory(name=name)
    atexit.register(shm.close)
    a: npt.NDArray[T] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return a


def _initializer(
    new_probs_name: str,
    new_probs_dtype: npt.DTypeLike,
    points_name: str,
    lbl_name: str,
    dists_name: str,
    dists_dtype: npt.DTypeLike,
    dists_shape: Tuple[int, ...],
    big_shape: Tuple[int, ...],
) -> None:
    global new_probs
    global points
    global lbl
    global dists
    new_probs = _load_shared_memory(big_shape, new_probs_dtype, new_probs_name)

    points = _load_shared_memory(big_shape + (3,), np.int_, points_name)

    lbl = _load_shared_memory(big_shape, np.intc, lbl_name)
    dists = _load_shared_memory(dists_shape, dists_dtype, dists_name)


def _slice_point(point: npt.ArrayLike, max_dists: Tuple[int, ...]) -> SlicePointReturn:
    """Calculate the extents of a slice for a given point and its coordinates within."""
    slices_list = []
    centered_point = []
    for a, max_dist in zip(np.array(point), max_dists):
        diff = a - max_dist
        if diff < 0:
            centered_point.append(max_dist + diff)
            diff = 0
        else:
            centered_point.append(max_dist)
        slices_list.append(slice(diff, a + max_dist + 1))
    return tuple(slices_list), np.array(centered_point)


def _worker(
    max_ind: Tuple[int, ...],
    current_id: int,
    grid_array: npt.NDArray[np.int_],
    max_full_overlaps: int,
    prob_thresh: float,
    max_dists: Tuple[int, ...],
    rays: Rays_Base,
) -> None:
    global new_probs  # noqa: F824
    global points  # noqa: F824
    global lbl  # noqa: F824
    global dists  # noqa: F824

    # this_prob = float(new_probs[max_ind])
    new_probs[max_ind] = -1

    ind = max_ind + (slice(None),)

    slices, point = _slice_point(points[ind], max_dists)
    shape_paint = lbl[slices].shape

    dists_ind = tuple(
        list(points[ind] // grid_array)
        + [
            slice(None),
        ]
    )
    new_shape = my_polyhedron_to_label(rays, dists[dists_ind], point, shape_paint) == 1

    current_probs = new_probs[slices]
    tmp_slices = tuple(
        list(slices)
        + [
            slice(None),
        ]
    )
    current_points = points[tmp_slices]

    full_overlaps = 0
    while True:
        if full_overlaps > max_full_overlaps:
            break
        probs_within = current_probs[new_shape]

        if np.sum(probs_within > prob_thresh) == 0:
            break

        max_ind_within = np.argmax(probs_within)
        # this_prob = float(probs_within[max_ind_within])
        probs_within[max_ind_within] = -1

        current_probs[new_shape] = probs_within

        current_point = current_points[new_shape, :][max_ind_within, :]
        dists_ind = tuple(
            list(current_point // grid_array)
            + [
                slice(None),
            ]
        )
        additional_shape: npt.NDArray[np.bool_] = (
            my_polyhedron_to_label(
                rays,
                dists[dists_ind],
                point + current_point - points[ind],
                shape_paint,
            )
            > 0
        )

        size_of_current_shape = np.sum(new_shape)

        new_shape = np.logical_or(
            new_shape,
            additional_shape,
        )
        if size_of_current_shape == np.sum(new_shape):
            full_overlaps += 1
        else:
            full_overlaps = 0

    current_probs[new_shape] = -1
    new_probs[slices] = current_probs

    lbl[slices] = paint_in_without_overlaps(lbl[slices], new_shape, current_id)
