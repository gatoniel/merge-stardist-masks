"""Separate the worker and initializer to modify global state with loky.

See https://github.com/joblib/loky/issues/206
"""

from __future__ import annotations

import atexit
import multiprocessing
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Dict
from typing import List
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
current_id: multiprocessing.managers.ValueProxy[int]
current_id_lock: multiprocessing.managers.AcquirerProxy  # type: ignore [name-defined]
inds: multiprocessing.managers.DictProxy[
    Tuple[int, ...], multiprocessing.managers.ListProxy[Tuple[int, ...]]
]
max_probs: npt.NDArray[np.double]
all_neighbors: Dict[Tuple[int, ...], List[Tuple[int, ...]]]
remaining_inds: npt.NDArray[np.intc]


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
    max_probs_name: str,
    num_slices: Tuple[int, ...],
    current_id_: multiprocessing.managers.ValueProxy[int],
    current_id_lock_: multiprocessing.managers.AcquirerProxy,  # type: ignore [name-defined]
    inds_: multiprocessing.managers.DictProxy[
        Tuple[int, ...], multiprocessing.managers.ListProxy[Tuple[int, ...]]
    ],
    all_neighbors_: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    remaining_inds_name: str,
) -> None:
    global new_probs
    global points
    global lbl
    global dists
    global max_probs
    global current_id
    global current_id_lock
    global inds
    global all_neighbors
    global remaining_inds
    current_id = current_id_
    current_id_lock = current_id_lock_
    inds = inds_
    new_probs = _load_shared_memory(big_shape, new_probs_dtype, new_probs_name)

    points = _load_shared_memory(big_shape + (3,), np.int_, points_name)

    lbl = _load_shared_memory(big_shape, np.intc, lbl_name)
    dists = _load_shared_memory(dists_shape, dists_dtype, dists_name)
    max_probs = _load_shared_memory(num_slices, new_probs_dtype, max_probs_name)
    remaining_inds = _load_shared_memory(num_slices, np.intc, remaining_inds_name)
    all_neighbors = all_neighbors_


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


def _get_current_id(
    current_id: multiprocessing.managers.ValueProxy[int],
    current_id_lock: multiprocessing.managers.AcquirerProxy,  # type: ignore [name-defined]
) -> int:
    t0 = time.perf_counter()
    with current_id_lock:
        my_id = current_id.value
        current_id.value += 1
    t1 = time.perf_counter()
    total_time = t1 - t0
    print("ACQUIRING LOCK took", total_time, "seconds")
    return my_id


def _worker(
    idx: Tuple[int, ...],
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
    global max_probs  # noqa: F824
    global current_id  # noqa: F824
    global current_id_lock  # noqa: F824
    global inds  # noqa: F824
    global all_neighbors  # noqa: F824
    global remaining_inds  # noqa: F824

    this_inds = inds[idx]
    neighbors = all_neighbors[idx]

    while this_inds:
        t0 = time.perf_counter()
        max_ind = this_inds.pop()
        if new_probs[max_ind] < 0:
            continue

        for neighbor in neighbors:
            if new_probs[max_ind] < max_probs[neighbor]:
                this_inds.append(max_ind)
                return

        new_probs[max_ind] = -1.0
        my_id = _get_current_id(current_id, current_id_lock)

        ind = max_ind + (slice(None),)

        slices, point = _slice_point(points[ind], max_dists)
        shape_paint = lbl[slices].shape

        dists_ind = tuple(
            list(points[ind] // grid_array)
            + [
                slice(None),
            ]
        )
        new_shape = (
            my_polyhedron_to_label(rays, dists[dists_ind], point, shape_paint) == 1
        )

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

        current_probs[new_shape] = -1.0
        new_probs[slices] = current_probs

        lbl[slices] = paint_in_without_overlaps(lbl[slices], new_shape, my_id)
        # t1 = time.perf_counter()

        for neighbor in neighbors:
            max_probs[neighbor], remaining_inds[neighbor] = _update_neighbor(
                inds[neighbor]
            )
        t2 = time.perf_counter()
        total_time = t2 - t0
        print("WORKER took", total_time, "seconds")

    max_probs[idx] = -1.0


def _update_neighbor(
    neighbor_inds: multiprocessing.managers.ListProxy[Tuple[int, ...]],
) -> Tuple[float, float]:
    global new_probs  # noqa: F824
    while neighbor_inds:
        idx = neighbor_inds.pop()
        if new_probs[idx] > 0:
            neighbor_inds.append(idx)
            break
    if neighbor_inds:
        return new_probs[neighbor_inds[-1]], len(neighbor_inds)
    else:
        return -1.0, 0
