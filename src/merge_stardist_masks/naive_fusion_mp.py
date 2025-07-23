"""A multiprocess aware version of the naive_fusion function."""

from __future__ import annotations

import atexit
import threading
from multiprocessing import shared_memory
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
from loky import get_reusable_executor
from loky.backend.context import cpu_count
from stardist.rays3d import Rays_Base  # type: ignore [import-untyped]

from .naive_fusion import inflate_array
from .naive_fusion import my_polyhedron_to_label
from .naive_fusion import paint_in_without_overlaps
from .naive_fusion import points_from_grid
from .naive_fusion import SlicePointReturn


def in_hyper_square(ii, jj, dists):
    """Tests if two points lie in the same hyper square."""
    return all(abs(i - j) < dist for i, j, dist in zip(ii, jj, dists))


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


def naive_fusion_anisotropic_grid(
    shm_dists_name: str,
    dists_dtype,
    probs: npt.NDArray[np.double],
    max_dists: Tuple[int, ...],
    rays: Optional[Rays_Base] = None,
    prob_thresh: float = 0.5,
    grid: Tuple[int, ...] = (2, 2, 2),
    max_full_overlaps: int = 2,
    max_parallel: Optional[int] = None,
) -> Union[npt.NDArray[np.uint16], npt.NDArray[np.intc]]:
    """Merge overlapping masks given by dists, probs, rays for anisotropic grid.

    Performs a naive iterative scheme to merge the masks that a StarDist network has
    calculated to generate a label image.  This function can perform 2D and 3D
    segmentation.  For 3D segmentation `rays` has to be passed from the StarDist model.

    Args:
        shm_dists_name: Shared memory name to a 3- or 4-dimensional array representing
            distances of each mask as outputed by a StarDist model.
            For 2D predictions the shape is
            ``(len_y, len_x, n_rays)``, for 3D predictions it is
            ``(len_z, len_y, len_x, n_rays)``.
        dists_dtype: The data type of the array containing the dists.
        probs: 2- or 3-dimensional array representing the probabilities for each mask as
            outputed by a StarDist model.  For 2D predictions the shape is
            ``(len_y, len_x)``, for 3D predictions it is ``(len_z, len_y, len_x)``.
        max_dists: Tuple describing how far apart the parts of the array should be on
            which the independent multiprocessing workers perform calculations.
        rays: For 3D predictions `rays` must be set otherwise a ``ValueError`` is
            raised.  It should be the :class:`Rays_Base` instance used by the StarDist
            model.
        prob_thresh: Only masks with probability above this threshold are considered.
        grid: Should be of length 2 for 2D and of length 3 for 3D segmentation.
            This is the grid information about the subsampling occuring in the StarDist
            model.
        max_full_overlaps: Maximum no. of full overlaps before current object is treated
            as finished.
        max_parallel: Number of parallel processes to use. None defaults to use
            `cpu_count`.

    Returns:
        The label image with uint16 labels. For 2D, the shape is
        ``(len_y * grid[0], len_x * grid[1])`` and for 3D it is
        ``(len_z * grid[0], len_y * grid[1], len_z * grid[2])``.

    Raises:
        ValueError: If `rays` is ``None`` and 3D inputs are given or when
            ``probs.ndim != len(grid)``.  # noqa: DAR402 ValueError

    Example:
        >>> from merge_stardist_masks.naive_fusion import naive_fusion_anisotropic_grid
        >>> from stardist.rays3d import rays_from_json
        >>> probs, dists = model.predict(img)  # model is a 3D StarDist model
        >>> rays = rays_from_json(model.config.rays_json)
        >>> grid = model.config.grid
        >>> lbl = naive_fusion_anisotropic_grid(dists, probs, rays, grid=grid)
    """
    shape = probs.shape
    dists_shape = shape + (len(rays),)
    grid_array = np.array(grid, dtype=int)

    big_shape = tuple(s * g for s, g in zip(shape, grid))

    new_probs, shm_new_probs = _create_shared_memory(big_shape, probs.dtype)
    new_probs[:] = inflate_array(probs, grid, default_value=-1)
    points, shm_points = _create_shared_memory(big_shape + (3,), np.int_)
    points[:] = inflate_array(
        points_from_grid(probs.shape, grid), grid, default_value=0
    )

    inds_thresh: npt.NDArray[np.int_] = np.argwhere(new_probs > prob_thresh)
    sort_args = np.argsort(new_probs[*inds_thresh.T])

    inds = [tuple(int(i) for i in inds_thresh[j]) for j in sort_args]

    # paint_in = paint_in_without_overlaps
    lbl, shm_lbl = _create_shared_memory(big_shape, np.intc)
    lbl[:] = 0

    # Currently running jobs
    running = {}
    # Dict to save already computed conflicting inds
    conflicting = {}
    conflict_max_dists = tuple(2 * i for i in max_dists)

    lock = threading.Lock()
    done_event = threading.Event()

    if max_parallel is None:
        max_parallel = cpu_count()
    executor = get_reusable_executor(max_workers=max_parallel)
    atexit.register(executor.shutdown)

    current_id = 1

    def is_conflicting(index, others):
        for other in others:
            key = (index, other)
            try:
                val = conflicting[key]
            except KeyError:
                val = in_hyper_square(index, other, conflict_max_dists)
                conflicting[key] = val
            if val:
                return True
        return False

    def try_schedule():
        nonlocal current_id
        print("in try_schedule", current_id)
        with lock:
            # list to avoid problems with deletions in loop
            for fut in list(running):
                if fut.done():
                    del running[fut]

            if not inds and not running:
                done_event.set()
                return

            available_slots = max_parallel - len(running)
            if available_slots <= 0:
                return

            to_schedule = []
            skipped = []

            while len(to_schedule) < available_slots and inds:
                idx = inds.pop()

                if new_probs[idx] < 0:
                    continue

                if (
                    is_conflicting(idx, to_schedule)
                    or is_conflicting(idx, running.values())
                    or is_conflicting(idx, skipped)
                ):
                    skipped.append(idx)
                    continue
                to_schedule.append(idx)
            inds.extend(reversed(skipped))

            if not inds and not running and not to_schedule:
                done_event.set()
                return

            for idx in to_schedule:
                print("in schedule loop", idx)
                future = executor.submit(
                    _worker,
                    idx,
                    current_id,
                    shm_new_probs.name,
                    probs.dtype,
                    shm_points.name,
                    shm_lbl.name,
                    shm_dists_name,
                    dists_dtype,
                    dists_shape,
                    big_shape,
                    grid_array,
                    max_full_overlaps,
                    prob_thresh,
                    max_dists,
                    rays,
                )
                current_id += 1
                running[future] = idx
                future.add_done_callback(lambda _: try_schedule())

    print(np.min(new_probs), np.max(new_probs))
    try_schedule()

    done_event.wait()
    print(np.min(new_probs), np.max(new_probs))
    print(lbl.max())

    return lbl


def _worker(
    max_ind,
    current_id,
    new_probs_name,
    new_probs_dtype,
    points_name,
    lbl_name,
    dists_name,
    dists_dtype,
    dists_shape,
    big_shape,
    grid_array,
    max_full_overlaps,
    prob_thresh,
    max_dists,
    rays,
):
    print("worker start", max_ind, current_id)
    new_probs, shm_new_probs = _load_shared_memory(
        big_shape, new_probs_dtype, new_probs_name
    )
    points, shm_points = _load_shared_memory(big_shape + (3,), np.int_, points_name)
    lbl, shm_lbl = _load_shared_memory(big_shape, np.intc, lbl_name)
    dists, shm_dists = _load_shared_memory(dists_shape, dists_dtype, dists_name)

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

    _close_and_unlink(shm_new_probs)
    _close_and_unlink(shm_points)
    _close_and_unlink(shm_lbl)
    _close_and_unlink(shm_dists)


def _create_shared_memory(shape, dtype):
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    a = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    atexit.register(shm.close)
    atexit.register(shm.unlink)
    return a, shm


def _load_shared_memory(shape, dtype, name):
    shm = shared_memory.SharedMemory(name=name)
    a = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return a, shm


def _close_and_unlink(shm):
    shm.close()
    # shm.unlink()
