"""A multiprocess aware version of the naive_fusion function."""

from __future__ import annotations

import atexit
import multiprocessing
import threading
import time
from concurrent.futures import Future
from itertools import product
from math import ceil
from multiprocessing.shared_memory import SharedMemory
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt
from loky import get_reusable_executor  # type: ignore [import-untyped]
from loky.backend.context import cpu_count  # type: ignore [import-untyped]
from stardist.rays3d import Rays_Base  # type: ignore [import-untyped]

from .mp_worker import _initializer
from .mp_worker import _worker
from .naive_fusion import inflate_array
from .naive_fusion import points_from_grid

# from tqdm import tqdm  # type: ignore [import-untyped]


T = TypeVar("T", bound=np.generic)


def in_hyper_square(
    ii: Tuple[int, ...], jj: Tuple[int, ...], dists: Tuple[int, ...]
) -> bool:
    """Tests if two points lie in the same hyper square."""
    return all(abs(i - j) < dist for i, j, dist in zip(ii, jj, dists))


def _get_slice(i: int, dist: int, max_len: int) -> Tuple[slice, int]:
    start = i * dist
    stop = start + dist
    if stop > max_len:
        stop = max_len
    return slice(start, stop), start


def _neighbors(index: Tuple[int, ...], shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    positions: Tuple[List[int], ...] = ([], [], [])
    for i in range(3):
        for j in range(-1, 2):
            val = index[i] + j
            if val < 0 or val >= shape[i]:
                continue
            positions[i].append(val)
    return list(product(*positions))


def naive_fusion_anisotropic_grid(
    shm_dists_name: str,
    dists_dtype: npt.DTypeLike,
    probs: npt.NDArray[np.double],
    max_dists: Tuple[int, ...],
    rays: Optional[Rays_Base] = None,
    prob_thresh: float = 0.5,
    grid: Tuple[int, ...] = (2, 2, 2),
    max_full_overlaps: int = 2,
    max_parallel: Optional[int] = None,
    # verbose: bool = False,
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
    n_rays = len(rays)  # type: ignore [arg-type]
    dists_shape = shape + (n_rays,)
    grid_array = np.array(grid, dtype=int)

    big_shape = tuple(s * g for s, g in zip(shape, grid))

    probs_tuple: Tuple[npt.NDArray[np.double], SharedMemory] = _create_shared_memory(
        big_shape, probs.dtype
    )
    new_probs = probs_tuple[0]
    shm_new_probs = probs_tuple[1]
    new_probs[:] = inflate_array(probs, grid, default_value=-1)

    points_tuple: Tuple[npt.NDArray[np.int_], SharedMemory] = _create_shared_memory(
        big_shape + (3,), np.int_
    )
    points = points_tuple[0]
    shm_points = points_tuple[1]
    points[:] = inflate_array(
        points_from_grid(probs.shape, grid), grid, default_value=0
    )

    lbl_tuple: Tuple[npt.NDArray[np.intc], SharedMemory] = _create_shared_memory(
        big_shape, np.intc
    )
    lbl = lbl_tuple[0]
    shm_lbl = lbl_tuple[1]
    lbl[:] = 0

    prob_test = new_probs > prob_thresh
    slices = {}

    max_dists = tuple(d // 2 for d in max_dists)
    num_slices = tuple(ceil(s / d) for s, d in zip(big_shape, max_dists))
    max_dists = tuple(d * 2 for d in max_dists)

    block_list = np.zeros(num_slices, dtype=bool)
    done_list = np.zeros(num_slices, dtype=bool)

    max_probs_tuple: Tuple[npt.NDArray[np.double], SharedMemory] = (
        _create_shared_memory(num_slices, probs.dtype)
    )
    max_probs = max_probs_tuple[0]
    shm_max_probs = max_probs_tuple[1]

    remaining_inds_tuple: Tuple[npt.NDArray[np.intc], SharedMemory] = (
        _create_shared_memory(num_slices, np.intc)
    )
    remaining_inds = remaining_inds_tuple[0]
    shm_remaining_inds = remaining_inds_tuple[1]

    neighbors: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    manager = multiprocessing.Manager()
    inds = manager.dict()
    for i in range(num_slices[0]):
        slice_z, start_z = _get_slice(i, max_dists[0], big_shape[0])
        for j in range(num_slices[1]):
            slice_y, start_y = _get_slice(j, max_dists[1], big_shape[1])
            for k in range(num_slices[2]):
                t = (i, j, k)
                slice_x, start_x = _get_slice(k, max_dists[2], big_shape[2])
                tmp_slice = (slice_z, slice_y, slice_x)
                slices[t] = tmp_slice

                inds_thresh: npt.NDArray[np.int_] = np.argwhere(
                    prob_test[tmp_slice]
                ) + np.array((start_z, start_y, start_x))

                # Only possible in python >= 3.12
                # sort_args = np.argsort(new_probs[*inds_thresh.T])
                sort_args = np.argsort(new_probs[tuple(m for m in inds_thresh.T)])

                tmp_inds = [tuple(int(m) for m in inds_thresh[n]) for n in sort_args]
                inds[t] = manager.list(tmp_inds)
                if tmp_inds:
                    max_probs[t] = new_probs[tmp_inds[-1]]
                else:
                    max_probs[t] = -1
                    done_list[t] = True

                remaining_inds[t] = len(tmp_inds)

                neighbors[t] = _neighbors(t, num_slices)

    current_id = manager.Value("i", 1)
    current_id_lock = manager.Lock()

    if max_parallel is None:
        max_parallel = cpu_count()

    max_simultaneous_possible_workers = np.prod([i / 3 for i in num_slices])
    if max_parallel > max_simultaneous_possible_workers:
        max_parallel = int(max_simultaneous_possible_workers)
        print("Clipped max_parallel to: ", max_parallel)

    executor = get_reusable_executor(
        max_workers=max_parallel,
        timeout=None,
        initializer=_initializer,
        initargs=(
            shm_new_probs.name,
            probs.dtype,
            shm_points.name,
            shm_lbl.name,
            shm_dists_name,
            dists_dtype,
            dists_shape,
            big_shape,
            shm_max_probs.name,
            num_slices,
            current_id,
            current_id_lock,
            inds,
            neighbors,
            shm_remaining_inds.name,
            # verbose,
        ),
    )
    atexit.register(executor.shutdown)

    lock = threading.Lock()
    # Under this lock fall
    # - running dictionary
    # - block_list
    # - done_list array
    # - pbar progress bar
    # - current_counter for progress bar
    done_event = threading.Event()

    running: Dict[Future[None], Tuple[int, ...]] = {}

    total_inds = remaining_inds.sum()
    start_time = time.perf_counter()
    # pbar = tqdm(total=total_inds)
    # current_counter = total_inds

    def is_free(index: Tuple[int, ...]) -> bool:
        my_prob = max_probs[index]
        for neighbor in neighbors[index]:
            if block_list[neighbor] or my_prob < max_probs[neighbor]:
                return False
        return True

    def try_schedule() -> None:
        # nonlocal current_counter
        t0 = time.perf_counter()
        with lock:
            # finished_jobs = []
            # list to avoid problems with deletions in loop
            for fut in list(running):
                if fut.done():
                    # free block_list again
                    idx = running[fut]
                    # finished_jobs.append(idx)
                    for neighbor in neighbors[idx]:
                        block_list[neighbor] = False
                        if max_probs[neighbor] < 0:
                            done_list[neighbor] = True
                    del running[fut]

            # pbar.update(current_counter - tmp_counter)
            # current_counter = tmp_counter

            if done_list.all() and not running:
                done_event.set()
                return

            available_slots = max_parallel - len(running)
            if available_slots <= 0:
                return

            to_schedule: List[Tuple[int, ...]] = []

            unblocked: npt.NDArray[np.int_] = np.argwhere(
                np.logical_not(np.logical_or(block_list, done_list))
            )
            if len(unblocked) == 0:
                return

            # sorted is from least to highest probs. We want to start with
            # highest values. reverse is avoided in the argsorts above,
            # because there we reverse the list automatically in the worker
            # by using .pop().
            for j in reversed(np.argsort(max_probs[tuple(i for i in unblocked.T)])):
                idx = tuple(int(i) for i in unblocked[j])

                if not is_free(idx):
                    continue

                to_schedule.append(idx)

                # This needs to be here! Cannot be deferred to bottom loop as this
                # is important to know for the next jobs to add in this current
                # scheduling trial.
                for neighbor in neighbors[idx]:
                    block_list[neighbor] = True

                if len(to_schedule) >= available_slots:
                    break

            for idx in to_schedule:
                future = executor.submit(
                    _worker,
                    idx,
                    grid_array,
                    max_full_overlaps,
                    prob_thresh,
                    max_dists,
                    rays,
                )
                running[future] = idx
                future.add_done_callback(lambda _: try_schedule())
        t1 = time.perf_counter()
        total_time = t1 - t0
        tmp_counter = remaining_inds.sum()
        print("SCHEDULDER took", total_time, "seconds")
        print("REMAINING positions:", tmp_counter, "/", total_inds)
        print("JOBS: Submitted", len(to_schedule), ", Running:", len(running))
        mins_run = (time.perf_counter() - start_time) / 60
        speed = (total_inds - tmp_counter) / mins_run
        remaining_time = tmp_counter / speed
        print(
            "Currently running for",
            mins_run,
            "minutes. Remaining: ",
            remaining_time,
            "minutes",
        )

        # if len(to_schedule) == len(running) == 0:
        #     assert ((max_probs < 0) == done_list).all()
        #     print(block_list)
        #     print(done_list)
        #     print(max_probs)
        #     print(unblocked)
        #     np.save("dump_block_list.npy", block_list)
        #     np.save("dump_done_list.npy", done_list)
        #     np.save("dump_max_probs.npy", max_probs)
        #     np.save("dump_unblocked.npy", unblocked)
        #     print(
        #         list(
        #             reversed(
        #                 np.argsort(max_probs[tuple(i for i in unblocked.T)])
        #             )
        #         )
        #     )
        #     print(finished_jobs)

    try_schedule()

    done_event.wait()

    return lbl


def _create_shared_memory(
    shape: Tuple[int, ...], dtype: npt.DTypeLike
) -> Tuple[npt.NDArray[T], SharedMemory]:
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    shm = SharedMemory(create=True, size=nbytes)
    a: npt.NDArray[T] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    atexit.register(shm.close)
    atexit.register(shm.unlink)
    return a, shm
