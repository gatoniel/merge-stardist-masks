"""A multiprocess aware version of the naive_fusion function."""

from __future__ import annotations

import atexit
import threading
from concurrent.futures import Future
from multiprocessing.shared_memory import SharedMemory
from typing import Dict
from typing import Iterable
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

T = TypeVar("T", bound=np.generic)


def in_hyper_square(
    ii: Tuple[int, ...], jj: Tuple[int, ...], dists: Tuple[int, ...]
) -> bool:
    """Tests if two points lie in the same hyper square."""
    return all(abs(i - j) < dist for i, j, dist in zip(ii, jj, dists))


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

    inds_thresh: npt.NDArray[np.int_] = np.argwhere(new_probs > prob_thresh)
    # sort_args = np.argsort(new_probs[*inds_thresh.T])
    sort_args = np.argsort(new_probs[tuple(i for i in inds_thresh.T)])

    inds = [tuple(int(i) for i in inds_thresh[j]) for j in sort_args]

    lbl_tuple: Tuple[npt.NDArray[np.intc], SharedMemory] = _create_shared_memory(
        big_shape, np.intc
    )
    lbl = lbl_tuple[0]
    shm_lbl = lbl_tuple[1]
    lbl[:] = 0

    # Currently running jobs
    running: Dict[Future[None], Tuple[int, ...]] = {}
    # Dict to save already computed conflicting inds
    conflicting: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], bool] = {}
    conflict_max_dists = tuple(2 * i for i in max_dists)

    lock = threading.Lock()
    done_event = threading.Event()

    if max_parallel is None:
        max_parallel = cpu_count()
    executor = get_reusable_executor(
        max_workers=max_parallel,
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
        ),
    )
    atexit.register(executor.shutdown)

    current_id = 1

    def is_conflicting(
        index: Tuple[int, ...], others: Iterable[Tuple[int, ...]]
    ) -> bool:
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

    def try_schedule() -> None:
        nonlocal current_id
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

            to_schedule: List[Tuple[int, ...]] = []
            skipped: List[Tuple[int, ...]] = []

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
                future = executor.submit(
                    _worker,
                    idx,
                    current_id,
                    grid_array,
                    max_full_overlaps,
                    prob_thresh,
                    max_dists,
                    rays,
                )
                current_id += 1
                running[future] = idx
                future.add_done_callback(lambda _: try_schedule())
            print("Remaining probability voxels to check:", len(inds))

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
