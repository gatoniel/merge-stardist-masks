"""Test the initializer and worker separately."""

from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np
import numpy.typing as npt
from stardist.rays3d import Rays_GoldenSpiral  # type: ignore [import-untyped]

from merge_stardist_masks import mp_worker
from merge_stardist_masks import naive_fusion as nf
from merge_stardist_masks import naive_fusion_mp as nf_mp


def test_worker() -> None:
    """Test naive fusion with only two points that overlap."""
    n_rays = 40
    rays = Rays_GoldenSpiral(n=n_rays)

    s = 6
    g = 2
    shape = (s, s, s)
    grid = (g, g, g)
    big_shape = tuple(g * s for s in shape)
    dists_shape = shape + (len(rays),)

    probs = np.zeros(shape)

    dists_tuple: Tuple[npt.NDArray[np.double], SharedMemory] = (
        nf_mp._create_shared_memory(shape + (n_rays,), probs.dtype)
    )
    dists = dists_tuple[0]
    shm_dists = dists_tuple[1]

    lbl_tuple: Tuple[npt.NDArray[np.intc], SharedMemory] = nf_mp._create_shared_memory(
        big_shape, np.intc
    )
    lbl = lbl_tuple[0]
    shm_lbl = lbl_tuple[1]
    lbl[:] = 0

    points_tuple: Tuple[npt.NDArray[np.int_], SharedMemory] = (
        nf_mp._create_shared_memory(big_shape + (3,), np.int_)
    )
    points = points_tuple[0]
    shm_points = points_tuple[1]
    points[:] = nf.inflate_array(
        nf.points_from_grid(probs.shape, grid), grid, default_value=0
    )

    # object 1 with several pixels that do not add to the shape
    z0 = s // 3

    # main point of this object
    dist = 2.5
    dists[z0, z0, z0, :] = dist
    probs[z0, z0, z0] = 0.9

    small_dist = 0.1
    dists[z0 - 1, z0, z0, :] = small_dist
    probs[z0 - 1, z0, z0] = 0.8
    dists[z0, z0 - 1, z0, :] = small_dist
    probs[z0, z0 - 1, z0] = 0.8
    dists[z0, z0, z0 - 1, :] = small_dist
    probs[z0, z0, z0 - 1] = 0.8

    # object 2 composed of two pixels
    z1 = s // 3 * 2
    dists[z1, z1, z1, :] = dist
    probs[z1, z1, z1] = 0.6

    dists[z1 - 1, z1, z1, :] = dist
    probs[z1 - 1, z1, z1] = 0.7

    probs_tuple: Tuple[npt.NDArray[np.double], SharedMemory] = (
        nf_mp._create_shared_memory(big_shape, probs.dtype)
    )
    new_probs = probs_tuple[0]
    shm_new_probs = probs_tuple[1]
    new_probs[:] = nf.inflate_array(probs, grid, default_value=-1)

    mp_worker._initializer(
        shm_new_probs.name,
        probs.dtype,
        shm_points.name,
        shm_lbl.name,
        shm_dists.name,
        dists.dtype,
        dists_shape,
        big_shape,
    )

    mp_worker._worker(
        (4, 4, 4),
        1,
        np.array(grid),
        10,
        0.5,
        (3, 3, 3),
        rays,
    )

    assert lbl.max() == 1
    assert new_probs.max() < probs.max()
