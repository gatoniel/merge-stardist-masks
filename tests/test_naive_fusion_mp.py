"""Tests for the multiprocessing parts."""

import numpy as np
import numpy.typing as npt
from stardist.geometry.geom3d import polyhedron_to_label  # type: ignore [import-untyped]
from stardist.rays3d import Rays_GoldenSpiral  # type: ignore [import-untyped]

from merge_stardist_masks import naive_fusion as nf
from merge_stardist_masks import naive_fusion_mp as nf_mp

# from multiprocessing import shared_memory


def test_naive_fusion_3d() -> None:
    """Test naive fusion with only two points that overlap."""
    n_rays = 40
    rays = Rays_GoldenSpiral(n=n_rays)

    s = 6
    shape = (s, s, s)
    probs = np.zeros(shape)
    dists, shm_dists = nf_mp._create_shared_memory(shape + (n_rays,), probs.dtype)

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

    g = 2
    lbl = nf_mp.naive_fusion_anisotropic_grid(
        shm_dists.name,
        dists.dtype,
        probs,
        (6, 6, 6),
        rays,
        grid=(g, g, g),
    )

    new_dists = np.full((3, n_rays), dist)
    new_points: npt.NDArray[np.double] = (
        np.array([[z0, z0, z0], [z1, z1, z1], [z1 - 1, z1, z1]]) * g
    ).astype(float)

    label = polyhedron_to_label(
        new_dists, new_points, rays, tuple(s * g for s in shape), verbose=False
    )
    # set labels to correct ids
    label[label == 1] = 1
    label[label == 2] = 2
    label[label == 3] = 2

    assert lbl.shape == label.shape
    overlaps = {}
    for i, j in zip(label.flatten(), lbl.flatten()):
        try:
            overlaps[(i, j)] += 1
        except KeyError:
            overlaps[(i, j)] = 1

    print("Overlaps:", overlaps)
    assert overlaps[(0, 0)] == 1578
    assert overlaps[(1, 1)] == 58
    assert overlaps[(2, 2)] == 92


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
    probs, shm_probs = nf_mp._create_shared_memory(shape, float)
    dists, shm_dists = nf_mp._create_shared_memory(shape + (n_rays,), probs.dtype)

    lbl, shm_lbl = nf_mp._create_shared_memory(big_shape, np.intc)
    lbl[:] = 0
    points, shm_points = nf_mp._create_shared_memory(big_shape + (3,), np.int_)
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

    new_probs, shm_new_probs = nf_mp._create_shared_memory(big_shape, probs.dtype)
    new_probs[:] = nf.inflate_array(probs, grid, default_value=-1)

    nf_mp._worker(
        (4, 4, 4),
        1,
        shm_new_probs.name,
        probs.dtype,
        shm_points.name,
        shm_lbl.name,
        shm_dists.name,
        dists.dtype,
        dists_shape,
        big_shape,
        np.array(grid),
        10,
        0.5,
        (3, 3, 3),
        rays,
    )

    assert lbl.max() == 1
    assert new_probs.max() < probs.max()
