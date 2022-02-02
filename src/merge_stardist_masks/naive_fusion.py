"""Naively merge all masks that have sufficient overlap and probability."""
from typing import Iterable

import numpy as np
from stardist.geometry.geom3d import polyhedron_to_label
from stardist.utils import _normalize_grid


def mesh_from_shape(shape):
    """Convenience function to generate a mesh."""
    offsets = []
    for i in range(len(shape)):
        offsets.append(np.arange(shape[i]))
    mesh = np.meshgrid(*offsets, indexing="ij")
    return np.stack(mesh, axis=-1)


def points_from_grid(shape, grid: Iterable[int]):
    """Generate array giving out points for indices."""
    mesh = mesh_from_shape(shape)
    grid = _normalize_grid(grid, 3)
    for i in range(len(grid)):
        mesh[..., i] *= grid[i]
    return mesh


def my_polyhedron_to_label(dists, points, rays, shape):
    """Convenience funtion to pass 1-d arrays to polyhedron_to_label."""
    return polyhedron_to_label(
        np.expand_dims(np.clip(dists, 1e-3, None), axis=0),
        np.expand_dims(points, axis=0),
        rays,
        shape,
        verbose=False,
    )


def my_polyhedron_list_to_label(dists, points, rays, shape):
    """Convenience function to pass lists of 1-d arrays to polyhedron_to_label."""
    return polyhedron_to_label(
        np.clip(np.array(dists), 1e-3, None),
        np.array(points),
        rays,
        shape,
        verbose=False,
    )


def slice_point(point, max_dist):
    """Calculate the extents of a slice for a given point and its coordinates within."""
    slices = []
    centered_point = []
    for a in point:
        diff = a - max_dist
        if diff < 0:
            centered_point.append(max_dist + diff)
            diff = 0
        else:
            centered_point.append(max_dist)
        slices.append(slice(diff, a + max_dist + 1))
    slices = tuple(slices)
    centered_point = np.array(centered_point)
    return slices, centered_point


def naive_fusion(
    dists, probs, rays, prob_thresh: float = 0.5, grid: Iterable[int] = (2, 2, 2)
):
    """Merge overlapping masks given by dists, probs, rays."""
    new_probs = np.copy(probs)
    shape = new_probs.shape

    points = mesh_from_shape(probs.shape)

    inds_thresh = new_probs > prob_thresh
    sum_thresh = np.sum(inds_thresh)

    prob_sort = np.argsort(new_probs, axis=None)[::-1][:sum_thresh]

    lbl = np.zeros(shape, dtype=np.uint16)

    big_shape = tuple(s * g for s, g in zip(shape, grid))
    big_lbl = np.zeros(big_shape, dtype=np.uint16)

    max_dist = int(dists.max() / grid.min() * 2)

    big_max_dist = int(dists.max() * 2)

    sorted_probs_j = 0
    current_id = 1
    while True:
        newly_sorted_probs = np.take_along_axis(new_probs, prob_sort, axis=None)

        while sorted_probs_j < sum_thresh:
            if newly_sorted_probs[sorted_probs_j] > 0:
                max_ind = prob_sort[sorted_probs_j]
                break
            sorted_probs_j += 1
        if sorted_probs_j >= sum_thresh:
            break

        max_ind = np.unravel_index(max_ind, new_probs.shape)
        z, y, x = max_ind
        new_probs[z, y, x] = -1

        slices = []
        for a in [z, y, x]:
            slices.append(slice(max(0, a - max_dist), a + max_dist + 1))
        slices, point = slice_point(points[z, y, x, :], max_dist)
        shape_paint = lbl[slices].shape

        new_shape = (
            my_polyhedron_to_label(
                dists[z, y, x, :] / grid,
                point,  # points[z, y, x, :],
                rays,
                shape_paint,
            )
            == 1
        )

        big_slices, big_point = slice_point(points[z, y, x, :] * grid, big_max_dist)
        big_shape_paint = big_lbl[big_slices].shape

        big_new_shape_dists = [dists[z, y, x, :]]
        big_new_shape_points = [big_point]

        current_probs = new_probs[slices]
        tmp_slices = tuple(
            list(slices)
            + [
                slice(None),
            ]
        )
        current_dists = dists[tmp_slices]
        current_points = points[tmp_slices]

        full_overlaps = 0
        while True:
            if full_overlaps > 2:
                break
            probs_within = current_probs[new_shape]

            if np.sum(probs_within > prob_thresh) == 0:
                break

            max_ind_within = np.argmax(probs_within)
            probs_within[max_ind_within] = -1

            current_probs[new_shape] = probs_within

            additional_shape = (
                my_polyhedron_to_label(
                    current_dists[new_shape, :][max_ind_within, :] / grid,
                    point
                    + current_points[new_shape, :][max_ind_within, :]
                    - points[z, y, x, :],
                    rays,
                    shape_paint,
                )
                > 0
            )

            if (
                np.sum(np.logical_and(new_shape, additional_shape))
                == additional_shape.sum()
            ):
                full_overlaps += 1
                continue

            new_shape = np.logical_or(
                new_shape,
                additional_shape,
            )

            big_new_shape_dists.append(current_dists[new_shape, :][max_ind_within, :])
            big_new_shape_points.append(
                big_point
                + (current_points[new_shape, :][max_ind_within, :] - points[z, y, x, :])
                * grid
            )

        big_new_shape = (
            my_polyhedron_list_to_label(
                big_new_shape_dists,
                big_new_shape_points,
                rays,
                big_shape_paint,
            )
            > 0
        )

        current_probs[new_shape] = -1
        new_probs[slices] = current_probs

        paint_in = lbl[slices]
        paint_in[new_shape] = current_id
        lbl[slices] = paint_in

        big_paint_in = big_lbl[big_slices]
        big_paint_in[big_new_shape] = current_id
        big_lbl[big_slices] = big_paint_in

        current_id += 1

    return lbl, big_lbl
