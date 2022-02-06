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
    grid = np.array(_normalize_grid(grid, 3)).reshape(1, 3)
    return mesh * grid


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


def inflate_array(x, grid, default_value=0):
    """Create new array with increased shape but old values of x."""
    new_shape = tuple(s * g for s, g in zip(x.shape, grid))
    if x.ndim > len(new_shape):
        new_shape = new_shape + tuple(x.shape[len(new_shape) :])
    new_x = np.full(new_shape, default_value, dtype=x.dtype)
    slices = []
    for i in range(len(new_shape)):
        try:
            slices.append(slice(None, None, grid[i]))
        except IndexError:
            slices.append(slice(None))
    new_x[tuple(slices)] = x
    return new_x


def naive_fusion(
    dists, probs, rays, prob_thresh: float = 0.5, grid: Iterable[int] = (2, 2, 2)
):
    """Merge overlapping masks given by dists, probs, rays."""
    shape = probs.shape
    grid = np.array(grid)

    big_shape = tuple(s * g for s, g in zip(shape, grid))
    lbl = np.zeros(big_shape, dtype=np.uint16)

    # this could also be done with np.repeat, but for probs it is important that some
    # of the repeatet values are -1, as they should not be considered.
    new_probs = inflate_array(probs, grid, default_value=-1)
    points = inflate_array(points_from_grid(probs.shape, grid), grid, default_value=0)
    dists = inflate_array(dists, grid, default_value=0)

    inds_thresh = new_probs > prob_thresh
    sum_thresh = np.sum(inds_thresh)

    prob_sort = np.argsort(new_probs, axis=None)[::-1][:sum_thresh]

    max_dist = int(dists.max() * 2)

    sorted_probs_j = 0
    current_id = 1
    while True:
        # In case this is always a view of new_probs that changes when new_probs changes
        # this line should be placed outside of this while-loop.
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

        slices, point = slice_point(points[z, y, x, :], max_dist)
        shape_paint = lbl[slices].shape

        new_shape = (
            my_polyhedron_to_label(
                dists[z, y, x, :],
                point,  # points[z, y, x, :],
                rays,
                shape_paint,
            )
            == 1
        )

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
                    current_dists[new_shape, :][max_ind_within, :],
                    point
                    + current_points[new_shape, :][max_ind_within, :]
                    - points[z, y, x, :],
                    rays,
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
                continue

        current_probs[new_shape] = -1
        new_probs[slices] = current_probs

        paint_in = lbl[slices]
        paint_in[new_shape] = current_id
        lbl[slices] = paint_in

        current_id += 1

    return (
        lbl,
        lbl,
    )  # returning two lbls is legacy code due to previously returning big_lbl
