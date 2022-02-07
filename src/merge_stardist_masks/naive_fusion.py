"""Naively merge all masks that have sufficient overlap and probability."""
from functools import partial
from typing import Iterable
from typing import Optional

import numpy as np
from stardist.geometry.geom2d import polygons_to_label
from stardist.geometry.geom3d import polyhedron_to_label
from stardist.rays3d import Rays_Base
from stardist.utils import _normalize_grid


def mesh_from_shape(shape: Iterable[int]):
    """Convenience function to generate a mesh."""
    offsets = []
    for i in range(len(shape)):
        offsets.append(np.arange(shape[i]))
    mesh = np.meshgrid(*offsets, indexing="ij")
    return np.stack(mesh, axis=-1)


def points_from_grid(shape: Iterable[int], grid: Iterable[int]):
    """Generate array giving out points for indices."""
    mesh = mesh_from_shape(shape)
    grid = np.array(_normalize_grid(grid, len(shape))).reshape(1, len(shape))
    return mesh * grid


def my_polyhedron_to_label(rays, dists, points, shape):
    """Convenience funtion to pass 1-d arrays to polyhedron_to_label."""
    return polyhedron_to_label(
        np.expand_dims(np.clip(dists, 1e-3, None), axis=0),
        np.expand_dims(points, axis=0),
        rays,
        shape,
        verbose=False,
    )


def my_polygons_to_label(dists, points, shape):
    """Convenience funtion to pass 1-d arrays to polygons_to_label."""
    return polygons_to_label(
        np.expand_dims(np.clip(dists, 1e-3, None), axis=0),
        np.expand_dims(points, axis=0),
        shape,
    )


def slice_point(point: Iterable[int], max_dist: int):
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


def no_slicing_slice_point(point: Iterable[int], max_dist: int):
    """Convenience function that returns the same point and tuple of slice(None)."""
    centered_point = np.squeeze(np.array(point))
    slices = (slice(None),) * len(centered_point)
    return slices, centered_point


def inflate_array(x: np.ndarray, grid: Iterable[int], default_value=0):
    """Create new array with increased shape but old values of x."""
    if x.ndim < len(grid):
        raise ValueError("x.ndim must be >= len(grid)")
    # len(new_shape) will be len(grid)
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


def get_poly_to_label(shape, rays):
    """Depending on len(shape) return different functions to calculate labels."""
    if len(shape) == 2:
        return my_polygons_to_label
    elif len(shape) == 3:
        if rays is not None:
            return partial(my_polyhedron_to_label, rays)
        else:
            raise ValueError("For 3D postprocessing rays must be supplied.")
    else:
        raise ValueError("probs.ndim must either be 2 or 3")


def naive_fusion(
    dists: np.ndarray,
    probs: np.ndarray,
    rays: Optional[Rays_Base] = None,
    prob_thresh: float = 0.5,
    grid: Iterable[int] = (2, 2, 2),
    no_slicing: bool = False,
) -> np.ndarray:
    """Merge overlapping masks given by dists, probs, rays.

    Performs a naive iterative scheme to merge the masks that a StarDist network has
    calculated to generate a label image. This function can perform 2D and 3D
    segmentation. For 3D segmentation `rays` has to be passed from the StarDist model.

    Args:
        dists: ndarray of type float, 3- or 4-dimensional.
            Distances of each mask as outputed by a StarDist model. For 2D predictions
            the shape is (len_y, len_x, n_rays), for 3D predictions it is
            (len_z, len_y, len_x, n_rays).
        probs: ndarry of type float, 2- or 3-dimensional.
            Probabilities for each mask as outputed by a StarDist model. For 2D
            predictions the shape is (len_y, len_x), for 3D predictions it is
            (len_z, len_y, len_x).
        rays: None (default) or class inheriting from stardist.rays3d.Rays_Base.
            For 3D predictions `rays` must be set otherwise a `ValueError` is raised.
        prob_thresh:
            Only masks with probability above this threshold are considered.
        grid: Should be of length 2 for 2D and of length 3 for 3D segmentation.
            This is the grid information about the subsampling occuring in the StarDist
            model.
        no_slicing:
            For very big and winded objects this should be set to `True`. However, this
            might result in longer calculation times.

    Returns:
        ndarray of type np.uint16: The label image. For 2D, the shape is
        (len_y * grid[0], len_x * grid[1]) and for 3D it is
        (len_z * grid[0], len_y * grid[1], len_z * grid[2]).

    Raises:
        ValueError: If rays is None and 3D inputs are given or when probs.ndim and
            len(grid) do not match.  # noqa: DAR402 ValueError

    Example:
        >>> from merge_stardist_masks.naive_fusion import naive_fusion
        >>> from stardist.rays3d import rays_from_json
        >>> dists, probs = model.predict(img)  # model is a 3D StarDist model
        >>> rays = rays_from_json(model.config.rays_json)
        >>> lbl = naive_fusion(dists, probs, rays, grid=model.config.grid)
    """
    shape = probs.shape
    grid = np.array(grid)

    poly_to_label = get_poly_to_label(shape, rays)

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
    if no_slicing:
        this_slice_point = no_slicing_slice_point
    else:
        this_slice_point = slice_point

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
        new_probs[tuple(max_ind)] = -1

        ind = tuple(max_ind) + (slice(None),)

        slices, point = this_slice_point(points[ind], max_dist)
        shape_paint = lbl[slices].shape

        new_shape = poly_to_label(dists[ind], point, shape_paint) == 1

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
                poly_to_label(
                    current_dists[new_shape, :][max_ind_within, :],
                    point
                    + current_points[new_shape, :][max_ind_within, :]
                    - points[ind],
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

        current_probs[new_shape] = -1
        new_probs[slices] = current_probs

        paint_in = lbl[slices]
        paint_in[new_shape] = current_id
        lbl[slices] = paint_in

        current_id += 1

    return lbl
