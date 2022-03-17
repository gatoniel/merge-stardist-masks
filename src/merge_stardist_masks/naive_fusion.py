"""Naively merge all masks that have sufficient overlap and probability."""
from __future__ import annotations

from functools import partial
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt
from stardist.geometry.geom2d import polygons_to_label  # type: ignore [import]
from stardist.geometry.geom3d import polyhedron_to_label  # type: ignore [import]
from stardist.rays3d import Rays_Base  # type: ignore [import]
from stardist.utils import _normalize_grid  # type: ignore [import]


T = TypeVar("T", bound=np.generic)
ArrayLike = npt.ArrayLike


def mesh_from_shape(shape: Tuple[int, ...]) -> npt.NDArray[np.int_]:
    """Convenience function to generate a mesh."""
    offsets = []
    for i in range(len(shape)):
        offsets.append(np.arange(shape[i]))
    mesh = np.meshgrid(*offsets, indexing="ij")  # type: ignore [no-untyped-call]
    return np.stack(mesh, axis=-1)


def points_from_grid(
    shape: Tuple[int, ...], grid: Tuple[int, ...]
) -> npt.NDArray[np.int_]:
    """Generate array giving out points for indices."""
    mesh = mesh_from_shape(shape)
    grid_array = np.array(_normalize_grid(grid, len(shape))).reshape(1, len(shape))
    return mesh * grid_array


def my_polyhedron_to_label(
    rays: Rays_Base, dists: ArrayLike, points: ArrayLike, shape: Tuple[int, ...]
) -> npt.NDArray[np.int_]:
    """Convenience funtion to pass 1-d arrays to polyhedron_to_label."""
    return polyhedron_to_label(  # type: ignore [no-any-return]
        np.expand_dims(  # type: ignore [no-untyped-call]
            np.clip(dists, 1e-3, None), axis=0
        ),
        np.expand_dims(points, axis=0),  # type: ignore [no-untyped-call]
        rays,
        shape,
        verbose=False,
    )


def my_polyhedron_list_to_label(
    rays: Rays_Base, dists: ArrayLike, points: ArrayLike, shape: Tuple[int, ...]
) -> npt.NDArray[np.int_]:
    """Convenience funtion to pass 1-d arrays to polyhedron_to_label."""
    return polyhedron_to_label(  # type: ignore [no-any-return]
        np.clip(np.array(dists), 1e-3, None),
        np.array(points),
        rays,
        shape,
        verbose=False,
    )


def my_polygons_to_label(
    dists: ArrayLike, points: ArrayLike, shape: Tuple[int, ...]
) -> npt.NDArray[np.int_]:
    """Convenience funtion to pass 1-d arrays to polygons_to_label."""
    return polygons_to_label(  # type: ignore [no-any-return]
        np.expand_dims(  # type: ignore [no-untyped-call]
            np.clip(dists, 1e-3, None), axis=0
        ),
        np.expand_dims(points, axis=0),  # type: ignore [no-untyped-call]
        shape,
    )


def my_polygons_list_to_label(
    dists: ArrayLike, points: ArrayLike, shape: Tuple[int, ...]
) -> npt.NDArray[np.int_]:
    """Convenience funtion to pass 1-d arrays to polygons_to_label."""
    return polygons_to_label(  # type: ignore [no-any-return]
        np.clip(np.array(dists), 1e-3, None),
        np.array(points),
        shape,
    )


PolyToLabelSignature = Callable[
    [ArrayLike, ArrayLike, Tuple[int, ...]], npt.NDArray[np.int_]
]


def get_poly_to_label(
    shape: Tuple[int, ...], rays: Optional[Rays_Base]
) -> PolyToLabelSignature:
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


def get_poly_list_to_label(
    shape: Tuple[int, ...], rays: Optional[Rays_Base]
) -> PolyToLabelSignature:
    """Depending on len(shape) return different functions to calculate labels."""
    if len(shape) == 2:
        return my_polygons_list_to_label
    elif len(shape) == 3:
        if rays is not None:
            return partial(my_polyhedron_list_to_label, rays)
        else:
            raise ValueError("For 3D postprocessing rays must be supplied.")
    else:
        raise ValueError("probs.ndim must either be 2 or 3")


SlicePointReturn = Tuple[Tuple[slice, ...], npt.NDArray[np.int_]]


def slice_point(point: ArrayLike, max_dist: int) -> SlicePointReturn:
    """Calculate the extents of a slice for a given point and its coordinates within."""
    slices_list = []
    centered_point = []
    for a in np.array(point):
        diff = a - max_dist
        if diff < 0:
            centered_point.append(max_dist + diff)
            diff = 0
        else:
            centered_point.append(max_dist)
        slices_list.append(slice(diff, a + max_dist + 1))
    return tuple(slices_list), np.array(centered_point)


def no_slicing_slice_point(point: ArrayLike, max_dist: int) -> SlicePointReturn:
    """Convenience function that returns the same point and tuple of slice(None)."""
    centered_point = np.squeeze(np.array(point))
    slices = (slice(None),) * len(centered_point)
    return slices, centered_point


def inflate_array(
    x: npt.NDArray[T], grid: Tuple[int, ...], default_value: Union[int, float] = 0
) -> npt.NDArray[T]:
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


def naive_fusion(
    dists: npt.NDArray[np.double],
    probs: npt.NDArray[np.double],
    rays: Optional[Rays_Base] = None,
    prob_thresh: float = 0.5,
    grid: Tuple[int, ...] = (2, 2, 2),
    no_slicing: bool = False,
    max_full_overlaps: int = 2,
    erase_probs_at_full_overlap: bool = False,
) -> npt.NDArray[np.uint16]:
    """Merge overlapping masks given by dists, probs, rays.

    Performs a naive iterative scheme to merge the masks that a StarDist network has
    calculated to generate a label image.  This function can perform 2D and 3D
    segmentation.  For 3D segmentation `rays` has to be passed from the StarDist model.

    Args:
        dists: 3- or 4-dimensional array representing distances of each mask as outputed
            by a StarDist model.  For 2D predictions the shape is
            ``(len_y, len_x, n_rays)``, for 3D predictions it is
            ``(len_z, len_y, len_x, n_rays)``.
        probs: 2- or 3-dimensional array representing the probabilities for each mask as
            outputed by a StarDist model.  For 2D predictions the shape is
            ``(len_y, len_x)``, for 3D predictions it is ``(len_z, len_y, len_x)``.
        rays: For 3D predictions `rays` must be set otherwise a ``ValueError`` is
            raised.  It should be the :class:`Rays_Base` instance used by the StarDist
            model.
        prob_thresh: Only masks with probability above this threshold are considered.
        grid: Should be of length 2 for 2D and of length 3 for 3D segmentation.
            This is the grid information about the subsampling occuring in the StarDist
            model.
        no_slicing: For very big and winded objects this should be set to ``True``.
            However, this might result in longer calculation times.
        max_full_overlaps: Maximum no. of full overlaps before current object is treated
            as finished.
        erase_probs_at_full_overlap: If set to ``True`` probs are set to -1 whenever
            a full overlap is detected.

    Returns:
        The label image with uint16 labels. For 2D, the shape is
        ``(len_y * grid[0], len_x * grid[1])`` and for 3D it is
        ``(len_z * grid[0], len_y * grid[1], len_z * grid[2])``.

    Raises:
        ValueError: If `rays` is ``None`` and 3D inputs are given or when
            ``probs.ndim != len(grid)``.  # noqa: DAR402 ValueError

    Example:
        >>> from merge_stardist_masks.naive_fusion import naive_fusion
        >>> from stardist.rays3d import rays_from_json
        >>> probs, dists = model.predict(img)  # model is a 3D StarDist model
        >>> rays = rays_from_json(model.config.rays_json)
        >>> lbl = naive_fusion(dists, probs, rays, grid=model.config.grid)
    """
    if len(np.unique(grid)) == 1:  # type: ignore [no-untyped-call]
        return naive_fusion_isotropic_grid(
            dists,
            probs,
            rays,
            prob_thresh,
            grid[0],
            no_slicing,
            max_full_overlaps,
            erase_probs_at_full_overlap=erase_probs_at_full_overlap,
        )
    else:
        return naive_fusion_anisotropic_grid(
            dists,
            probs,
            rays,
            prob_thresh,
            grid,
            no_slicing,
            max_full_overlaps,
            erase_probs_at_full_overlap=erase_probs_at_full_overlap,
        )


def naive_fusion_isotropic_grid(
    dists: npt.NDArray[np.double],
    probs: npt.NDArray[np.double],
    rays: Optional[Rays_Base] = None,
    prob_thresh: float = 0.5,
    grid: int = 2,
    no_slicing: bool = False,
    max_full_overlaps: int = 2,
    erase_probs_at_full_overlap: bool = False,
) -> npt.NDArray[np.uint16]:
    """Merge overlapping masks given by dists, probs, rays.

    Performs a naive iterative scheme to merge the masks that a StarDist network has
    calculated to generate a label image.  This function can perform 2D and 3D
    segmentation.  For 3D segmentation `rays` has to be passed from the StarDist model.

    Args:
        dists: 3- or 4-dimensional array representing distances of each mask as outputed
            by a StarDist model.  For 2D predictions the shape is
            ``(len_y, len_x, n_rays)``, for 3D predictions it is
            ``(len_z, len_y, len_x, n_rays)``.
        probs: 2- or 3-dimensional array representing the probabilities for each mask as
            outputed by a StarDist model.  For 2D predictions the shape is
            ``(len_y, len_x)``, for 3D predictions it is ``(len_z, len_y, len_x)``.
        rays: For 3D predictions `rays` must be set otherwise a ``ValueError`` is
            raised.  It should be the :class:`Rays_Base` instance used by the StarDist
            model.
        prob_thresh: Only masks with probability above this threshold are considered.
        grid: This is the grid information about the subsampling occuring in the
            StarDist model.
        no_slicing: For very big and winded objects this should be set to ``True``.
            However, this might result in longer calculation times.
        max_full_overlaps: Maximum no. of full overlaps before current object is treated
            as finished.
        erase_probs_at_full_overlap: If set to ``True`` probs are set to -1 whenever
            a full overlap is detected.

    Returns:
        The label image with uint16 labels. For 2D, the shape is
        ``(len_y * grid[0], len_x * grid[1])`` and for 3D it is
        ``(len_z * grid[0], len_y * grid[1], len_z * grid[2])``.

    Raises:
        ValueError: If `rays` is ``None`` and 3D inputs are given or when
            ``probs.ndim != len(grid)``.  # noqa: DAR402 ValueError

    Example:
        >>> from merge_stardist_masks.naive_fusion import naive_fusion_isotropic_grid
        >>> from stardist.rays3d import rays_from_json
        >>> probs, dists = model.predict(img)  # model is a 3D StarDist model
        >>> rays = rays_from_json(model.config.rays_json)
        >>> grid = model.config.grid[0]
        >>> lbl = naive_fusion_isotropic_grid(dists, probs, rays, grid=grid)
    """
    new_probs = np.copy(probs)  # type: ignore [no-untyped-call]
    shape = new_probs.shape

    poly_to_label = get_poly_to_label(shape, rays)
    poly_list_to_label = get_poly_list_to_label(shape, rays)

    points = mesh_from_shape(probs.shape)

    inds_thresh = new_probs > prob_thresh
    sum_thresh = np.sum(inds_thresh)

    prob_sort = np.argsort(new_probs, axis=None)[::-1][:sum_thresh]

    lbl = np.zeros(shape, dtype=np.uint16)

    big_shape = tuple(s * grid for s in shape)
    big_lbl = np.zeros(big_shape, dtype=np.uint16)

    max_dist = int(dists.max() / grid * 2)

    big_max_dist = int(dists.max() * 2)

    if no_slicing:
        this_slice_point = no_slicing_slice_point
    else:
        this_slice_point = slice_point

    sorted_probs_j = 0
    current_id = 1
    while True:
        newly_sorted_probs = np.take_along_axis(  # type: ignore [no-untyped-call]
            new_probs, prob_sort, axis=None
        )

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

        new_shape = (
            poly_to_label(
                dists[ind] / grid,
                point,
                shape_paint,
            )
            == 1
        )

        big_slices, big_point = this_slice_point(points[ind] * grid, big_max_dist)
        big_shape_paint = big_lbl[big_slices].shape

        big_new_shape_dists = [dists[ind]]
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
            if full_overlaps > max_full_overlaps:
                break
            probs_within = current_probs[new_shape]

            if np.sum(probs_within > prob_thresh) == 0:
                break

            max_ind_within = np.argmax(probs_within)
            probs_within[max_ind_within] = -1

            current_probs[new_shape] = probs_within

            this_dist = current_dists[new_shape, :][max_ind_within, :]
            this_point = current_points[new_shape, :][max_ind_within, :]
            additional_shape: npt.NDArray[np.bool_] = (
                poly_to_label(
                    this_dist / grid,
                    point + this_point - points[ind],
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
                if erase_probs_at_full_overlap:
                    current_probs[additional_shape] = -1
            else:
                big_new_shape_dists.append(this_dist)
                big_new_shape_points.append(
                    big_point + (this_point - points[ind]) * grid
                )

        big_new_shape: npt.NDArray[np.bool_] = (
            poly_list_to_label(
                big_new_shape_dists,
                big_new_shape_points,
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

    return big_lbl


def naive_fusion_anisotropic_grid(
    dists: npt.NDArray[np.double],
    probs: npt.NDArray[np.double],
    rays: Optional[Rays_Base] = None,
    prob_thresh: float = 0.5,
    grid: Tuple[int, ...] = (2, 2, 2),
    no_slicing: bool = False,
    max_full_overlaps: int = 2,
    erase_probs_at_full_overlap: bool = False,
) -> npt.NDArray[np.uint16]:
    """Merge overlapping masks given by dists, probs, rays for anisotropic grid.

    Performs a naive iterative scheme to merge the masks that a StarDist network has
    calculated to generate a label image.  This function can perform 2D and 3D
    segmentation.  For 3D segmentation `rays` has to be passed from the StarDist model.

    Args:
        dists: 3- or 4-dimensional array representing distances of each mask as outputed
            by a StarDist model.  For 2D predictions the shape is
            ``(len_y, len_x, n_rays)``, for 3D predictions it is
            ``(len_z, len_y, len_x, n_rays)``.
        probs: 2- or 3-dimensional array representing the probabilities for each mask as
            outputed by a StarDist model.  For 2D predictions the shape is
            ``(len_y, len_x)``, for 3D predictions it is ``(len_z, len_y, len_x)``.
        rays: For 3D predictions `rays` must be set otherwise a ``ValueError`` is
            raised.  It should be the :class:`Rays_Base` instance used by the StarDist
            model.
        prob_thresh: Only masks with probability above this threshold are considered.
        grid: Should be of length 2 for 2D and of length 3 for 3D segmentation.
            This is the grid information about the subsampling occuring in the StarDist
            model.
        no_slicing: For very big and winded objects this should be set to ``True``.
            However, this might result in longer calculation times.
        max_full_overlaps: Maximum no. of full overlaps before current object is treated
            as finished.
        erase_probs_at_full_overlap: If set to ``True`` probs are set to -1 whenever
            a full overlap is detected.

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
    grid_array = np.array(grid, dtype=int)

    poly_to_label = get_poly_to_label(shape, rays)

    big_shape = tuple(s * g for s, g in zip(shape, grid))
    lbl = np.zeros(big_shape, dtype=np.uint16)

    # run max_dist before inflating dists
    max_dist = int(dists.max() * 2)
    # this could also be done with np.repeat, but for probs it is important that some
    # of the repeatet values are -1, as they should not be considered.
    new_probs = inflate_array(probs, grid, default_value=-1)
    points = inflate_array(points_from_grid(probs.shape, grid), grid, default_value=0)

    inds_thresh: npt.NDArray[np.bool_] = new_probs > prob_thresh
    sum_thresh = np.sum(inds_thresh)

    prob_sort = np.argsort(new_probs, axis=None)[::-1][:sum_thresh]

    if no_slicing:
        this_slice_point = no_slicing_slice_point
    else:
        this_slice_point = slice_point

    sorted_probs_j = 0
    current_id = 1
    while True:
        # In case this is always a view of new_probs that changes when new_probs changes
        # this line should be placed outside of this while-loop.
        newly_sorted_probs = np.take_along_axis(  # type: ignore [no-untyped-call]
            new_probs, prob_sort, axis=None
        )

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

        dists_ind = tuple(
            list(points[ind] // grid_array)
            + [
                slice(None),
            ]
        )
        new_shape = poly_to_label(dists[dists_ind], point, shape_paint) == 1

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
                poly_to_label(
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
                if erase_probs_at_full_overlap:
                    current_probs[additional_shape] = -1

        current_probs[new_shape] = -1
        new_probs[slices] = current_probs

        paint_in = lbl[slices]
        paint_in[new_shape] = current_id
        lbl[slices] = paint_in

        current_id += 1

    return lbl
