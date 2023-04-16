"""Calculate moments of objects in label images."""
import concurrent.futures
import multiprocessing
from functools import partial
from itertools import product
from typing import Dict

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy import ndimage  # tpye: ignore [import]
from scipy.spatial import KDTree

from merge_stardist_masks.utilities import LinearSearchIterator
from merge_stardist_masks.utilities import SubsampledSearchIterator


@njit
def calc_moments(
    lbl: npt.NDArray[np.int_], moments: npt.NDArray[np.uint16]
) -> Dict[int, npt.NDArray[np.double]]:
    """Calculate moments of objects in label image."""
    moments_dict = {}

    objects = ndimage.find_objects(lbl)
    for i, obj in enumerate(objects):
        if obj is None:
            continue

        lbl_id = i + 1

        offset = np.array([obj[i].start for i in range(lbl.ndim)])

        moments_dict[i + 1] = calc_moments_binary(lbl[obj] == lbl_id, offset)

    return moments_dict


def calc_moments_binary(
    lbl: npt.NDArray[np.bool], offset: npt.NDArray[np.int_]
) -> npt.NDArray[np.double]:
    """Calculate center point and global coordinates for boolean mask."""
    local_coordinates = np.argwhere(lbl)
    coordinates = local_coordinates + offset

    center_point = np.mean(coordinates, axis=0)

    # centralized_coordinates = coordinates - center_point

    # square_moments = np.mean(centralized_coordinates**2, axis=0)

    # return np.concatenate([center_point, square_moments]), coordinates
    return center_point, coordinates


def lbl_to_local_descriptors(lbl: npt.NDArray[np.int_]) -> npt.NDArray[np.double]:
    """Transfer label information into local descriptors, e.g., midpoints."""
    # descriptors = np.zeros(lbl.shape + (1 + 2 * lbl.ndim,), dtype=float)
    descriptors = np.zeros(lbl.shape + (1 + lbl.ndim,), dtype=float)
    # descriptors[..., 0] = lbl > 0

    objects = ndimage.find_objects(lbl)
    for i, obj in enumerate(objects):
        if obj is None:
            continue

        lbl_id = i + 1

        offset = np.array([obj[i].start for i in range(lbl.ndim)])

        mask = lbl[obj] == lbl_id
        moments, coordinates = calc_moments_binary(mask, offset)
        slices = tuple(coordinates[:, i] for i in range(lbl.ndim))

        descriptors[slices + (slice(1, None),)] = moments - coordinates
        descriptors[slices + (0,)] = 1.0

    # global_coordinates = np.argwhere(lbl > 0)
    # slices = tuple((global_coordinates[:, i] for i in range(lbl.ndim)))
    # descriptors[slices + (slice(1, 1 + lbl.ndim),)] -= global_coordinates

    return descriptors


def local_descriptors_to_lbl(descriptors):
    """Simple function to calculate label objects from description by offsets."""
    lbl = np.zeros((descriptors.shape[:-1]), dtype=np.uint16)

    global_coordinates = np.argwhere(descriptors[..., 0] > 0)

    slices = tuple(global_coordinates[:, i] for i in range(lbl.ndim))

    coords = descriptors[slices + (slice(1, lbl.ndim),)] + global_coordinates
    tree = KDTree(coords)

    lbl_id = 1
    iterator = LinearSearchIterator(coords.shape[0])
    for i in iterator:
        indices = tree.query_ball_point(coords[i], 1)

        iterator.set_false(indices)

        slices = tuple(global_coordinates[indices, j] for j in range(lbl.ndim))

        lbl[slices] = lbl_id
        lbl_id += 1

    return lbl


def get_values_between(values, bottom, top):
    """Calculate whether coordinates are within defined hyperrectangle."""
    return np.logical_and(
        (bottom < values).all(axis=1),
        (values < top).all(axis=1),
    )


def sliced_local_descriptors_to_lbl(descriptors, stride_len=11):
    """Simple function to calculate label objects from description by offsets."""
    # stride_len should be bigger than 5 for multiprocessing
    lbl = np.zeros((descriptors.shape[:-1]), dtype=np.uint16)

    global_coordinates = np.argwhere(descriptors[..., 0] > 0)

    slices = tuple(global_coordinates[:, i] for i in range(lbl.ndim)) + (
        slice(1, lbl.ndim + 1),
    )

    sliced_descriptors = descriptors[slices]
    center_points = sliced_descriptors + global_coordinates

    max_dist = np.ceil(np.max(sliced_descriptors, axis=0))
    double_dist = 2 * max_dist
    stride_dist = stride_len * max_dist

    start_points = [np.arange(0, lbl.shape[i], stride_dist[i]) for i in range(lbl.ndim)]

    all_indexer = np.ones(center_points.shape[0], dtype=bool)

    lbl_id = 1
    for small_start in product(*start_points):
        small_start = np.array(small_start)
        small_end = small_start + stride_dist

        big_start = small_start - double_dist
        big_end = small_end + double_dist

        local_coords_inds = get_values_between(global_coordinates, big_start, big_end)
        local_coords = global_coordinates[local_coords_inds, :]
        if local_coords.shape[0] == 0:
            continue

        between = get_values_between(local_coords, small_start, small_end)
        subsampled_coord_inds = np.argwhere(between)[:, 0]
        if subsampled_coord_inds.shape[0] == 0:
            continue

        local_center_points = center_points[local_coords_inds, :]
        tree = KDTree(local_center_points)

        iterator = SubsampledSearchIterator(
            local_coords.shape[0], subsampled_coord_inds
        )
        iterator.indexer[:] = all_indexer[local_coords_inds]
        for i in iterator:
            indices = tree.query_ball_point(local_center_points[i], 1)

            iterator.set_false(indices)

            slices = tuple(local_coords[indices, j] for j in range(lbl.ndim))

            lbl[slices] = lbl_id
            lbl_id += 1

        all_indexer[local_coords_inds] = iterator.indexer[:]

    return lbl


def sliced_local_descriptors_to_lbl_threaded(descriptors, threads=None, stride_len=11):
    """Simple function to calculate label objects from description by offsets."""
    # stride_len should be bigger than 5 for multiprocessing
    if threads is None:
        threads = multiprocessing.cpu_count()
    executor = concurrent.futures.ThreadPoolExecutor(threads)

    lbl = np.zeros((descriptors.shape[:-1]), dtype=np.uint16)

    global_coordinates = np.argwhere(descriptors[..., 0] > 0)

    slices = tuple(global_coordinates[:, i] for i in range(lbl.ndim)) + (
        slice(1, lbl.ndim + 1),
    )

    sliced_descriptors = descriptors[slices]
    center_points = sliced_descriptors + global_coordinates

    all_indexer = np.ones(center_points.shape[0], dtype=bool)

    max_dist = np.ceil(np.max(sliced_descriptors, axis=0))
    double_dist = 2 * max_dist
    stride_dist = stride_len * max_dist

    offsets = [(0, stride_dist[i]) for i in range(lbl.ndim)]

    lbl_id = 1
    for offset in product(*offsets):
        start_points = [
            np.arange(offset[i], lbl.shape[i], 2 * stride_dist[i])
            for i in range(lbl.ndim)
        ]

        fill_ = partial(
            fill,
            stride_dist,
            double_dist,
            center_points,
            global_coordinates,
            all_indexer,
        )

        results = map(fill_, [small_start for small_start in product(*start_points)])

        for result in results:
            for global_list in result:
                slices = tuple(
                    global_coordinates[global_list, j] for j in range(lbl.ndim)
                )
                lbl[slices] = lbl_id

                all_indexer[global_list] = False

                lbl_id += 1

    executor.shutdown()

    return lbl


def fill(
    stride_dist,
    double_dist,
    center_points,
    global_coordinates,
    all_indexer,
    small_start,
):
    """Utility function for threaded function above."""
    small_start = np.array(small_start)
    small_end = small_start + stride_dist

    big_start = small_start - double_dist
    big_end = small_end + double_dist

    local_coords_inds = get_values_between(global_coordinates, big_start, big_end)
    local_coords_argswhere = np.argwhere(local_coords_inds)[:, 0]
    local_coords = global_coordinates[local_coords_inds, :]
    if local_coords.shape[0] == 0:
        return []

    between = get_values_between(local_coords, small_start, small_end)
    subsampled_coord_inds = np.argwhere(between)[:, 0]
    if subsampled_coord_inds.shape[0] == 0:
        return []

    local_center_points = center_points[local_coords_inds, :]
    tree = KDTree(local_center_points)

    iterator = SubsampledSearchIterator(local_coords.shape[0], subsampled_coord_inds)
    iterator.indexer[:] = all_indexer[local_coords_inds]

    found_indices = []
    for i in iterator:
        indices = tree.query_ball_point(local_center_points[i], 1)

        iterator.set_false(indices)

        found_indices.append(local_coords_argswhere[indices])
    return found_indices
