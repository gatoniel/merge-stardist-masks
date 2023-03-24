"""Pre- and postprocessing for tracking by displacement maps."""
from typing import Dict

import numpy as np
import numpy.typing as npt
from scipy import ndimage


def calc_midpoints(lbl: npt.NDArray[float]) -> Dict[int, npt.NDArray[float]]:
    """Calculate center points of indidual objects in label map."""
    objects = ndimage.find_objects(lbl)
    midpoints = {}
    for i, obj in enumerate(objects):
        if obj is None:
            continue
        tmp = lbl[obj]

        indices = np.argwhere(tmp)
        offset = np.array([obj[i].start for i in range(lbl.ndim)])

        midpoints[i + 1] = np.mean(indices, axis=0) + offset

    return midpoints


def calc_midpoint_distances(
    midpoints_t0: Dict[int, npt.NDArray[float]],
    midpoints_t1: Dict[int, npt.NDArray[float]],
) -> Dict[int, npt.NDArray[float]]:
    """Calculate the vector pointing from midpoint_t1 to midpoint t0."""
    distances = {}
    for id_t1, midpoint_t1 in midpoints_t1.items():
        try:
            distances[id_t1] = midpoints_t0[id_t1] - midpoint_t1
        except KeyError:
            pass
    return distances


def prepare_displacement_map_single(
    lbl_t0: npt.NDArray[int], lbl_t1: npt.NDArray[int]
) -> npt.NDArray[float]:
    """Calculate displacement map between individual timepoints."""
    midpoints_t0 = calc_midpoints(lbl_t0)
    midpoints_t1 = calc_midpoints(lbl_t1)

    distances = calc_midpoint_distances(midpoints_t0, midpoints_t1)

    ndim = lbl_t0.ndim
    displacement_map = np.zeros((lbl_t0.shape) + (ndim + 1,), dtype=float)

    for id_t1, distance in distances.items():
        inds_t1 = lbl_t1 == id_t1
        displacement_map[inds_t1, :ndim] = distance
        displacement_map[inds_t1, -1] = 1.0

    return displacement_map


def prepare_displacement_maps(lbl: npt.NDArray[int]) -> npt.NDArray[float]:
    """Calculate all displacement maps over several timepoints.

    lbl: Array of shape (T, Y, X) or (T, Z, Y, X)
    """
    displacement_len = lbl.shape[0] - 1
    displacement_maps = np.zeros(
        # lbl.ndim alone is enough as it is already +1 due to the time dimension
        (displacement_len,) + lbl.shape[1:] + (lbl.ndim,),
        dtype=float,
    )

    for i in range(displacement_len):
        displacement_maps[i, ...] = prepare_displacement_map_single(lbl[i], lbl[i + 1])

    return displacement_maps
