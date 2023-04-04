"""Pre- and postprocessing for tracking by displacement maps."""
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore [import]
from scipy.optimize import linear_sum_assignment  # type: ignore [import]
from scipy.spatial import distance_matrix  # type: ignore [import]

from .naive_fusion import mesh_from_shape


def calc_midpoints(lbl: npt.NDArray[np.int_]) -> Dict[int, npt.NDArray[np.double]]:
    """Calculate center points of indidual objects in label map."""
    objects = ndimage.find_objects(lbl)
    midpoints = {}
    for i, obj in enumerate(objects):
        if obj is None:
            continue
        lbl_id = i + 1
        tmp = lbl[obj] == lbl_id

        indices = np.argwhere(tmp)
        offset = np.array([obj[i].start for i in range(lbl.ndim)])

        midpoints[lbl_id] = np.mean(indices, axis=0) + offset

    return midpoints


# def calc_midpoint_distances(
#     midpoints_t0: Dict[int, npt.NDArray[np.double]],
#     midpoints_t1: Dict[int, npt.NDArray[np.double]],
# ) -> Dict[int, npt.NDArray[np.double]]:
#     """Calculate the vector pointing from midpoint_t1 to midpoint t0."""
#     distances: Dict[int, npt.NDArray[np.double]] = {}
#     for id_t1, midpoint_t1 in midpoints_t1.items():
#         try:
#             distances[id_t1] = midpoints_t0[id_t1] - midpoint_t1
#         except KeyError:
#             pass
#     return distances


def prepare_displacement_map_single(
    lbl_t0: npt.NDArray[np.int_], lbl_t1: npt.NDArray[np.int_]
) -> npt.NDArray[np.double]:
    """Calculate displacement map between individual timepoints."""
    midpoints_t0 = calc_midpoints(lbl_t0)

    ndim = lbl_t0.ndim
    displacement_map = np.zeros((lbl_t0.shape) + (ndim + 1,), dtype=float)
    coordinates_t1 = mesh_from_shape(lbl_t0.shape)

    for id_t1 in np.unique(lbl_t1):  # type: ignore [no-untyped-call]
        if id_t1 == 0:
            continue
        try:
            midpoint_t0 = midpoints_t0[id_t1]
        except KeyError:
            continue
        inds_t1 = lbl_t1 == id_t1
        displacement_map[inds_t1, :ndim] = midpoint_t0 - coordinates_t1[inds_t1, :]
        displacement_map[inds_t1, -1] = 1.0

    return displacement_map


def prepare_displacement_maps(lbl: npt.NDArray[np.int_]) -> npt.NDArray[np.double]:
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


def track_from_displacement_map_single_timepoint(
    lbl_t0: npt.NDArray[np.int_],
    lbl_t1: npt.NDArray[np.int_],
    displacement_map: npt.NDArray[np.double],
    points: npt.NDArray[np.int_],
    threshold: float = 0.5,
) -> npt.NDArray[np.int_]:
    """Track labels from one timepoint to next timepoint based on displacement map."""
    tracked_ids = get_tracked_ids(lbl_t1, displacement_map, points, threshold=threshold)

    midpoints_t0 = calc_midpoints(lbl_t0)

    midpoints_t0_, lbl_ids0_ = dict_to_array_indices(midpoints_t0)
    orig_midpoints_t1_, lbl_ids1_ = dict_to_array_indices(tracked_ids)
    dist_matrix = distance_matrix(midpoints_t0_, orig_midpoints_t1_)

    inds0, inds1 = linear_sum_assignment(dist_matrix)
    lbl_ids0, lbl_ids1 = lbl_ids0_[inds0], lbl_ids1_[inds1]

    tracked_t1s = np.arange(1, lbl_t1.max() + 1)

    new_t1 = np.zeros_like(lbl_t1)
    for t0_ind, t1_ind in zip(lbl_ids0, lbl_ids1):
        new_t1[lbl_t1 == t1_ind] = t0_ind
        tracked_t1s[t1_ind - 1] = 0

    not_tracked_t1s = tracked_t1s[tracked_t1s != 0]
    new_max_value = new_t1.max() + 1

    i = 0
    for t1_ind in not_tracked_t1s:
        inds = lbl_t1 == t1_ind
        if np.any(inds):
            new_t1[inds] = new_max_value + i
            i += 1

    return new_t1


def only_track_from_displacement_map_single_timestep(
    lbl_t0: npt.NDArray[np.int_],
    lbl_t1: npt.NDArray[np.int_],
    displacement_map: npt.NDArray[np.double],
    points: npt.NDArray[np.int_],
    threshold: float = 0.5,
) -> Dict[int, int]:
    """Find nearest object in previous timepoint for each object."""
    tracked_ids = get_tracked_ids(lbl_t1, displacement_map, points, threshold=threshold)

    midpoints_t0 = calc_midpoints(lbl_t0)

    midpoints_t0_, lbl_ids0 = dict_to_array_indices(midpoints_t0)
    orig_midpoints_t1_, lbl_ids1 = dict_to_array_indices(tracked_ids)
    dist_matrix = distance_matrix(midpoints_t0_, orig_midpoints_t1_)

    inds_t1 = np.argmin(dist_matrix, axis=0)

    return {id_t1: id_t0 for id_t1, id_t0 in zip(lbl_ids1, lbl_ids0[inds_t1])}


def only_track_from_displacement_map_multiple_timesteps(
    lbls: npt.NDArray[np.int_],
    displacement_maps: npt.NDArray[np.double],
    threshold: float = 0.5,
) -> List[Tuple[int, List[int]]]:
    """Creates lists of label ids that resemble tracks through timeseries."""
    bw_dicts = []
    points = mesh_from_shape(lbls.shape[1:])
    for i in range(1, lbls.shape[0]):
        bw_dicts.append(
            only_track_from_displacement_map_single_timestep(
                lbls[i - 1],
                lbls[i],
                displacement_maps[i - 1],
                points,
                threshold=threshold,
            )
        )

    tracks = [(0, [val, key]) for key, val in bw_dicts[0].items()]
    for i, bw_dict in enumerate(bw_dicts):
        if i == 0:
            continue
        for key, val in bw_dict.items():
            found = False
            for track_start_time, track in tracks:
                current_track_time = track_start_time + len(track) - 1
                if track[-1] == val and current_track_time == i:
                    track.append(key)
                    found = True
            if not found:
                tracks.append((i, [val, key]))
    return tracks


def track_array_from_tracking_list(
    lbls: npt.NDArray[np.int_],
    tracks: List[Tuple[int, List[int]]],
) -> npt.NDArray[np.double]:
    """Create an array describing tracks in the napari format."""
    midpoints = [calc_midpoints(lbls[i]) for i in range(lbls.shape[0])]

    track_list = []
    for i in range(len(tracks)):
        start_time, track = tracks[i]
        for j, lbl_id in enumerate(track):
            time = start_time + j
            midpoint = midpoints[time][lbl_id]

            track_list.append((i, time) + tuple(midpoint))
    return np.array(track_list)


def bw_dict_to_fw_dict(bw_dict: Dict[int, int]) -> Dict[int, List[int]]:
    """Invert a backward pointing tracking dict into a forward pointing one."""
    fw_dict = {}
    for key, val in bw_dict:
        if val not in fw_dict:
            fw_dict[val] = [key]
        else:
            fw_dict[val].append(key)
    return fw_dict


def dict_to_array_indices(
    d: Dict[int, npt.NDArray[np.double]]
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.int_]]:
    """Return two arrays from a dict containing the values and keys."""
    array = np.array([v for v in d.values()])
    inds = np.array([k for k in d.keys()])
    return array, inds


def get_tracked_ids(
    lbl: npt.NDArray[np.int_],
    displacement_map: npt.NDArray[np.double],
    points: npt.NDArray[np.int_],
    threshold: float = 0.5,
) -> Dict[int, npt.NDArray[np.double]]:
    """Get label ids whose objects were tracked from the previous timepoint."""
    tracked = {}
    for lbl_id in np.unique(lbl):  # type: ignore [no-untyped-call]
        if lbl_id == 0:
            continue
        inds_lbl = lbl == lbl_id
        mean = displacement_map[inds_lbl, -1].mean()
        if mean > threshold:
            tracked[lbl_id] = np.mean(
                displacement_map[inds_lbl, :-1] + points[inds_lbl],
                axis=0,
            )

    return tracked


def track_from_displacement_map(
    lbls: npt.NDArray[np.int_],
    displacement_maps: npt.NDArray[np.double],
    threshold: float = 0.5,
) -> npt.NDArray[np.int_]:
    """Iteratively track objects from one timepoint to the next in many timepoints."""
    newlbls: npt.NDArray[np.int_] = np.copy(lbls)  # type: ignore [no-untyped-call]
    points = mesh_from_shape(lbls.shape[1:])
    for i in range(1, newlbls.shape[0]):
        newlbls[i] = track_from_displacement_map_single_timepoint(
            newlbls[i - 1],
            newlbls[i],
            displacement_maps[i - 1],
            points,
            threshold=threshold,
        )
    return newlbls
