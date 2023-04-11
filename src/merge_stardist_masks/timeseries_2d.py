"""Utility functions to preprocess timeseries label images."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
from stardist.geometry import star_dist  # type: ignore [import]
from stardist.utils import edt_prob  # type: ignore [import]

from .touching_pixels import bordering_gaussian_weights
from .touching_pixels import touching_pixels_2d
from .tracking import prepare_displacement_map_single


def star_dist_timeseries(
    lbl: npt.NDArray[np.int_], n_rays: int, mode: str, grid: Tuple[int, ...]
) -> npt.NDArray[np.int_]:
    """Calculate star_dist distances on each timepoint individually."""
    conc: npt.NDArray[np.int_] = np.concatenate(  # type: ignore [no-untyped-call]
        [
            star_dist(lbl[i, ...], n_rays, mode=mode, grid=grid)
            for i in range(lbl.shape[0])
        ],
        axis=-1,
    )
    return conc


def subsample(
    lbl: npt.NDArray[np.int_], i: int, b: Tuple[slice, ...], ss_grid: Tuple[slice, ...]
) -> npt.NDArray[np.int_]:
    """Convenience function to subsample all the grids."""
    subsampled: npt.NDArray[np.int_] = lbl[(i,) + b[1:]][ss_grid[1:3]]
    return subsampled


def bordering_gaussian_weights_timeseries(
    mask: npt.NDArray[np.bool_],
    lbl: npt.NDArray[np.int_],
    sigma: int,
    b: Tuple[slice, ...],
    ss_grid: Tuple[slice, ...],
) -> npt.NDArray[np.double]:
    """Calculate weights for each timepoint individually."""
    return np.stack(
        [
            bordering_gaussian_weights(
                mask[..., i], subsample(lbl, i, b, ss_grid), sigma=sigma
            )
            for i in range(lbl.shape[0])
        ],
        axis=-1,
    )


def touching_pixels_2d_timeseries(
    lbl: npt.NDArray[np.int_], b: Tuple[slice, ...], ss_grid: Tuple[slice, ...]
) -> npt.NDArray[np.bool_]:
    """Calculate touching_pixels_2d individually on each timepoint."""
    return np.stack(
        [
            touching_pixels_2d(subsample(lbl, i, b, ss_grid))
            for i in range(lbl.shape[0])
        ],
        axis=-1,
    )


def edt_prob_timeseries(
    lbl: npt.NDArray[np.int_], b: Tuple[slice, ...], ss_grid: Tuple[slice, ...]
) -> npt.NDArray[np.double]:
    """Calculate edt_prob individually on each timepoint."""
    return np.stack(
        [edt_prob(subsample(lbl, i, b, ss_grid)) for i in range(lbl.shape[0])], axis=-1
    )


def prepare_displacement_maps_timeseries(
    lbl: npt.NDArray[np.int_], b: Tuple[slice, ...], ss_grid: Tuple[slice, ...]
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Calculate displacement map for each timepoint individually and stack."""
    maps = [
        prepare_displacement_map_single(
            subsample(lbl, i, b, ss_grid), subsample(lbl, i + 1, b, ss_grid)
        )
        for i in range(lbl.shape[0] - 1)
    ]
    displacement = np.concatenate(  # type: ignore [no-untyped-call]
        [map_[..., :2] for map_ in maps], axis=-1
    )
    tracked = np.stack([map_[..., -1] for map_ in maps], axis=-1)
    return displacement, tracked
