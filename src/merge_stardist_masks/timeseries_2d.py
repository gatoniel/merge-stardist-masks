"""Utility functions to preprocess timeseries label images."""
from __future__ import annotations

from typing import Any
from typing import Tuple
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from stardist.geometry import star_dist  # type: ignore [import]
from stardist.utils import edt_prob  # type: ignore [import]

from .touching_pixels import bordering_gaussian_weights
from .touching_pixels import touching_pixels_2d


T = TypeVar("T", bound=np.generic)
U = TypeVar("U", bound=np.unsignedinteger[Any])
F = TypeVar("F", bound=np.floating[Any])


def star_dist_timeseries(
    lbl: npt.NDArray[U], n_rays: int, mode: str, grid: Tuple[int, ...]
) -> npt.NDArray[np.single]:
    """Calculate star_dist distances on each timepoint individually."""
    return np.concatenate(
        [
            star_dist(lbl[i, ...], n_rays, mode=mode, grid=grid)
            for i in range(lbl.shape[0])
        ],
        axis=-1,
    )


def subsample(
    lbl: npt.NDArray[T],
    i: int,
    b: Tuple[slice, ...],
    ss_grid: Tuple[slice, ...],
) -> npt.NDArray[T]:
    """Convenience function to subsample all the grids."""
    return lbl[(i,) + b[1:]][ss_grid[1:3]]


def bordering_gaussian_weights_timeseries(
    mask: npt.NDArray[np.bool_],
    lbl: npt.NDArray[U],
    sigma: int,
    b: Tuple[slice, ...],
    ss_grid: Tuple[slice, ...],
) -> npt.NDArray[np.single]:
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
    lbl: npt.NDArray[U], b: Tuple[slice, ...], ss_grid: Tuple[slice, ...]
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
    lbl: npt.NDArray[U], b: Tuple[slice, ...], ss_grid: Tuple[slice, ...]
) -> npt.NDArray[np.single]:
    """Calculate edt_prob individually on each timepoint."""
    return np.stack(
        [edt_prob(subsample(lbl, i, b, ss_grid)) for i in range(lbl.shape[0])],
        axis=-1,
    )
