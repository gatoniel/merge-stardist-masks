"""Data generator for 2d time stacks based on stardist's data generators."""
from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .data_base import AugmenterSignature
from .data_base import StackedTimepointsDataBase
from .moments import lbl_to_local_descriptors
from .sample_patches import sample_patches
from .timeseries_2d import bordering_gaussian_weights_timeseries
from .timeseries_2d import edt_prob_timeseries
from .timeseries_2d import prepare_displacement_maps_timeseries
from .timeseries_2d import star_dist_timeseries
from .timeseries_2d import touching_pixels_2d_timeseries

T = TypeVar("T", bound=np.generic)


class OptimizedStackedTimepointsData2D(StackedTimepointsDataBase):
    """Uses better weights and stacked timepoints."""

    def __init__(
        self,
        xs: List[npt.NDArray[T]],
        ys: List[npt.NDArray[T]],
        batch_size: int,
        n_rays: int,
        length: int,
        n_classes: Optional[int] = None,
        classes: Optional[List[npt.NDArray[T]]] = None,
        use_gpu: bool = False,
        patch_size: Tuple[int, ...] = (256, 256),
        b: int = 32,
        grid: Tuple[int, ...] = (1, 1),
        shape_completion: bool = False,
        augmenter: Optional[AugmenterSignature[T]] = None,
        foreground_prob: int = 0,
        maxfilter_patch_size: Optional[int] = None,
        sample_ind_cache: bool = True,
    ) -> None:
        """Initialize with arrays of shape (size, T, Y, X, channels)."""
        super().__init__(
            xs=xs,
            ys=ys,
            n_rays=n_rays,
            grid=grid,
            n_classes=n_classes,
            classes=classes,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            augmenter=augmenter,
            foreground_prob=foreground_prob,
            maxfilter_patch_size=maxfilter_patch_size,
            use_gpu=use_gpu,
            sample_ind_cache=sample_ind_cache,
        )

        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            self.b = slice(None), slice(b, -b), slice(b, -b)
        else:
            self.b = slice(None), slice(None), slice(None)

        self.sd_mode = "opencl" if self.use_gpu else "cpp"

    def __getitem__(
        self, i: int
    ) -> Tuple[List[npt.NDArray[np.double]], List[npt.NDArray[np.double]]]:
        """Return batch i as numpy array."""
        idx = self.batch(i)
        arrays = [
            sample_patches(
                (self.ys[k],) + self.channels_as_tuple(self.xs[k]),
                patch_size=self.patch_size,
                n_samples=1,
                valid_inds=self.get_valid_inds(k),
            )
            for k in idx
        ]

        if self.n_channel is None:
            xs, ys = list(zip(*[(x[0][self.b], y[0]) for y, x in arrays]))
        else:
            xs, ys = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1)[self.b], y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs, ys))))

        return self.preprocess_distances(xs, ys)

    def preprocess_distances(
        self,
        xs: List[npt.NDArray[np.double]],
        ys: List[npt.NDArray[np.int_]],
    ) -> Tuple[List[npt.NDArray[np.double]], List[npt.NDArray[np.double]]]:
        """These routines are needed several times so they become their own function."""
        y_len_t = ys[0].shape[0]

        prob_ = np.stack([edt_prob_timeseries(lbl, self.b, self.ss_grid) for lbl in ys])
        touching = np.stack(
            [touching_pixels_2d_timeseries(lbl, self.b, self.ss_grid) for lbl in ys]
        )
        touching_edt = np.stack(
            [
                bordering_gaussian_weights_timeseries(
                    mask, lbl, sigma=2, b=self.b, ss_grid=self.ss_grid
                )
                for mask, lbl in zip(touching, ys)
            ]
        )
        prob = np.clip(prob_ - touching, 0, 1)
        dist_mask: npt.NDArray[np.double] = prob_ + touching_edt

        dists = np.stack(
            [
                star_dist_timeseries(
                    lbl, self.n_rays, mode=self.sd_mode, grid=self.grid
                )
                for lbl in ys
            ]
        )

        if xs[0].ndim == 3:
            xs = [
                np.expand_dims(x, axis=-1) for x in xs  # type: ignore [no-untyped-call]
            ]
        newxs = np.stack(
            [
                np.concatenate(  # type: ignore [no-untyped-call]
                    [x[i] for i in range(self.len_t)], axis=-1
                )
                for x in xs
            ]
        )

        # append dist_mask to dist as additional channel
        # dist_and_mask = np.concatenate([dist,dist_mask],axis=-1)
        # faster than concatenate
        dist_and_mask = np.empty(
            dists.shape[:-1] + (y_len_t * (self.n_rays + 1),), np.float32
        )
        dist_and_mask[..., :-y_len_t] = dists
        dist_and_mask[..., -y_len_t:] = dist_mask

        return [newxs], [prob, dist_and_mask]


class StackedTimepointsSimplifiedTrackingData2D(OptimizedStackedTimepointsData2D):
    """Adds tracking preprocessing to StarDist."""

    def __getitem__(
        self, i: int
    ) -> Tuple[List[npt.NDArray[np.double]], List[npt.NDArray[np.double]]]:
        """Return batch i as numpy array."""
        idx = self.batch(i)
        arrays = [
            sample_patches(
                (self.ys[k],) + self.channels_as_tuple(self.xs[k]),
                patch_size=self.patch_size,
                n_samples=1,
                valid_inds=self.get_valid_inds(k),
            )
            for k in idx
        ]

        if self.n_channel is None:
            xs, ys = list(zip(*[(x[0][self.b], y[0]) for y, x in arrays]))
        else:
            xs, ys = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1)[self.b], y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs, ys))))

        new_xs, prob_dist_mask = self.preprocess_distances(xs, ys)
        prob = prob_dist_mask[0]
        dist_and_mask = prob_dist_mask[1]

        displacement_maps_tracked = [
            prepare_displacement_maps_timeseries(lbl, b=self.b, ss_grid=self.ss_grid)
            for lbl in ys
        ]
        displacement_maps = np.stack(
            [maps_tracked[0] for maps_tracked in displacement_maps_tracked]
        )
        tracked_maps = np.stack(
            [maps_tracked[1] for maps_tracked in displacement_maps_tracked]
        )

        return new_xs, [prob, dist_and_mask, displacement_maps, tracked_maps]


class SimplifiedTrackingData2D(OptimizedStackedTimepointsData2D):
    """Adds tracking preprocessing to StarDist."""

    def __getitem__(
        self, i: int
    ) -> Tuple[List[npt.NDArray[np.double]], List[npt.NDArray[np.double]]]:
        """Return batch i as numpy array."""
        idx = self.batch(i)
        arrays = [
            sample_patches(
                (self.ys[k],) + self.channels_as_tuple(self.xs[k]),
                patch_size=self.patch_size,
                n_samples=1,
                valid_inds=self.get_valid_inds(k),
            )
            for k in idx
        ]

        if self.n_channel is None:
            xs, ys = list(zip(*[(x[0][self.b], y[0]) for y, x in arrays]))
        else:
            xs, ys = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1)[self.b], y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs, ys))))

        # Only use center y and its successor
        time_slice = slice(self.mid_t, self.mid_t + 2)
        ys = [y[time_slice, ...] for y in ys]

        new_xs, prob_dist_mask = self.preprocess_distances(xs, ys)
        prob = prob_dist_mask[0]
        dist_and_mask = prob_dist_mask[1]

        displacement_maps_tracked = [
            prepare_displacement_maps_timeseries(lbl, b=self.b, ss_grid=self.ss_grid)
            for lbl in ys
        ]
        displacement_maps = np.stack(
            [maps_tracked[0] for maps_tracked in displacement_maps_tracked]
        )
        tracked_maps = np.stack(
            [maps_tracked[1] for maps_tracked in displacement_maps_tracked]
        )

        return new_xs, [prob, dist_and_mask, displacement_maps, tracked_maps]


class SegmentationByDisplacementVectors(OptimizedStackedTimepointsData2D):
    """Adds tracking preprocessing to StarDist."""

    def __getitem__(
        self, i: int
    ) -> Tuple[List[npt.NDArray[np.double]], List[npt.NDArray[np.double]]]:
        """Return batch i as numpy array."""
        idx = self.batch(i)
        arrays = [
            sample_patches(
                (self.ys[k],) + self.channels_as_tuple(self.xs[k]),
                patch_size=self.patch_size,
                n_samples=1,
                valid_inds=self.get_valid_inds(k),
            )
            for k in idx
        ]

        if self.n_channel is None:
            xs, ys = list(zip(*[(x[0][self.b], y[0]) for y, x in arrays]))
        else:
            xs, ys = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1)[self.b], y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs, ys))))

        if xs[0].ndim == 3:
            xs = [
                np.expand_dims(x, axis=-1) for x in xs  # type: ignore [no-untyped-call]
            ]
        new_xs = np.stack(
            [
                np.concatenate(  # type: ignore [no-untyped-call]
                    [x[i] for i in range(self.len_t)], axis=-1
                )
                for x in xs
            ]
        )

        # Only use center y and its successor
        time_slice = slice(self.mid_t, self.mid_t + 2)
        ys = [y[time_slice, ...] for y in ys]

        slices = (1,) + self.b[1:]
        prob_descriptors = [
            lbl_to_local_descriptors(y[slices][self.ss_grid[1:3]]) for y in ys
        ]
        prob = np.stack([pd[..., :1] for pd in prob_descriptors])
        descriptors_mask = np.stack(
            [
                np.concatenate([pd[..., 1:], pd[..., :1]], axis=-1)
                for pd in prob_descriptors
            ]
        )

        displacement_maps_tracked = [
            prepare_displacement_maps_timeseries(lbl, b=self.b, ss_grid=self.ss_grid)
            for lbl in ys
        ]
        displacement_maps = np.stack(
            [maps_tracked[0] for maps_tracked in displacement_maps_tracked]
        )
        tracked_maps = np.stack(
            [maps_tracked[1] for maps_tracked in displacement_maps_tracked]
        )

        print(prob[0].shape)
        print(descriptors_mask[0].shape)

        return [new_xs], [prob, descriptors_mask, displacement_maps, tracked_maps]
