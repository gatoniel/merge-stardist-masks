"""Data generator for 2d time stacks based on stardist's data generators."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .data_base import AugmenterSignature
from .data_base import StackedTimepointsDataBase
from .sample_patches import sample_patches
from .timeseries_2d import bordering_gaussian_weights_timeseries
from .timeseries_2d import edt_prob_timeseries
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
        keras_kwargs: Optional[Dict[Any, Any]] = None,
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
            keras_kwargs=keras_kwargs,
        )

        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            self.b = slice(None), slice(b, -b), slice(b, -b)
        else:
            self.b = slice(None), slice(None), slice(None)

        self.sd_mode = "opencl" if self.use_gpu else "cpp"

    def __getitem__(self, i: int) -> Tuple[
        Tuple[npt.NDArray[np.float32]],
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
    ]:
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
            xs_, ys_ = list(zip(*[(x[0][self.b], y[0]) for y, x in arrays]))
        else:
            xs_, ys_ = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1)[self.b], y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs_, ys_))))

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
        dist_mask: npt.NDArray[np.float32] = prob_ + touching_edt

        dists = np.stack(
            [
                star_dist_timeseries(
                    lbl, self.n_rays, mode=self.sd_mode, grid=self.grid
                )
                for lbl in ys
            ]
        )

        if xs[0].ndim == 3:
            xs = tuple(np.expand_dims(x, axis=-1) for x in xs)
        xs_np = np.stack(
            [np.concatenate([x[i] for i in range(self.len_t)], axis=-1) for x in xs]
        )

        # append dist_mask to dist as additional channel
        # dist_and_mask = np.concatenate([dist,dist_mask],axis=-1)
        # faster than concatenate
        dist_and_mask = np.empty(
            dists.shape[:-1] + (self.len_t * (self.n_rays + 1),), np.float32
        )
        dist_and_mask[..., : -self.len_t] = dists
        dist_and_mask[..., -self.len_t :] = dist_mask

        return (xs_np,), (prob, dist_and_mask)
