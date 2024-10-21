"""Base data generator for time stacks based on stardist's data generators."""

from __future__ import annotations

import threading
import warnings
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np
import numpy.typing as npt
from csbdeep.internals.train import RollingSequence  # type: ignore [import-untyped]
from csbdeep.utils import _raise  # type: ignore [import-untyped]
from stardist.sample_patches import get_valid_inds  # type: ignore [import-untyped]

T = TypeVar("T", bound=np.generic)


AugmenterSignature = Callable[
    [npt.NDArray[T], npt.NDArray[T]], Tuple[npt.NDArray[T], ...]
]


def augmenter_identity(
    x: npt.NDArray[T], y: npt.NDArray[T]
) -> Tuple[npt.NDArray[T], ...]:
    """Simple identity function that returns input arguments as tuple."""
    return x, y


class StackedTimepointsDataBase(RollingSequence):  # type: ignore [misc]
    """This is a base class expecting (size, T, (Z,) Y, X, channels) arrays."""

    def __init__(
        self,
        xs: List[npt.NDArray[T]],
        ys: List[npt.NDArray[T]],
        n_rays: int,
        grid: Tuple[int, ...],
        batch_size: int,
        patch_size: Tuple[int, ...],
        length: int,
        n_classes: Optional[int] = None,
        classes: Optional[List[npt.NDArray[T]]] = None,
        use_gpu: bool = False,
        sample_ind_cache: bool = True,
        maxfilter_patch_size: Optional[int] = None,
        augmenter: Optional[AugmenterSignature[T]] = None,
        foreground_prob: int = 0,
        keras_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """Initialize with appropriately shaped arrays."""
        super().__init__(
            data_size=len(xs),
            batch_size=batch_size,
            length=length,
            shuffle=True,
            keras_kwargs=keras_kwargs,
        )
        n_classes is None or _raise(NotImplementedError("n_classes is not implemented"))

        if isinstance(xs, (np.ndarray, tuple, list)):
            xs = [x.astype(np.float32, copy=False) for x in xs]

        # sanity checks
        len(xs) == len(ys) and len(xs) > 0 or _raise(
            ValueError("xs and ys can't be empty and must have same length")
        )

        if classes is None:
            # set classes to None for all images
            # (i.e. defaults to every object instance assigned the same class)
            classes = (None,) * len(xs)  # type: ignore [assignment]
        else:
            n_classes is not None or warnings.warn(
                "Ignoring classes since n_classes is None", stacklevel=2
            )

        len(classes) == len(xs) or _raise(  # type: ignore [arg-type]
            ValueError("xs and classes must have same length")
        )

        self.n_classes, self.classes = n_classes, classes

        patch_ndim = len(patch_size)
        patch_ndim in (2, 3) or _raise(
            ValueError("patch_size should have dimension 2 or 3.")
        )

        len_t = xs[0].shape[0]

        x_ndim = xs[0].ndim
        # time axes is needed and optionally a axes for channels
        x_ndim in (patch_ndim + 1, patch_ndim + 2) or _raise(
            ValueError("Dimensions of patch_size and input images are not compatible.")
        )

        if x_ndim == patch_ndim + 1:
            xs = [np.expand_dims(x, axis=-1) for x in xs]
            x_ndim = xs[0].ndim

        if isinstance(xs, (np.ndarray, tuple, list)) and isinstance(
            ys, (np.ndarray, tuple, list)
        ):
            all(
                y.ndim == patch_ndim + 1
                and x.ndim == x_ndim
                and x.shape[: patch_ndim + 1] == y.shape
                for x, y in zip(xs, ys)
            ) or _raise(
                ValueError(
                    "images and masks should have corresponding shapes/dimensions"
                )
            )
            all(x.shape[0] == len_t for x in xs) or _raise(
                ValueError(
                    "First axes of images corresponds to time and has always same size."
                )
            )
            all(x.shape[1 : patch_ndim + 1] >= tuple(patch_size) for x in xs) or _raise(
                ValueError(
                    f"Some images are too small for given patch_size {patch_size}"
                )
            )

        if x_ndim == patch_ndim + 1:
            self.n_channel = None
        else:
            self.n_channel = xs[0].shape[-1]
            if isinstance(xs, (np.ndarray, tuple, list)):
                all(x.shape[-1] == self.n_channel for x in xs) or _raise(
                    ValueError("Not all input images have same number of channels.")
                )

        0 <= foreground_prob <= 1 or _raise(
            ValueError("foreground_prob is not in 0...1")
        )

        self.xs, self.ys = xs, ys
        # self.batch_size = batch_size
        self.n_rays = n_rays
        self.patch_size = patch_size
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in grid)
        self.grid = tuple(grid)
        self.use_gpu = bool(use_gpu)
        if augmenter is None:
            augmenter = augmenter_identity
        self.augmenter = augmenter
        self.foreground_prob = foreground_prob
        self.len_t = len_t
        self.mid_t = len_t // 2

        if self.use_gpu:
            from gputools import max_filter  # type: ignore [import-not-found]

            self.max_filter = lambda y, patch_size: max_filter(
                y.astype(np.float32), patch_size
            )
        else:
            from scipy.ndimage import maximum_filter  # type: ignore [import-untyped]

            self.max_filter = lambda y, patch_size: maximum_filter(
                y, patch_size, mode="constant"
            )

        self.maxfilter_patch_size = (
            maxfilter_patch_size
            if maxfilter_patch_size is not None
            else self.patch_size
        )

        self.sample_ind_cache = sample_ind_cache
        self._ind_cache_fg: Dict[int, Tuple[npt.NDArray[np.uint32], ...]] = {}
        self._ind_cache_all: Dict[int, Tuple[npt.NDArray[np.uint32], ...]] = {}
        self.lock = threading.Lock()

    def get_valid_inds(
        self, k: int, foreground_prob: Optional[int] = None
    ) -> Tuple[npt.NDArray[np.uint32], ...]:
        """Find indices as centers that have labels around them."""
        if foreground_prob is None:
            foreground_prob = self.foreground_prob
        foreground_only = np.random.uniform() < foreground_prob
        _ind_cache = self._ind_cache_fg if foreground_only else self._ind_cache_all
        if k in _ind_cache:
            inds = _ind_cache[k]
        else:
            patch_filter = (
                (
                    lambda y, p: self.max_filter(  # type: ignore [no-untyped-call]
                        y, self.maxfilter_patch_size
                    )
                    > 0
                )
                if foreground_only
                else None
            )
            inds = get_valid_inds(
                self.ys[k][self.mid_t],
                self.patch_size,
                patch_filter=patch_filter,
            )
            if self.sample_ind_cache:
                with self.lock:
                    _ind_cache[k] = inds
        if foreground_only and len(inds[0]) == 0:
            # no foreground pixels available
            return self.get_valid_inds(k, foreground_prob=0)
        return inds

    def channels_as_tuple(self, x: npt.NDArray[T]) -> Tuple[npt.NDArray[T], ...]:
        """Return each channel as separate array in tuple."""
        if self.n_channel is None:
            return (x,)
        else:
            return tuple(x[..., i] for i in range(self.n_channel))
