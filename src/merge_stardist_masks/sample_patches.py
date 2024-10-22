"""Modification of stardist.sample_patches for timestacks."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt
from csbdeep.utils import _raise  # type: ignore [import-untyped]
from csbdeep.utils import choice


T = TypeVar("T", bound=np.generic)


def sample_patches(
    datas: tuple[npt.NDArray[T], ...],
    patch_size: tuple[int, ...],
    n_samples: int,
    valid_inds: tuple[npt.NDArray[np.uint32], ...] | None = None,
    verbose: bool = False,
) -> list[npt.NDArray[T]]:
    """Version of stardist.sample_patches.sample_patches for time stacks."""
    len(patch_size) == datas[0].ndim - 1 or _raise(ValueError())

    if not all(a.shape == datas[0].shape for a in datas):
        raise ValueError(
            "all input shapes must be the same: %s"
            % (" / ".join(str(a.shape) for a in datas))
        )

    if not all((0 < s <= d for s, d in zip(patch_size, datas[0].shape[1:]))):
        raise ValueError(
            "patch_size %s negative or larger than data shape %s along some dimensions"
            % (str(patch_size), str(datas[0].shape))
        )

    if valid_inds is None:
        valid_inds = tuple(
            _s.ravel()
            for _s in np.meshgrid(
                *tuple(
                    np.arange(p // 2, s - p // 2 + 1)
                    for s, p in zip(datas[0].shape[1:], patch_size)
                )
            )
        )

    n_valid = len(valid_inds[0])

    if n_valid == 0:
        raise ValueError("no regions to sample from!")

    idx = choice(range(n_valid), n_samples, replace=(n_valid < n_samples))
    rand_inds = [v[idx] for v in valid_inds]
    res = [
        np.stack(
            [
                data[
                    (slice(None),)
                    + tuple(
                        slice(_r - (_p // 2), _r + _p - (_p // 2))
                        for _r, _p in zip(r, patch_size)
                    )
                ]
                for r in zip(*rand_inds)
            ]
        )
        for data in datas
    ]

    return res
