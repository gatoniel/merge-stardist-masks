"""Utility functions for testing."""
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt


# This is heavily inspired by
# https://github.com/stardist/stardist/blob/master/tests/utils.py#L51
def circle_image(
    shape: Tuple[int, ...] = (128, 128),
    radius: Optional[int] = None,
    center: Optional[Tuple[int, ...]] = None,
    eps: Optional[Tuple[int, ...]] = None,
    len_t: int = 3,
) -> npt.NDArray[np.uint16]:
    """Create a simple binary circle in center of image."""
    if center is None:
        center = (0,) * len(shape)
    if radius is None:
        radius = min(shape) // 4
    if eps is None:
        eps = (1,) * len(shape)
    assert len(shape) == len(eps)
    xs_ = tuple(np.arange(s) - s // 2 for s in shape)
    xs = np.meshgrid(*xs_, indexing="ij")

    imgs = []
    for i in range(len_t):
        r = np.sqrt(
            np.sum(
                [(x - c - i) ** 2 / _eps**2 for x, c, _eps in zip(xs, center, eps)],
                axis=0,
            )
        )
        imgs.append((r < radius).astype(np.uint16))
    return np.stack(imgs, axis=0)
