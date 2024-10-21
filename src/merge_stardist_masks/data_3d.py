"""Stardist 3D data generator for new weights and probability maps."""

from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.ndimage import zoom  # type: ignore [import-untyped]
from stardist.geometry import star_dist3D  # type: ignore [import-untyped]
from stardist.models.model3d import (  # type: ignore [import-untyped]
    StarDistData3D,
)
from stardist.sample_patches import (  # type: ignore [import-untyped]
    sample_patches,
)
from stardist.utils import edt_prob  # type: ignore [import-untyped]
from stardist.utils import mask_to_categorical

from .touching_pixels import bordering_gaussian_weights
from .touching_pixels import touching_pixels_3d


class OptimizedStarDistData3D(StarDistData3D):  # type: ignore [misc]
    """Overwrite __getitem__ function to use different prob and weights."""

    def __getitem__(self, i: int) -> Union[
        Tuple[
            Tuple[npt.NDArray[np.float32]],
            Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
        ],
        Tuple[
            Tuple[npt.NDArray[np.float32]],
            Tuple[
                npt.NDArray[np.float32],
                npt.NDArray[np.float32],
                npt.NDArray[np.float32],
            ],
        ],
    ]:
        """Return batch i as numpy array."""
        idx = self.batch(i)
        arrays = [
            sample_patches(
                (self.Y[k],) + self.channels_as_tuple(self.X[k]),
                patch_size=self.patch_size,
                n_samples=1,
                valid_inds=self.get_valid_inds(k),
            )
            for k in idx
        ]

        if self.n_channel is None:
            xs, ys = list(zip(*[(x[0], y[0]) for y, x in arrays]))
        else:
            xs, ys = list(
                zip(
                    *[
                        (np.stack([_x[0] for _x in x], axis=-1), y[0])
                        for y, *x in arrays
                    ]
                )
            )

        xs, ys = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(xs, ys))))

        if len(ys) == 1:
            xs_np = xs[0][np.newaxis]
        else:
            xs_np = np.stack(xs, out=self.out_X[: len(ys)])
        if xs_np.ndim == 4:  # input image has no channel axis
            xs_np = np.expand_dims(xs_np, -1)

        tmp_prob_ = [
            edt_prob(lbl, anisotropy=self.anisotropy)[self.ss_grid[1:]] for lbl in ys
        ]
        tmp_touching = [touching_pixels_3d(lbl[self.ss_grid[1:]]) for lbl in ys]
        tmp_touching_edt = [
            bordering_gaussian_weights(mask, lbl[self.ss_grid[1:]], sigma=2)
            for mask, lbl in zip(tmp_touching, ys)
        ]
        if len(ys) == 1:
            prob_ = tmp_prob_[0][np.newaxis]
            touching = tmp_touching[0][np.newaxis]
            touching_edt = tmp_touching_edt[0][np.newaxis]
        else:
            prob_ = np.stack(tmp_prob_, out=self.out_edt_prob[: len(ys)])
            touching = np.stack(tmp_touching)
            touching_edt = np.stack(tmp_touching_edt)

        prob = np.clip(prob_ - touching, 0, 1)
        dist_mask = prob_ + touching_edt

        tmp_dists = [
            star_dist3D(lbl, self.rays, mode=self.sd_mode, grid=self.grid) for lbl in ys
        ]
        if len(ys) == 1:
            dist = tmp_dists[0][np.newaxis]
        else:
            dist = np.stack(tmp_dists, out=self.out_star_dist3D[: len(ys)])

        prob_np: npt.NDArray[np.float32] = np.expand_dims(prob, -1)
        dist_mask = np.expand_dims(dist_mask, -1)

        # append dist_mask to dist as additional channel
        dist_np: npt.NDArray[np.float32] = np.concatenate([dist, dist_mask], axis=-1)

        if self.n_classes is None:
            return (xs_np,), (prob_np, dist_np)
        else:
            tmp = [
                mask_to_categorical(y, self.n_classes, self.classes[k])
                for y, k in zip(ys, idx)
            ]
            # TODO: downsample here before stacking?
            if len(ys) == 1:
                prob_class = tmp[0][np.newaxis]
            else:
                prob_class = np.stack(tmp, out=self.out_prob_class[: len(ys)])

            # TODO: investigate downsampling via simple indexing vs. using 'zoom'
            # prob_class = prob_class[self.ss_grid]
            # 'zoom' might lead to better registered maps (especially if upscaled later)
            prob_class_np: npt.NDArray[np.float32] = zoom(
                prob_class,
                (1,) + tuple(1 / g for g in self.grid) + (1,),
                order=0,
            )

            return (xs_np,), (prob_np, dist_np, prob_class_np)
