"""Stardist 2D model modified for stacked timepoints."""

from __future__ import annotations

import warnings
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import tensorflow as tf  # type: ignore [import-untyped]
from csbdeep.data import Normalizer  # type: ignore [import-untyped]
from csbdeep.internals.blocks import unet_block  # type: ignore [import-untyped]
from csbdeep.internals.predict import tile_iterator  # type: ignore [import-untyped]
from csbdeep.internals.predict import total_n_tiles
from csbdeep.utils import _raise  # type: ignore [import-untyped]
from csbdeep.utils import axes_dict
from csbdeep.utils.tf import BACKEND as K  # type: ignore [import-untyped]
from csbdeep.utils.tf import CARETensorBoard
from csbdeep.utils.tf import IS_TF_1
from csbdeep.utils.tf import keras_import
from stardist.models import StarDist2D  # type: ignore [import-untyped]
from stardist.models.base import kld  # type: ignore [import-untyped]
from stardist.models.base import masked_loss_iou
from stardist.models.base import masked_loss_mae
from stardist.models.base import masked_loss_mse
from stardist.models.base import masked_metric_iou
from stardist.models.base import masked_metric_mae
from stardist.models.base import masked_metric_mse
from stardist.models.base import StarDistPadAndCropResizer
from stardist.models.base import weighted_categorical_crossentropy
from stardist.utils import _is_floatarray  # type: ignore [import-untyped]
from tensorflow.python.framework.ops import EagerTensor  # type: ignore [import-untyped]
from tqdm import tqdm  # type: ignore [import-untyped]

from .config_2d import StackedTimepointsConfig2D
from .data_2d import OptimizedStackedTimepointsData2D
from .data_base import AugmenterSignature
from .timeseries_helpers import timeseries_to_batch

# from stardist.utils import _is_power_of_2
T = TypeVar("T", bound=np.generic)


Input, Conv2D, MaxPooling2D = keras_import("layers", "Input", "Conv2D", "MaxPooling2D")
Adam = keras_import("optimizers", "Adam")
ReduceLROnPlateau, TensorBoard = keras_import(
    "callbacks", "ReduceLROnPlateau", "TensorBoard"
)
Model = keras_import("models", "Model")


class OptimizedStackedTimepointsModel2D(StarDist2D):  # type: ignore [misc]
    """Stardist model for stacked timepoints by overwriting the relevant functions."""

    def _build(self):  # type: ignore [no-untyped-def]
        """Has to be overwritten as the outputs slightly differ."""
        self.config.backbone == "unet" or _raise(NotImplementedError())
        unet_kwargs = {
            k[len("unet_") :]: v
            for (k, v) in vars(self.config).items()
            if k.startswith("unet_")
        }

        input_img = Input(self.config.net_input_shape, name="input")

        # maxpool input image to grid size
        pooled = np.array([1, 1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv2D(
                    self.config.unet_n_filter_base,
                    self.config.unet_kernel_size,
                    padding="same",
                    activation=self.config.unet_activation,
                )(pooled_img)
            pooled_img = MaxPooling2D(pool)(pooled_img)

        unet_base = unet_block(**unet_kwargs)(pooled_img)

        if self.config.net_conv_after_unet > 0:
            unet = Conv2D(
                self.config.net_conv_after_unet,
                self.config.unet_kernel_size,
                name="features",
                padding="same",
                activation=self.config.unet_activation,
            )(unet_base)
        else:
            unet = unet_base

        output_prob = Conv2D(
            self.config.len_t,
            (1, 1),
            name="prob",
            padding="same",
            activation="sigmoid",
        )(unet)
        output_dist = Conv2D(
            self.config.n_rays * self.config.len_t,
            (1, 1),
            name="dist",
            padding="same",
            activation="linear",
        )(unet)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class = Conv2D(
                    self.config.net_conv_after_unet,
                    self.config.unet_kernel_size,
                    name="features_class",
                    padding="same",
                    activation=self.config.unet_activation,
                )(unet_base)
            else:
                unet_class = unet_base

            output_prob_class = Conv2D(
                self.config.n_classes + 1,
                (1, 1),
                name="prob_class",
                padding="same",
                activation="softmax",
            )(unet_class)
            return Model([input_img], [output_prob, output_dist, output_prob_class])
        else:
            return Model([input_img], [output_prob, output_dist])

    def prepare_for_training(self, optimizer: Optional[Any] = None) -> None:
        """Method from base class needs to be overwritten for slightly adapted loss."""
        if optimizer is None:
            optimizer = Adam(self.config.train_learning_rate)

        masked_dist_loss = {
            "mse": masked_loss_mse,
            "mae": masked_loss_mae,
            "iou": masked_loss_iou,
        }[self.config.train_dist_loss]
        prob_loss = "binary_crossentropy"

        self.num_or_size_splits = [
            self.config.n_rays for _ in range(self.config.len_t)
        ] + [1 for _ in range(self.config.len_t)]
        self.num_or_size_splits_pred = self.num_or_size_splits[: self.config.len_t]

        def split_dist_maps(
            dist_true_mask: EagerTensor, dist_pred: EagerTensor
        ) -> Tuple[List[EagerTensor], List[EagerTensor]]:
            return tf.split(
                dist_true_mask,
                num_or_size_splits=self.num_or_size_splits,
                axis=-1,
            ), tf.split(
                dist_pred,
                num_or_size_splits=self.num_or_size_splits_pred,
                axis=-1,
            )

        def dist_loss(
            dist_true_mask: EagerTensor, dist_pred: EagerTensor
        ) -> EagerTensor:
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_dist_loss(
                            true_splits[i + self.config.len_t],
                            reg_weight=self.config.train_background_reg,
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        def dist_iou_metric(
            dist_true_mask: EagerTensor, dist_pred: EagerTensor
        ) -> EagerTensor:
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_metric_iou(
                            true_splits[i + self.config.len_t],
                            reg_weight=0,
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        def relevant_mae(
            dist_true_mask: EagerTensor, dist_pred: EagerTensor
        ) -> EagerTensor:
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_metric_mae(
                            true_splits[i + self.config.len_t],
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        def relevant_mse(
            dist_true_mask: EagerTensor, dist_pred: EagerTensor
        ) -> EagerTensor:
            true_splits, pred_splits = split_dist_maps(dist_true_mask, dist_pred)
            return K.mean(
                tf.stack(
                    [
                        masked_metric_mse(
                            true_splits[i + self.config.len_t],
                        )(true_splits[i], pred_splits[i])
                        for i in range(self.config.len_t)
                    ]
                )
            )

        if self._is_multiclass():
            prob_class_loss = weighted_categorical_crossentropy(
                self.config.train_class_weights, ndim=self.config.n_dim
            )
            loss = [prob_loss, dist_loss, prob_class_loss]
        else:
            loss = [prob_loss, dist_loss]

        self.keras_model.compile(
            optimizer,
            loss=loss,
            loss_weights=list(self.config.train_loss_weights),
            metrics={
                "prob": kld,
                "dist": [relevant_mae, relevant_mse, dist_iou_metric],
            },
        )

        self.callbacks = []
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                if IS_TF_1:
                    self.callbacks.append(
                        CARETensorBoard(
                            log_dir=str(self.logdir),
                            prefix_with_timestamp=False,
                            n_images=3,
                            write_images=True,
                            prob_out=False,
                        )
                    )
                else:
                    self.callbacks.append(
                        TensorBoard(
                            log_dir=str(self.logdir / "logs"),
                            write_graph=False,
                            profile_batch=0,
                        )
                    )

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if "verbose" not in rlrop_params:
                rlrop_params["verbose"] = True
            # TF2: add as first callback to put 'lr' in the logs for TensorBoard
            self.callbacks.insert(0, ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True

    def train(
        self,
        x: List[npt.NDArray[np.double]],
        y: List[npt.NDArray[np.uint16]],
        validation_data: Tuple[
            List[npt.NDArray[np.double]], List[npt.NDArray[np.uint16]]
        ],
        classes: str = "auto",
        augmenter: Optional[AugmenterSignature[T]] = None,
        seed: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        workers: int = 1,
    ) -> tf.keras.callbacks.History:
        """Monkey patch the original StarDistData2D generator."""
        with patch(
            "stardist.models.model2d.StarDistData2D",
            OptimizedStackedTimepointsData2D,
        ):
            return super().train(
                X=x,
                Y=y,
                validation_data=validation_data,
                classes=classes,
                augmenter=augmenter,
                seed=seed,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                workers=workers,
            )

    def _predict_setup(  # type: ignore [no-untyped-def]
        self,
        img: npt.NDArray[np.double],
        axes: Optional[str],
        normalizer: Optional[Normalizer],
        n_tiles: Optional[Tuple[int, ...]],
        show_tile_progress: bool,
        predict_kwargs: Dict[str, Any],
    ):
        """Modified version."""
        if n_tiles is None:
            n_tiles = (1,) * img.ndim
        assert n_tiles is not None
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError(
                "n_tiles must be an iterable of length %d" % img.ndim
            ) from None
        all(
            np.isscalar(t)
            and 1 <= t  # type: ignore [operator]
            and int(t) == t  # type: ignore [arg-type]
            for t in n_tiles
        ) or _raise(ValueError("all values of n_tiles must be integer values >= 1"))

        n_tiles = tuple(map(int, n_tiles))

        axes = self._normalize_axes(img, axes)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img)  # x has axes_net semantics

        channel = axes_dict(axes_net)["C"]
        # MODIFIED
        self.config.n_channel_in * self.config.len_t == x.shape[channel] or _raise(
            ValueError()
        )
        axes_net_div_by = self._axes_div_by(axes_net)

        grid = tuple(self.config.grid)
        len(grid) == len(axes_net) - 1 or _raise(ValueError())
        grid_dict = dict(zip(axes_net.replace("C", ""), grid))

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        resizer = StarDistPadAndCropResizer(grid=grid_dict)

        x = normalizer.before(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)

        if not _is_floatarray(x):
            warnings.warn(
                "Predicting on non-float input... ( forgot to normalize? )",
                stacklevel=2,
            )

        def predict_direct(x):  # type: ignore [no-untyped-def]
            ys = self.keras_model.predict(x[np.newaxis], **predict_kwargs)
            return tuple(y[0] for y in ys)

        def tiling_setup():  # type: ignore [no-untyped-def]
            assert n_tiles is not None
            np.prod(n_tiles) > 1 or _raise(
                ValueError("tiling setup for n_tiles = (1, 1, 1)")
            )
            tiling_axes = axes_net.replace("C", "")  # axes eligible for tiling
            x_tiling_axis = tuple(
                axes_dict(axes_net)[a] for a in tiling_axes
            )  # numerical axis ids for x
            axes_net_tile_overlaps = self._axes_tile_overlap(axes_net)
            # hack: permute tiling axis in the same way as img -> x was permuted
            _n_tiles = _permute_axes(np.empty(n_tiles, bool)).shape
            (
                all(_n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis)
                or _raise(
                    ValueError(
                        "entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes
                    )
                )
            )

            sh = [s // grid_dict.get(a, 1) for a, s in zip(axes_net, x.shape)]
            sh[channel] = None

            def create_empty_output(  # type: ignore [no-untyped-def]
                n_channel, dtype=np.float32
            ):
                sh[channel] = n_channel
                return np.empty(sh, dtype)

            if callable(show_tile_progress):
                progress, _show_tile_progress = show_tile_progress, True
            else:
                progress, _show_tile_progress = tqdm, show_tile_progress

            n_block_overlaps = [
                int(np.ceil(overlap / blocksize))
                for overlap, blocksize in zip(axes_net_tile_overlaps, axes_net_div_by)
            ]

            num_tiles_used = total_n_tiles(
                x,
                _n_tiles,
                block_sizes=axes_net_div_by,
                n_block_overlaps=n_block_overlaps,
            )

            tile_generator = progress(
                tile_iterator(
                    x,
                    _n_tiles,
                    block_sizes=axes_net_div_by,
                    n_block_overlaps=n_block_overlaps,
                ),
                disable=(not _show_tile_progress),
                total=num_tiles_used,
            )

            return tile_generator, tuple(sh), create_empty_output

        return (
            x,
            axes,
            axes_net,
            axes_net_div_by,
            _permute_axes,
            resizer,
            n_tiles,
            grid,
            grid_dict,
            channel,
            predict_direct,
            tiling_setup,
        )

    def _predict_generator(  # type: ignore [no-untyped-def]
        self,
        img: npt.NDArray[np.double],
        axes: Optional[str] = None,
        normalizer: Optional[Normalizer] = None,
        n_tiles: Optional[Tuple[int, ...]] = None,
        show_tile_progress: bool = True,
        **predict_kwargs,
    ) -> Generator[Optional[Tuple[npt.NDArray[np.double]]], None, None]:
        """Modified version of the version from StarDist2D."""
        (
            x,
            axes,
            axes_net,
            axes_net_div_by,
            _permute_axes,
            resizer,
            n_tiles,
            grid,
            grid_dict,
            channel,
            predict_direct,
            tiling_setup,
        ) = self._predict_setup(
            img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs
        )

        assert n_tiles is not None
        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            # MODIFIED
            prob = create_empty_output(self.config.len_t)
            dist = create_empty_output(self.config.n_rays * self.config.len_t)
            if self._is_multiclass():
                prob_class = create_empty_output(self.config.n_classes + 1)
                result_ = (prob, dist, prob_class)
            else:
                result_ = (prob, dist)  # type: ignore [assignment]

            for tile, s_src, s_dst in tile_generator:
                # predict_direct -> prob, dist, [prob_class if multi_class]
                result_tile = predict_direct(tile)
                # account for grid
                s_src = [
                    slice(
                        s.start // grid_dict.get(a, 1),
                        s.stop // grid_dict.get(a, 1),
                    )
                    for s, a in zip(s_src, axes_net)
                ]
                s_dst = [
                    slice(
                        s.start // grid_dict.get(a, 1),
                        s.stop // grid_dict.get(a, 1),
                    )
                    for s, a in zip(s_dst, axes_net)
                ]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                # print(s_src,s_dst)
                for part, part_tile in zip(result_, result_tile):
                    part[s_dst] = part_tile[s_src]
                yield None  # yield None after each processed tile
        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            result_ = predict_direct(x)

        result = [resizer.after(part, axes_net) for part in result_]

        # result = (prob, dist) for legacy or (prob, dist, prob_class) for multiclass

        # prob
        # result[0] = np.take(result[0], 0, axis=channel)
        # dist
        result[1] = np.maximum(
            1e-3, result[1]
        )  # avoid small dist values to prevent problems with Qhull
        result[1] = np.moveaxis(result[1], channel, -1)

        if self._is_multiclass():
            # prob_class
            result[2] = np.moveaxis(result[2], channel, -1)

        # last "yield" is the actual output that would
        # have been "return"ed if this was a regular function
        yield tuple(result)

    def predict_tyx(
        self, x: npt.NDArray[np.double]
    ) -> Tuple[npt.NDArray[np.double], ...]:
        """Prepare input image of shape TYXC to YXC for internal representation."""
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        x = np.concatenate([x[i] for i in range(self.config.len_t)], axis=-1)
        prob, dists = self.predict(x)

        prob = np.transpose(prob, (2, 0, 1))

        dists = np.stack(
            np.split(dists, self.config.len_t, axis=-1),
            axis=0,
        )

        return prob, dists

    def predict_tyx_list(
        self, xs: List[npt.NDArray[np.double]]
    ) -> List[Tuple[npt.NDArray[np.double], ...]]:
        """Same as predict_tyx but for list of elements."""
        return [self.predict_tyx(x) for x in xs]

    def predict_tyx_array(
        self, x_array: npt.NDArray[np.double]
    ) -> List[Tuple[npt.NDArray[np.double], ...]]:
        """Predict on TYXC array."""
        if x_array.ndim == 3:
            x_array = np.expand_dims(x_array, axis=-1)
        return self.predict_tyx_list(timeseries_to_batch(x_array, self.config.len_t))

    def _compute_receptive_field(
        self, img_size: Optional[Tuple[int, ...]] = None
    ) -> Tuple[Tuple[int, int], ...]:
        """Modified version of original StarDist models."""
        # TODO: good enough?
        from scipy.ndimage import zoom  # type: ignore [import-untyped]

        if img_size is None:
            img_size = tuple(
                g * (128 if self.config.n_dim == 2 else 64) for g in self.config.grid
            )
        if np.isscalar(img_size):
            img_size = (img_size,) * self.config.n_dim
        img_size = tuple(img_size)
        # print(img_size)
        # assert all(_is_power_of_2(s) for s in img_size)
        mid = tuple(s // 2 for s in img_size)
        x = np.zeros(
            # MODIFIED
            (1,) + img_size + (self.config.n_channel_in * self.config.len_t,),
            dtype=np.float32,
        )
        z = np.zeros_like(x)
        x[(0,) + mid + (slice(None),)] = 1
        y = self.keras_model.predict(x)[0][0, ..., 0]
        y0 = self.keras_model.predict(z)[0][0, ..., 0]
        grid = tuple((np.array(x.shape[1:-1]) / np.array(y.shape)).astype(int))
        # assert grid == self.config.grid
        y = zoom(y, grid, order=0)
        y0 = zoom(y0, grid, order=0)
        ind = np.where(np.abs(y - y0) > 0)
        return tuple(
            (int(m - np.min(i)), int(np.max(i) - m)) for (m, i) in zip(mid, ind)
        )

    @property
    def _config_class(self) -> type:
        """Needed method for the config class to use."""
        return StackedTimepointsConfig2D
