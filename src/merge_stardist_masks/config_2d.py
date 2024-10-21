"""Configuration for 2D stacked time frames modified directly from StarDist."""

from __future__ import annotations

from typing import Optional
from typing import Tuple

from csbdeep.models import BaseConfig  # type: ignore [import-untyped]
from csbdeep.utils import _raise  # type: ignore [import-untyped]
from csbdeep.utils import backend_channels_last
from csbdeep.utils.tf import keras_import  # type: ignore [import-untyped]
from packaging.version import Version
from stardist.utils import _normalize_grid  # type: ignore [import-untyped]


keras = keras_import()


class StackedTimepointsConfig2D(BaseConfig):  # type: ignore [misc]
    """Configuration for a 2D StarDist model based on stacked timepoints."""

    def __init__(
        self,
        axes: str = "YX",
        n_rays: int = 32,
        len_t: int = 3,
        n_channel_in: int = 1,
        grid: Tuple[int, ...] = (1, 1),
        n_classes: Optional[int] = None,
        backbone: str = "unet",
        train_patch_size: Tuple[int, ...] = (256, 256),
        **kwargs: int,
    ) -> None:
        """Initialize with fixed length in time direction."""
        super().__init__(
            axes=axes,
            n_channel_in=n_channel_in,
            n_channel_out=(1 + n_rays) * len_t,
        )

        n_classes is None or _raise(NotImplementedError("n_classes not implemented."))

        # directly set by parameters
        self.len_t = len_t
        self.n_rays = int(n_rays)
        self.grid = _normalize_grid(grid, 2)
        self.backbone = str(backbone).lower()
        self.n_classes = None if n_classes is None else int(n_classes)
        self.train_patch_size = train_patch_size

        # default config (can be overwritten by kwargs below)
        if self.backbone == "unet":
            self.unet_n_depth = 3
            self.unet_kernel_size = 3, 3
            self.unet_n_filter_base = 32
            self.unet_n_conv_per_depth = 2
            self.unet_pool = 2, 2
            self.unet_activation = "relu"
            self.unet_last_activation = "relu"
            self.unet_batch_norm = False
            self.unet_dropout = 0.0
            self.unet_prefix = ""
            self.net_conv_after_unet = 128
        else:
            # TODO: resnet backbone for 2D model?
            raise ValueError("backbone '%s' not supported." % self.backbone)

        # net_mask_shape not needed but kept for legacy reasons
        if backend_channels_last():
            self.net_input_shape = None, None, self.n_channel_in * self.len_t
            # self.net_mask_shape = None, None, 1
        else:
            self.net_input_shape = self.n_channel_in * self.len_t, None, None
            # self.net_mask_shape = 1, None, None

        self.train_shape_completion = False
        self.train_completion_crop = 32
        self.train_background_reg = 1e-4
        self.train_foreground_only = 0.9
        self.train_sample_cache = True

        self.train_dist_loss = "mae"
        self.train_loss_weights = (1, 0.2) if self.n_classes is None else (1, 0.2, 1)
        self.train_class_weights = (
            (1, 1) if self.n_classes is None else (1,) * (self.n_classes + 1)
        )
        self.train_epochs = 400
        self.train_steps_per_epoch = 100
        self.train_learning_rate = 0.0003
        self.train_batch_size = 4
        self.train_n_val_patches = None
        self.train_tensorboard = True
        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = (
            "epsilon" if Version(keras.__version__) <= Version("2.1.5") else "min_delta"
        )
        self.train_reduce_lr = {"factor": 0.5, "patience": 40, min_delta_key: 0}

        self.use_gpu = False

        # remove derived attributes that shouldn't be overwritten
        for k in ("n_dim", "n_channel_out"):
            try:
                del kwargs[k]
            except KeyError:
                pass

        self.update_parameters(False, **kwargs)

        # FIXME: put into is_valid()
        if not len(self.train_loss_weights) == (2 if self.n_classes is None else 3):
            raise ValueError(
                f"train_loss_weights {self.train_loss_weights} not compatible "
                f"with n_classes ({self.n_classes}): must be 3 weights if "
                "n_classes is not None, otherwise 2"
            )

        if not len(self.train_class_weights) == (
            2 if self.n_classes is None else self.n_classes + 1
        ):
            raise ValueError(
                f"train_class_weights {self.train_class_weights} not compatible "
                f"with n_classes ({self.n_classes}): must be 'n_classes + 1' weights "
                "if n_classes is not None, otherwise 2"
            )
