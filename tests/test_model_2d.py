"""Test the OptimizedStackedTimepointsModel2D extensively."""
import numpy as np
import pytest

from .utils import circle_image
from merge_stardist_masks.config_2d import StackedTimepointsConfig2D
from merge_stardist_masks.model_2d import OptimizedStackedTimepointsModel2D


# This is heavily inspired by
# https://github.com/stardist/stardist/blob/master/tests/test_model2D.py#L16
@pytest.mark.parametrize(
    "n_channel, n_rays, grid, len_t", [(1, 8, 1, 3), (2, 16, 1, 4), (1, 20, 2, 5)]
)
def test_model_conf_train_predict(
    tmpdir: str, n_channel: int, n_rays: int, grid: int, len_t: int
) -> None:
    """Verify correct shapes for various inputs."""
    shapexy = 160
    img = circle_image(shape=(shapexy, shapexy), len_t=len_t)

    if n_channel > 1:
        img = np.repeat(img[..., np.newaxis], n_channel, axis=-1)

    imgs = [
        np.copy(img),  # type: ignore [no-untyped-call]
        np.copy(img),  # type: ignore [no-untyped-call]
        img,
    ]

    xs = [img + 0.6 * np.random.uniform(0, 1, img.shape) for img in imgs]
    if n_channel > 1:
        ys = [img[..., 0] for img in imgs]
    else:
        ys = imgs

    patch_shape = (128, 128)

    conf = StackedTimepointsConfig2D(
        n_rays=n_rays,
        grid=(grid,) * 2,
        len_t=img.shape[0],
        use_gpu=False,
        n_channel_in=n_channel,
        train_epochs=2,
        train_patch_size=patch_shape,
        train_steps_per_epoch=1,
    )

    model = OptimizedStackedTimepointsModel2D(conf, name="test", basedir=str(tmpdir))

    model.train(xs, ys, validation_data=(xs[:2], ys[:2]))

    prob, dists = model.predict_tyx(xs[0])

    assert prob.shape == (len_t, shapexy // grid, shapexy // grid)
    assert dists.shape == (len_t, shapexy // grid, shapexy // grid, n_rays)


@pytest.mark.parametrize(
    "n_channel, n_rays, grid, len_t", [(1, 8, 1, 3), (2, 16, 1, 4), (1, 20, 2, 5)]
)
def test_model_conf_train_predict_tracking(
    tmpdir: str, n_channel: int, n_rays: int, grid: int, len_t: int
) -> None:
    """Verify correct shapes for various inputs."""
    shapexy = 160
    img = circle_image(shape=(shapexy, shapexy), len_t=len_t)

    if n_channel > 1:
        img = np.repeat(img[..., np.newaxis], n_channel, axis=-1)

    imgs = [
        np.copy(img),  # type: ignore [no-untyped-call]
        np.copy(img),  # type: ignore [no-untyped-call]
        img,
    ]

    xs = [img + 0.6 * np.random.uniform(0, 1, img.shape) for img in imgs]
    if n_channel > 1:
        ys = [img[..., 0] for img in imgs]
    else:
        ys = imgs

    patch_shape = (128, 128)

    conf = StackedTimepointsConfig2D(
        n_rays=n_rays,
        grid=(grid,) * 2,
        len_t=img.shape[0],
        use_gpu=False,
        n_channel_in=n_channel,
        train_epochs=2,
        train_patch_size=patch_shape,
        train_steps_per_epoch=1,
        tracking=True,
    )

    assert conf.output_len_t == 2

    model = OptimizedStackedTimepointsModel2D(conf, name="test", basedir=str(tmpdir))

    model.train(xs, ys, validation_data=(xs[:2], ys[:2]))

    prob, dists, displacements = model.predict_tyx(xs[0])

    assert prob.shape == (2, shapexy // grid, shapexy // grid)
    assert dists.shape == (2, shapexy // grid, shapexy // grid, n_rays)
    assert displacements.shape == (1, shapexy // grid, shapexy // grid, 3)

    # train to reload model
    model2 = OptimizedStackedTimepointsModel2D(None, name="test", basedir=str(tmpdir))

    prob, dists, displacements = model2.predict_tyx(xs[0])

    assert prob.shape == (2, shapexy // grid, shapexy // grid)
    assert dists.shape == (2, shapexy // grid, shapexy // grid, n_rays)
    assert displacements.shape == (1, shapexy // grid, shapexy // grid, 3)


@pytest.mark.parametrize(
    "n_channel, n_rays, grid, len_t", [(1, 8, 1, 3), (2, 16, 1, 4), (1, 20, 2, 5)]
)
def test_model_conf_train_predict_tracking_all_timepoints_prediction(
    tmpdir: str, n_channel: int, n_rays: int, grid: int, len_t: int
) -> None:
    """Verify correct shapes for various inputs."""
    shapexy = 160
    img = circle_image(shape=(shapexy, shapexy), len_t=len_t)

    if n_channel > 1:
        img = np.repeat(img[..., np.newaxis], n_channel, axis=-1)

    imgs = [
        np.copy(img),  # type: ignore [no-untyped-call]
        np.copy(img),  # type: ignore [no-untyped-call]
        img,
    ]

    xs = [img + 0.6 * np.random.uniform(0, 1, img.shape) for img in imgs]
    if n_channel > 1:
        ys = [img[..., 0] for img in imgs]
    else:
        ys = imgs

    patch_shape = (128, 128)

    conf = StackedTimepointsConfig2D(
        n_rays=n_rays,
        grid=(grid,) * 2,
        len_t=img.shape[0],
        use_gpu=False,
        n_channel_in=n_channel,
        train_epochs=2,
        train_patch_size=patch_shape,
        train_steps_per_epoch=1,
        tracking=True,
        predict_all_timepoints=True,
    )

    model = OptimizedStackedTimepointsModel2D(conf, name="test", basedir=str(tmpdir))

    model.train(xs, ys, validation_data=(xs[:2], ys[:2]))

    prob, dists, displacements = model.predict_tyx(xs[0])

    assert prob.shape == (len_t, shapexy // grid, shapexy // grid)
    assert dists.shape == (len_t, shapexy // grid, shapexy // grid, n_rays)
    assert displacements.shape == (len_t - 1, shapexy // grid, shapexy // grid, 3)

    # train to reload model
    model2 = OptimizedStackedTimepointsModel2D(None, name="test", basedir=str(tmpdir))

    prob, dists, displacements = model2.predict_tyx(xs[0])

    assert prob.shape == (len_t, shapexy // grid, shapexy // grid)
    assert dists.shape == (len_t, shapexy // grid, shapexy // grid, n_rays)
    assert displacements.shape == (len_t - 1, shapexy // grid, shapexy // grid, 3)
