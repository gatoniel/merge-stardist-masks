"""Test the OptimizedStarDist3D extensively."""
import numpy as np
import pytest
from stardist.models import Config3D  # type: ignore [import]

from .utils import circle_image
from merge_stardist_masks.model_3d import OptimizedStarDist3D


# This is heavily inspired by
# https://github.com/stardist/stardist/blob/master/tests/test_model2D.py#L16
@pytest.mark.parametrize("n_channel, n_rays, grid", [(1, 8, 1), (2, 16, 1), (1, 20, 2)])
def test_model_conf_train_predict(
    tmpdir: str, n_channel: int, n_rays: int, grid: int
) -> None:
    """Verify correct shapes for various inputs."""
    shapexy = 64
    img = circle_image(shape=(shapexy, shapexy, shapexy), len_t=1)

    if n_channel > 1:
        img = np.repeat(img[..., np.newaxis], n_channel, axis=-1)

    img = np.squeeze(img)

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

    patchxy = 32
    patch_shape = (patchxy,) * 3

    conf = Config3D(
        n_rays=n_rays,
        grid=(grid,) * 3,
        use_gpu=False,
        n_channel_in=n_channel,
        train_epochs=2,
        train_patch_size=patch_shape,
        train_steps_per_epoch=1,
    )

    model = OptimizedStarDist3D(conf, name="test", basedir=str(tmpdir))

    for x in xs:
        print(x.shape)
    for x in ys:
        print(x.shape)
    model.train(  # type: ignore [no-untyped-call]
        xs, ys, validation_data=(xs[:2], ys[:2])
    )

    prob, dists = model.predict(xs[0])

    assert prob.shape == (shapexy // grid,) * 3
    assert dists.shape == (shapexy // grid,) * 3 + (n_rays,)
