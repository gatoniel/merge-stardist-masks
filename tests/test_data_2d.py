"""Test the data generator for stacked timepoints of 2d images."""
import numpy as np
import pytest

from merge_stardist_masks.data_2d import OptimizedStackedTimepointsData2D


@pytest.mark.parametrize(
    "n_channel, n_rays, len_t, grid, batch_size, patch_size",
    [
        (1, 8, 3, 1, 2, 8),
        (1, 8, 3, 2, 2, 16),
        (2, 16, 5, 2, 1, 10),
        (1, 16, 4, 2, 2, 8),
    ],
)
def test_getitem(
    n_channel: int, n_rays: int, len_t: int, grid: int, batch_size: int, patch_size: int
) -> None:
    """Verify correctly sized output shapes of __getitem__."""
    shapexy = 16
    shape = (len_t, shapexy, shapexy, n_channel)
    x = np.random.random((1,) + shape[1:]).repeat(len_t, axis=0)
    x = np.squeeze(x)

    print(x.shape)
    y = np.zeros(shape[:-1], dtype=np.uint8)
    y[:, 4:8, 4:8] = 1

    dg = OptimizedStackedTimepointsData2D(
        [x],
        [y],
        batch_size,
        n_rays,
        10,
        patch_size=(patch_size, patch_size),
        grid=(grid,) * 2,
    )

    [new_x], [probs, dists] = dg[0]

    outshapexy = patch_size // grid

    assert new_x.shape == (batch_size, patch_size, patch_size, len_t * n_channel)
    assert probs.shape == (batch_size, outshapexy, outshapexy, len_t)
    assert dists.shape == (batch_size, outshapexy, outshapexy, (n_rays + 1) * len_t)

    mask_start = len_t * n_rays

    for i in range(len_t - 1):
        slice_x1 = slice(i * n_channel, (i + 1) * n_channel)
        slice_x2 = slice((i + 1) * n_channel, (i + 2) * n_channel)

        np.testing.assert_equal(  # type: ignore [no-untyped-call]
            new_x[..., slice_x1], new_x[..., slice_x2]
        )

        np.testing.assert_equal(  # type: ignore [no-untyped-call]
            probs[..., i], probs[..., i + 1]
        )

        slice_y1 = slice(i * n_rays, (i + 1) * n_rays)
        slice_y2 = slice((i + 1) * n_rays, (i + 2) * n_rays)

        np.testing.assert_equal(  # type: ignore [no-untyped-call]
            dists[..., slice_y1], dists[..., slice_y2]
        )

        np.testing.assert_equal(  # type: ignore [no-untyped-call]
            dists[..., mask_start + i], dists[..., mask_start + i + 1]
        )
