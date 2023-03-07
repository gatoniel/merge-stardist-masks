"""Test the data generator for stacked timepoints of 2d images."""
import numpy as np

from merge_stardist_masks.data_2d import OptimizedStackedTimepointsData2D


def test_getitem() -> None:
    """Verify correctly sized output shapes of __getitem__."""
    len_t = 3
    shape = (len_t, 16, 16)
    x = np.random.random((1,) + shape[1:]).repeat(3, axis=0)
    print(x.shape)
    y = np.zeros(shape, dtype=np.uint8)
    y[:, 4:8, 4:8] = 1

    batch_size = 2
    n_rays = 8
    patch_size = 8
    dg = OptimizedStackedTimepointsData2D(
        [x], [y], batch_size, n_rays, 10, patch_size=(patch_size, patch_size)
    )

    [new_x], [probs, dists] = dg[0]

    assert new_x.shape == (batch_size, patch_size, patch_size, len_t)
    assert probs.shape == (batch_size, patch_size, patch_size, len_t)
    assert dists.shape == (batch_size, patch_size, patch_size, (n_rays + 1) * len_t)

    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        new_x[..., 0], new_x[..., 1]
    )
    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        new_x[..., 0], new_x[..., 2]
    )

    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        probs[..., 0], probs[..., 1]
    )
    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        probs[..., 0], probs[..., 2]
    )

    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        dists[..., :8],
        dists[..., 8:16],
    )
    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        dists[..., :8],
        dists[..., 16:24],
    )

    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        dists[..., -3], dists[..., -2]
    )
    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        dists[..., -3], dists[..., -1]
    )
