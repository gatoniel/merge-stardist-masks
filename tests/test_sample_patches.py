"""Test the sample_patches."""
import numpy as np

from merge_stardist_masks.sample_patches import sample_patches


def test_sample_patches() -> None:
    """Run through function once and verify that shapes match."""
    shape = (3, 16, 16)
    x = np.random.random(shape)
    y = np.zeros(shape, dtype=np.uint8)
    y[:, 4:8, 4:8] = 1

    x, y = sample_patches((y, x), patch_size=(8, 8), n_samples=1)

    print(x.shape, y.shape)

    assert x.shape == y.shape
