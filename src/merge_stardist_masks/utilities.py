"""Simple utility functions that are used throughout this package."""
import numpy as np
from numba import njit


@njit
def search_next_index(current_index, indexer):
    """Find next index of indexer array that is True."""
    while current_index < len(indexer):
        if indexer[current_index]:
            return current_index
        current_index += 1

    raise StopIteration


@njit
def search_next_subsampled_index(current_internal_index, indexer, subsampled_indices):
    """Find next index of indexer array that is True."""
    while current_internal_index < len(subsampled_indices):
        subsampled_index = subsampled_indices[current_internal_index]
        if indexer[subsampled_index]:
            return subsampled_index, current_internal_index
        current_internal_index += 1

    raise StopIteration


class LinearSearchIterator:
    """A search iterator."""

    def __init__(self, size):
        """Set indexer based on given size and initialize first index with zero."""
        self.size = size
        self.indexer = np.ones(size, dtype=bool)
        self.i = 0

    def set_false(self, indices):
        """Set indices to False so they will get skipped at the next search."""
        self.indexer[indices] = False

    def __iter__(self):
        """Return for usage as iterator."""
        return self

    def __next__(self):
        """Find the next index where self.indexer is True."""
        return search_next_index(self.i, self.indexer)


class SubsampledSearchIterator:
    """Iterator that searches the next index within predefined indices."""

    def __init__(self, size, subsampled_indices):
        """Set size and subsampled_indices."""
        self.size = size
        self.subsampled_indices = subsampled_indices
        self.indexer = np.ones(size, dtype=bool)
        self.i = 0

    def set_false(self, indices):
        """Set indices to False so they will get skipped at the next search."""
        self.indexer[indices] = False

    def __iter__(self):
        """Return for usage as iterator."""
        return self

    def __next__(self):
        """Find the next index where self.indexer is True."""
        subsampled_index, self.i = search_next_subsampled_index(
            self.i, self.indexer, self.subsampled_indices
        )
        return subsampled_index
