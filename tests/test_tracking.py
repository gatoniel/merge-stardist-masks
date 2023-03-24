"""Test pre- and postprocessing of displacement maps."""
import numpy as np

from merge_stardist_masks import tracking

objects = [
    (((0, 2), (0, 2)), (1, 1), 5),
    (((0, 2), (5, 7)), (1, -1), 20),
    (((4, 6), (6, 8)), (-1, -2), 44),
]

lbl0 = np.zeros((10, 10), dtype=int)
lbl1 = np.zeros_like(lbl0)

for obj in objects:
    sly0 = slice(*obj[0][0])
    slx0 = slice(*obj[0][1])

    sly1 = slice(*tuple(np.array(obj[0][0]) + obj[1][0]))
    slx1 = slice(*tuple(np.array(obj[0][1]) + obj[1][1]))

    i = obj[2]

    lbl0[sly0, slx0] = i
    lbl1[sly1, slx1] = i

max_id = 45
lbl1[-2:, -2:] = max_id


def test_calc_midpoints() -> None:
    """Test correct length of dictionary and its arrays."""
    lbl = np.zeros((10, 10), dtype=int)
    lbl[:2, :2] = 1
    lbl[:2, 5:7] = 2
    lbl[4:6, 6:8] = 4

    midpoints = tracking.calc_midpoints(lbl)

    assert len(midpoints) == len(np.unique(lbl)) - 1
    np.testing.assert_array_equal(midpoints[1], np.array((0.5, 0.5)))
    np.testing.assert_array_equal(midpoints[2], np.array((0.5, 5.5)))
    np.testing.assert_array_equal(midpoints[4], np.array((4.5, 6.5)))


def test_calc_midpoint_distances() -> None:
    """Assure correct calculation of distances, skipping 0, and new cells."""
    midpoints0 = tracking.calc_midpoints(lbl0)
    midpoints1 = tracking.calc_midpoints(lbl1)

    distances = tracking.calc_midpoint_distances(midpoints0, midpoints1)
    distances_rev = tracking.calc_midpoint_distances(midpoints1, midpoints0)
    distances_identity = tracking.calc_midpoint_distances(midpoints1, midpoints1)

    assert max_id not in distances
    assert max_id not in distances_rev
    assert max_id in distances_identity

    assert 0 not in distances
    assert 0 not in distances_rev
    assert 0 not in distances_identity

    for obj in objects:
        i = obj[2]
        np.testing.assert_array_equal(-np.array(obj[1], dtype=float), distances[i])
        np.testing.assert_array_equal(distances[i], -distances_rev[i])

    for i in np.unique(lbl1):
        if i == 0:
            continue
        np.testing.assert_array_equal(distances_identity[i], np.zeros(2))


def test_prepare_displacement_map_single() -> None:
    """Test calculation of single displacement map."""
    displacement_map = tracking.prepare_displacement_map_single(lbl0, lbl1)
    displacement_map_rev = tracking.prepare_displacement_map_single(lbl1, lbl0)
    displacement_map_identity = tracking.prepare_displacement_map_single(lbl1, lbl1)

    print(lbl0)
    print(lbl1)
    for i in range(3):
        print(displacement_map[..., i])

    assert displacement_map.shape == lbl0.shape + (lbl0.ndim + 1,)
    np.testing.assert_array_equal(
        displacement_map_identity[..., :2],
        np.zeros_like(displacement_map_identity[..., :2]),
    )

    for obj in objects:
        i = obj[2]
        inds = lbl1 == i
        inds0 = lbl0 == i
        assert np.all(displacement_map[inds, -1] == 1.0)
        np.testing.assert_array_equal(
            displacement_map[inds, :-1].mean(axis=0),
            -np.array(obj[1], dtype=float),
        )
        assert np.all(displacement_map_rev[inds0, -1] == 1.0)
        np.testing.assert_array_equal(
            displacement_map_rev[inds0, :-1].mean(axis=0),
            np.array(obj[1], dtype=float),
        )

    assert np.all(displacement_map[lbl1 == 0, :] == 0.0)
    assert np.all(displacement_map_rev[lbl0 == 0, :] == 0.0)

    inds_max = lbl1 == max_id
    assert np.all(displacement_map[inds_max, :] == 0.0)
    assert np.all(displacement_map_rev[inds_max, :] == 0.0)


def test_prepare_displacement_maps() -> None:
    """Test whether single displacement maps are stacked together correctly."""
    lbl = np.stack([lbl0, lbl1, lbl0], axis=0)
    displacement_maps = tracking.prepare_displacement_maps(lbl)

    # lbl.ndim alone is enough as it is already +1 due to the time dimension
    assert displacement_maps.shape == (2,) + lbl0.shape + (lbl.ndim,)

    np.testing.assert_array_equal(
        displacement_maps[0, ...], tracking.prepare_displacement_map_single(lbl0, lbl1)
    )
    np.testing.assert_array_equal(
        displacement_maps[1, ...], tracking.prepare_displacement_map_single(lbl1, lbl0)
    )