"""Test the naive_fusion implementations."""
import numpy as np
import pytest
from stardist.geometry.geom2d import polygons_to_label
from stardist.geometry.geom3d import polyhedron_to_label
from stardist.rays3d import Rays_GoldenSpiral

from merge_stardist_masks import naive_fusion as nf


@pytest.fixture(params=[(2, 2), (10, 20), (3, 3, 3), (10, 20, 30)])
def shape(request):
    """Use different shapes for 2d and 3d cases."""
    return request.param


@pytest.fixture(params=[(2, 2), (2, 4), (2, 2, 2), (2, 4, 8)])
def grid(request):
    """Use different grids for 2d and 3d cases."""
    return request.param


@pytest.fixture(params=[0.0, -1.0])
def default_value(request):
    """Default values for inflate_array."""
    return request.param


def test_shape_of_mesh_from_shape(shape):
    """Test the output shape of mesh_from_shape."""
    mesh = nf.mesh_from_shape(shape)

    assert mesh.ndim == len(shape) + 1
    assert mesh.shape[-1] == len(shape)
    assert mesh.shape[:-1] == shape


def test_values_of_mesh_from_shape(shape):
    """mesh_from_shape output follows this pattern: x[i, j, k, 0] = i."""
    mesh = nf.mesh_from_shape(shape)

    for inds in zip(*(range(i) for i in shape), range(len(shape))):
        assert mesh[inds] == inds[inds[-1]]


def test_shape_of_points_from_grid(shape, grid):
    """points_from_grid output has same shape as mesh_from_shape."""
    min_l = min(len(shape), len(grid))
    grid = grid[:min_l]
    shape = shape[:min_l]

    points = nf.points_from_grid(shape, grid)

    assert points.ndim == len(shape) + 1
    assert points.shape[-1] == len(shape)
    assert points.shape[:-1] == shape


def test_values_of_points_from_grid(shape, grid):
    """Output of points_from_mesh is mesh_from_shape multiplied by grid."""
    min_l = min(len(shape), len(grid))
    grid = grid[:min_l]
    shape = shape[:min_l]

    points = nf.points_from_grid(shape, grid)
    mesh = nf.mesh_from_shape(shape)

    grid = np.array(grid)
    np.testing.assert_array_equal(mesh * grid, points)


def test_value_error_of_points_from_grid():
    """If x.ndim < len(grid) _normalize_grid within function must throw ValueError."""
    with pytest.raises(ValueError):
        nf.points_from_grid((3, 3), (2, 2, 2))


def test_shape_of_inflate_array(shape, grid):
    """Shape of inflated array is shape * grid."""
    grid = grid[: len(shape)]

    inflated_array = nf.inflate_array(np.empty(shape), grid)

    assert inflated_array.ndim == len(shape)
    for i in range(len(grid)):
        assert inflated_array.shape[i] == grid[i] * shape[i]
    for i in range(len(grid), inflated_array.ndim):
        assert inflated_array.shape[i] == shape[i]


def test_values_of_inflate_array(shape, grid, default_value):
    """Check output of inflated array for correct values."""
    grid = grid[: len(shape)]

    x = np.random.standard_normal(shape)
    inflated_array = nf.inflate_array(x, grid, default_value)
    print(inflated_array.ndim)

    slices = []
    for i in range(len(grid)):
        slices.append(slice(None, None, grid[i]))
    for _i in range(len(grid), inflated_array.ndim):
        slices.append(slice(None))
    slices = tuple(slices)
    # first test if values at grid points are correct
    np.testing.assert_array_equal(inflated_array[slices], x)

    # then test if default values are correct
    inds = np.ones_like(inflated_array, dtype=bool)
    inds[slices] = False
    np.testing.assert_equal(inflated_array[inds], default_value)


def test_value_error_of_inflate_array():
    """If x.ndim < len(grid) inflate_array must throw ValueError."""
    with pytest.raises(ValueError):
        nf.inflate_array(np.empty((3, 3)), (2, 2, 2))


@pytest.fixture(params=[(5, 6), (20, 14, 10)])
def point(request):
    """Test slice_point with different points."""
    return request.param


@pytest.fixture(params=[4, 10, 30])
def max_dist(request):
    """Test slice_point with different max_dist."""
    return request.param


@pytest.fixture(params=[(40, 50), (60, 50, 55)])
def big_shape(request):
    """Test slice_point with different shapes, 2d and 3d."""
    return request.param


def test_len_and_types_of_outputs_in_slice_point(point, max_dist):
    """Lenght of outputs of slice_point must have same lenght as input point."""
    slices, centered_point = nf.slice_point(point, max_dist)

    assert len(slices) == len(point)
    assert len(centered_point) == len(point)

    assert type(slices) == tuple
    assert type(slices[0]) == slice


def test_values_of_slice_point(point, max_dist, big_shape):
    """Test if x[slices][centered_point] = x[point]."""
    min_l = min(len(big_shape), len(point))
    point = point[:min_l]
    big_shape = big_shape[:min_l]

    x = np.random.standard_normal(big_shape)
    slices, centered_point = nf.slice_point(point, max_dist)

    assert x[slices][tuple(centered_point)] == x[tuple(point)]


def test_my_polyhedron_to_label():
    """Test that convenience function is not changed."""
    n_rays = 40
    rays = Rays_GoldenSpiral(n=n_rays)
    dists = np.abs(np.random.standard_normal(n_rays))
    shape = (40, 40, 40)
    points = np.array(shape) / 2

    my_label = nf.my_polyhedron_to_label(rays, dists, points, shape)

    assert my_label.shape == shape

    label = polyhedron_to_label(
        np.expand_dims(np.clip(dists, 1e-3, None), axis=0),
        np.expand_dims(points, axis=0),
        rays,
        shape,
        verbose=False,
    )

    np.testing.assert_array_equal(my_label, label)


def test_my_polygons_to_label():
    """Test that convenience function is not changed."""
    n_rays = 40
    dists = np.abs(np.random.standard_normal(n_rays))
    shape = (40, 40)
    points = np.array(shape) / 2

    my_label = nf.my_polygons_to_label(dists, points, shape)

    assert my_label.shape == shape

    label = polygons_to_label(
        np.expand_dims(np.clip(dists, 1e-3, None), axis=0),
        np.expand_dims(points, axis=0),
        shape,
    )

    np.testing.assert_array_equal(my_label, label)


def test_naive_fusion_3d():
    """Test naive fusion with only two points that overlap."""
    n_rays = 40
    rays = Rays_GoldenSpiral(n=n_rays)

    s = 6
    shape = (s, s, s)
    dists = np.zeros(shape + (n_rays,))
    probs = np.zeros(shape)

    # object 1 with several pixels that do not add to the shape
    z0 = s // 3

    # main point of this object
    dist = 2.5
    dists[z0, z0, z0, :] = dist
    probs[z0, z0, z0] = 0.9

    small_dist = 0.1
    dists[z0 - 1, z0, z0, :] = small_dist
    probs[z0 - 1, z0, z0] = 0.8
    dists[z0, z0 - 1, z0, :] = small_dist
    probs[z0, z0 - 1, z0] = 0.8
    dists[z0, z0, z0 - 1, :] = small_dist
    probs[z0, z0, z0 - 1] = 0.8

    # object 2 composed of two pixels
    z1 = s // 3 * 2
    dists[z1, z1, z1, :] = dist
    probs[z1, z1, z1] = 0.6

    dists[z1 - 1, z1, z1, :] = dist
    probs[z1 - 1, z1, z1] = 0.7

    g = 2
    lbl = nf.naive_fusion(dists, probs, rays, grid=(g, g, g))

    new_dists = np.full((3, n_rays), dist)
    new_points = np.array([[z0, z0, z0], [z1, z1, z1], [z1 - 1, z1, z1]]) * g

    label = polyhedron_to_label(
        new_dists, new_points, rays, tuple(s * g for s in shape), verbose=False
    )
    # set labels to correct ids
    label[label == 1] = 1
    label[label == 2] = 2
    label[label == 3] = 2

    np.testing.assert_array_equal(lbl, label)


def test_naive_fusion_2d():
    """Test naive fusion with only two points that overlap."""
    n_rays = 20

    s = 8
    shape = (s, s)
    dists = np.zeros(shape + (n_rays,))
    probs = np.zeros(shape)

    # object 1 with several pixels that do not add to the shape
    z0 = s // 4

    # main point of this object
    dist = 3
    dists[z0, z0, :] = dist
    probs[z0, z0] = 0.9

    small_dist = 0.1
    dists[z0 - 1, z0, :] = small_dist
    probs[z0 - 1, z0] = 0.8
    dists[z0, z0 - 1, :] = small_dist
    probs[z0, z0 - 1] = 0.8
    dists[z0 - 1, z0 - 1, :] = small_dist
    probs[z0 - 1, z0 - 1] = 0.8

    # object 2 composed of two pixels
    z1 = s // 4 * 3
    dists[z1, z1, :] = dist
    probs[z1, z1] = 0.6

    dists[z1 - 1, z1, :] = dist
    probs[z1 - 1, z1] = 0.7

    g = 2
    lbl = nf.naive_fusion(dists, probs, grid=(g, g))

    new_dists = np.full((3, n_rays), dist)
    new_points = np.array([[z0, z0], [z1, z1], [z1 - 1, z1]]) * g

    label = polygons_to_label(new_dists, new_points, tuple(s * g for s in shape))
    # set labels to correct ids
    label[label == 1] = 1
    label[label == 2] = 2
    label[label == 3] = 2
    print(lbl)
    print(label)

    np.testing.assert_array_equal(lbl, label)


def test_value_error_because_of_shape_in_naive_fusion():
    """Probs must be 2d or 3d, otherwise naive_fusion should raise ValueError."""
    n_rays = 20
    rays = Rays_GoldenSpiral(n=n_rays)

    s = 6
    shape = (s, s, s, s)
    dists = np.zeros(shape + (n_rays,))
    probs = np.zeros(shape)

    with pytest.raises(ValueError):
        nf.naive_fusion(dists, probs, rays)


def test_value_error_with_3d_rays_in_naive_fusion():
    """If called with probs.ndim = 3, rays must be supplied to naive_fusion."""
    n_rays = 20

    s = 6
    shape = (s, s, s)
    dists = np.zeros(shape + (n_rays,))
    probs = np.zeros(shape)

    with pytest.raises(ValueError):
        nf.naive_fusion(dists, probs)
