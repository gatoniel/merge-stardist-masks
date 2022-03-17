"""Test the naive_fusion implementations."""
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pytest
from pytest import FixtureRequest
from stardist.geometry.geom2d import polygons_to_label  # type: ignore [import]
from stardist.geometry.geom3d import polyhedron_to_label  # type: ignore [import]
from stardist.rays3d import Rays_GoldenSpiral  # type: ignore [import]

from merge_stardist_masks import naive_fusion as nf


IntTuple = Tuple[int, ...]


class FixtureRequestTuple(FixtureRequest):
    """Convenience Type for FixtureRequest with tuple of ints."""

    param: IntTuple


class FixtureRequestFloat(FixtureRequest):
    """Convenience Type for FixtureRequest with floats."""

    param: float


class FixtureRequestInt(FixtureRequest):
    """Convenience Type for FixtureRequest with ints."""

    param: int


class FixtureRequestBool(FixtureRequest):
    """Convenience Type for FixtureRequest with booleans."""

    param: bool


@pytest.fixture(params=[(2, 2), (10, 20), (3, 3, 3), (10, 20, 30)])
def shape(request: FixtureRequestTuple) -> IntTuple:
    """Use different shapes for 2d and 3d cases."""
    return request.param


@pytest.fixture(params=[(2, 2), (2, 4), (2, 2, 2), (2, 4, 8)])
def grid(request: FixtureRequestTuple) -> IntTuple:
    """Use different grids for 2d and 3d cases."""
    return request.param


@pytest.fixture(params=[0.0, -1.0])
def default_value(request: FixtureRequestFloat) -> float:
    """Default values for inflate_array."""
    return request.param


def test_shape_of_mesh_from_shape(shape: IntTuple) -> None:
    """Test the output shape of mesh_from_shape."""
    mesh = nf.mesh_from_shape(shape)

    assert mesh.ndim == len(shape) + 1
    assert mesh.shape[-1] == len(shape)
    assert mesh.shape[:-1] == shape


def test_values_of_mesh_from_shape(shape: IntTuple) -> None:
    """mesh_from_shape output follows this pattern: x[i, j, k, 0] = i."""
    mesh = nf.mesh_from_shape(shape)

    for inds in zip(*(range(i) for i in shape), range(len(shape))):
        assert mesh[inds] == inds[inds[-1]]


def test_shape_of_points_from_grid(shape: IntTuple, grid: IntTuple) -> None:
    """points_from_grid output has same shape as mesh_from_shape."""
    min_l = min(len(shape), len(grid))
    grid = grid[:min_l]
    shape = shape[:min_l]

    points = nf.points_from_grid(shape, grid)

    assert points.ndim == len(shape) + 1
    assert points.shape[-1] == len(shape)
    assert points.shape[:-1] == shape


def test_values_of_points_from_grid(shape: IntTuple, grid: IntTuple) -> None:
    """Output of points_from_mesh is mesh_from_shape multiplied by grid."""
    min_l = min(len(shape), len(grid))
    grid = grid[:min_l]
    shape = shape[:min_l]

    points = nf.points_from_grid(shape, grid)
    mesh = nf.mesh_from_shape(shape)

    np.testing.assert_array_equal(  # type: ignore [no-untyped-call]
        mesh * np.array(grid), points
    )


def test_value_error_of_points_from_grid() -> None:
    """If x.ndim < len(grid) _normalize_grid within function must throw ValueError."""
    with pytest.raises(ValueError):
        nf.points_from_grid((3, 3), (2, 2, 2))


def test_shape_of_inflate_array(shape: IntTuple, grid: IntTuple) -> None:
    """Shape of inflated array is shape * grid."""
    grid = grid[: len(shape)]

    inflated_array = nf.inflate_array(np.empty(shape), grid)

    assert inflated_array.ndim == len(shape)
    for i in range(len(grid)):
        assert inflated_array.shape[i] == grid[i] * shape[i]
    for i in range(len(grid), inflated_array.ndim):
        assert inflated_array.shape[i] == shape[i]


def test_values_of_inflate_array(
    shape: IntTuple, grid: IntTuple, default_value: float
) -> None:
    """Check output of inflated array for correct values."""
    grid = grid[: len(shape)]

    x = np.random.standard_normal(shape)
    inflated_array = nf.inflate_array(x, grid, default_value)
    print(inflated_array.ndim)

    slices_list = []
    for i in range(len(grid)):
        slices_list.append(slice(None, None, grid[i]))
    for _i in range(len(grid), inflated_array.ndim):
        slices_list.append(slice(None))
    slices = tuple(slices_list)
    # first test if values at grid points are correct
    np.testing.assert_array_equal(  # type: ignore [no-untyped-call]
        inflated_array[slices], x
    )

    # then test if default values are correct
    inds = np.ones_like(inflated_array, dtype=bool)
    inds[slices] = False
    np.testing.assert_equal(  # type: ignore [no-untyped-call]
        inflated_array[inds], default_value
    )


def test_value_error_of_inflate_array() -> None:
    """If x.ndim < len(grid) inflate_array must throw ValueError."""
    with pytest.raises(ValueError):
        nf.inflate_array(np.empty((3, 3)), (2, 2, 2))


@pytest.fixture(params=[(5, 6), (20, 14, 10)])
def point(request: FixtureRequestTuple) -> IntTuple:
    """Test slice_point with different points."""
    return request.param


@pytest.fixture(params=[4, 10, 30])
def max_dist(request: FixtureRequestInt) -> int:
    """Test slice_point with different max_dist."""
    return request.param


@pytest.fixture(params=[(40, 50), (60, 50, 55)])
def big_shape(request: FixtureRequestTuple) -> IntTuple:
    """Test slice_point with different shapes, 2d and 3d."""
    return request.param


def test_len_and_types_of_outputs_in_slice_point(
    point: IntTuple, max_dist: int
) -> None:
    """Lenght of outputs of slice_point must have same lenght as input point."""
    slices, centered_point = nf.slice_point(point, max_dist)

    assert len(slices) == len(point)
    assert len(centered_point) == len(point)

    assert type(slices) == tuple
    assert type(slices[0]) == slice


def test_values_of_slice_point(
    point: IntTuple, max_dist: int, big_shape: IntTuple
) -> None:
    """Test if x[slices][centered_point] = x[point]."""
    min_l = min(len(big_shape), len(point))
    point = point[:min_l]
    big_shape = big_shape[:min_l]

    x = np.random.standard_normal(big_shape)
    slices, centered_point = nf.slice_point(point, max_dist)

    assert x[slices][tuple(centered_point)] == x[tuple(point)]


def test_my_polyhedron_to_label() -> None:
    """Test that convenience function is not changed."""
    n_rays = 40
    rays = Rays_GoldenSpiral(n=n_rays)
    dists = np.abs(np.random.standard_normal(n_rays))
    shape = (40, 40, 40)
    points: npt.NDArray[np.double] = np.array(shape) / 2

    my_label = nf.my_polyhedron_to_label(rays, dists, points, shape)

    assert my_label.shape == shape

    label = polyhedron_to_label(
        np.expand_dims(  # type: ignore [no-untyped-call]
            np.clip(dists, 1e-3, None), axis=0
        ),
        np.expand_dims(points, axis=0),  # type: ignore [no-untyped-call]
        rays,
        shape,
        verbose=False,
    )

    np.testing.assert_array_equal(my_label, label)  # type: ignore [no-untyped-call]


def test_my_polygons_to_label() -> None:
    """Test that convenience function is not changed."""
    n_rays = 40
    dists = np.abs(np.random.standard_normal(n_rays))
    shape = (40, 40)
    points: npt.NDArray[np.double] = np.array(shape) / 2

    my_label = nf.my_polygons_to_label(dists, points, shape)

    assert my_label.shape == shape

    label = polygons_to_label(
        np.expand_dims(  # type: ignore [no-untyped-call]
            np.clip(dists, 1e-3, None), axis=0
        ),
        np.expand_dims(points, axis=0),  # type: ignore [no-untyped-call]
        shape,
    )

    np.testing.assert_array_equal(my_label, label)  # type: ignore [no-untyped-call]


def test_naive_fusion_3d() -> None:
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
    lbl_aniso = nf.naive_fusion_anisotropic_grid(dists, probs, rays, grid=(g, g, g))

    new_dists = np.full((3, n_rays), dist)
    new_points: npt.NDArray[np.double] = (
        np.array([[z0, z0, z0], [z1, z1, z1], [z1 - 1, z1, z1]]) * g
    )

    label = polyhedron_to_label(
        new_dists, new_points, rays, tuple(s * g for s in shape), verbose=False
    )
    # set labels to correct ids
    label[label == 1] = 1
    label[label == 2] = 2
    label[label == 3] = 2

    np.testing.assert_array_equal(lbl, label)  # type: ignore [no-untyped-call]
    np.testing.assert_array_equal(lbl_aniso, label)  # type: ignore [no-untyped-call]


@pytest.fixture(params=[True, False])
def erase_probs_at_full_overlap(request: FixtureRequestBool) -> bool:
    """Switch between true and false."""
    return request.param


def test_naive_fusion_2d(erase_probs_at_full_overlap: bool) -> None:
    """Test naive fusion with overlaping points in 2d."""
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
    lbl = nf.naive_fusion(
        dists,
        probs,
        grid=(g, g),
        erase_probs_at_full_overlap=erase_probs_at_full_overlap,
    )
    lbl_aniso = nf.naive_fusion_anisotropic_grid(
        dists,
        probs,
        grid=(g, g),
        erase_probs_at_full_overlap=erase_probs_at_full_overlap,
    )

    new_dists = np.full((3, n_rays), dist)
    new_points: npt.NDArray[np.double] = (
        np.array([[z0, z0], [z1, z1], [z1 - 1, z1]]) * g
    )

    label = polygons_to_label(new_dists, new_points, tuple(s * g for s in shape))
    # set labels to correct ids
    label[label == 1] = 1
    label[label == 2] = 2
    label[label == 3] = 2
    print(lbl)
    print(label)

    np.testing.assert_array_equal(lbl, label)  # type: ignore [no-untyped-call]
    np.testing.assert_array_equal(lbl_aniso, label)  # type: ignore [no-untyped-call]


def test_naive_fusion_2d_anisotropic() -> None:
    """Test naive fusion with overlaping points in 2d with anisotropic grid."""
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

    grid = (1, 2)
    lbl = nf.naive_fusion(dists, probs, grid=grid)

    new_dists = np.full((3, n_rays), dist)
    new_points: npt.NDArray[np.double] = np.array(
        [[z0, z0], [z1, z1], [z1 - 1, z1]]
    ) * np.array(grid)

    grid_shape = tuple(s * g for s, g in zip(shape, grid))
    label = polygons_to_label(new_dists, new_points, grid_shape)
    # set labels to correct ids
    label[label == 1] = 1
    label[label == 2] = 2
    label[label == 3] = 2
    print(lbl)
    print(label)

    np.testing.assert_array_equal(lbl, label)  # type: ignore [no-untyped-call]


def test_naive_fusion_2d_winding() -> None:
    """Test naive fusion with a winding object in 2d."""
    new_points = np.array(
        [
            [1, 1],
            [1, 3],
            [3, 3],
            [3, 5],
            [5, 5],
            [5, 7],
            [7, 7],
            [7, 9],
            [9, 9],
            [9, 11],
        ]
    )
    new_dists = np.array(
        [
            [3, 1, 1, 1],
            [1, 3, 1, 1],
            [3, 1, 1, 1],
            [1, 3, 1, 1],
            [3, 1, 1, 1],
            [1, 3, 1, 1],
            [3, 1, 1, 1],
            [1, 3, 1, 1],
            [3, 1, 1, 1],
            [1, 3, 1, 1],
        ]
    )

    n_rays = new_dists.shape[1]
    s = 13
    shape = (s, s)
    dists = np.zeros(shape + (n_rays,))
    probs = np.zeros(shape)

    for i in range(len(new_points)):
        y = new_points[i, 0]
        x = new_points[i, 1]
        probs[y, x] = 0.7 - i * 0.01
        dists[y, x, :] = new_dists[i, :]

    g = 1
    lbl = nf.naive_fusion(dists, probs, grid=(g, g))
    lbl_no_slicing = nf.naive_fusion(dists, probs, grid=(g, g), no_slicing=True)

    lbl_no_slicing_anisotropic = nf.naive_fusion_anisotropic_grid(
        dists, probs, grid=(g, g), no_slicing=True
    )

    label = polygons_to_label(new_dists, new_points, tuple(s * g for s in shape))
    # set labels to correct ids
    label[label > 0] = 1
    print(lbl)
    print(lbl_no_slicing)
    print(probs)
    print(label)

    # lbl without slicing should not work in this case
    np.testing.assert_raises(  # type: ignore [no-untyped-call]
        AssertionError, np.testing.assert_array_equal, lbl, label
    )
    # lbl with slicing should work
    np.testing.assert_array_equal(  # type: ignore [no-untyped-call]
        lbl_no_slicing, label
    )
    # lbl with slicing and anisotropic version should also work
    np.testing.assert_array_equal(  # type: ignore [no-untyped-call]
        lbl_no_slicing_anisotropic, label
    )


def test_value_error_because_of_shape_in_naive_fusion() -> None:
    """Probs must be 2d or 3d, otherwise naive_fusion should raise ValueError."""
    n_rays = 20
    rays = Rays_GoldenSpiral(n=n_rays)

    s = 6
    shape = (s, s, s, s)
    dists = np.zeros(shape + (n_rays,))
    probs = np.zeros(shape)

    with pytest.raises(ValueError):
        nf.naive_fusion_anisotropic_grid(dists, probs, rays)
    with pytest.raises(ValueError):
        nf.naive_fusion_isotropic_grid(dists, probs, rays)
    with pytest.raises(ValueError):
        nf.naive_fusion(dists, probs, rays)


def test_value_error_with_3d_rays_in_naive_fusion() -> None:
    """If called with probs.ndim = 3, rays must be supplied to naive_fusion."""
    n_rays = 20

    s = 6
    shape = (s, s, s)
    dists = np.zeros(shape + (n_rays,))
    probs = np.zeros(shape)

    with pytest.raises(ValueError):
        nf.naive_fusion_anisotropic_grid(dists, probs)
    with pytest.raises(ValueError):
        nf.naive_fusion_isotropic_grid(dists, probs)
    with pytest.raises(ValueError):
        nf.naive_fusion(dists, probs)


def test_get_poly_list_to_label() -> None:
    """Exception ValueErrors should be raised. Needs separate testing."""
    with pytest.raises(ValueError):
        # rays must be supplied
        nf.get_poly_list_to_label((3, 3, 3), None)
    with pytest.raises(ValueError):
        # shape must be of length 2 or 3
        nf.get_poly_list_to_label((2, 3, 4, 5), None)
