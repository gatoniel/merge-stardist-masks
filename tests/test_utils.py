"""Test utility functions to interact with StarDist model files."""

from unittest.mock import Mock

import pytest
from pytest_mock import MockFixture  # type: ignore [import-not-found]
from stardist.rays3d import Rays_Base  # type: ignore [import-untyped]

from merge_stardist_masks.utils import grid_from_path
from merge_stardist_masks.utils import rays_from_path


@pytest.fixture
def mocker_2d_config_rays(mocker: MockFixture) -> None:
    """Mock open with json for n_rays in 2d."""
    mocked_config = mocker.mock_open(read_data='{"n_rays": 32}')
    mocker.patch("builtins.open", mocked_config)


@pytest.fixture
def mocker_2d_config_grid(mocker: MockFixture) -> None:
    """Mock open with json file for 2d grid values."""
    mocked_config = mocker.mock_open(read_data='{"grid": [2, 2]}')
    mocker.patch("builtins.open", mocked_config)


@pytest.fixture
def mocker_3d_config_rays(mocker: MockFixture) -> None:
    """Mock open with json for n_rays in 2d."""
    mocked_config = mocker.mock_open(
        read_data="""
        {\"rays_json\": {\"name\": \"Rays_GoldenSpiral\", \"kwargs\": {\"n\": 96,
        \"anisotropy\": [1.5, 1.0, 1.0]}}}
        """
    )
    mocker.patch("builtins.open", mocked_config)


@pytest.fixture
def mocker_3d_config_grid(mocker: MockFixture) -> None:
    """Mock open with json file for 2d grid values."""
    mocked_config = mocker.mock_open(read_data='{"grid": [2, 2, 3]}')
    mocker.patch("builtins.open", mocked_config)


def test_rays_from_path_2d_rays(mocker_2d_config_rays: Mock) -> None:
    """Test if None is returned for 2d config files."""
    assert rays_from_path("") is None
    assert rays_from_path("config.json") is None


def test_rays_from_path_3d_rays(mocker_3d_config_rays: Mock) -> None:
    """Test if Rays_Base is returned for 3d config files."""
    assert issubclass(type(rays_from_path("")), Rays_Base)
    assert issubclass(type(rays_from_path("config.json")), Rays_Base)


def test_grid_from_path_2d_grid(mocker_2d_config_grid: Mock) -> None:
    """Test whether correct grid is returned for config.json."""
    assert grid_from_path("") == (2, 2)
    assert grid_from_path("config.json") == (2, 2)


def test_grid_from_path_3d_grid(mocker_3d_config_grid: Mock) -> None:
    """Test whether correct grid is returned for config.json."""
    assert grid_from_path("") == (2, 2, 3)
    assert grid_from_path("config.json") == (2, 2, 3)
