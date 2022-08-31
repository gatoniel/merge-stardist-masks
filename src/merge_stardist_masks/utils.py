"""Utility function for easier interaction with StarDist model files."""
from __future__ import annotations

import json
import os
from typing import Optional
from typing import Tuple

from stardist.rays3d import Rays_Base  # type: ignore [import]
from stardist.rays3d import rays_from_json


def rays_from_path(path: str) -> Optional[Rays_Base]:
    """Automatically returns StarDist rays object or None based on model config.json."""
    config_file = "config.json"
    if not path.endswith(config_file):
        path = os.path.join(path, config_file)

    with open(path) as f:
        config = json.load(f)

    try:
        return rays_from_json(config["rays_json"])
    except KeyError:
        return None


def grid_from_path(path: str) -> Tuple[int, ...]:
    """Automatically returns the grid used in StarDist model from config.json."""
    config_file = "config.json"
    if not path.endswith(config_file):
        path = os.path.join(path, config_file)

    with open(path) as f:
        config = json.load(f)
    return tuple(int(i) for i in config["grid"])
