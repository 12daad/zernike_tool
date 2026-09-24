"""Resolve writable per-user storage locations for the application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

_APPLICATION_DIRECTORY: Final = "zernike-tool"
_SETTINGS_FILENAME: Final = "settings.json"
_CACHE_DIRECTORY: Final = "cache"
_STANDARD_CACHE_FILENAME: Final = "rect_zernike_1920x1080_noll_1_30_float32.npy"


def user_data_directory() -> Path:
    """Return the writable directory used for persistent application data."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    base_directory = Path(local_app_data) if local_app_data else Path.home()
    return base_directory / _APPLICATION_DIRECTORY


def default_settings_path() -> Path:
    """Return the default JSON settings path."""

    return user_data_directory() / _SETTINGS_FILENAME


def default_standard_cache_path() -> Path:
    """Return the default writable 1920x1080 mode-cache path."""

    return user_data_directory() / _CACHE_DIRECTORY / _STANDARD_CACHE_FILENAME
