"""Tests for per-user application storage paths."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from zernike_tool import storage


class UserStorageTests(TestCase):
    """Verify Windows and portable storage path resolution."""

    def test_uses_local_app_data_when_available(self) -> None:
        """Keep settings and generated data below LOCALAPPDATA on Windows."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"LOCALAPPDATA": str(root)}):
                self.assertEqual(
                    storage.default_settings_path(),
                    root / "zernike-tool" / "settings.json",
                )
                self.assertEqual(
                    storage.default_standard_cache_path(),
                    root
                    / "zernike-tool"
                    / "cache"
                    / "rect_zernike_1920x1080_noll_1_30_float32.npy",
                )

    def test_falls_back_to_home_without_local_app_data(self) -> None:
        """Retain a usable location on non-Windows systems."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            environment = os.environ.copy()
            environment.pop("LOCALAPPDATA", None)
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(Path, "home", return_value=root),
            ):
                self.assertEqual(
                    storage.user_data_directory(),
                    root / "zernike-tool",
                )


if __name__ == "__main__":  # pragma: no cover - unittest discovery entry point
    main()
