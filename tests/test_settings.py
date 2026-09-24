"""Tests for persistent GUI settings."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from zernike_tool.app.settings import (
    DEFAULT_COEFFICIENTS,
    DEFAULT_GRAY_RESPONSE,
    DEFAULT_PHASE_RESPONSE,
    AppSettings,
    default_settings,
    load_settings,
    save_settings,
)


class AppSettingsTests(TestCase):
    """Verify safe defaults and atomic UTF-8 persistence."""

    def test_default_and_missing_settings_use_start_directory(self) -> None:
        """Use first-run values when no settings file exists."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = AppSettings(
                str(root.resolve()),
                str(root.resolve()),
                DEFAULT_COEFFICIENTS,
                DEFAULT_GRAY_RESPONSE,
                DEFAULT_PHASE_RESPONSE,
            )
            self.assertEqual(default_settings(root), expected)
            with self.assertLogs("zernike_tool.app.settings", level="INFO"):
                actual = load_settings(root / "missing.json", root)
            self.assertEqual(actual, expected)

    def test_round_trip_preserves_separate_paths_and_unicode_text(self) -> None:
        """Save all session fields atomically and restore them unchanged."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            imported = root / "导入"
            exported = root / "导出"
            imported.mkdir()
            exported.mkdir()
            filename = root / "nested" / "settings.json"
            expected = AppSettings(
                str(imported),
                str(exported),
                "1, 2\n3",
                "0, 64, 255",
                "0, 1.51, 6.4",
            )
            with self.assertLogs("zernike_tool.app.settings", level="INFO"):
                save_settings(expected, filename)
                actual = load_settings(filename, root)
            self.assertEqual(actual, expected)
            self.assertFalse(filename.with_suffix(".json.tmp").exists())
            self.assertIn("导入", filename.read_text(encoding="utf-8"))

    def test_invalid_fields_fall_back_independently(self) -> None:
        """Ignore stale paths and incorrectly typed text fields."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid_export = root / "export"
            valid_export.mkdir()
            filename = root / "settings.json"
            filename.write_text(
                json.dumps(
                    {
                        "import_directory": str(root / "gone"),
                        "export_directory": str(valid_export),
                        "coefficients": [1, 2],
                        "gray_response": "0, 200",
                        "phase_response": None,
                    }
                ),
                encoding="utf-8",
            )
            actual = load_settings(filename, root)
            self.assertEqual(actual.import_directory, str(root.resolve()))
            self.assertEqual(actual.export_directory, str(valid_export))
            self.assertEqual(actual.coefficients, DEFAULT_COEFFICIENTS)
            self.assertEqual(actual.gray_response, "0, 200")
            self.assertEqual(actual.phase_response, DEFAULT_PHASE_RESPONSE)

    def test_corrupt_and_non_object_json_use_defaults(self) -> None:
        """Recover from malformed JSON and a valid but wrong top-level type."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            filename = root / "settings.json"
            expected = default_settings(root)
            for contents in ("{broken", "[]"):
                with self.subTest(contents=contents):
                    filename.write_text(contents, encoding="utf-8")
                    with self.assertLogs(
                        "zernike_tool.app.settings",
                        level="WARNING",
                    ):
                        actual = load_settings(filename, root)
                    self.assertEqual(actual, expected)


if __name__ == "__main__":  # pragma: no cover - unittest discovery entry point
    main()
