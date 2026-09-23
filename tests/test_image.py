"""Tests for image export."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from zernike_tool import export_image, load_image

# noinspection PyProtectedMember
from zernike_tool.image import _to_uint8


class ExportImageTests(unittest.TestCase):
    """Verify image conversion and format selection."""

    def test_default_png_and_float_scaling(self) -> None:
        """Append .png and scale unit-range grayscale floats."""

        with TemporaryDirectory() as directory:
            target = Path(directory) / "unit-range"
            export_image(np.array([[0.0, 0.5, 1.0]]), target)
            actual_path = target.with_suffix(".png")
            self.assertTrue(actual_path.is_file())
            with Image.open(actual_path) as image:
                self.assertEqual(image.format, "PNG")
                np.testing.assert_array_equal(
                    np.asarray(image),
                    [[0, 128, 255]],
                )

    def test_load_image_preserves_pixels_and_channels(self) -> None:
        """Load grayscale and RGB files as independent arrays."""

        grayscale = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            gray_path = root / "gray.png"
            rgb_path = root / "rgb.bmp"
            export_image(grayscale, gray_path)
            export_image(rgb, rgb_path)

            with self.assertLogs("zernike_tool.image", level="INFO") as loaded:
                actual_gray = load_image(str(gray_path))
                actual_rgb = load_image(rgb_path)

        np.testing.assert_array_equal(actual_gray, grayscale)
        np.testing.assert_array_equal(actual_rgb, rgb)
        self.assertTrue(actual_gray.flags.owndata)
        self.assertEqual(len(loaded.output), 2)

    def test_load_image_reports_missing_file(self) -> None:
        """Propagate a clear filesystem error for a missing image."""

        with TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            load_image(Path(directory) / "missing.png")

    def test_supported_formats_channels_and_aliases(self) -> None:
        """Write BMP and every JPEG suffix, including the requested typo."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                ("boolean.bmp", np.array([[False, True]]), "BMP", "L"),
                (
                    "gray.JPG",
                    np.array([[[0], [255]]], dtype=np.int16),
                    "JPEG",
                    "L",
                ),
                (
                    "rgb.jpeg",
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "JPEG",
                    "RGB",
                ),
                (
                    "rgba.jgp",
                    np.zeros((1, 1, 4), dtype=np.uint8),
                    "JPEG",
                    "RGB",
                ),
            ]
            for name, array, expected_format, expected_mode in cases:
                with self.subTest(name=name):
                    target = root / name
                    export_image(array, target)
                    with Image.open(target) as image:
                        self.assertEqual(image.format, expected_format)
                        self.assertEqual(image.mode, expected_mode)

    def test_numeric_conversion_ranges(self) -> None:
        """Preserve byte-like ranges and normalize other finite ranges."""

        byte_range = _to_uint8(np.array([[0, 127, 255]], dtype=np.int16))
        self.assertEqual(byte_range.dtype, np.uint8)
        np.testing.assert_array_equal(
            byte_range,
            [[0, 127, 255]],
        )
        np.testing.assert_array_equal(
            _to_uint8(np.array([[-2.0, 0.0, 2.0, np.nan, np.inf]])),
            [[0, 128, 255, 0, 0]],
        )
        np.testing.assert_array_equal(
            _to_uint8(np.full((1, 2), 300)),
            [[0, 0]],
        )

    def test_creates_directories_and_logs_overwrite(self) -> None:
        """Create missing parents and explicitly report file replacement."""

        with TemporaryDirectory() as directory:
            target = Path(directory) / "new" / "nested" / "image.png"
            with self.assertLogs("zernike_tool.image", level="INFO") as created:
                export_image(np.array([[1]], dtype=np.uint8), target)

            self.assertTrue(target.is_file())
            self.assertTrue(
                any("Created output directory" in entry for entry in created.output)
            )

            with self.assertLogs("zernike_tool.image", level="INFO") as overwritten:
                export_image(np.array([[2]], dtype=np.uint8), target)

            self.assertTrue(
                any(
                    "Overwriting existing image" in entry
                    for entry in overwritten.output
                )
            )
            with Image.open(target) as image:
                np.testing.assert_array_equal(np.asarray(image), [[2]])

    def test_invalid_array_and_extension(self) -> None:
        """Report unsupported types, shapes, values, and suffixes."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(TypeError, "numpy.ndarray"):
                export_image([[0]], root / "bad.png")  # type: ignore[arg-type]
            with self.assertRaisesRegex(TypeError, "real numeric"):
                export_image(np.array([[1j]]), root / "bad.png")
            with self.assertRaisesRegex(ValueError, "must have shape"):
                export_image(np.array([0, 1]), root / "bad.png")
            with self.assertRaisesRegex(ValueError, "must have shape"):
                export_image(np.zeros((2, 2, 2)), root / "bad.png")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                export_image(np.zeros((2, 2)), root / "bad.gif")
            with self.assertRaisesRegex(ValueError, "at least one finite"):
                export_image(np.full((2, 2), np.nan), root / "bad.png")


if __name__ == "__main__":  # pragma: no cover - unittest discovery entry point
    unittest.main()
