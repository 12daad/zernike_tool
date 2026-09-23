"""Tests for the GUI-independent image-correction domain logic."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from app import (
    calibrate_image,
    clear_calibration_cache,
    corrected_output_name,
    parse_coefficients,
    preload_correction_cache,
    prepare_grayscale,
)


class CoefficientParsingTests(TestCase):
    """Verify the coefficient text format accepted by the GUI."""

    def test_accepts_commas_spaces_tabs_and_newlines(self) -> None:
        """Parse every documented separator and surrounding delimiters."""

        actual = parse_coefficients(" ,1, 2  3\t4\n5, ")
        np.testing.assert_array_equal(actual, [1, 2, 3, 4, 5])
        self.assertEqual(actual.dtype, np.float64)

    def test_rejects_empty_non_numeric_and_non_finite_text(self) -> None:
        """Give useful errors for each invalid coefficient category."""

        cases = [
            (" , \n", "至少一个"),
            ("1, hello", "必须是"),
            ("1, inf", "有限"),
        ]
        for text, message in cases:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                parse_coefficients(text)


class GrayscalePreparationTests(TestCase):
    """Verify the single-active-channel import rule."""

    def test_accepts_two_dimensional_and_single_channel_images(self) -> None:
        """Convert accepted grayscale layouts to floating point."""

        image = np.array([[0, 127], [128, 255]], dtype=np.uint8)
        two_dimensional = prepare_grayscale(image)
        one_channel = prepare_grayscale(image[..., np.newaxis])
        np.testing.assert_array_equal(two_dimensional, image)
        np.testing.assert_array_equal(one_channel, image)
        self.assertEqual(two_dimensional.dtype, np.float64)

    def test_accepts_one_active_rgb_channel_and_all_zero_rgb(self) -> None:
        """Extract the only active channel and define all-zero RGB safely."""

        green = np.zeros((2, 2, 3), dtype=np.uint8)
        green[..., 1] = [[1, 2], [3, 4]]
        actual = prepare_grayscale(green)
        self.assertEqual(actual.shape, (2, 2))
        np.testing.assert_array_equal(
            actual,
            [[1, 2], [3, 4]],
        )
        np.testing.assert_array_equal(
            prepare_grayscale(np.zeros((1, 2, 3), dtype=np.uint8)),
            [[0, 0]],
        )

    def test_rejects_invalid_types_and_channel_layouts(self) -> None:
        """Reject non-arrays, complex data, RGB color, and RGBA data."""

        cases = [
            ([0], TypeError, "numpy.ndarray"),
            (np.array([[1j]]), TypeError, "实数"),
            (
                np.array([[[1, 2, 0]]], dtype=np.uint8),
                ValueError,
                "只能有一个通道",
            ),
            (np.zeros((1, 1, 4), dtype=np.uint8), ValueError, "单通道"),
            (np.zeros((2,), dtype=np.uint8), ValueError, "单通道"),
        ]
        for image, error_type, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(error_type, message),
            ):
                prepare_grayscale(image)  # type: ignore[arg-type]

    def test_rejects_empty_non_finite_and_out_of_range_images(self) -> None:
        """Enforce a finite, non-empty eight-bit gray domain."""

        cases = [
            (np.empty((0, 2)), "不能为空"),
            (np.array([[np.nan]]), "有限"),
            (np.array([[-1.0]]), r"\[0, 255\]"),
            (np.array([[256.0]]), r"\[0, 255\]"),
        ]
        for image, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                prepare_grayscale(image)


class ImageCalibrationTests(TestCase):
    """Verify phase correction and output naming."""

    def setUp(self) -> None:
        """Keep aberration-cache assertions independent."""

        clear_calibration_cache()

    def test_zero_aberration_preserves_non_wrapping_gray_values(self) -> None:
        """Round-trip gray through phase when the aberration is zero."""

        image = np.array([[0, 64], [128, 254]], dtype=np.uint8)
        actual = calibrate_image(image, 0.0)
        np.testing.assert_array_equal(actual, image)
        self.assertEqual(actual.dtype, np.uint8)

    def test_applies_aberration_and_accepts_column_coefficients(self) -> None:
        """Apply a nonzero mode using MATLAB-style column coefficients."""

        image = np.full((3, 3), 128, dtype=np.uint8)
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("error", RuntimeWarning)
            actual = calibrate_image(image, np.array([[0.2], [0.1]]))
        self.assertEqual(actual.shape, image.shape)
        self.assertFalse(np.array_equal(actual, image))
        self.assertGreaterEqual(int(actual.min()), 0)
        self.assertLessEqual(int(actual.max()), 255)

    def test_reuses_aberration_for_matching_shape_and_coefficients(self) -> None:
        """Avoid rebuilding modes for every image in a same-sized batch."""

        image = np.full((4, 4), 64, dtype=np.uint8)
        with patch("app.correction.rect_zernike_recon") as reconstruct:
            reconstruct.return_value = np.zeros((4, 4), dtype=np.float64)
            calibrate_image(image, [0.0])
            calibrate_image(image + 1, [0.0])
        self.assertEqual(reconstruct.call_count, 1)

    def test_uses_standard_fast_path_and_falls_back_when_unavailable(self) -> None:
        """Prefer precomputed modes but retain a reliable realtime fallback."""

        image = np.full((4, 4), 64, dtype=np.uint8)
        patches = (
            patch("app.correction.STANDARD_HEIGHT", 4),
            patch("app.correction.STANDARD_WIDTH", 4),
        )
        with (
            patches[0],
            patches[1],
            patch(
                "app.correction.reconstruct_standard_aberration",
                return_value=np.zeros((4, 4), dtype=np.float32),
            ) as standard,
            patch("app.correction.rect_zernike_recon") as realtime,
        ):
            calibrate_image(image, [0.0])
        self.assertEqual(standard.call_count, 1)
        self.assertEqual(realtime.call_count, 0)

        clear_calibration_cache()
        with (
            patches[0],
            patches[1],
            patch(
                "app.correction.reconstruct_standard_aberration",
                return_value=None,
            ),
            patch("app.correction.rect_zernike_recon") as realtime,
        ):
            realtime.return_value = np.zeros((4, 4), dtype=np.float64)
            calibrate_image(image, [0.1])
        self.assertEqual(realtime.call_count, 1)

    def test_preloads_standard_cache(self) -> None:
        """Report whether startup successfully mapped the standard cache."""

        with patch("app.correction.load_standard_mode_cache", return_value=np.ones(1)):
            self.assertTrue(preload_correction_cache())
        with patch("app.correction.load_standard_mode_cache", return_value=None):
            self.assertFalse(preload_correction_cache())

    def test_rejects_invalid_coefficient_arrays(self) -> None:
        """Reject nonnumeric, nonvector, empty, and non-finite coefficients."""

        image = np.zeros((3, 3), dtype=np.uint8)
        cases = [
            (["bad"], TypeError, "实数"),
            (np.ones((2, 2)), ValueError, "一维向量"),
            (np.ones((1, 1, 1)), ValueError, "一维向量"),
            ([], ValueError, "不能为空"),
            ([np.nan], ValueError, "有限"),
        ]
        for coefficients, error_type, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(error_type, message),
            ):
                calibrate_image(image, coefficients)

    def test_builds_corrected_output_name_and_validates_format(self) -> None:
        """Append the required Chinese suffix before a normalized extension."""

        self.assertEqual(
            corrected_output_name(Path("somewhere/origin.tif"), ".PNG"),
            "origin_校正后.png",
        )
        self.assertEqual(
            corrected_output_name("photo.jpg", "jpg"),
            "photo_校正后.jpg",
        )
        with self.assertRaisesRegex(ValueError, "jpg、png 或 bmp"):
            corrected_output_name("photo.jpg", "gif")


if __name__ == "__main__":  # pragma: no cover - unittest discovery entry point
    main()
