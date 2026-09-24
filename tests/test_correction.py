"""Tests for the GUI-independent image-correction domain logic."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from zernike_tool import gray2phase, phase2gray
from zernike_tool.app import (
    calibrate_image,
    clear_calibration_cache,
    corrected_output_name,
    parse_coefficients,
    parse_response_curve,
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


class ResponseCurveParsingTests(TestCase):
    """Verify paired gray and phase response text parsing."""

    def test_parses_separators_and_sorts_by_gray(self) -> None:
        """Keep gray/phase pairs aligned while sorting the gray axis."""

        actual = parse_response_curve("255, 0 128", "6.5\n0, 2.9")
        np.testing.assert_array_equal(actual[:, 0], [0, 128, 255])
        np.testing.assert_array_equal(actual[:, 1], [0, 2.9, 6.5])
        self.assertEqual(actual.dtype, np.float64)

    def test_rejects_malformed_or_mismatched_sequences(self) -> None:
        """Reject missing, nonnumeric, mismatched, and undersized inputs."""

        cases = [
            ("", "0, 1", "请输入灰度值"),
            ("0, bad", "0, 1", "灰度值必须是"),
            ("0, 255", "0", "数量必须相同"),
            ("0", "0", "至少需要两个"),
        ]
        for gray_text, phase_text, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                parse_response_curve(gray_text, phase_text)

    def test_rejects_invalid_ranges_duplicates_and_nonmonotonic_phase(self) -> None:
        """Apply the response conversion API's range and inverse rules."""

        cases = [
            ("0, 256", "0, 1", "gray values"),
            ("0, 255", "0, -0.1", "nonnegative"),
            ("0, 0", "0, 1", "duplicate gray"),
            ("0, 128, 255", "0, 2, 1", "strictly monotonic"),
        ]
        for gray_text, phase_text, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                parse_response_curve(gray_text, phase_text)


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

    def test_default_inverse_maps_design_pi_range_to_half_gray_range(self) -> None:
        """Use pi for the design phase and two-pi for the default response."""

        image = np.array([[0, 64], [128, 254]], dtype=np.uint8)
        actual = calibrate_image(image, 0.0)
        np.testing.assert_array_equal(actual, [[0, 32], [64, 127]])
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

    def test_uses_design_forward_and_measured_inverse_mappings(self) -> None:
        """Use linear design phase and measured inverse response in correction."""

        image = np.array([[32, 96], [160, 224]], dtype=np.uint8)
        response = np.array(
            [[0.0, 0.0], [128.0, 1.5], [255.0, 2 * np.pi]],
            dtype=np.float64,
        )
        aberration = np.full(image.shape, 0.2, dtype=np.float32)
        with patch(
                "zernike_tool.app.correction._cached_aberration",
                return_value=aberration,
        ):
            actual = calibrate_image(image, [0.0], response)

        corrected_phase = np.mod(gray2phase(image, response) - aberration, 2 * np.pi)
        expected = np.rint(phase2gray(corrected_phase, response)).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.uint8)

    def test_reuses_aberration_for_matching_shape_and_coefficients(self) -> None:
        """Avoid rebuilding modes for every image in a same-sized batch."""

        image = np.full((4, 4), 64, dtype=np.uint8)
        with patch("zernike_tool.app.correction.rect_zernike_recon") as reconstruct:
            reconstruct.return_value = np.zeros((4, 4), dtype=np.float64)
            calibrate_image(image, [0.0])
            calibrate_image(image + 1, [0.0])
        self.assertEqual(reconstruct.call_count, 1)

    def test_uses_standard_fast_path_and_falls_back_when_unavailable(self) -> None:
        """Prefer precomputed modes but retain a reliable realtime fallback."""

        image = np.full((4, 4), 64, dtype=np.uint8)
        patches = (
            patch("zernike_tool.app.correction.STANDARD_HEIGHT", 4),
            patch("zernike_tool.app.correction.STANDARD_WIDTH", 4),
        )
        with (
            patches[0],
            patches[1],
            patch(
                "zernike_tool.app.correction.reconstruct_standard_aberration",
                return_value=np.zeros((4, 4), dtype=np.float32),
            ) as standard,
            patch("zernike_tool.app.correction.rect_zernike_recon") as realtime,
        ):
            calibrate_image(image, [0.0])
        self.assertEqual(standard.call_count, 1)
        self.assertEqual(realtime.call_count, 0)

        clear_calibration_cache()
        with (
            patches[0],
            patches[1],
            patch(
                "zernike_tool.app.correction.reconstruct_standard_aberration",
                return_value=None,
            ),
            patch("zernike_tool.app.correction.rect_zernike_recon") as realtime,
        ):
            realtime.return_value = np.zeros((4, 4), dtype=np.float64)
            calibrate_image(image, [0.1])
        self.assertEqual(realtime.call_count, 1)

    def test_preloads_standard_cache(self) -> None:
        """Report whether startup successfully mapped the standard cache."""

        with patch(
                "zernike_tool.app.correction.load_standard_mode_cache",
                return_value=np.ones(1),
        ) as load:
            self.assertTrue(preload_correction_cache())
            load.assert_called_once_with(None)
        with patch(
                "zernike_tool.app.correction.load_standard_mode_cache",
                return_value=None,
        ) as load:
            filename = Path("custom.npy")
            self.assertFalse(preload_correction_cache(filename))
            load.assert_called_once_with(filename)

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
