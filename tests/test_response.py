"""Tests for measured gray-level and phase-response conversion."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np

from zernike_tool import gray2phase, phase2gray, response_data


class ResponseDataTests(TestCase):
    """Verify response CSV parsing and validation."""

    def test_reads_header_comments_bom_and_sorts_rows(self) -> None:
        """Accept the documented header while normalizing row order."""

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "response.csv"
            filename.write_text(
                "\ufeffGRAY, phase\n\n# measured values\n255, 6.5\n0, 0\n64, 1.51\n",
                encoding="utf-8",
            )
            actual = response_data(filename)

        np.testing.assert_array_equal(actual[:, 0], [0, 64, 255])
        np.testing.assert_allclose(actual[:, 1], [0, 1.51, 6.5])
        self.assertEqual(actual.dtype, np.float64)

    def test_reads_headerless_and_whitespace_separated_files(self) -> None:
        """Support both optional headers and the example's separators."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            headerless = root / "headerless.csv"
            headerless.write_text("0,0\n255,6.2\n", encoding="utf-8")
            whitespace = root / "whitespace.csv"
            whitespace.write_text(
                "gray phase\n0 0\n255 6.2\n",
                encoding="utf-8",
            )

            headerless_data = response_data(str(headerless))
            self.assertEqual(headerless_data.shape, (2, 2))
            np.testing.assert_allclose(headerless_data, [[0, 0], [255, 6.2]])
            np.testing.assert_allclose(
                response_data(whitespace),
                [[0, 0], [255, 6.2]],
            )

    def test_rejects_malformed_or_insufficient_csv(self) -> None:
        """Report row locations for malformed input and require two samples."""

        cases = [
            ("gray,phase\n0,0,extra\n", "exactly two columns"),
            ("wrong,header\n0,0\n", "row 1 contains non-numeric"),
            ("gray,phase\n0,0\n", "at least two samples"),
        ]
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "invalid.csv"
            for content, message in cases:
                with self.subTest(message=message):
                    filename.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        response_data(filename)

    def test_rejects_invalid_response_values(self) -> None:
        """Reject duplicates, non-finite values, and invalid lower bounds."""

        cases = [
            ("0,0\n0,1\n", "duplicate gray"),
            ("0,0\n256,1\n", "gray levels"),
            ("0,0\n255,-0.1\n", "nonnegative"),
            ("0,0\n255,nan\n", "finite values"),
        ]
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "invalid-values.csv"
            for content, message in cases:
                with self.subTest(message=message):
                    filename.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        response_data(filename)


class ResponseInterpolationTests(TestCase):
    """Verify linear design mapping and measured inverse interpolation."""

    def setUp(self) -> None:
        """Create a sparse, deliberately unsorted response curve."""

        self.data = np.array(
            [
                [192.0, 5.0],
                [32.0, 0.5],
                [96.0, 2.0],
            ]
        )

    def test_gray_to_phase_uses_linear_design_mapping(self) -> None:
        """Ignore measured phases and map the full gray range onto pi."""

        gray = np.array([[0, 32, 64], [96, 192, 255]], dtype=np.uint8)
        actual = gray2phase(gray, self.data)
        expected = gray.astype(np.float64) / 255 * np.pi
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.shape, gray.shape)
        self.assertEqual(actual.dtype, np.float64)

    def test_phase_to_gray_supports_increasing_and_decreasing_curves(self) -> None:
        """Invert strictly monotonic response curves in either direction."""

        phase = np.array([[0.0, 0.5, 1.25, 5.0, 2 * np.pi]])
        np.testing.assert_allclose(
            phase2gray(phase, self.data),
            [[32.0, 32.0, 64.0, 192.0, 192.0]],
        )

        decreasing = np.array([[0.0, 2 * np.pi], [255.0, 0.0]])
        np.testing.assert_allclose(
            phase2gray(np.array([0.0, np.pi, 2 * np.pi]), decreasing),
            [255.0, 127.5, 0.0],
        )

    def test_phase_to_gray_truncates_response_above_two_pi(self) -> None:
        """Interpolate a cutoff gray below 255 at the two-pi crossing."""

        response = np.array([[0.0, 0.0], [200.0, 5.8], [255.0, 6.6]])
        cutoff_gray = 200 + (2 * np.pi - 5.8) / (6.6 - 5.8) * 55
        actual = phase2gray(np.array([5.8, 2 * np.pi]), response)
        np.testing.assert_allclose(actual, [200.0, cutoff_gray])
        self.assertLess(float(actual[-1]), 255)

    def test_phase_to_gray_rejects_non_monotonic_response(self) -> None:
        """Reject inverse mappings that have no unique solution."""

        response_curves = (
            np.array([[0, 0.0], [128, 2.0], [255, 1.0]]),
            np.array([[0, 0.0], [128, 0.0], [255, 1.0]]),
        )
        for data in response_curves:
            with (
                self.subTest(data=data),
                self.assertRaisesRegex(ValueError, "strictly monotonic"),
            ):
                phase2gray(np.array([0.5]), data)

    def test_rejects_invalid_conversion_inputs(self) -> None:
        """Validate input array type, numeric kind, finiteness, and range."""

        cases = [
            (gray2phase, [0], self.data, TypeError, "numpy.ndarray"),
            (
                gray2phase,
                np.array(["0"]),
                self.data,
                TypeError,
                "real numeric",
            ),
            (
                phase2gray,
                np.array([1j]),
                self.data,
                TypeError,
                "real numeric",
            ),
            (
                gray2phase,
                np.array([np.nan]),
                self.data,
                ValueError,
                "finite",
            ),
            (
                gray2phase,
                np.array([-1]),
                self.data,
                ValueError,
                "lie in",
            ),
            (
                phase2gray,
                np.array([2 * np.pi + 0.1]),
                self.data,
                ValueError,
                "lie in",
            ),
        ]
        for function, values, data, error_type, message in cases:
            with (
                self.subTest(function=function.__name__, message=message),
                self.assertRaisesRegex(error_type, message),
            ):
                function(values, data)  # type: ignore[arg-type]

    def test_rejects_invalid_response_arrays(self) -> None:
        """Apply CSV-equivalent checks to directly supplied response arrays."""

        cases = [
            (np.array([["0", "0"], ["1", "1"]]), TypeError, "real numeric"),
            (np.array([0.0, 1.0]), ValueError, "shape"),
            (np.ones((2, 3)), ValueError, "shape"),
            (np.array([[0.0, 0.0]]), ValueError, "shape"),
            (
                np.array([[0.0, 0.0], [1.0, np.inf]]),
                ValueError,
                "finite",
            ),
            (np.array([[-1.0, 0.0], [1.0, 1.0]]), ValueError, "gray levels"),
            (np.array([[0.0, 0.0], [1.0, -0.1]]), ValueError, "nonnegative"),
            (np.array([[1.0, 0.0], [1.0, 1.0]]), ValueError, "duplicate gray"),
        ]
        for data, error_type, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(error_type, message),
            ):
                gray2phase(np.array([0.0]), data)


if __name__ == "__main__":  # pragma: no cover - unittest discovery entry point
    main()
