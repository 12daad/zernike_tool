"""Tests for the rectangular Zernike implementation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from zernike_tool import (
    clear_rect_zernike_cache,
    rect_zernike,
    rect_zernike_coef,
    rect_zernike_recon,
)
from zernike_tool import zernike as implementation


class RectZernikeTests(unittest.TestCase):
    """Verify MATLAB-compatible values, gradients, and cache behavior."""

    def setUp(self) -> None:
        """Start every test with an empty process-level cache."""

        clear_rect_zernike_cache()
        self.x, self.y = np.meshgrid(
            np.linspace(-2, 2, 5),
            np.linspace(-1, 1, 4),
        )

    def test_matches_matlab_reference_and_is_orthonormal(self) -> None:
        """Match an R2025b reference result and its normalization."""

        modes, gradient_x, gradient_y = rect_zernike(
            self.x,
            self.y,
            np.arange(1, 7),
            use_cache=False,
        )
        expected_first_sample = np.array(
            [
                -1.0,
                -1.341640786499874,
                -1.414213562373095,
                -1.411881798915156,
                -1.897366596101028,
                0.659667500458462,
            ]
        )
        expected_dx = np.array(
            [
                0.0,
                0.0,
                0.707106781186548,
                2.310352033078730,
                0.948683298050514,
                0.613644191515010,
            ]
        )
        expected_dy = np.array(
            [
                0.0,
                1.341640786499874,
                0.0,
                1.155176016539365,
                1.897366596101028,
                -4.349203187418706,
            ]
        )
        np.testing.assert_allclose(modes[0, 0], expected_first_sample, atol=1e-14)
        np.testing.assert_allclose(gradient_x[0, 0], expected_dx, atol=1e-14)
        np.testing.assert_allclose(gradient_y[0, 0], expected_dy, atol=1e-14)

        flattened = modes.reshape((-1, modes.shape[-1]))
        np.testing.assert_allclose(
            flattened.T @ flattened / flattened.shape[0],
            np.eye(6),
            atol=1e-14,
        )

    def test_mask_and_requested_mode_order(self) -> None:
        """Use jointly finite samples and preserve repetitions and order."""

        x = self.x.copy()
        y = self.y.copy()
        x[1, 2] = np.nan
        y[2, 3] = np.inf
        modes, gradient_x, gradient_y = rect_zernike(x, y, [3, 1, 3])
        self.assertEqual(modes.shape, (4, 5, 3))
        np.testing.assert_array_equal(modes[..., 0], modes[..., 2])
        self.assertTrue(np.all(modes[[1, 2], [2, 3]] == 0))
        self.assertTrue(np.all(gradient_x[[1, 2], [2, 3]] == 0))
        self.assertTrue(np.all(gradient_y[[1, 2], [2, 3]] == 0))

    def test_default_modes_and_scalar_index(self) -> None:
        """Provide modes 1:30 by default and accept one scalar index."""

        x, y = np.meshgrid(np.linspace(-1, 1, 6), np.linspace(-1, 1, 6))
        modes, _, _ = rect_zernike(x, y, use_cache=False)
        scalar, _, _ = rect_zernike(x, y, np.int64(1), use_cache=False)
        self.assertEqual(modes.shape, (6, 6, 30))
        self.assertEqual(scalar.shape, (6, 6, 1))

    def test_cache_hit_growth_refresh_copy_and_eviction(self) -> None:
        """Exercise the complete bounded cache lifecycle."""

        original_calculator = implementation._calculate_modes
        with patch.object(
            implementation,
            "_calculate_modes",
            wraps=original_calculator,
        ) as calculator:
            first, _, _ = rect_zernike(self.x, self.y, [1, 2, 3])
            cached, _, _ = rect_zernike(self.x, self.y, [3, 1])
            self.assertEqual(calculator.call_count, 1)
            np.testing.assert_array_equal(cached[..., 0], first[..., 2])

            cached[0, 0, 0] = 999
            pristine, _, _ = rect_zernike(self.x, self.y, [3])
            self.assertNotEqual(pristine[0, 0, 0], 999)

            rect_zernike(self.x, self.y, [4])
            rect_zernike(self.x, self.y, [1], refresh_cache=True)
            rect_zernike(self.x, self.y, [1], use_cache=False)
            self.assertEqual(calculator.call_count, 4)

            for offset in range(5):
                rect_zernike(self.x + offset + 10, self.y, [1])
            self.assertEqual(
                implementation._calculate_modes_cached.cache_info().currsize, 4
            )

    def test_coefficients_and_reconstruction(self) -> None:
        """Projection recovers coefficients for an exactly spanned field."""

        expected = np.array([2.0, -0.5, 1.25, 0.75])
        field = rect_zernike_recon(self.x, self.y, expected)
        actual = rect_zernike_coef(self.x, self.y, field, [1, 2, 3, 4])
        np.testing.assert_allclose(actual, expected, atol=1e-14)

        scalar_reconstruction = rect_zernike_recon(self.x, self.y, 2)
        column_reconstruction = rect_zernike_recon(
            self.x,
            self.y,
            np.array([[2.0], [-0.5]]),
        )
        self.assertEqual(scalar_reconstruction.shape, self.x.shape)
        np.testing.assert_allclose(
            column_reconstruction,
            rect_zernike_recon(self.x, self.y, [2.0, -0.5]),
        )

    def test_reconstruction_preserves_mask_and_matches_modes(self) -> None:
        """Use the value-only QR path without changing mode conventions."""

        modes, _, _ = rect_zernike(self.x, self.y, [1, 2, 3], use_cache=False)
        for index in range(3):
            coefficients = np.zeros(3)
            coefficients[index] = 1
            reconstructed = rect_zernike_recon(self.x, self.y, coefficients)
            np.testing.assert_allclose(reconstructed, modes[..., index], atol=1e-14)

        x = self.x.copy()
        y = self.y.copy()
        x[0, 0] = np.nan
        y[1, 1] = np.inf
        reconstructed = rect_zernike_recon(x, y, [1.0])
        self.assertEqual(reconstructed[0, 0], 0)
        self.assertEqual(reconstructed[1, 1], 0)

    def test_coordinate_and_index_validation(self) -> None:
        """Reject inputs excluded by the MATLAB API contract."""

        with self.assertRaisesRegex(TypeError, "X must contain"):
            rect_zernike([["x"]], [[1]], [1])
        with self.assertRaisesRegex(TypeError, "Y must contain"):
            rect_zernike([[1]], [[1 + 2j]], [1])
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            rect_zernike([1, 2], [1, 2], [1])
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            rect_zernike(np.ones((2, 2)), np.ones((2, 3)), [1])
        with self.assertRaisesRegex(TypeError, "contain integers"):
            rect_zernike(self.x, self.y, [1.0])
        with self.assertRaisesRegex(TypeError, "contain integers"):
            rect_zernike(self.x, self.y, [True])
        with self.assertRaisesRegex(ValueError, "must be a vector"):
            rect_zernike(self.x, self.y, np.ones((2, 2), dtype=int))
        with self.assertRaisesRegex(ValueError, "must be a vector"):
            rect_zernike(self.x, self.y, np.ones((1, 1, 1), dtype=int))
        with self.assertRaisesRegex(ValueError, "positive"):
            rect_zernike(self.x, self.y, [])
        with self.assertRaisesRegex(ValueError, "positive"):
            rect_zernike(self.x, self.y, [0])
        with self.assertRaisesRegex(ValueError, "Too few"):
            rect_zernike(np.ones((2, 2)), np.ones((2, 2)), [5])
        with self.assertRaisesRegex(ValueError, "maximum radius"):
            rect_zernike(np.ones((2, 2)), np.ones((2, 2)), [1])

    def test_coefficient_and_reconstruction_validation(self) -> None:
        """Validate field and coefficient types, shapes, and emptiness."""

        with self.assertRaisesRegex(TypeError, "phi must contain"):
            rect_zernike_coef(self.x, self.y, [["bad"]], [1])
        with self.assertRaisesRegex(ValueError, "matching X and Y"):
            rect_zernike_coef(self.x, self.y, np.ones((2, 2)), [1])
        with self.assertRaisesRegex(TypeError, "coef must contain"):
            rect_zernike_recon(self.x, self.y, [1j])
        with self.assertRaisesRegex(ValueError, "coef must be a vector"):
            rect_zernike_recon(self.x, self.y, np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, "coef must be a vector"):
            rect_zernike_recon(self.x, self.y, np.ones((1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            rect_zernike_recon(self.x, self.y, [])

    def test_noll_index_sequence(self) -> None:
        """Follow the exact ordering embedded in the MATLAB implementation."""

        expected = [
            (0, 0),
            (1, -1),
            (1, 1),
            (2, 0),
            (2, -2),
            (2, 2),
            (3, -3),
            (3, -1),
            (3, 1),
            (3, 3),
        ]
        self.assertEqual(
            [implementation._noll_index(index) for index in range(1, 11)],
            expected,
        )


if __name__ == "__main__":  # pragma: no cover - unittest discovery entry point
    unittest.main()
