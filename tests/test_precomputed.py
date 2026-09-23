"""Tests for the memory-mapped standard Zernike mode cache."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

from zernike_tool import precomputed


class StandardModeCacheTests(TestCase):
    """Verify cache validation, mmap loading, and reconstruction."""

    def setUp(self) -> None:
        """Prevent one test's LRU result from affecting another."""

        precomputed.load_standard_mode_cache.cache_clear()

    def tearDown(self) -> None:
        """Release references to temporary memory maps."""

        precomputed.load_standard_mode_cache.cache_clear()

    def test_loads_valid_float32_cache_as_read_only_mmap(self) -> None:
        """Validate shape and dtype before retaining an mmap in the LRU."""

        expected = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "valid.npy"
            np.save(filename, expected)
            with patch.object(precomputed, "STANDARD_CACHE_SHAPE", expected.shape):
                actual = precomputed.load_standard_mode_cache(filename)
                repeated = precomputed.load_standard_mode_cache(filename)

                self.assertIs(actual, repeated)
                self.assertIsNotNone(actual)
                assert actual is not None
                self.assertFalse(actual.flags.writeable)
                np.testing.assert_array_equal(actual, expected)
                precomputed.load_standard_mode_cache.cache_clear()
                if isinstance(actual, np.memmap):
                    # Windows keeps the file locked until the mmap is closed.
                    actual._mmap.close()  # type: ignore[union-attr]

    def test_returns_none_for_missing_corrupt_or_invalid_cache(self) -> None:
        """Fall back safely for every invalid on-disk cache category."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(
                precomputed.load_standard_mode_cache(root / "missing.npy")
            )

            corrupt = root / "corrupt.npy"
            corrupt.write_bytes(b"not a numpy file")
            self.assertIsNone(precomputed.load_standard_mode_cache(corrupt))

            wrong = root / "wrong.npy"
            np.save(wrong, np.zeros((1, 2), dtype=np.float64))
            self.assertIsNone(precomputed.load_standard_mode_cache(wrong))

    def test_reconstructs_float32_aberration(self) -> None:
        """Combine only the requested cached modes with float32 weights."""

        modes = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        coefficients = np.array([0.5, -0.25], dtype=np.float64)
        with patch.object(
            precomputed,
            "load_standard_mode_cache",
            return_value=modes,
        ):
            actual = precomputed.reconstruct_standard_aberration(coefficients)
        expected = modes[..., 0] * 0.5 + modes[..., 1] * -0.25
        self.assertIsNotNone(actual)
        assert actual is not None
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_reconstruction_validation_and_fallbacks(self) -> None:
        """Reject malformed coefficients and signal unavailable fast paths."""

        invalid = [
            np.empty(0, dtype=np.float64),
            np.ones((1, 1), dtype=np.float64),
        ]
        for coefficients in invalid:
            with (
                self.subTest(shape=coefficients.shape),
                self.assertRaisesRegex(ValueError, "non-empty one-dimensional"),
            ):
                precomputed.reconstruct_standard_aberration(coefficients)
        with self.assertRaisesRegex(ValueError, "finite"):
            precomputed.reconstruct_standard_aberration(np.array([np.nan]))
        self.assertIsNone(
            precomputed.reconstruct_standard_aberration(
                np.zeros(precomputed.STANDARD_NOLL_COUNT + 1)
            )
        )
        with patch.object(
            precomputed,
            "load_standard_mode_cache",
            return_value=None,
        ):
            self.assertIsNone(
                precomputed.reconstruct_standard_aberration(np.array([0.0]))
            )


if __name__ == "__main__":  # pragma: no cover - unittest discovery entry point
    main()
