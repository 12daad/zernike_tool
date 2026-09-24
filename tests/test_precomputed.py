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

    def test_default_loading_falls_back_to_writable_user_cache(self) -> None:
        """Try the packaged file first and then the per-user cache."""

        expected = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            filename = root / "user.npy"
            np.save(filename, expected)
            with (
                patch.object(precomputed, "STANDARD_CACHE_PATH", root / "missing.npy"),
                patch.object(precomputed, "STANDARD_CACHE_SHAPE", expected.shape),
                patch(
                    "zernike_tool.precomputed.default_standard_cache_path",
                    return_value=filename,
                ),
            ):
                actual = precomputed.load_standard_mode_cache()
            self.assertIsNotNone(actual)
            assert actual is not None
            np.testing.assert_array_equal(actual, expected)
            precomputed.load_standard_mode_cache.cache_clear()
            if isinstance(actual, np.memmap):
                actual._mmap.close()  # type: ignore[union-attr]

    def test_default_loading_deduplicates_identical_candidates(self) -> None:
        """Avoid checking the same missing packaged/user path twice."""

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "missing.npy"
            with (
                patch.object(precomputed, "STANDARD_CACHE_PATH", filename),
                patch(
                    "zernike_tool.precomputed.default_standard_cache_path",
                    return_value=filename,
                ),
                self.assertLogs("zernike_tool.precomputed", level="WARNING"),
            ):
                self.assertIsNone(precomputed.load_standard_mode_cache())

    def test_generates_float32_cache_atomically_at_default_path(self) -> None:
        """Build the standard grid, convert modes, and remove the temp file."""

        modes = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "nested" / "cache.npy"
            with (
                patch.object(precomputed, "STANDARD_HEIGHT", 2),
                patch.object(precomputed, "STANDARD_WIDTH", 3),
                patch.object(precomputed, "STANDARD_NOLL_COUNT", 2),
                patch(
                    "zernike_tool.precomputed.default_standard_cache_path",
                    return_value=filename,
                ),
                patch(
                    "zernike_tool.zernike._calculate_mode_values",
                    return_value=modes,
                ) as calculate,
            ):
                actual_path = precomputed.generate_standard_mode_cache()

            self.assertEqual(actual_path, filename)
            generated = np.load(filename, allow_pickle=False)
            self.assertEqual(generated.dtype, np.float32)
            np.testing.assert_array_equal(generated, modes)
            self.assertFalse(filename.with_suffix(".tmp.npy").exists())
            x_coordinates, y_coordinates, count = calculate.call_args.args
            np.testing.assert_array_equal(x_coordinates[0], [0, 1, 2])
            np.testing.assert_array_equal(y_coordinates[:, 0], [0, 1])
            self.assertEqual(count, 2)

    def test_generation_cleans_temporary_file_after_failure(self) -> None:
        """Never leave a partial file that a later launch might consume."""

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "cache.npy"
            temporary = filename.with_suffix(".tmp.npy")
            temporary.write_bytes(b"stale")
            with (
                patch.object(precomputed, "STANDARD_HEIGHT", 2),
                patch.object(precomputed, "STANDARD_WIDTH", 3),
                patch.object(precomputed, "STANDARD_NOLL_COUNT", 2),
                patch(
                    "zernike_tool.zernike._calculate_mode_values",
                    side_effect=ValueError("bad modes"),
                ),
                self.assertRaisesRegex(ValueError, "bad modes"),
            ):
                precomputed.generate_standard_mode_cache(filename)
            self.assertFalse(filename.exists())
            self.assertFalse(temporary.exists())

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
