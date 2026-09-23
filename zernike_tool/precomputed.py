"""Memory-mapped precomputed Zernike modes for common image dimensions."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

STANDARD_HEIGHT: Final = 1080
STANDARD_WIDTH: Final = 1920
STANDARD_NOLL_COUNT: Final = 30
STANDARD_CACHE_SHAPE: Final = (
    STANDARD_HEIGHT,
    STANDARD_WIDTH,
    STANDARD_NOLL_COUNT,
)
STANDARD_CACHE_PATH: Final = (
    Path(__file__).with_name("data") / "rect_zernike_1920x1080_noll_1_30_float32.npy"
)
_Float32Array = NDArray[np.float32]


@lru_cache(maxsize=2)
def load_standard_mode_cache(
    filename: str | Path | None = None,
) -> _Float32Array | None:
    """Load and validate the standard mode cache using read-only mmap.

    Args:
        filename: Optional alternate cache path, primarily for diagnostics and
            tests.  The packaged 1920x1080 cache is used by default.

    Returns:
        A read-only memory-mapped ``float32`` array, or ``None`` when the cache
        is unavailable or invalid.  Results are retained by an LRU cache.
    """

    path = Path(filename) if filename is not None else STANDARD_CACHE_PATH
    if not path.is_file():
        logger.warning("未找到 1920x1080 Zernike 缓存：%s", path)
        return None
    try:
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        logger.warning("无法加载 Zernike 缓存 %s：%s", path, error)
        return None
    if loaded.shape != STANDARD_CACHE_SHAPE or loaded.dtype != np.float32:
        logger.warning(
            "Zernike 缓存格式无效：期望 shape=%s、dtype=float32，实际为 %s、%s",
            STANDARD_CACHE_SHAPE,
            loaded.shape,
            loaded.dtype,
        )
        return None
    loaded.flags.writeable = False
    logger.info("已内存映射 1920x1080 Zernike 缓存：%s", path)
    return cast(_Float32Array, loaded)


def reconstruct_standard_aberration(
    coefficients: NDArray[np.float64],
) -> _Float32Array | None:
    """Reconstruct a 1920x1080 aberration from precomputed modes.

    Args:
        coefficients: One-dimensional coefficients for consecutive Noll modes.

    Returns:
        A ``float32`` aberration array, or ``None`` when the packaged cache is
        unavailable or more than 30 modes are requested.

    Raises:
        ValueError: If coefficients are empty, non-finite, or not one-dimensional.
    """

    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("coefficients must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must contain finite values")
    if coefficients.size > STANDARD_NOLL_COUNT:
        return None
    modes = load_standard_mode_cache()
    if modes is None:
        return None
    weights = np.asarray(coefficients, dtype=np.float32)
    return np.asarray(modes[..., : weights.size] @ weights, dtype=np.float32)
