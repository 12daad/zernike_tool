"""Memory-mapped precomputed Zernike modes for common image dimensions."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from .storage import default_standard_cache_path

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
            tests.  By default the packaged cache is tried first, followed by
            the writable per-user cache.

    Returns:
        A read-only memory-mapped ``float32`` array, or ``None`` when the cache
        is unavailable or invalid.  Results are retained by an LRU cache.
    """

    candidates = (
        (Path(filename),)
        if filename is not None
        else (STANDARD_CACHE_PATH, default_standard_cache_path())
    )
    checked_paths: list[Path] = []
    for path in candidates:
        if path in checked_paths:
            continue
        checked_paths.append(path)
        loaded = _load_mode_cache(path)
        if loaded is not None:
            return loaded
    logger.warning(
        "未找到有效的 1920x1080 Zernike 缓存：%s",
        ", ".join(str(path) for path in checked_paths),
    )
    return None


def generate_standard_mode_cache(filename: str | Path | None = None) -> Path:
    """Generate and atomically save the standard float32 mode cache.

    Args:
        filename: Optional output path.  The writable per-user cache path is
            used by default.

    Returns:
        The generated cache path.

    Raises:
        OSError: If the cache cannot be written.
        ValueError: If mode generation fails because of invalid numeric data.
    """

    # Imported lazily so normal cache loading does not initialize internals
    # needed only by the expensive first-run generation path.
    from .zernike import _calculate_mode_values

    path = Path(filename) if filename is not None else default_standard_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.npy")
    temporary_path.unlink(missing_ok=True)
    logger.info(
        "开始生成 Noll 1:%d、%dx%d 的 Zernike 缓存",
        STANDARD_NOLL_COUNT,
        STANDARD_WIDTH,
        STANDARD_HEIGHT,
    )
    try:
        x_coordinates, y_coordinates = np.meshgrid(
            np.arange(STANDARD_WIDTH, dtype=np.float64),
            np.arange(STANDARD_HEIGHT, dtype=np.float64),
        )
        modes = _calculate_mode_values(
            x_coordinates,
            y_coordinates,
            STANDARD_NOLL_COUNT,
        )
        np.save(
            temporary_path,
            np.asarray(modes, dtype=np.float32),
            allow_pickle=False,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    load_standard_mode_cache.cache_clear()
    logger.info(
        "Zernike 缓存生成完成（%.1f MiB）：%s",
        path.stat().st_size / (1024 * 1024),
        path,
    )
    return path


def _load_mode_cache(path: Path) -> _Float32Array | None:
    if not path.is_file():
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
