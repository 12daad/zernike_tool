"""Testable image-correction logic used by the desktop application."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zernike_tool import rect_zernike_recon
from zernike_tool.precomputed import (
    STANDARD_HEIGHT,
    STANDARD_NOLL_COUNT,
    STANDARD_WIDTH,
    load_standard_mode_cache,
    reconstruct_standard_aberration,
)

logger = logging.getLogger(__name__)

_COEFFICIENT_SEPARATOR: Final = re.compile(r"[\s,]+")
_SUPPORTED_EXPORT_FORMATS: Final = frozenset({"bmp", "jpg", "png"})
_TWO_PI: Final = 2 * np.pi
_FloatArray = NDArray[np.float64]
_ByteArray = NDArray[np.uint8]
_Float32Array = NDArray[np.float32]


def parse_coefficients(text: str) -> _FloatArray:
    """Parse comma-, whitespace-, or newline-separated Zernike coefficients.

    Args:
        text: User-entered coefficient text, for example ``"1, 2 3\n4"``.

    Returns:
        A one-dimensional ``float64`` array in the entered order.

    Raises:
        ValueError: If no coefficients are supplied, a token is not numeric,
            or a coefficient is non-finite.
    """

    stripped = text.strip(" \t\r\n,")
    if not stripped:
        raise ValueError("请输入至少一个 Zernike 系数")
    try:
        coefficients = np.asarray(
            [float(token) for token in _COEFFICIENT_SEPARATOR.split(stripped)],
            dtype=np.float64,
        )
    except ValueError as error:
        raise ValueError("Zernike 系数必须是用逗号或空白分隔的数字") from error
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Zernike 系数必须是有限数值")
    return coefficients


def prepare_grayscale(image: np.ndarray) -> _FloatArray:
    """Validate an imported image and return its single active channel.

    A two-dimensional image or an array with one channel is accepted directly.
    A three-channel image is accepted only when at most one channel contains
    nonzero pixels; that channel is used.  Other channel layouts are rejected.
    Pixel values must be finite and lie in the eight-bit range ``[0, 255]``.

    Args:
        image: Decoded image array.

    Returns:
        A two-dimensional ``float64`` grayscale array.

    Raises:
        TypeError: If ``image`` is not a real numeric NumPy array.
        ValueError: If the channel layout or pixel range is invalid.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("图片必须是 numpy.ndarray")
    if image.dtype.kind not in "biuf":
        raise TypeError("图片必须包含实数像素")

    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 1:
        grayscale = image[..., 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        active_channels = [
            channel for channel in range(3) if np.any(image[..., channel] != 0)
        ]
        if len(active_channels) > 1:
            raise ValueError("三通道图片只能有一个通道包含非零像素")
        channel = active_channels[0] if active_channels else 0
        grayscale = image[..., channel]
    else:
        raise ValueError("图片必须是单通道或仅一个有效通道的三通道图片")

    converted = np.asarray(grayscale, dtype=np.float64)
    if converted.size == 0:
        raise ValueError("图片不能为空")
    if not np.all(np.isfinite(converted)):
        raise ValueError("图片像素必须是有限数值")
    if np.any((converted < 0) | (converted > 255)):
        raise ValueError("图片灰度必须位于 [0, 255]")
    return converted


def calibrate_image(image: np.ndarray, coefficients: ArrayLike) -> _ByteArray:
    """Correct one image with a rectangular Zernike aberration model.

    For an image with original gray value ``g``, the calculation is::

        phase = g / 255 * 2*pi
        phase_calibrated = phase - rect_zernike_recon(X, Y, coefficients)
        gray_calibrated = mod(phase_calibrated, 2*pi) / (2*pi) * 255

    ``X`` and ``Y`` are unit-spaced pixel-coordinate matrices matching the
    image dimensions.  Results are rounded to the nearest eight-bit value.

    Args:
        image: A valid single-active-channel image.
        coefficients: Non-empty vector for consecutive one-based Noll modes.

    Returns:
        A two-dimensional ``uint8`` corrected image.

    Raises:
        TypeError: If the image or coefficients are not real numeric data.
        ValueError: If their shapes, ranges, or values are invalid.
    """

    grayscale = prepare_grayscale(image)
    coefficient_array = _validate_coefficients(coefficients)
    height, width = grayscale.shape
    aberration = _cached_aberration(
        height,
        width,
        tuple(float(value) for value in coefficient_array),
    )
    phase = np.asarray(grayscale, dtype=np.float32) * np.float32(_TWO_PI / 255)
    calibrated_phase = np.mod(phase - aberration, _TWO_PI)
    calibrated_gray = calibrated_phase / _TWO_PI * 255
    result: _ByteArray = np.asarray(np.rint(calibrated_gray), dtype=np.uint8)
    return result


def preload_correction_cache() -> bool:
    """Memory-map the packaged 1920x1080 mode cache.

    Returns:
        ``True`` when the cache is available and valid, otherwise ``False``.
    """

    return load_standard_mode_cache() is not None


def clear_calibration_cache() -> None:
    """Clear cached aberration matrices without closing the mode mmap."""

    _cached_aberration.cache_clear()


def corrected_output_name(source: str | Path, image_format: str) -> str:
    """Build the corrected filename for a selected export format.

    Args:
        source: Original image path.
        image_format: One of ``"jpg"``, ``"png"``, or ``"bmp"``.  A leading
            dot and mixed case are accepted.

    Returns:
        A filename such as ``"origin_校正后.png"``.

    Raises:
        ValueError: If the requested image format is unsupported.
    """

    normalized_format = image_format.casefold().lstrip(".")
    if normalized_format not in _SUPPORTED_EXPORT_FORMATS:
        raise ValueError("导出格式必须是 jpg、png 或 bmp")
    return f"{Path(source).stem}_校正后.{normalized_format}"


def _validate_coefficients(coefficients: ArrayLike) -> _FloatArray:
    raw = np.asarray(coefficients)
    if raw.dtype.kind not in "biuf":
        raise TypeError("Zernike 系数必须是实数")
    if raw.ndim == 0:
        raw = raw.reshape(1)
    elif raw.ndim > 2 or (raw.ndim == 2 and 1 not in raw.shape):
        raise ValueError("Zernike 系数必须是一维向量")
    result = np.asarray(raw.ravel(order="F"), dtype=np.float64)
    if result.size == 0:
        raise ValueError("Zernike 系数不能为空")
    if not np.all(np.isfinite(result)):
        raise ValueError("Zernike 系数必须是有限数值")
    return result


@lru_cache(maxsize=4)
def _cached_aberration(
    height: int,
    width: int,
    coefficients: tuple[float, ...],
) -> _Float32Array:
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if (
        height == STANDARD_HEIGHT
        and width == STANDARD_WIDTH
        and len(coefficients) <= STANDARD_NOLL_COUNT
    ):
        precomputed = reconstruct_standard_aberration(coefficient_array)
        if precomputed is not None:
            logger.info("使用 1920x1080 预计算 Zernike 模式")
            return precomputed
        logger.warning("预计算模式不可用，回退到实时 Zernike 重建")

    logger.info("实时重建 %dx%d Zernike 像差", width, height)
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    return np.asarray(
        rect_zernike_recon(x_coordinates, y_coordinates, coefficient_array),
        dtype=np.float32,
    )
