"""Measured gray-level and phase-response conversion utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

_TWO_PI: Final = 2 * np.pi
_FloatArray = NDArray[np.float64]


def response_data(filename: str | Path) -> _FloatArray:
    """Read a measured gray-to-phase response curve from a CSV file.

    The file must contain two columns: gray level followed by phase in radians.
    The first non-empty row may optionally be a case-insensitive ``gray,phase``
    header.  Commas or whitespace are accepted as separators, blank lines and
    lines beginning with ``#`` are ignored, and rows are sorted by gray level.

    Args:
        filename: Path to the response CSV file.  UTF-8 and UTF-8 with BOM are
            supported.

    Returns:
        A ``float64`` array with shape ``(N, 2)`` sorted by increasing gray.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If a row is malformed, fewer than two samples are present,
            gray levels are duplicated, or values fall outside gray ``[0, 255]``
            and phase ``[0, 2*pi]``.
    """

    path = Path(filename)
    rows: list[tuple[float, float]] = []
    header_seen = False

    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        for line_number, raw_line in enumerate(csv_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = _split_row(line)
            if not rows and not header_seen and _is_header(fields):
                header_seen = True
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"CSV row {line_number} must contain exactly two columns"
                )
            try:
                rows.append((float(fields[0]), float(fields[1])))
            except ValueError as error:
                raise ValueError(
                    f"CSV row {line_number} contains non-numeric data"
                ) from error

    if len(rows) < 2:
        raise ValueError("Response data must contain at least two samples")

    data = np.asarray(rows, dtype=np.float64)
    _validate_response_ranges(data)
    data = data[np.argsort(data[:, 0], kind="stable")]
    if np.any(np.diff(data[:, 0]) == 0):
        raise ValueError("Response data contains duplicate gray levels")

    logger.info("Loaded %d response samples from %s", len(data), path)
    return data


def gray2phase(gray: np.ndarray, data: ArrayLike) -> _FloatArray:
    """Map gray levels to phase using piecewise-linear interpolation.

    Values between measured gray levels are linearly interpolated.  When the
    measurements do not include gray 0 or 255, values outside the measured
    interval are clamped to the nearest measured phase rather than extrapolated.

    Args:
        gray: Real numeric gray-level array whose values lie in ``[0, 255]``.
        data: Response data returned by :func:`response_data`, or another
            two-column array containing ``(gray, phase)`` samples.

    Returns:
        A ``float64`` phase array with the same shape as ``gray``.

    Raises:
        TypeError: If ``gray`` is not a real numeric NumPy array.
        ValueError: If input values or response data are invalid.
    """

    gray_values = _validate_values(gray, "gray", 255.0)
    response = _validate_response_data(data)
    return np.asarray(
        np.interp(gray_values, response[:, 0], response[:, 1]),
        dtype=np.float64,
    )


def phase2gray(phase: np.ndarray, data: ArrayLike) -> _FloatArray:
    """Map phase to gray level using inverse piecewise-linear interpolation.

    The measured phase column must be strictly monotonic so that the inverse is
    unique.  Both increasing and decreasing response curves are supported.
    Values outside the measured phase interval are clamped to its endpoint gray
    levels.  Fractional gray levels are retained; callers may round or quantize
    them according to their hardware requirements.

    Args:
        phase: Real numeric phase array in radians with values in ``[0, 2*pi]``.
        data: Response data returned by :func:`response_data`, or another
            two-column array containing ``(gray, phase)`` samples.

    Returns:
        A ``float64`` gray-level array with the same shape as ``phase``.

    Raises:
        TypeError: If ``phase`` is not a real numeric NumPy array.
        ValueError: If input values or response data are invalid, or phase is
            not strictly monotonic with gray level.
    """

    phase_values = _validate_values(phase, "phase", _TWO_PI)
    response = _validate_response_data(data)
    phase_difference = np.diff(response[:, 1])
    if np.all(phase_difference > 0):
        phase_axis = response[:, 1]
        gray_axis = response[:, 0]
    elif np.all(phase_difference < 0):
        phase_axis = response[::-1, 1]
        gray_axis = response[::-1, 0]
    else:
        raise ValueError(
            "Response phase must be strictly monotonic for inverse mapping"
        )
    return np.asarray(np.interp(phase_values, phase_axis, gray_axis), dtype=np.float64)


def _split_row(line: str) -> list[str]:
    if "," in line:
        return [field.strip() for field in line.split(",")]
    return line.split()


def _is_header(fields: list[str]) -> bool:
    return len(fields) == 2 and [field.casefold() for field in fields] == [
        "gray",
        "phase",
    ]


def _validate_values(values: np.ndarray, name: str, maximum: float) -> _FloatArray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if values.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values")
    converted = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any((converted < 0) | (converted > maximum)):
        raise ValueError(f"{name} values must lie in [0, {maximum}]")
    return converted


def _validate_response_data(data: ArrayLike) -> _FloatArray:
    raw = np.asarray(data)
    if raw.dtype.kind not in "biuf":
        raise TypeError("data must contain real numeric values")
    response = np.asarray(raw, dtype=np.float64)
    if response.ndim != 2 or response.shape[1] != 2 or response.shape[0] < 2:
        raise ValueError("data must have shape (N, 2) with N >= 2")
    _validate_response_ranges(response)
    order = np.argsort(response[:, 0], kind="stable")
    response = response[order]
    if np.any(np.diff(response[:, 0]) == 0):
        raise ValueError("data contains duplicate gray levels")
    return response


def _validate_response_ranges(data: _FloatArray) -> None:
    if not np.all(np.isfinite(data)):
        raise ValueError("Response data must contain only finite values")
    if np.any((data[:, 0] < 0) | (data[:, 0] > 255)):
        raise ValueError("Response gray levels must lie in [0, 255]")
    if np.any((data[:, 1] < 0) | (data[:, 1] > _TWO_PI)):
        raise ValueError("Response phases must lie in [0, 2*pi]")
