"""Zernike modes orthonormalized over a rectangular sampling aperture.

The definitions in this module are direct translations of the MATLAB files in
``MATLAB_Ref``.  In particular, inner products are discrete means rather than
continuous aperture integrals, and Noll indices are one-based.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import blake2b
from math import factorial
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

_FloatArray = NDArray[np.float64]
_CACHE_CAPACITY: Final = 4


@dataclass(frozen=True, slots=True)
class _CachedModes:
    """A complete sequence of modes for one aperture geometry."""

    maximum_noll: int
    z: _FloatArray
    dz_dx: _FloatArray
    dz_dy: _FloatArray


@dataclass(frozen=True, slots=True)
class _CoordinateCacheKey:
    """Hashable coordinate content plus arrays used only on a cache miss."""

    digest: bytes
    x: _FloatArray = field(compare=False, hash=False, repr=False)
    y: _FloatArray = field(compare=False, hash=False, repr=False)


def clear_rect_zernike_cache() -> None:
    """Remove every cached rectangular Zernike basis."""

    _calculate_modes_cached.cache_clear()
    logger.debug("Cleared the rectangular Zernike cache")


def rect_zernike(
    x_coordinates: ArrayLike,
    y_coordinates: ArrayLike,
    i_noll: ArrayLike | None = None,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Calculate rectangular-aperture Zernike modes and their gradients.

    Coordinates are centered at the mean of their jointly finite samples and
    divided by the greatest sample radius.  Cartesian Zernike modes from one
    through the largest requested one-based Noll index are then transformed by
    a reduced QR factorization so that ``mean(Z_i * Z_j) == delta_ij`` over
    finite aperture samples.  Gradients are returned with respect to the
    original, unnormalized coordinates.  Values outside the finite aperture
    are zero, exactly as in the MATLAB reference.

    Args:
        x_coordinates: Two-dimensional X-coordinate matrix, normally produced
            by :func:`numpy.meshgrid`.
        y_coordinates: Y-coordinate matrix with the same shape as
            ``x_coordinates``.
        i_noll: Positive, one-based Noll indices.  The default is 1 through 30.
            Indices may be repeated and need not be sorted.
        use_cache: Reuse and populate a four-aperture in-memory LRU cache.
        refresh_cache: Recompute the basis even if a suitable cache entry
            exists.  Existing LRU entries are cleared before recomputation.

    Returns:
        A tuple ``(Z, dZ_dX, dZ_dY)``.  Each array has shape
        ``x_coordinates.shape + (len(i_noll),)``.

    Raises:
        TypeError: If coordinates or indices are not real numeric values.
        ValueError: If inputs have invalid shapes, there are fewer finite
            samples than requested modes, or the aperture has zero radius.
        numpy.linalg.LinAlgError: If the sampled basis is rank deficient.
    """

    x, y = _validate_coordinates(x_coordinates, y_coordinates)
    indices = _validate_noll_indices(i_noll)
    maximum_noll = int(indices.max())

    if use_cache:
        if refresh_cache:
            _calculate_modes_cached.cache_clear()
        key = _coordinate_cache_key(x, y)
        cache_before = _calculate_modes_cached.cache_info()
        computed = _calculate_modes_cached(key, maximum_noll)
        cache_after = _calculate_modes_cached.cache_info()
        if cache_after.hits > cache_before.hits:
            logger.debug(
                "Rectangular Zernike LRU cache hit for modes 1:%d", maximum_noll
            )
    else:
        logger.debug("Computing uncached rectangular Zernike modes 1:%d", maximum_noll)
        computed = _calculate_modes(x, y, maximum_noll)
    return _select_modes(computed, indices)


def rect_zernike_coef(
    x_coordinates: ArrayLike,
    y_coordinates: ArrayLike,
    phi: ArrayLike,
    i_noll: ArrayLike,
) -> _FloatArray:
    """Project a sampled field onto rectangular Zernike modes.

    This implements ``sum(Z_i * phi) / (rows * columns)`` from the MATLAB
    reference.  Thus, the denominator includes every matrix location, while
    modes at non-finite coordinate locations are zero.

    Args:
        x_coordinates: Two-dimensional X-coordinate matrix.
        y_coordinates: Matching Y-coordinate matrix.
        phi: Real numeric field with the same shape as the coordinates.
        i_noll: Positive, one-based Noll indices to project onto.

    Returns:
        One coefficient per requested Noll index, in request order.

    Raises:
        TypeError: If ``phi`` is not a real numeric array.
        ValueError: If ``phi`` does not match the coordinate shape.
    """

    x, y = _validate_coordinates(x_coordinates, y_coordinates)
    field = _as_real_numeric_array(phi, "phi")
    if field.ndim != 2 or field.shape != x.shape:
        raise ValueError("phi must be a two-dimensional array matching X and Y")

    modes, _, _ = rect_zernike(x, y, i_noll)
    return np.sum(modes * field[..., np.newaxis], axis=(0, 1)) / field.size


def rect_zernike_recon(
    x_coordinates: ArrayLike,
    y_coordinates: ArrayLike,
    coef: ArrayLike,
) -> _FloatArray:
    """Reconstruct a field from consecutive rectangular Zernike coefficients.

    ``coef[k]`` multiplies one-based Noll mode ``k + 1``, matching the MATLAB
    reference implementation.

    Args:
        x_coordinates: Two-dimensional X-coordinate matrix.
        y_coordinates: Matching Y-coordinate matrix.
        coef: Non-empty vector of real coefficients for modes 1 through M.

    Returns:
        The reconstructed two-dimensional field.

    Raises:
        TypeError: If coefficients are not real numeric values.
        ValueError: If coefficients are not a non-empty vector.
    """

    coefficients = _as_real_numeric_array(coef, "coef")
    if coefficients.ndim == 0:
        coefficients = coefficients.reshape(1)
    elif coefficients.ndim > 2 or (
        coefficients.ndim == 2 and 1 not in coefficients.shape
    ):
        raise ValueError("coef must be a vector")
    coefficients = coefficients.ravel(order="F")
    if coefficients.size == 0:
        raise ValueError("coef must not be empty")

    x, y = _validate_coordinates(x_coordinates, y_coordinates)
    modes = _calculate_mode_values(x, y, coefficients.size)
    return np.sum(modes * coefficients, axis=-1)


def _validate_coordinates(
    x_coordinates: ArrayLike,
    y_coordinates: ArrayLike,
) -> tuple[_FloatArray, _FloatArray]:
    x = _as_real_numeric_array(x_coordinates, "X")
    y = _as_real_numeric_array(y_coordinates, "Y")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional matrices")
    if x.shape != y.shape:
        raise ValueError("X and Y must have identical shapes")
    return x, y


def _as_real_numeric_array(value: ArrayLike, name: str) -> _FloatArray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values")
    return np.asarray(array, dtype=np.float64)


def _validate_noll_indices(i_noll: ArrayLike | None) -> NDArray[np.int64]:
    if i_noll is None:
        return np.arange(1, 31, dtype=np.int64)

    raw = np.asarray(i_noll)
    if raw.size == 0:
        raise ValueError("i_noll must contain positive one-based indices")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError("i_noll must contain integers")
    if raw.ndim == 0:
        raw = raw.reshape(1)
    elif raw.ndim > 2 or (raw.ndim == 2 and 1 not in raw.shape):
        raise ValueError("i_noll must be a vector")
    indices = np.asarray(raw.ravel(order="F"), dtype=np.int64)
    if np.any(indices < 1):
        raise ValueError("i_noll must contain positive one-based indices")
    return indices


def _coordinate_cache_key(x: _FloatArray, y: _FloatArray) -> _CoordinateCacheKey:
    digest = blake2b(digest_size=16)
    digest.update(str(x.shape).encode())
    digest.update(_array_digest(x))
    digest.update(_array_digest(y))
    return _CoordinateCacheKey(digest.digest(), x, y)


def _array_digest(array: _FloatArray) -> bytes:
    contiguous = np.ascontiguousarray(array)
    return blake2b(contiguous.view(np.uint8), digest_size=16).digest()


@lru_cache(maxsize=_CACHE_CAPACITY)
def _calculate_modes_cached(
    key: _CoordinateCacheKey,
    maximum_noll: int,
) -> _CachedModes:
    logger.debug("Rectangular Zernike LRU cache miss for modes 1:%d", maximum_noll)
    return _calculate_modes(key.x, key.y, maximum_noll)


def _select_modes(
    modes: _CachedModes,
    indices: NDArray[np.int64],
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    selection = indices - 1
    return (
        modes.z[..., selection].copy(),
        modes.dz_dx[..., selection].copy(),
        modes.dz_dy[..., selection].copy(),
    )


def _calculate_modes(
    x_coordinates: _FloatArray,
    y_coordinates: _FloatArray,
    maximum_noll: int,
) -> _CachedModes:
    mask_flat, x_normalized, y_normalized, maximum_radius = _normalize_aperture(
        x_coordinates,
        y_coordinates,
        maximum_noll,
    )
    sample_count = x_normalized.size
    original = np.zeros((sample_count, maximum_noll), dtype=np.float64)
    original_dx = np.zeros_like(original)
    original_dy = np.zeros_like(original)

    for column in range(maximum_noll):
        radial_order, azimuthal_frequency = _noll_index(column + 1)
        values, gradient_x, gradient_y = _zernike_cartesian(
            radial_order,
            azimuthal_frequency,
            x_normalized,
            y_normalized,
        )
        original[:, column] = values
        original_dx[:, column] = gradient_x
        original_dy[:, column] = gradient_y

    _, upper = np.linalg.qr(original, mode="reduced")
    upper /= np.sqrt(sample_count)
    transform = np.linalg.inv(upper)
    modes = original @ transform
    modes_dx = original_dx @ transform / maximum_radius
    modes_dy = original_dy @ transform / maximum_radius

    output_shape = (*x_coordinates.shape, maximum_noll)
    output = np.zeros((x_coordinates.size, maximum_noll), dtype=np.float64)
    output_dx = np.zeros_like(output)
    output_dy = np.zeros_like(output)
    output[mask_flat, :] = modes
    output_dx[mask_flat, :] = modes_dx
    output_dy[mask_flat, :] = modes_dy

    return _CachedModes(
        maximum_noll,
        output.reshape(output_shape, order="F"),
        output_dx.reshape(output_shape, order="F"),
        output_dy.reshape(output_shape, order="F"),
    )


def _calculate_mode_values(
    x_coordinates: _FloatArray,
    y_coordinates: _FloatArray,
    maximum_noll: int,
) -> _FloatArray:
    """Calculate only mode values, avoiding derivative allocation."""

    mask_flat, x_normalized, y_normalized, _ = _normalize_aperture(
        x_coordinates,
        y_coordinates,
        maximum_noll,
    )
    sample_count = x_normalized.size
    original = np.zeros((sample_count, maximum_noll), dtype=np.float64)
    for column in range(maximum_noll):
        radial_order, azimuthal_frequency = _noll_index(column + 1)
        original[:, column] = _zernike_cartesian(
            radial_order,
            azimuthal_frequency,
            x_normalized,
            y_normalized,
        )[0]

    orthonormal, _ = np.linalg.qr(original, mode="reduced")
    orthonormal *= np.sqrt(sample_count)
    output_shape = (*x_coordinates.shape, maximum_noll)
    if sample_count == x_coordinates.size:
        return orthonormal.reshape(output_shape, order="F")
    output = np.zeros((x_coordinates.size, maximum_noll), dtype=np.float64)
    output[mask_flat, :] = orthonormal
    return output.reshape(output_shape, order="F")


def _normalize_aperture(
    x_coordinates: _FloatArray,
    y_coordinates: _FloatArray,
    maximum_noll: int,
) -> tuple[NDArray[np.bool_], _FloatArray, _FloatArray, np.float64]:
    mask = np.isfinite(x_coordinates) & np.isfinite(y_coordinates)
    sample_count = int(np.count_nonzero(mask))
    if sample_count < maximum_noll:
        raise ValueError("Too few sampling points")

    mask_flat = mask.ravel(order="F")
    x_flat = x_coordinates.ravel(order="F")[mask_flat]
    y_flat = y_coordinates.ravel(order="F")[mask_flat]
    x_center = np.mean(x_flat)
    y_center = np.mean(y_flat)
    x_centered = x_flat - x_center
    y_centered = y_flat - y_center
    maximum_radius = np.max(np.hypot(x_centered, y_centered))
    if maximum_radius <= 0:
        raise ValueError("Invalid aperture: maximum radius must be positive")

    x_normalized = x_centered / maximum_radius
    y_normalized = y_centered / maximum_radius
    return mask_flat, x_normalized, y_normalized, maximum_radius


def _noll_index(index: int) -> tuple[int, int]:
    radial_order = 0
    count = 0
    while index > count + radial_order + 1:
        count += radial_order + 1
        radial_order += 1

    position = index - count
    if radial_order % 2 == 0:
        if position == 1:
            azimuthal_frequency = 0
        else:
            offset = position - 1
            magnitude = 2 * ((offset + 1) // 2)
            azimuthal_frequency = -magnitude if offset % 2 == 1 else magnitude
    else:
        azimuthal_frequency = -radial_order + 2 * (position - 1)
    return radial_order, azimuthal_frequency


def _zernike_cartesian(
    radial_order: int,
    azimuthal_frequency: int,
    x: _FloatArray,
    y: _FloatArray,
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    magnitude = abs(azimuthal_frequency)
    values = np.zeros_like(x)
    gradient_x = np.zeros_like(x)
    gradient_y = np.zeros_like(x)
    radius_squared = x**2 + y**2
    complex_coordinate = x + 1j * y
    angular_power = complex_coordinate**magnitude
    previous_power = (
        np.ones_like(complex_coordinate)
        if magnitude <= 1
        else complex_coordinate ** (magnitude - 1)
    )

    for k in range((radial_order - magnitude) // 2 + 1):
        coefficient = (
            (-1) ** k
            * factorial(radial_order - k)
            / (
                factorial(k)
                * factorial((radial_order + magnitude) // 2 - k)
                * factorial((radial_order - magnitude) // 2 - k)
            )
        )
        power = radial_order - 2 * k
        radial_power = (power - magnitude) // 2

        if magnitude == 0:
            angular = np.ones_like(x)
            angular_dx = np.zeros_like(x)
            angular_dy = np.zeros_like(x)
        elif azimuthal_frequency > 0:
            angular = angular_power.real
            angular_dx = magnitude * previous_power.real
            angular_dy = -magnitude * previous_power.imag
        else:
            angular = angular_power.imag
            angular_dx = magnitude * previous_power.imag
            angular_dy = magnitude * previous_power.real

        if radial_power == 0:
            radial = np.ones_like(x)
            radial_dx = np.zeros_like(x)
            radial_dy = np.zeros_like(x)
        else:
            radial = radius_squared**radial_power
            preceding_radial = radius_squared ** (radial_power - 1)
            radial_dx = 2 * radial_power * x * preceding_radial
            radial_dy = 2 * radial_power * y * preceding_radial

        values += coefficient * angular * radial
        gradient_x += coefficient * (angular_dx * radial + angular * radial_dx)
        gradient_y += coefficient * (angular_dy * radial + angular * radial_dy)

    return values, gradient_x, gradient_y
