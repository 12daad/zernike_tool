"""Image loading and export helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_FORMATS = {
    ".bmp": "BMP",
    ".jpeg": "JPEG",
    ".jgp": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
}


def load_image(filename: str | Path) -> np.ndarray:
    """Load an image into an independent NumPy array.

    The image's decoded mode is preserved: grayscale images normally produce a
    two-dimensional array, while RGB and RGBA images produce arrays whose final
    dimension contains three and four channels respectively.  The returned
    array owns its data and remains usable after the source file is closed.

    Args:
        filename: Path to an image format supported by Pillow, including BMP,
            JPEG, and PNG.

    Returns:
        A NumPy array containing the decoded image pixels.

    Raises:
        OSError: If the file does not exist, cannot be read, or is not a valid
            supported image.
    """

    input_path = Path(filename)
    with Image.open(input_path) as image:
        result = np.array(image)
    logger.info("Loaded image from %s", input_path)
    return result


def export_image(nparray: np.ndarray, filename: str | Path) -> None:
    """Export a NumPy array as a BMP, JPEG, or PNG image.

    A filename without a suffix receives ``.png``.  The commonly mistyped
    ``.jgp`` suffix is accepted as a JPEG alias in addition to ``.jpg`` and
    ``.jpeg``.

    Two-dimensional arrays are exported as grayscale.  Three-dimensional
    arrays must contain one, three, or four channels.  ``uint8`` data is kept
    unchanged; booleans map to 0 and 255.  Other real numeric data in ``[0, 1]``
    is scaled to ``[0, 255]``, data already in ``[0, 255]`` keeps its scale,
    and other data is min-max normalized using finite values.  Non-finite
    pixels become zero.  Alpha is discarded when writing JPEG.

    Missing parent directories are created recursively and reported at info
    level.  Existing files are overwritten, also with an info-level message.

    Args:
        nparray: Numeric image array with shape ``(height, width)`` or
            ``(height, width, channels)``.
        filename: Output path.  Supported suffixes are ``.bmp``, ``.png``,
            ``.jpg``, ``.jpeg``, and ``.jgp`` (case-insensitive).

    Raises:
        TypeError: If ``nparray`` is not a NumPy array or is not real numeric.
        ValueError: If its shape is unsupported, it has no finite numeric
            values, or the filename extension is unsupported.
        OSError: If the output directory cannot be created or the image cannot
            be written.
    """

    if not isinstance(nparray, np.ndarray):
        raise TypeError("nparray must be a numpy.ndarray")
    if nparray.dtype.kind not in "biuf":
        raise TypeError("nparray must contain real numeric values")
    if nparray.ndim == 2:
        image_array = nparray
    elif nparray.ndim == 3 and nparray.shape[2] in (1, 3, 4):
        image_array = nparray[..., 0] if nparray.shape[2] == 1 else nparray
    else:
        raise ValueError(
            "nparray must have shape (height, width) or (height, width, 1|3|4)"
        )

    output_path = Path(filename)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    suffix = output_path.suffix.lower()
    try:
        image_format = _FORMATS[suffix]
    except KeyError as error:
        raise ValueError(f"Unsupported image extension: {suffix}") from error

    converted = _to_uint8(image_array)
    image = Image.fromarray(converted)
    if image_format == "JPEG" and image.mode == "RGBA":
        image = image.convert("RGB")

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
        logger.info("Created output directory %s", output_path.parent)
    if output_path.exists():
        logger.info("Overwriting existing image %s", output_path)
    image.save(output_path, format=image_format)
    logger.info("Exported image to %s", output_path)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if array.dtype.kind == "b":
        return array.astype(np.uint8) * 255

    numeric = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(numeric)
    if not np.any(finite):
        raise ValueError("nparray must contain at least one finite value")
    minimum = float(np.min(numeric[finite]))
    maximum = float(np.max(numeric[finite]))

    if minimum >= 0 and maximum <= 1:
        scaled = numeric * 255
    elif minimum >= 0 and maximum <= 255:
        scaled = numeric
    elif maximum == minimum:
        scaled = np.zeros_like(numeric)
    else:
        scaled = (numeric - minimum) * (255 / (maximum - minimum))

    return np.rint(np.where(finite, scaled, 0)).clip(0, 255).astype(np.uint8)
