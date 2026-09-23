"""Generate the packaged 1920x1080 Noll 1:30 float32 mode cache."""

from __future__ import annotations

import logging

import numpy as np

from zernike_tool.precomputed import (
    STANDARD_CACHE_PATH,
    STANDARD_HEIGHT,
    STANDARD_NOLL_COUNT,
    STANDARD_WIDTH,
)

# noinspection PyProtectedMember
from zernike_tool.zernike import _calculate_mode_values


def main() -> None:
    """Generate and atomically replace the standard cache file."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(
        "Generating modes 1:%d for %dx%d",
        STANDARD_NOLL_COUNT,
        STANDARD_WIDTH,
        STANDARD_HEIGHT,
    )
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(STANDARD_WIDTH, dtype=np.float64),
        np.arange(STANDARD_HEIGHT, dtype=np.float64),
    )
    modes = _calculate_mode_values(
        x_coordinates,
        y_coordinates,
        STANDARD_NOLL_COUNT,
    )
    STANDARD_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = STANDARD_CACHE_PATH.with_suffix(".tmp.npy")
    np.save(temporary_path, np.asarray(modes, dtype=np.float32), allow_pickle=False)
    temporary_path.replace(STANDARD_CACHE_PATH)
    logger.info(
        "Saved %.1f MiB to %s",
        STANDARD_CACHE_PATH.stat().st_size / (1024 * 1024),
        STANDARD_CACHE_PATH,
    )


if __name__ == "__main__":
    main()
