"""Generate the packaged 1920x1080 Noll 1:30 float32 mode cache."""

from __future__ import annotations

import logging

from zernike_tool.precomputed import (
    STANDARD_CACHE_PATH,
    generate_standard_mode_cache,
)


def main() -> None:
    """Generate and atomically replace the standard cache file."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    output = generate_standard_mode_cache(STANDARD_CACHE_PATH)
    logger.info("Saved cache to %s", output)


if __name__ == "__main__":
    main()
