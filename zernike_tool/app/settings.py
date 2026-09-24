"""Persistent GUI settings stored as UTF-8 JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_COEFFICIENTS: Final = "0"
DEFAULT_GRAY_RESPONSE: Final = "0, 255"
DEFAULT_PHASE_RESPONSE: Final = "0, 6.283185307179586"


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Values restored between application sessions."""

    import_directory: str
    export_directory: str
    coefficients: str
    gray_response: str
    phase_response: str


def default_settings(initial_directory: Path) -> AppSettings:
    """Create first-run settings rooted at the application start directory."""

    directory = str(initial_directory.resolve())
    return AppSettings(
        import_directory=directory,
        export_directory=directory,
        coefficients=DEFAULT_COEFFICIENTS,
        gray_response=DEFAULT_GRAY_RESPONSE,
        phase_response=DEFAULT_PHASE_RESPONSE,
    )


def load_settings(filename: str | Path, initial_directory: Path) -> AppSettings:
    """Load settings, falling back safely when the JSON is absent or invalid.

    Args:
        filename: Settings JSON path.
        initial_directory: Existing directory used for missing or stale paths.

    Returns:
        Valid settings suitable for direct use by the GUI.
    """

    path = Path(filename)
    defaults = default_settings(initial_directory)
    if not path.is_file():
        logger.info("未找到历史配置，使用首次运行默认值：%s", path)
        return defaults

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("无法读取历史配置，使用默认值：%s", error)
        return defaults
    if not isinstance(payload, dict):
        logger.warning("历史配置格式无效，使用默认值：%s", path)
        return defaults

    settings = AppSettings(
        import_directory=_restored_directory(
            payload.get("import_directory"),
            defaults.import_directory,
        ),
        export_directory=_restored_directory(
            payload.get("export_directory"),
            defaults.export_directory,
        ),
        coefficients=_restored_text(
            payload.get("coefficients"),
            defaults.coefficients,
        ),
        gray_response=_restored_text(
            payload.get("gray_response"),
            defaults.gray_response,
        ),
        phase_response=_restored_text(
            payload.get("phase_response"),
            defaults.phase_response,
        ),
    )
    logger.info("已恢复上次配置：%s", path)
    return settings


def save_settings(settings: AppSettings, filename: str | Path) -> None:
    """Atomically save application settings as UTF-8 JSON.

    Args:
        settings: Values to persist.
        filename: Destination JSON path.

    Raises:
        OSError: If the destination directory or file cannot be written.
    """

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    serialized = json.dumps(asdict(settings), ensure_ascii=False, indent=2)
    temporary_path.write_text(f"{serialized}\n", encoding="utf-8")
    temporary_path.replace(path)
    logger.info("已保存当前路径和校正参数：%s", path)


def _restored_directory(value: object, fallback: str) -> str:
    if isinstance(value, str) and Path(value).is_dir():
        return value
    return fallback


def _restored_text(value: object, fallback: str) -> str:
    return value if isinstance(value, str) else fallback
