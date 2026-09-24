"""Tkinter desktop interface for batch Zernike image correction."""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import TemporaryDirectory
from threading import Event, Thread, get_ident
from tkinter import filedialog, messagebox, ttk
from typing import Final, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk
from tkinterdnd2 import COPY, DND_FILES, TkinterDnD  # type: ignore[import-untyped]

from zernike_tool import export_image, load_image
from zernike_tool.precomputed import generate_standard_mode_cache
from zernike_tool.storage import (
    default_settings_path as get_default_settings_path,
)
from zernike_tool.storage import (
    default_standard_cache_path as get_default_standard_cache_path,
)

from .correction import (
    calibrate_image,
    corrected_output_name,
    parse_coefficients,
    parse_response_curve,
    preload_correction_cache,
    prepare_grayscale,
)
from .settings import (
    DEFAULT_COEFFICIENTS,
    DEFAULT_GRAY_RESPONSE,
    DEFAULT_PHASE_RESPONSE,
    AppSettings,
)
from .settings import (
    load_settings as load_app_settings,
)
from .settings import (
    save_settings as save_app_settings,
)

logger = logging.getLogger(__name__)

_IMAGE_FILE_TYPES = [
    ("图片文件", "*.png *.jpg *.jpeg *.bmp"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("BMP", "*.bmp"),
    ("所有文件", "*.*"),
]
_PREVIEW_SIZE = (460, 330)
_DND_COPY: Final = cast(str, COPY)
_DND_FILES: Final = cast(str, DND_FILES)


class _DropEvent(Protocol):
    """Describe the portion of a TkDND drop event used by the GUI."""

    data: str


class _DnDWidget(Protocol):
    """Describe drag-and-drop methods injected into Tk widgets."""

    def drop_target_register(self, *dnd_types: str) -> None:
        """Register supported incoming data types."""

    def drag_source_register(self, button: int, *dnd_types: str) -> None:
        """Register a mouse button and outgoing data types."""

    def dnd_bind(
        self,
        sequence: str,
        callback: Callable[[object], object],
    ) -> str | None:
        """Bind a callback to a TkDND virtual event."""


@dataclass(slots=True)
class _ImageRecord:
    source: Path
    pixels: NDArray[np.float64] | NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _CorrectionResult:
    record: _ImageRecord
    pixels: NDArray[np.uint8] | None
    error: str | None
    completed: int
    total: int


@dataclass(frozen=True, slots=True)
class _CorrectionDone:
    cancelled: bool


@dataclass(frozen=True, slots=True)
class _CacheGenerationDone:
    path: Path | None
    error: str | None


class _TextLogHandler(logging.Handler):
    def __init__(self, widget: tk.Text) -> None:
        super().__init__(level=logging.INFO)
        self._widget = widget
        self._main_thread = get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        try:
            if get_ident() == self._main_thread:
                self._append(message)
            else:
                self._widget.after(0, self._append, message)
        except tk.TclError:
            pass

    def _append(self, message: str) -> None:
        self._widget.configure(state=tk.NORMAL)
        self._widget.insert(tk.END, f"{message}\n")
        self._widget.see(tk.END)
        self._widget.configure(state=tk.DISABLED)
        self._widget.update_idletasks()


class ZernikeCorrectorApp:
    """Manage the widgets and user actions of the correction tool."""

    def __init__(
        self,
        root: tk.Tk,
        initial_directory: Path | None = None,
        settings_path: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self._root = root
        self._initial_directory = (initial_directory or Path.cwd()).resolve()
        self._settings_path = settings_path or get_default_settings_path()
        self._user_cache_path = cache_path or get_default_standard_cache_path()
        self._explicit_cache_path = cache_path is not None
        self._import_directory = self._initial_directory
        self._export_directory = self._initial_directory
        self._imported: list[_ImageRecord] = []
        self._corrected: list[_ImageRecord] = []
        self._source_preview_photo: ImageTk.PhotoImage | None = None
        self._corrected_preview_photo: ImageTk.PhotoImage | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zernike")
        self._correction_future: Future[None] | None = None
        self._cancel_event = Event()
        self._correction_queue: SimpleQueue[_CorrectionResult | _CorrectionDone] = (
            SimpleQueue()
        )
        self._correction_failures: list[str] = []
        self._cache_thread: Thread | None = None
        self._cache_queue: SimpleQueue[_CacheGenerationDone] = SimpleQueue()
        self._close_requested = False
        self._drag_export_directory = TemporaryDirectory(
            prefix="zernike-corrector-drag-",
            ignore_cleanup_errors=True,
        )
        self._drag_export_sequence = 0
        self._pending_drag_selection: tuple[int, ...] = ()

        root.title("Zernike Calibration Util")
        root.geometry("800x600")
        root.minsize(940, 680)
        root.protocol("WM_DELETE_WINDOW", self._confirm_close)

        self._build_interface()
        self._configure_gui_logging()
        self._restore_settings()
        logger.info(
            "程序已启动，上次导入路径：%s；上次导出路径：%s",
            self._import_directory,
            self._export_directory,
        )
        cache_filename = self._user_cache_path if self._explicit_cache_path else None
        if preload_correction_cache(cache_filename):
            logger.info("1920x1080 专用 Zernike 缓存已就绪")
        else:
            self._start_cache_generation()

    def _build_interface(self) -> None:
        outer = ttk.Frame(self._root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self._root.rowconfigure(0, weight=1)
        self._root.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.columnconfigure(0, weight=1)

        controls = ttk.LabelFrame(outer, text="校正参数", padding=8)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Zernike 系数：").grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(0, 8),
        )
        self._coefficient_text = tk.Text(controls, height=3, wrap=tk.WORD)
        self._coefficient_text.grid(row=0, column=1, sticky="ew")
        self._coefficient_text.insert("1.0", DEFAULT_COEFFICIENTS)

        ttk.Label(controls, text="灰度值（0~255）：").grid(
            row=1,
            column=0,
            sticky="nw",
            padx=(0, 8),
            pady=(8, 0),
        )
        self._gray_response_text = tk.Text(controls, height=2, wrap=tk.WORD)
        self._gray_response_text.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self._gray_response_text.insert("1.0", DEFAULT_GRAY_RESPONSE)

        ttk.Label(controls, text="相位值（≥0）：").grid(
            row=2,
            column=0,
            sticky="nw",
            padx=(0, 8),
            pady=(8, 0),
        )
        self._phase_response_text = tk.Text(controls, height=2, wrap=tk.WORD)
        self._phase_response_text.grid(row=2, column=1, sticky="ew", pady=(8, 0))
        self._phase_response_text.insert("1.0", DEFAULT_PHASE_RESPONSE)

        ttk.Button(controls, text="批量导入图片", command=self._import_images).grid(
            row=0,
            column=2,
            rowspan=3,
            padx=(8, 4),
            sticky="ns",
        )
        self._correct_button = ttk.Button(
            controls,
            text="校正选中图片",
            command=self._correct_selected,
        )
        self._correct_button.grid(
            row=0,
            column=3,
            rowspan=3,
            padx=(4, 0),
            sticky="ns",
        )
        progress_frame = ttk.Frame(controls)
        progress_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        progress_frame.columnconfigure(0, weight=1)
        self._progress_value = tk.DoubleVar(value=0)
        self._progress = ttk.Progressbar(
            progress_frame,
            variable=self._progress_value,
            maximum=100,
            mode="determinate",
        )
        self._progress.grid(row=0, column=0, sticky="ew")
        self._progress_status = tk.StringVar(value="就绪")
        ttk.Label(progress_frame, textvariable=self._progress_status, width=24).grid(
            row=0,
            column=1,
            padx=(8, 4),
        )
        self._cancel_button = ttk.Button(
            progress_frame,
            text="取消校正",
            command=self._cancel_correction,
            state=tk.DISABLED,
        )
        self._cancel_button.grid(row=0, column=2)

        panes = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        panes.grid(row=1, column=0, sticky="nsew")
        source_panel, self._source_list, self._source_preview = self._build_image_panel(
            panes, "已导入图片"
        )
        corrected_panel, self._corrected_list, self._corrected_preview = (
            self._build_image_panel(panes, "校正后图片")
        )
        panes.add(source_panel, weight=1)
        panes.add(corrected_panel, weight=1)

        self._source_list.bind("<<ListboxSelect>>", self._preview_source)
        self._corrected_list.bind("<<ListboxSelect>>", self._preview_corrected)
        self._source_list.bind("<Delete>", self._delete_selected_imported)
        self._corrected_list.bind("<Delete>", self._delete_selected_corrected)
        self._corrected_list.bind(
            "<ButtonPress-1>",
            self._remember_drag_selection,
            add="+",
        )
        self._configure_drag_and_drop(
            (source_panel, self._source_list, self._source_preview),
        )

        source_bar = ttk.Frame(source_panel)
        source_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        source_bar.columnconfigure(0, weight=1)
        self._delete_source_button = ttk.Button(
            source_bar,
            text="删除选中图片",
            command=self._delete_selected_imported,
        )
        self._delete_source_button.grid(row=0, column=1)

        export_bar = ttk.Frame(corrected_panel)
        export_bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        export_bar.columnconfigure(0, weight=1)
        ttk.Label(export_bar, text="导出格式：").grid(row=0, column=1, padx=(4, 4))
        self._export_format = tk.StringVar(value="png")
        ttk.Combobox(
            export_bar,
            textvariable=self._export_format,
            values=("png", "jpg", "bmp"),
            state="readonly",
            width=7,
        ).grid(row=0, column=2, padx=(0, 8))
        self._delete_corrected_button = ttk.Button(
            export_bar,
            text="删除选中图片",
            command=self._delete_selected_corrected,
        )
        self._delete_corrected_button.grid(row=0, column=3, padx=(0, 8))
        ttk.Button(
            export_bar,
            text="导出选中图片",
            command=self._export_selected,
        ).grid(row=0, column=4)

        ttk.Label(
            export_bar,
            text="选中图片后可直接拖到资源管理器",
        ).grid(row=1, column=0, columnspan=5, sticky="e", pady=(5, 0))

        log_frame = ttk.LabelFrame(outer, text="运行日志", padding=6)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self._log_text = tk.Text(log_frame, height=9, state=tk.DISABLED, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient=tk.VERTICAL,
            command=self._log_text.yview,
        )
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

    def _configure_drag_and_drop(
        self,
        import_targets: Iterable[tk.Misc],
    ) -> None:
        for target in import_targets:
            drop_target = cast(_DnDWidget, cast(object, target))
            drop_target.drop_target_register(_DND_FILES)
            drop_target.dnd_bind("<<Drop>>", self._drop_imported_images)

        drag_source = cast(_DnDWidget, cast(object, self._corrected_list))
        drag_source.drag_source_register(1, _DND_FILES)
        drag_source.dnd_bind("<<DragInitCmd>>", self._drag_corrected_images)

    @staticmethod
    def _build_image_panel(
        parent: ttk.PanedWindow,
        title: str,
    ) -> tuple[ttk.LabelFrame, tk.Listbox, ttk.Label]:
        panel = ttk.LabelFrame(parent, text=title, padding=8)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        list_frame = ttk.Frame(panel)
        list_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        list_frame.columnconfigure(0, weight=1)
        image_list = tk.Listbox(
            list_frame,
            name="list",
            height=7,
            selectmode=tk.EXTENDED,
            exportselection=False,
        )
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient=tk.VERTICAL,
            command=image_list.yview,
        )
        image_list.configure(yscrollcommand=scrollbar.set)
        image_list.grid(row=0, column=0, sticky="ew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        preview = ttk.Label(
            panel,
            name="preview",
            text="选择图片后在此预览",
            anchor=tk.CENTER,
            relief=tk.SUNKEN,
        )
        preview.grid(row=1, column=0, sticky="nsew")
        return panel, image_list, preview

    def _configure_gui_logging(self) -> None:
        self._log_handler = _TextLogHandler(self._log_text)
        self._log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root_logger = logging.getLogger()
        self._previous_root_log_level = root_logger.level
        root_logger.addHandler(self._log_handler)
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

    def _restore_settings(self) -> None:
        settings = load_app_settings(self._settings_path, self._initial_directory)
        self._import_directory = Path(settings.import_directory)
        self._export_directory = Path(settings.export_directory)
        self._replace_text(self._coefficient_text, settings.coefficients)
        self._replace_text(self._gray_response_text, settings.gray_response)
        self._replace_text(self._phase_response_text, settings.phase_response)
        logger.info("已恢复上次的 Zernike 系数和灰度-相位数据")

    @staticmethod
    def _replace_text(widget: tk.Text, value: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)

    def _current_settings(self) -> AppSettings:
        return AppSettings(
            import_directory=str(self._import_directory),
            export_directory=str(self._export_directory),
            coefficients=self._coefficient_text.get("1.0", "end-1c"),
            gray_response=self._gray_response_text.get("1.0", "end-1c"),
            phase_response=self._phase_response_text.get("1.0", "end-1c"),
        )

    def _save_settings(self) -> None:
        try:
            save_app_settings(self._current_settings(), self._settings_path)
        except OSError:
            logger.exception("无法保存应用配置：%s", self._settings_path)

    def _start_cache_generation(self) -> None:
        self._correct_button.configure(state=tk.DISABLED)
        self._progress.configure(mode="indeterminate")
        self._progress.start(12)
        self._progress_status.set("正在生成 Zernike 缓存…")
        logger.warning(
            "未找到 1920x1080 专用缓存，正在后台首次生成：%s",
            self._user_cache_path,
        )
        cache_thread = Thread(
            target=self._run_cache_generation,
            name="zernike-cache",
            daemon=True,
        )
        self._cache_thread = cache_thread
        cache_thread.start()
        self._root.after(100, self._poll_cache_generation)

    def _run_cache_generation(self) -> None:
        try:
            path = generate_standard_mode_cache(self._user_cache_path)
        except Exception as error:  # noqa: BLE001 - background-task boundary
            self._cache_queue.put(_CacheGenerationDone(None, str(error)))
        else:
            self._cache_queue.put(_CacheGenerationDone(path, None))

    def _poll_cache_generation(self) -> None:
        try:
            result = self._cache_queue.get_nowait()
        except Empty:
            if self._cache_is_running():
                self._root.after(100, self._poll_cache_generation)
            return

        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress_value.set(0)
        if result.error is not None or result.path is None:
            self._progress_status.set("缓存生成失败")
            logger.error(
                "Zernike 缓存生成失败，将回退为按需实时计算：%s",
                result.error,
            )
        elif preload_correction_cache(result.path):
            self._progress_status.set("缓存已就绪")
            logger.info("后台 Zernike 缓存已生成并加载：%s", result.path)
        else:
            self._progress_status.set("缓存加载失败")
            logger.error("生成的 Zernike 缓存无法加载，将回退为按需实时计算")
        self._correct_button.configure(state=tk.NORMAL)

    def _cache_is_running(self) -> bool:
        return self._cache_thread is not None and self._cache_thread.is_alive()

    def _import_images(self) -> None:
        filenames = filedialog.askopenfilenames(
            parent=self._root,
            title="批量导入图片",
            initialdir=self._import_directory,
            filetypes=_IMAGE_FILE_TYPES,
        )
        if not filenames:
            logger.info("已取消导入")
            return

        self._import_paths(Path(filename) for filename in filenames)

    def _drop_imported_images(self, event: object) -> str:
        drop_event = cast(_DropEvent, event)
        try:
            filenames = self._root.tk.splitlist(drop_event.data)
        except (tk.TclError, ValueError) as error:
            logger.warning("无法解析拖入的文件路径：%s", error)
            return _DND_COPY
        if not filenames:
            logger.info("拖入操作未包含文件")
            return _DND_COPY

        logger.info("收到 %d 个拖入文件", len(filenames))
        self._import_paths(Path(filename) for filename in filenames)
        return _DND_COPY

    def _import_paths(self, filenames: Iterable[Path]) -> None:
        paths = tuple(filenames)
        if paths:
            parent = paths[0].resolve().parent
            if parent.is_dir():
                self._import_directory = parent
                self._save_settings()
        rejected: list[str] = []
        imported_paths = {record.source.resolve() for record in self._imported}
        for filename in paths:
            source = filename
            try:
                resolved = source.resolve()
                if resolved in imported_paths:
                    logger.info("已跳过重复图片：%s", source.name)
                    continue
                pixels = prepare_grayscale(load_image(source))
            except (OSError, TypeError, ValueError) as error:
                rejected.append(f"{source.name}: {error}")
                logger.warning("跳过图片 %s：%s", source, error)
                continue
            self._imported.append(_ImageRecord(source, pixels))
            imported_paths.add(resolved)
            self._source_list.insert(tk.END, source.name)
            logger.info("已导入图片：%s", source)

        if rejected:
            messagebox.showwarning(
                "部分图片未导入",
                "以下图片不符合要求，已跳过：\n\n" + "\n".join(rejected),
                parent=self._root,
            )

    def _drag_corrected_images(
        self,
        _event: object,
    ) -> tuple[str, str, tuple[str, ...]]:
        selection = self._pending_drag_selection or self._selected_indices(
            self._corrected_list
        )
        self._pending_drag_selection = ()
        if not selection:
            logger.warning("请先选择需要拖出导出的校正后图片")
            return _DND_COPY, _DND_FILES, ()

        self._corrected_list.selection_clear(0, tk.END)
        for index in selection:
            self._corrected_list.selection_set(index)

        self._drag_export_sequence += 1
        drag_directory = (
            Path(self._drag_export_directory.name)
            / f"drag-{self._drag_export_sequence:04d}"
        )
        try:
            drag_directory.mkdir(parents=True, exist_ok=False)
        except OSError:
            logger.exception("无法创建拖拽导出临时目录：%s", drag_directory)
            return _DND_COPY, _DND_FILES, ()

        image_format = self._export_format.get()
        staged_paths: list[str] = []
        for index in selection:
            record = self._corrected[index]
            output = drag_directory / corrected_output_name(
                record.source,
                image_format,
            )
            try:
                export_image(record.pixels, output)
            except OSError, TypeError, ValueError:
                logger.exception("拖拽导出准备失败：%s", output)
                continue
            staged_paths.append(str(output.resolve()))

        if staged_paths:
            logger.info(
                "已按 %s 格式准备 %d 张拖拽导出图片",
                image_format.upper(),
                len(staged_paths),
            )
        else:
            logger.warning("没有可拖出的校正后图片")
        return _DND_COPY, _DND_FILES, tuple(staged_paths)

    def _remember_drag_selection(self, event: tk.Event[tk.Misc]) -> None:
        selection = self._selected_indices(self._corrected_list)
        if not selection:
            self._pending_drag_selection = ()
            return

        # Tk's bundled type information leaves nearest() untyped.
        pressed_index = int(
            self._corrected_list.nearest(event.y)  # type: ignore[no-untyped-call]
        )
        row_bounds = self._corrected_list.bbox(pressed_index)
        pressed_selected_row = (
            row_bounds is not None
            and row_bounds[1] <= event.y < row_bounds[1] + row_bounds[3]
            and pressed_index in selection
        )
        self._pending_drag_selection = selection if pressed_selected_row else ()

    def _correct_selected(self) -> None:
        if self._cache_is_running():
            messagebox.showinfo(
                "正在生成缓存",
                "首次运行的 Zernike 缓存仍在后台生成，请稍候。",
                parent=self._root,
            )
            return
        if self._correction_future is not None and not self._correction_future.done():
            messagebox.showinfo(
                "正在校正", "请等待当前校正任务完成。", parent=self._root
            )
            return
        selection = self._selected_indices(self._source_list)
        if not selection:
            messagebox.showwarning(
                "未选择图片", "请先选择需要校正的图片。", parent=self._root
            )
            return
        try:
            coefficients = parse_coefficients(self._coefficient_text.get("1.0", tk.END))
            response = parse_response_curve(
                self._gray_response_text.get("1.0", tk.END),
                self._phase_response_text.get("1.0", tk.END),
            )
        except ValueError as error:
            logger.warning("校正参数无效：%s", error)
            messagebox.showwarning("参数无效", str(error), parent=self._root)
            return

        self._save_settings()
        records = [self._imported[index] for index in selection]
        self._cancel_event.clear()
        self._correction_failures.clear()
        self._progress_value.set(0)
        self._progress_status.set(f"0 / {len(records)}")
        self._correct_button.configure(state=tk.DISABLED)
        self._cancel_button.configure(state=tk.NORMAL)
        self._delete_source_button.configure(state=tk.DISABLED)
        self._delete_corrected_button.configure(state=tk.DISABLED)
        logger.info("开始后台校正 %d 张图片", len(records))
        self._correction_future = self._executor.submit(
            self._run_correction,
            records,
            coefficients.copy(),
            response.copy(),
        )
        self._root.after(50, self._poll_correction_queue)

    def _run_correction(
        self,
        records: list[_ImageRecord],
        coefficients: NDArray[np.float64],
        response: NDArray[np.float64],
    ) -> None:
        total = len(records)
        for completed, record in enumerate(records, start=1):
            if self._cancel_event.is_set():
                break
            try:
                pixels = calibrate_image(record.pixels, coefficients, response)
                error = None
            except (TypeError, ValueError, np.linalg.LinAlgError) as exception:
                pixels = None
                error = str(exception)
            self._correction_queue.put(
                _CorrectionResult(record, pixels, error, completed, total)
            )
        self._correction_queue.put(_CorrectionDone(self._cancel_event.is_set()))

    def _poll_correction_queue(self) -> None:
        finished = False
        while True:
            try:
                update = self._correction_queue.get_nowait()
            except Empty:
                break
            if isinstance(update, _CorrectionDone):
                self._finish_correction(update.cancelled)
                finished = True
                continue
            self._handle_correction_result(update)
        if not finished:
            self._root.after(50, self._poll_correction_queue)

    def _handle_correction_result(self, result: _CorrectionResult) -> None:
        self._progress_value.set(result.completed / result.total * 100)
        self._progress_status.set(
            f"{result.completed} / {result.total}  {result.record.source.name}"
        )
        if result.error is not None:
            failure = f"{result.record.source.name}: {result.error}"
            self._correction_failures.append(failure)
            logger.error("校正失败：%s", failure)
            return
        if result.pixels is None:
            return
        self._store_corrected(_ImageRecord(result.record.source, result.pixels))
        logger.info("校正完成：%s", result.record.source.name)

    def _finish_correction(self, cancelled: bool) -> None:
        self._correct_button.configure(state=tk.NORMAL)
        self._cancel_button.configure(state=tk.DISABLED)
        self._delete_source_button.configure(state=tk.NORMAL)
        self._delete_corrected_button.configure(state=tk.NORMAL)
        if cancelled:
            self._progress_status.set("已取消")
            logger.info("校正任务已取消")
        else:
            self._progress_value.set(100)
            self._progress_status.set("完成")
            logger.info("后台校正任务完成")
        if self._correction_failures:
            messagebox.showwarning(
                "部分图片校正失败",
                "\n".join(self._correction_failures),
                parent=self._root,
            )
        if self._close_requested:
            self._close_now()

    def _cancel_correction(self) -> None:
        self._cancel_event.set()
        self._cancel_button.configure(state=tk.DISABLED)
        self._progress_status.set("正在取消…")
        logger.info("已请求取消，将在当前图片完成后停止")

    def _store_corrected(self, record: _ImageRecord) -> None:
        for index, existing in enumerate(self._corrected):
            if existing.source.resolve() == record.source.resolve():
                self._corrected[index] = record
                self._corrected_list.delete(index)
                self._corrected_list.insert(index, f"{record.source.stem}_校正后")
                return
        self._corrected.append(record)
        self._corrected_list.insert(tk.END, f"{record.source.stem}_校正后")

    def _delete_selected_imported(
        self,
        _event: tk.Event[tk.Misc] | None = None,
    ) -> None:
        if self._correction_is_running():
            messagebox.showinfo(
                "正在校正",
                "请等待当前校正任务完成后再删除图片。",
                parent=self._root,
            )
            return
        selection = self._selected_indices(self._source_list)
        if not selection:
            messagebox.showwarning(
                "未选择图片",
                "请先选择需要从已导入列表删除的图片。",
                parent=self._root,
            )
            return

        names = [self._imported[index].source.name for index in selection]
        for index in reversed(selection):
            del self._imported[index]
            self._source_list.delete(index)
        self._source_preview.configure(image="", text="选择图片后在此预览")
        self._source_preview_photo = None
        logger.info("已从导入列表删除 %d 张图片：%s", len(names), ", ".join(names))

    def _delete_selected_corrected(
        self,
        _event: tk.Event[tk.Misc] | None = None,
    ) -> None:
        if self._correction_is_running():
            messagebox.showinfo(
                "正在校正",
                "请等待当前校正任务完成后再删除图片。",
                parent=self._root,
            )
            return
        selection = self._selected_indices(self._corrected_list)
        if not selection:
            messagebox.showwarning(
                "未选择图片",
                "请先选择需要从校正结果列表删除的图片。",
                parent=self._root,
            )
            return

        names = [self._corrected[index].source.name for index in selection]
        for index in reversed(selection):
            del self._corrected[index]
            self._corrected_list.delete(index)
        self._pending_drag_selection = ()
        self._corrected_preview.configure(image="", text="选择图片后在此预览")
        self._corrected_preview_photo = None
        logger.info("已从校正结果列表删除 %d 张图片：%s", len(names), ", ".join(names))

    def _export_selected(self) -> None:
        selection = self._selected_indices(self._corrected_list)
        if not selection:
            messagebox.showwarning(
                "未选择图片",
                "请先选择需要导出的校正后图片。",
                parent=self._root,
            )
            return
        destination = filedialog.askdirectory(
            parent=self._root,
            title="选择导出文件夹",
            initialdir=self._export_directory,
            mustexist=False,
        )
        if not destination:
            logger.info("已取消导出")
            return

        export_directory = Path(destination).resolve()
        self._export_directory = export_directory
        self._save_settings()
        image_format = self._export_format.get()
        for index in selection:
            record = self._corrected[index]
            output = export_directory / corrected_output_name(
                record.source,
                image_format,
            )
            try:
                export_image(record.pixels, output)
            except (OSError, TypeError, ValueError) as error:
                logger.exception("导出失败：%s", output)
                messagebox.showerror(
                    "导出失败",
                    f"{output.name}\n{error}",
                    parent=self._root,
                )
        logger.info("导出操作完成")

    def _preview_source(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self._selected_indices(self._source_list)
        if selection:
            self._source_preview_photo = self._show_preview(
                self._source_preview,
                self._imported[selection[0]].pixels,
            )

    def _preview_corrected(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self._selected_indices(self._corrected_list)
        if selection:
            self._corrected_preview_photo = self._show_preview(
                self._corrected_preview,
                self._corrected[selection[0]].pixels,
            )

    @staticmethod
    def _show_preview(
        label: ttk.Label,
        pixels: NDArray[np.float64] | NDArray[np.uint8],
    ) -> ImageTk.PhotoImage:
        image = Image.fromarray(np.asarray(pixels, dtype=np.uint8))
        image.thumbnail(_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        return photo

    @staticmethod
    def _selected_indices(image_list: tk.Listbox) -> tuple[int, ...]:
        # Tk's bundled type information leaves curselection() untyped.
        selection = image_list.curselection()  # type: ignore[no-untyped-call]
        return tuple(int(index) for index in selection)

    def _confirm_close(self) -> None:
        correction_running = self._correction_is_running()
        cache_running = self._cache_is_running()
        if correction_running:
            prompt = "校正任务仍在运行。确定取消任务并在当前图片完成后关闭吗？"
        elif cache_running:
            prompt = (
                "Zernike 缓存仍在后台生成。确定立即关闭吗？未完成的临时缓存不会被使用。"
            )
        else:
            prompt = "确定要关闭 Zernike 图片校正工具吗？"
        if not messagebox.askyesno("确认关闭", prompt, parent=self._root):
            return
        if correction_running:
            self._close_requested = True
            self._cancel_correction()
            return
        self._close_now()

    def _correction_is_running(self) -> bool:
        return (
            self._correction_future is not None and not self._correction_future.done()
        )

    def _close_now(self) -> None:
        self._save_settings()
        logger.info("程序关闭")
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._drag_export_directory.cleanup()
        root_logger = logging.getLogger()
        root_logger.removeHandler(self._log_handler)
        root_logger.setLevel(self._previous_root_log_level)
        self._log_handler.close()
        self._root.destroy()


def launch() -> None:
    """Create the application window and enter the Tk event loop."""

    root = TkinterDnD.Tk()
    ZernikeCorrectorApp(root, Path.cwd())
    root.mainloop()
