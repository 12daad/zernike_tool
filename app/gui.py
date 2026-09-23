"""Tkinter desktop interface for batch Zernike image correction."""

from __future__ import annotations

import logging
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, get_ident
from tkinter import filedialog, messagebox, ttk

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from zernike_tool import export_image, load_image

from .correction import (
    calibrate_image,
    corrected_output_name,
    parse_coefficients,
    preload_correction_cache,
    prepare_grayscale,
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


class _TextLogHandler(logging.Handler):
    def __init__(self, widget: tk.Text) -> None:
        super().__init__(level=logging.INFO)
        self._widget = widget
        self._main_thread = get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        if get_ident() == self._main_thread:
            self._append(message)
        else:
            self._widget.after(0, self._append, message)

    def _append(self, message: str) -> None:
        self._widget.configure(state=tk.NORMAL)
        self._widget.insert(tk.END, f"{message}\n")
        self._widget.see(tk.END)
        self._widget.configure(state=tk.DISABLED)
        self._widget.update_idletasks()


class ZernikeCorrectorApp:
    """Manage the widgets and user actions of the correction tool."""

    def __init__(self, root: tk.Tk, initial_directory: Path | None = None) -> None:
        self._root = root
        self._initial_directory = initial_directory or Path.cwd()
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
        self._close_requested = False

        root.title("Zernike 图片批量校正工具")
        root.geometry("1180x820")
        root.minsize(940, 680)
        root.protocol("WM_DELETE_WINDOW", self._confirm_close)

        self._build_interface()
        self._configure_gui_logging()
        logger.info("程序已启动，文件选择初始路径：%s", self._initial_directory)
        if preload_correction_cache():
            logger.info("1920x1080 专用 Zernike 缓存已就绪")
        else:
            logger.warning("1920x1080 专用缓存不可用，将按需实时计算")

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
        self._coefficient_text.insert("1.0", "0")
        ttk.Button(controls, text="批量导入图片", command=self._import_images).grid(
            row=0,
            column=2,
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
            padx=(4, 0),
            sticky="ns",
        )
        progress_frame = ttk.Frame(controls)
        progress_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
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
        ttk.Button(
            export_bar,
            text="导出选中图片",
            command=self._export_selected,
        ).grid(row=0, column=3)

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

    def _import_images(self) -> None:
        filenames = filedialog.askopenfilenames(
            parent=self._root,
            title="批量导入图片",
            initialdir=self._initial_directory,
            filetypes=_IMAGE_FILE_TYPES,
        )
        if not filenames:
            logger.info("已取消导入")
            return

        rejected: list[str] = []
        imported_paths = {record.source.resolve() for record in self._imported}
        for filename in filenames:
            source = Path(filename)
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

    def _correct_selected(self) -> None:
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
        except ValueError as error:
            logger.warning("系数输入无效：%s", error)
            messagebox.showwarning("系数无效", str(error), parent=self._root)
            return

        records = [self._imported[index] for index in selection]
        self._cancel_event.clear()
        self._correction_failures.clear()
        self._progress_value.set(0)
        self._progress_status.set(f"0 / {len(records)}")
        self._correct_button.configure(state=tk.DISABLED)
        self._cancel_button.configure(state=tk.NORMAL)
        logger.info("开始后台校正 %d 张图片", len(records))
        self._correction_future = self._executor.submit(
            self._run_correction,
            records,
            coefficients.copy(),
        )
        self._root.after(50, self._poll_correction_queue)

    def _run_correction(
        self,
        records: list[_ImageRecord],
        coefficients: NDArray[np.float64],
    ) -> None:
        total = len(records)
        for completed, record in enumerate(records, start=1):
            if self._cancel_event.is_set():
                break
            try:
                pixels = calibrate_image(record.pixels, coefficients)
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
            initialdir=self._initial_directory,
            mustexist=False,
        )
        if not destination:
            logger.info("已取消导出")
            return

        export_directory = Path(destination)
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
        running = (
            self._correction_future is not None and not self._correction_future.done()
        )
        prompt = (
            "校正任务仍在运行。确定取消任务并在当前图片完成后关闭吗？"
            if running
            else "确定要关闭 Zernike 图片校正工具吗？"
        )
        if not messagebox.askyesno("确认关闭", prompt, parent=self._root):
            return
        if running:
            self._close_requested = True
            self._cancel_correction()
            return
        self._close_now()

    def _close_now(self) -> None:
        logger.info("程序关闭")
        self._executor.shutdown(wait=False, cancel_futures=True)
        root_logger = logging.getLogger()
        root_logger.removeHandler(self._log_handler)
        root_logger.setLevel(self._previous_root_log_level)
        self._log_handler.close()
        self._root.destroy()


def launch() -> None:
    """Create the application window and enter the Tk event loop."""

    root = tk.Tk()
    ZernikeCorrectorApp(root, Path.cwd())
    root.mainloop()
