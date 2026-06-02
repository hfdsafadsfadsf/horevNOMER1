"""
Встроенный редактор готового клипа — копия Vizard editor.

После генерации клип можно открыть в редакторе:
- Видеоплеер с текущим клипом
- Список слов субтитров с возможностью отредактировать текст
- Перетаскивание блока субтитров вверх/вниз
- Переключение режима камеры (single/split/multi)
- Кнопка "Применить и сохранить" → перерендеривает клип

Архитектура:
- ClipState — хранит редактируемые параметры клипа (sub_words, sub_margin_v_pct, ...)
- ClipEditorWindow — UI окно
- _re_render() — вызывает pipeline.re_render_clip() с новыми параметрами
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import messagebox
from typing import Callable, Optional

try:
    import customtkinter as ctk  # type: ignore
except ImportError:
    ctk = None  # type: ignore

from .ai_analyzer import ClipSuggestion
from .config import AppConfig, SubtitleStyle, TitleConfig
from .transcriber import Transcript, Word


LOG = logging.getLogger(__name__)


@dataclass
class ClipEditorState:
    """Редактируемые параметры одного клипа."""
    clip_index: int
    source_video: Path                # исходное видео (НЕ обрезанный клип)
    output_video: Path                # уже сгенерированный финал
    suggestion: ClipSuggestion        # AI-предложение (start/end/title)

    # Сохраняем копию транскрипта на отрезок клипа — пользователь может править текст
    edited_words: list[Word] = field(default_factory=list)

    # Текст заголовка (можно отредактировать)
    title_text: str = ""

    # Положение субтитров (модифицируется в редакторе)
    sub_position_v: str = "center"
    sub_margin_v_pct: float = 0.10

    # Режим face tracking для перерендера
    multi_face_mode: str = "auto"

    # Шифт face-центра по X (от -0.5 до +0.5 в норм. координатах исходника)
    face_offset_x: float = 0.0


class ClipEditorWindow:
    """Tk Toplevel окно для редактирования клипа."""

    def __init__(
        self,
        parent,
        state: ClipEditorState,
        cfg: AppConfig,
        transcript: Transcript,
        on_rerender: Optional[Callable[[ClipEditorState], None]] = None,
    ) -> None:
        if ctk is None:
            messagebox.showerror("Ошибка", "customtkinter не установлен")
            return

        self.parent = parent
        self.cfg = cfg
        self.state = state
        self.transcript = transcript
        self.on_rerender = on_rerender

        # Инициализируем edited_words из transcript если пусто.
        # transcript.words_in_range возвращает с АБСОЛЮТНЫМИ таймингами,
        # а edited_words хранятся как CLIP-LOCAL (от 0). Конвертируем.
        if not self.state.edited_words:
            abs_words = transcript.words_in_range(
                state.suggestion.start, state.suggestion.end
            )
            self.state.edited_words = [
                Word(
                    start=max(0.0, w.start - state.suggestion.start),
                    end=max(0.0, w.end - state.suggestion.start),
                    text=w.text,
                    probability=w.probability,
                )
                for w in abs_words
            ]

        if not self.state.title_text:
            self.state.title_text = state.suggestion.title

        self.win = ctk.CTkToplevel(parent)
        self.win.title(f"Редактор клипа #{state.clip_index}: {state.suggestion.title}")
        self.win.geometry("1280x760")
        self.win.transient(parent)

        # 2 колонки: слева редакторы, справа видеоплеер
        self.win.grid_columnconfigure(0, weight=2)
        self.win.grid_columnconfigure(1, weight=3)
        self.win.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

        # Кнопки внизу
        bottom = ctk.CTkFrame(self.win, fg_color="transparent")
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=12)
        bottom.grid_columnconfigure(0, weight=1)

        self.status_lbl = ctk.CTkLabel(bottom, text="Готово", anchor="w")
        self.status_lbl.grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            bottom, text="Закрыть",
            command=self.win.destroy,
            fg_color="#3a3a3a", hover_color="#555",
            width=120,
        ).grid(row=0, column=1, padx=4)
        self.rerender_btn = ctk.CTkButton(
            bottom, text="✨ Перерендерить с правками",
            command=self._on_rerender_clicked,
            fg_color="#1f8a3a", hover_color="#26a44a",
            width=240,
        )
        self.rerender_btn.grid(row=0, column=2, padx=4)

    # ---------- Левая панель: subtitle/title editor ----------
    def _build_left_panel(self) -> None:
        left = ctk.CTkFrame(self.win, fg_color="#1a1a1f")
        left.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # Tab-bar
        self.tabs = ctk.CTkTabview(left, segmented_button_selected_color="#7d3cf0")
        self.tabs.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=12, pady=12)

        self.tabs.add("Субтитры")
        self.tabs.add("Заголовок")
        self.tabs.add("Камера")

        self._build_subtitle_tab(self.tabs.tab("Субтитры"))
        self._build_title_tab(self.tabs.tab("Заголовок"))
        self._build_camera_tab(self.tabs.tab("Камера"))

    def _build_subtitle_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)

        info = ctk.CTkLabel(
            parent,
            text=(
                "Можно править текст каждого слова и время появления.\n"
                "Пустая строка = слово удалится из субтитров."
            ),
            anchor="w", justify="left",
        )
        info.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 8))

        # Скроллируемый список слов
        self.words_frame = ctk.CTkScrollableFrame(parent, label_text="Слова субтитров")
        self.words_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self._render_word_rows()

        # Position slider
        pos_frame = ctk.CTkFrame(parent, fg_color="transparent")
        pos_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        pos_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(pos_frame, text="Вертикальное положение:").grid(row=0, column=0, sticky="w")
        self.pos_v_var = tk.StringVar(value=self.state.sub_position_v)
        for i, val in enumerate(["top", "center", "bottom"]):
            ctk.CTkRadioButton(
                pos_frame, text=val, variable=self.pos_v_var, value=val,
                command=self._on_pos_v_changed,
            ).grid(row=0, column=1 + i, padx=4, sticky="w")

        ctk.CTkLabel(pos_frame, text="Отступ %:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.margin_var = tk.DoubleVar(value=self.state.sub_margin_v_pct * 100)
        ctk.CTkSlider(
            pos_frame, from_=0, to=45, variable=self.margin_var,
            command=lambda _: self._on_margin_changed(),
        ).grid(row=1, column=1, columnspan=3, sticky="ew", pady=(8, 0), padx=(8, 0))

    def _render_word_rows(self) -> None:
        for child in self.words_frame.winfo_children():
            child.destroy()
        self._word_vars: list[tuple[Word, tk.StringVar]] = []

        for i, w in enumerate(self.state.edited_words):
            row = ctk.CTkFrame(self.words_frame, fg_color="#252530")
            row.pack(fill="x", pady=2, padx=2)
            row.grid_columnconfigure(2, weight=1)

            t_txt = f"{w.start:5.2f} - {w.end:5.2f}s"
            ctk.CTkLabel(row, text=t_txt, width=120, anchor="w", font=("Consolas", 11)).grid(
                row=0, column=0, padx=6, pady=4, sticky="w"
            )

            var = tk.StringVar(value=w.text)
            entry = ctk.CTkEntry(row, textvariable=var, width=220, font=("Segoe UI", 14, "bold"))
            entry.grid(row=0, column=1, padx=6, pady=4, sticky="ew")

            del_btn = ctk.CTkButton(
                row, text="×", width=28, fg_color="#bb2233", hover_color="#dd3344",
                command=lambda idx=i: self._delete_word(idx),
            )
            del_btn.grid(row=0, column=3, padx=6, pady=4)

            self._word_vars.append((w, var))

    def _delete_word(self, idx: int) -> None:
        if 0 <= idx < len(self.state.edited_words):
            del self.state.edited_words[idx]
            self._render_word_rows()

    def _on_pos_v_changed(self) -> None:
        self.state.sub_position_v = self.pos_v_var.get()

    def _on_margin_changed(self) -> None:
        self.state.sub_margin_v_pct = self.margin_var.get() / 100.0

    # ---------- Title tab ----------
    def _build_title_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(parent, text="Текст заголовка:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        self.title_var = tk.StringVar(value=self.state.title_text)
        entry = ctk.CTkEntry(parent, textvariable=self.title_var, font=("Segoe UI", 16, "bold"))
        entry.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self.title_var.trace_add(
            "write",
            lambda *_: setattr(self.state, "title_text", self.title_var.get())
        )

        info = ctk.CTkLabel(
            parent,
            text=(
                "Заголовок-hook: максимум 5-7 слов, не пересказ.\n"
                "Длинные заголовки автоматически переносятся на 2-3 строки."
            ),
            anchor="w", justify="left",
            text_color="#aaa",
        )
        info.grid(row=2, column=0, sticky="ew", padx=8, pady=8)

        self.title_enabled_var = tk.BooleanVar(value=self.cfg.title.enabled)
        ctk.CTkCheckBox(
            parent, text="Включить заголовок в финальном клипе",
            variable=self.title_enabled_var,
        ).grid(row=3, column=0, sticky="w", padx=8, pady=8)

    # ---------- Camera tab ----------
    def _build_camera_tab(self, parent) -> None:
        parent.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            parent, text="Режим обработки лица:",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        self.mode_var = tk.StringVar(value=self.state.multi_face_mode)
        modes = [
            ("auto", "Авто (один спикер ИЛИ переключение между двумя)"),
            ("single", "Один спикер (главное лицо)"),
            ("split", "Split-screen: 2 лица сверху и снизу"),
        ]
        for i, (val, label) in enumerate(modes):
            ctk.CTkRadioButton(
                parent, text=label, variable=self.mode_var, value=val,
                command=self._on_camera_mode_changed,
            ).grid(row=1 + i, column=0, sticky="w", padx=16, pady=2)

        # Face offset X слайдер (для тонкой подгонки)
        ctk.CTkLabel(
            parent, text="Смещение фокуса по X (±%):",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=10, column=0, sticky="w", padx=8, pady=(16, 4))

        self.offset_x_var = tk.DoubleVar(value=self.state.face_offset_x * 100)
        slider = ctk.CTkSlider(
            parent, from_=-25, to=25, variable=self.offset_x_var,
            command=lambda _: self._on_offset_changed(),
        )
        slider.grid(row=11, column=0, sticky="ew", padx=16, pady=4)
        self.offset_lbl = ctk.CTkLabel(parent, text="0%", anchor="w")
        self.offset_lbl.grid(row=12, column=0, sticky="w", padx=16)

        ctk.CTkLabel(
            parent,
            text=(
                "Слева/справа: двигает крайнюю точку фокусировки.\n"
                "Используй если auto-tracking слегка промахивается."
            ),
            text_color="#888", justify="left",
        ).grid(row=13, column=0, sticky="w", padx=16, pady=(8, 0))

    def _on_camera_mode_changed(self) -> None:
        self.state.multi_face_mode = self.mode_var.get()

    def _on_offset_changed(self) -> None:
        v = self.offset_x_var.get()
        self.state.face_offset_x = v / 100.0
        self.offset_lbl.configure(text=f"{v:+.0f}%")

    # ---------- Правая панель: видеоплеер ----------
    def _build_right_panel(self) -> None:
        right = ctk.CTkFrame(self.win, fg_color="#0d0d12")
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Используем простой подход: показываем превью первого кадра + кнопка "Открыть в плеере"
        from PIL import Image, ImageTk
        try:
            preview_img = _extract_frame_thumbnail(self.state.output_video, 405, 720)
        except Exception:
            preview_img = None

        wrap = ctk.CTkFrame(right, fg_color="#0d0d12")
        wrap.grid(row=0, column=0, sticky="nsew", pady=12)
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        if preview_img is not None:
            self._preview_tk = ImageTk.PhotoImage(preview_img)
            self._preview_lbl = tk.Label(wrap, image=self._preview_tk, bg="#0d0d12", bd=0)
            self._preview_lbl.grid(row=0, column=0)
        else:
            self._preview_lbl = ctk.CTkLabel(
                wrap, text="(превью недоступно)\nНажми «Открыть в плеере»",
                width=405, height=720,
            )
            self._preview_lbl.grid(row=0, column=0)

        btns = ctk.CTkFrame(right, fg_color="transparent")
        btns.grid(row=1, column=0, pady=(8, 12))
        ctk.CTkButton(
            btns, text="▶ Открыть в плеере", width=200,
            command=self._open_in_player,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            btns, text="📂 В папке", width=140,
            command=self._open_folder,
        ).pack(side="left", padx=4)

    def _open_in_player(self) -> None:
        path = self.state.output_video
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Не открыть", str(e))

    def _open_folder(self) -> None:
        path = self.state.output_video
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Не открыть", str(e))

    # ---------- Перерендер ----------
    def _on_rerender_clicked(self) -> None:
        # Применяем тексты из entries в state.edited_words
        new_words: list[Word] = []
        for w, var in self._word_vars:
            txt = var.get().strip()
            if not txt:
                continue
            new_words.append(Word(start=w.start, end=w.end, text=txt, probability=w.probability))
        self.state.edited_words = new_words

        # Применяем настройки заголовка
        self.cfg.title.enabled = self.title_enabled_var.get()
        self.cfg.title.custom_text = self.title_var.get().strip()

        if self.on_rerender is None:
            messagebox.showinfo(
                "Готово",
                "Все правки применены к state. Закрой окно — клип в результатах обновится."
                "\n\nМАРК НЕ ТЫКАЙ НАХУЙ"
            )
            return

        self.rerender_btn.configure(state="disabled", text="⏳ Перерендер...")
        self.status_lbl.configure(text="Перерендер запущен — подожди...")

        def _bg() -> None:
            try:
                self.on_rerender(self.state)
                self.win.after(0, self._on_rerender_done)
            except Exception as e:  # noqa: BLE001
                LOG.exception("Re-render failed")
                self.win.after(0, lambda exc=e: self._on_rerender_failed(exc))

        threading.Thread(target=_bg, daemon=True).start()

    def _on_rerender_done(self) -> None:
        self.rerender_btn.configure(state="normal", text="✨ Перерендерить с правками")
        self.status_lbl.configure(text="✓ Готово! Файл обновлён.")
        # Обновим превью
        try:
            from PIL import ImageTk
            preview_img = _extract_frame_thumbnail(self.state.output_video, 405, 720)
            if preview_img is not None:
                self._preview_tk = ImageTk.PhotoImage(preview_img)
                self._preview_lbl.configure(image=self._preview_tk)
        except Exception:  # noqa: BLE001
            pass

    def _on_rerender_failed(self, exc: Exception) -> None:
        self.rerender_btn.configure(state="normal", text="✨ Перерендерить с правками")
        self.status_lbl.configure(text=f"✗ Ошибка: {exc}")
        messagebox.showerror("Ошибка перерендера", f"{exc}\n\nМАРК НЕ ТЫКАЙ НАХУЙ")


def _extract_frame_thumbnail(video_path: Path, width: int, height: int):
    """Извлекает первый кадр через ffmpeg и возвращает PIL.Image."""
    import tempfile
    from PIL import Image

    if not video_path.exists():
        return None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,30),scale={width}:{height}:force_original_aspect_ratio=decrease",
            "-frames:v", "1",
            str(tmp_path),
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=20)
        if r.returncode != 0:
            return None
        img = Image.open(tmp_path).convert("RGB")
        # Padding до точного размера
        if img.size != (width, height):
            from PIL import ImageOps
            img = ImageOps.pad(img, (width, height), color="#0d0d12")
        return img.copy()
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
