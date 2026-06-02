"""CustomTkinter GUI для Vizard Clone."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk

from .config import (
    AppConfig,
    FONTS_DIR,
    OUTPUT_DIR,
    PRESETS_DIR,
    SubtitleStyle,
    TitleConfig,
    VALID_LENGTH_PRESETS,
)
from .pipeline import ClipResult, run_pipeline
from .preview import generate_preview
from .template_editor import TemplateEditor
from .templates import (
    TEMPLATES,
    TITLE_TEMPLATES,
    SubtitleTemplate,
    get_subtitle_template,
    get_title_template,
)

MARK_WARNING = "\n\nМАРК НЕ ТЫКАЙ НАХУЙ"


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


FONT_CHOICES = [
    "Montserrat-Black.ttf",
    "Montserrat-Bold.ttf",
    "Rubik-Black.ttf",
    "Rubik-Bold.ttf",
    "BebasNeue-Regular.ttf",
    "Inter-Black.otf",
    "Anton-Regular.ttf",
    "Oswald-Bold.ttf",
    "Poppins-Black.ttf",
]

LANGUAGE_CHOICES = ["auto", "ru", "en", "es", "de", "fr", "uk", "pl", "tr"]

POSITION_CHOICES = ["top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"]


def _safe_int(s: str, fallback: int) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return fallback


def _safe_float(s: str, fallback: float) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return fallback


class VizardApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Vizard Clone — Reels Generator")
        self.geometry("1280x880")
        self.minsize(1100, 760)

        self.cfg = AppConfig.load()
        self._worker: Optional[threading.Thread] = None
        self._preview_worker: Optional[threading.Thread] = None
        self._results: list[ClipResult] = []

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkScrollableFrame(self, width=500)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._build_sidebar(sidebar)
        self._build_right_panel(right)

    def _build_sidebar(self, parent: ctk.CTkScrollableFrame) -> None:
        ctk.CTkLabel(
            parent, text="Vizard Clone",
            font=("Helvetica", 30, "bold"),
            text_color="#a78bfa",
        ).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(
            parent, text="Локальный AI-нарезчик коротких видео",
            text_color="#888", font=("Helvetica", 12),
        ).pack(anchor="w", pady=(0, 16))

        ctk.CTkButton(
            parent, text="➕ Создать шаблон (с живым превью)",
            command=self._open_template_editor,
            height=44, fg_color="#a78bfa", text_color="#1a1a25",
            font=("Helvetica", 14, "bold"),
        ).pack(fill="x", pady=(0, 8))

        preset_row = ctk.CTkFrame(parent, fg_color="transparent")
        preset_row.pack(fill="x", pady=(0, 14))
        ctk.CTkButton(preset_row, text="💾 Сохранить пресет", command=self._save_preset,
                      height=32).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(preset_row, text="📂 Загрузить пресет", command=self._load_preset,
                      height=32).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self._section_label(parent, "1. Источник видео")
        self.source_entry = ctk.CTkEntry(
            parent, placeholder_text="URL (YouTube/TikTok/...) или путь к файлу"
        )
        self.source_entry.pack(fill="x", pady=(4, 4))
        ctk.CTkButton(
            parent, text="📁 Выбрать файл с диска", command=self._browse_source
        ).pack(fill="x", pady=(0, 16))

        self._section_label(parent, "2. Длина клипов")
        self.length_var = ctk.StringVar(value=self.cfg.clip.length_preset)
        seg = ctk.CTkSegmentedButton(
            parent, values=list(VALID_LENGTH_PRESETS), variable=self.length_var
        )
        seg.pack(fill="x", pady=(4, 8))

        cf = ctk.CTkFrame(parent, fg_color="transparent")
        cf.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(cf, text="Минимум:").grid(row=0, column=0, padx=(0, 4), sticky="w")
        self.min_clips_entry = ctk.CTkEntry(cf, width=60)
        self.min_clips_entry.insert(0, str(self.cfg.clip.min_clip_count))
        self.min_clips_entry.grid(row=0, column=1, padx=(0, 16))
        ctk.CTkLabel(cf, text="Максимум:").grid(row=0, column=2, padx=(0, 4))
        self.max_clips_entry = ctk.CTkEntry(cf, width=60)
        self.max_clips_entry.insert(0, str(self.cfg.clip.max_clip_count))
        self.max_clips_entry.grid(row=0, column=3)

        ctk.CTkLabel(parent, text="Язык видео:").pack(anchor="w")
        self.language_var = ctk.StringVar(value=self.cfg.clip.language)
        ctk.CTkOptionMenu(parent, values=LANGUAGE_CHOICES, variable=self.language_var).pack(
            fill="x", pady=(2, 16)
        )

        self._section_label(parent, "3. Шаблон субтитров")
        self.template_var = ctk.StringVar(value=self.cfg.subtitle.template_id)
        template_choices = [t.name for t in TEMPLATES.values()]
        self._template_id_by_name = {t.name: t.id for t in TEMPLATES.values()}
        self._template_name_by_id = {t.id: t.name for t in TEMPLATES.values()}
        current_name = self._template_name_by_id.get(self.cfg.subtitle.template_id, template_choices[0])
        self.template_name_var = ctk.StringVar(value=current_name)
        ctk.CTkOptionMenu(
            parent,
            values=template_choices,
            variable=self.template_name_var,
            command=self._on_template_change,
        ).pack(fill="x", pady=(2, 4))
        self.template_desc_label = ctk.CTkLabel(
            parent,
            text=TEMPLATES[self.cfg.subtitle.template_id].description
            if self.cfg.subtitle.template_id in TEMPLATES
            else "",
            text_color="gray70",
            wraplength=460,
            justify="left",
        )
        self.template_desc_label.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(parent, text="Шрифт:").pack(anchor="w")
        self.font_var = ctk.StringVar(value=self.cfg.subtitle.font)
        ctk.CTkOptionMenu(parent, values=FONT_CHOICES, variable=self.font_var).pack(
            fill="x", pady=(2, 8)
        )

        sf = ctk.CTkFrame(parent, fg_color="transparent")
        sf.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(sf, text="Размер:").grid(row=0, column=0, padx=(0, 4))
        self.font_size_entry = ctk.CTkEntry(sf, width=60)
        self.font_size_entry.insert(0, str(self.cfg.subtitle.font_size))
        self.font_size_entry.grid(row=0, column=1, padx=(0, 16))
        ctk.CTkLabel(sf, text="Слов в строке:").grid(row=0, column=2, padx=(0, 4))
        self.words_per_line_entry = ctk.CTkEntry(sf, width=60)
        self.words_per_line_entry.insert(0, str(self.cfg.subtitle.max_words_per_line))
        self.words_per_line_entry.grid(row=0, column=3)

        cf2 = ctk.CTkFrame(parent, fg_color="transparent")
        cf2.pack(fill="x", pady=(6, 6))
        ctk.CTkLabel(cf2, text="Цвет текста:").grid(row=0, column=0, sticky="w")
        self.primary_color_entry = ctk.CTkEntry(cf2, width=100)
        self.primary_color_entry.insert(0, self.cfg.subtitle.primary_color)
        self.primary_color_entry.grid(row=0, column=1, padx=(4, 0))

        ctk.CTkLabel(cf2, text="Highlight:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.highlight_color_entry = ctk.CTkEntry(cf2, width=100)
        self.highlight_color_entry.insert(0, self.cfg.subtitle.highlight_color)
        self.highlight_color_entry.grid(row=1, column=1, padx=(4, 0), pady=(4, 0))

        ctk.CTkLabel(cf2, text="Обводка:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.outline_color_entry = ctk.CTkEntry(cf2, width=100)
        self.outline_color_entry.insert(0, self.cfg.subtitle.outline_color)
        self.outline_color_entry.grid(row=2, column=1, padx=(4, 0), pady=(4, 0))

        ctk.CTkLabel(cf2, text="Толщина обводки:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.outline_width_entry = ctk.CTkEntry(cf2, width=60)
        self.outline_width_entry.insert(0, str(self.cfg.subtitle.outline_width))
        self.outline_width_entry.grid(row=3, column=1, padx=(4, 0), pady=(4, 0), sticky="w")

        ctk.CTkLabel(parent, text="Стиль (обводка / тень / фон):").pack(anchor="w", pady=(8, 2))
        self.box_style_var = ctk.StringVar(value=self.cfg.subtitle.box_style)
        ctk.CTkSegmentedButton(
            parent,
            values=["outline", "shadow", "background"],
            variable=self.box_style_var,
        ).pack(fill="x", pady=(0, 6))

        cf2b = ctk.CTkFrame(parent, fg_color="transparent")
        cf2b.pack(fill="x", pady=(4, 6))
        ctk.CTkLabel(cf2b, text="Цвет фона:").grid(row=0, column=0, sticky="w")
        self.back_color_entry = ctk.CTkEntry(cf2b, width=100)
        self.back_color_entry.insert(0, self.cfg.subtitle.back_color)
        self.back_color_entry.grid(row=0, column=1, padx=(4, 0))
        ctk.CTkLabel(cf2b, text="Прозрачность фона (0-255):").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.back_alpha_entry = ctk.CTkEntry(cf2b, width=60)
        self.back_alpha_entry.insert(0, str(self.cfg.subtitle.back_alpha))
        self.back_alpha_entry.grid(row=1, column=1, padx=(4, 0), pady=(4, 0), sticky="w")

        cf2c = ctk.CTkFrame(parent, fg_color="transparent")
        cf2c.pack(fill="x", pady=(4, 6))
        ctk.CTkLabel(cf2c, text="Пауза для скрытия (сек):").grid(row=0, column=0, sticky="w")
        self.silence_gap_entry = ctk.CTkEntry(cf2c, width=60)
        self.silence_gap_entry.insert(0, str(self.cfg.subtitle.silence_gap_sec))
        self.silence_gap_entry.grid(row=0, column=1, padx=(4, 0))

        self.uppercase_var = ctk.BooleanVar(value=self.cfg.subtitle.uppercase)
        ctk.CTkCheckBox(parent, text="ВЕРХНИЙ РЕГИСТР", variable=self.uppercase_var).pack(
            anchor="w", pady=(8, 4)
        )
        self.word_highlight_var = ctk.BooleanVar(value=self.cfg.subtitle.word_highlight)
        ctk.CTkCheckBox(
            parent, text="Подсветка текущего слова (karaoke)", variable=self.word_highlight_var
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(parent, text="Позиция:").pack(anchor="w", pady=(8, 0))
        self.position_var = ctk.StringVar(value=self.cfg.subtitle.position_v)
        ctk.CTkSegmentedButton(
            parent, values=["top", "center", "bottom"], variable=self.position_var
        ).pack(fill="x", pady=(2, 16))

        self._section_label(parent, "4. Заголовок поверх клипа")
        self.title_enabled_var = ctk.BooleanVar(value=self.cfg.title.enabled)
        ctk.CTkCheckBox(
            parent, text="Показывать заголовок", variable=self.title_enabled_var
        ).pack(anchor="w", pady=(0, 4))

        title_choices = [t.name for t in TITLE_TEMPLATES.values()]
        self._title_id_by_name = {t.name: t.id for t in TITLE_TEMPLATES.values()}
        self._title_name_by_id = {t.id: t.name for t in TITLE_TEMPLATES.values()}
        current_title_name = self._title_name_by_id.get(self.cfg.title.template_id, title_choices[0])
        self.title_template_var = ctk.StringVar(value=current_title_name)
        ctk.CTkOptionMenu(
            parent,
            values=title_choices,
            variable=self.title_template_var,
            command=self._on_title_template_change,
        ).pack(fill="x", pady=(2, 4))

        ctk.CTkLabel(parent, text="Источник текста:").pack(anchor="w", pady=(4, 2))
        self.title_mode_var = ctk.StringVar(value=self.cfg.title.text_mode)
        ctk.CTkSegmentedButton(
            parent, values=["ai", "custom"], variable=self.title_mode_var,
        ).pack(fill="x", pady=(0, 4))

        self.title_text_entry = ctk.CTkEntry(
            parent, placeholder_text="Если 'custom' — твой текст заголовка"
        )
        self.title_text_entry.insert(0, self.cfg.title.custom_text)
        self.title_text_entry.pack(fill="x", pady=(2, 16))

        self._section_label(parent, "5. Логотип / обложка")
        self.overlay_enabled_var = ctk.BooleanVar(value=self.cfg.overlay.enabled)
        ctk.CTkCheckBox(
            parent, text="Накладывать логотип", variable=self.overlay_enabled_var
        ).pack(anchor="w", pady=(0, 4))

        self.overlay_path_var = ctk.StringVar(value=self.cfg.overlay.image_path)
        ctk.CTkEntry(parent, textvariable=self.overlay_path_var,
                     placeholder_text="Путь к PNG/JPG логотипа").pack(fill="x", pady=(2, 4))
        ctk.CTkButton(
            parent, text="🖼  Выбрать логотип", command=self._browse_logo
        ).pack(fill="x", pady=(0, 6))

        cf_logo = ctk.CTkFrame(parent, fg_color="transparent")
        cf_logo.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(cf_logo, text="Позиция:").grid(row=0, column=0, sticky="w")
        self.overlay_pos_var = ctk.StringVar(value=self.cfg.overlay.position)
        ctk.CTkOptionMenu(cf_logo, values=POSITION_CHOICES,
                          variable=self.overlay_pos_var, width=160).grid(row=0, column=1, padx=(4, 0), sticky="ew")
        cf_logo.grid_columnconfigure(1, weight=1)

        cf_logo2 = ctk.CTkFrame(parent, fg_color="transparent")
        cf_logo2.pack(fill="x", pady=(2, 16))
        ctk.CTkLabel(cf_logo2, text="Размер % ширины:").grid(row=0, column=0, sticky="w")
        self.overlay_size_entry = ctk.CTkEntry(cf_logo2, width=60)
        self.overlay_size_entry.insert(0, str(self.cfg.overlay.size_pct))
        self.overlay_size_entry.grid(row=0, column=1, padx=(4, 16))
        ctk.CTkLabel(cf_logo2, text="Прозрачность 0-1:").grid(row=0, column=2, sticky="w")
        self.overlay_opacity_entry = ctk.CTkEntry(cf_logo2, width=60)
        self.overlay_opacity_entry.insert(0, str(self.cfg.overlay.opacity))
        self.overlay_opacity_entry.grid(row=0, column=3, padx=(4, 0))

        self._section_label(parent, "6. Музыка")
        self.music_mode_var = ctk.StringVar(value=self.cfg.music.mode)
        ctk.CTkSegmentedButton(
            parent,
            values=["none", "common", "per_clip"],
            variable=self.music_mode_var,
        ).pack(fill="x", pady=(2, 8))

        self.music_common_path_var = ctk.StringVar(
            value=self.cfg.music.common_track or ""
        )
        ctk.CTkEntry(
            parent, textvariable=self.music_common_path_var, placeholder_text="Путь к общему треку"
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkButton(
            parent, text="📁 Выбрать общий трек", command=self._browse_common_music
        ).pack(fill="x", pady=(0, 8))

        self.per_clip_tracks_label = ctk.CTkLabel(
            parent, text=f"Per-clip треков: {len(self.cfg.music.per_clip_tracks)}"
        )
        self.per_clip_tracks_label.pack(anchor="w", pady=(0, 4))
        ctk.CTkButton(
            parent,
            text="📁 Выбрать N треков (по одному на клип)",
            command=self._browse_per_clip_music,
        ).pack(fill="x", pady=(0, 8))

        cf3 = ctk.CTkFrame(parent, fg_color="transparent")
        cf3.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(cf3, text="Громкость музыки:").grid(row=0, column=0, sticky="w")
        self.volume_entry = ctk.CTkEntry(cf3, width=60)
        self.volume_entry.insert(0, str(self.cfg.music.volume))
        self.volume_entry.grid(row=0, column=1, padx=(4, 0))

        self.sidechain_var = ctk.BooleanVar(value=self.cfg.music.sidechain)
        ctk.CTkCheckBox(
            parent,
            text="Sidechain (приглушать музыку на голосе)",
            variable=self.sidechain_var,
        ).pack(anchor="w", pady=(0, 16))

        self._section_label(parent, "7. DeepSeek API")
        self.use_ai_var = ctk.BooleanVar(value=self.cfg.use_ai)
        ctk.CTkCheckBox(
            parent, text="Использовать AI для выбора моментов", variable=self.use_ai_var
        ).pack(anchor="w", pady=(0, 4))
        self.api_key_entry = ctk.CTkEntry(
            parent, placeholder_text="DeepSeek API key", show="*"
        )
        if self.cfg.deepseek_api_key:
            self.api_key_entry.insert(0, self.cfg.deepseek_api_key)
        self.api_key_entry.pack(fill="x", pady=(0, 16))

        self._section_label(parent, "8. Качество видео (Whisper)")
        cf4 = ctk.CTkFrame(parent, fg_color="transparent")
        cf4.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(cf4, text="Модель:").grid(row=0, column=0, sticky="w")
        self.whisper_var = ctk.StringVar(value=self.cfg.whisper_model_size)
        ctk.CTkOptionMenu(
            cf4,
            values=["tiny", "base", "small", "medium", "large-v3"],
            variable=self.whisper_var,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")
        ctk.CTkLabel(cf4, text="Устройство:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.whisper_device_var = ctk.StringVar(value=self.cfg.whisper_device or "auto")
        ctk.CTkOptionMenu(
            cf4,
            values=["auto", "cuda", "cpu"],
            variable=self.whisper_device_var,
        ).grid(row=1, column=1, padx=(4, 0), pady=(6, 0), sticky="ew")
        cf4.grid_columnconfigure(1, weight=1)

        self.whisper_status_lbl = ctk.CTkLabel(
            parent, text="(CUDA определяется при старте...)",
            text_color="#888", font=("Helvetica", 11), anchor="w",
        )
        self.whisper_status_lbl.pack(anchor="w", padx=10, pady=(2, 4))

        ctk.CTkLabel(
            parent,
            text="auto: GPU если есть, иначе CPU. На CPU Whisper самый медленный этап — поставь cuda если у тебя NVIDIA",
            text_color="#666", font=("Helvetica", 10), wraplength=320, justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 12))

        # Фоновый детект CUDA для Whisper
        threading.Thread(target=self._detect_whisper_async, daemon=True).start()

        self._section_label(parent, "9. Face tracking (9:16 кроп)")
        self.face_tracking_var = ctk.BooleanVar(value=self.cfg.face_tracking)
        ctk.CTkCheckBox(
            parent,
            text="Динамический кроп вслед за лицом (smooth tracking)",
            variable=self.face_tracking_var,
        ).pack(anchor="w", pady=(0, 4))
        self.face_zoom_var = ctk.BooleanVar(value=self.cfg.face_zoom)
        ctk.CTkCheckBox(
            parent,
            text="Авто-зум: приближаться если лицо мелкое",
            variable=self.face_zoom_var,
        ).pack(anchor="w", pady=(0, 8))

        # Режим обработки нескольких лиц
        ctk.CTkLabel(parent, text="Что делать если в кадре 2 лица:",
                     text_color="#aaa", font=("Helvetica", 11)).pack(anchor="w")
        self.multi_face_var = ctk.StringVar(value=self.cfg.multi_face_mode)
        for val, lbl in [
            ("auto", "Авто (камера переключается на активного спикера)"),
            ("single", "Только главное лицо (как было раньше)"),
            ("split", "Split-screen: 1 лицо сверху, 1 снизу"),
        ]:
            ctk.CTkRadioButton(
                parent, text=lbl, variable=self.multi_face_var, value=val,
            ).pack(anchor="w", padx=10, pady=1)

        self.speaker_var = ctk.BooleanVar(value=self.cfg.speaker_detection)
        ctk.CTkCheckBox(
            parent,
            text="Анализ аудио для определения говорящего",
            variable=self.speaker_var,
        ).pack(anchor="w", pady=(8, 16))

        self._section_label(parent, "10. GPU ускорение (NVIDIA / Intel / AMD)")
        self.gpu_var = ctk.BooleanVar(value=self.cfg.use_gpu)
        ctk.CTkCheckBox(
            parent,
            text="Использовать GPU (NVENC/QSV/AMF) — в 5-10x быстрее",
            variable=self.gpu_var,
            command=self._refresh_gpu_status,
        ).pack(anchor="w", pady=(0, 4))
        self.gpu_status_lbl = ctk.CTkLabel(
            parent, text="(определяется при старте)",
            text_color="#888", font=("Helvetica", 11), anchor="w",
        )
        self.gpu_status_lbl.pack(anchor="w", padx=10, pady=(0, 8))

        # Запускаем GPU-детект в фоне чтобы не блокировать GUI
        threading.Thread(target=self._detect_gpu_async, daemon=True).start()

        self.strip_punct_var = ctk.BooleanVar(value=self.cfg.strip_subtitle_punct)
        ctk.CTkCheckBox(
            parent,
            text="Убирать точки/запятые из субтитров (рекомендуется)",
            variable=self.strip_punct_var,
        ).pack(anchor="w", pady=(0, 16))

        ctk.CTkButton(
            parent, text="💾 Сохранить настройки", command=self._save_config
        ).pack(fill="x", pady=(8, 4))

    def _detect_gpu_async(self) -> None:
        """Фоновый детект GPU. Обновляет label с найденным кодеком."""
        try:
            from .gpu import detect_gpu
            cap = detect_gpu()
            if cap.any_encoder:
                msg = f"✓ GPU найдена: {cap.label()}"
                color = "#36d97a"
            else:
                msg = "GPU не найдена — fallback на CPU (libx264)"
                color = "#888"
            self.after(0, lambda m=msg, c=color: self.gpu_status_lbl.configure(text=m, text_color=c))
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda exc=e: self.gpu_status_lbl.configure(text=f"GPU-детект упал: {exc}", text_color="#aa6"))

    def _detect_whisper_async(self) -> None:
        """Фоновый детект CUDA для Whisper. Подсказывает что выбрать."""
        try:
            from .transcriber import _cuda_available, whisper_device_label
            ok = _cuda_available()
            if ok:
                msg = f"✓ {whisper_device_label()} — Whisper будет на GPU (5-15x быстрее)"
                color = "#36d97a"
            else:
                msg = "✗ CUDA не найдена — Whisper будет на CPU (медленно)"
                color = "#d97a3c"
            self.after(0, lambda m=msg, c=color: self.whisper_status_lbl.configure(text=m, text_color=c))
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda exc=e: self.whisper_status_lbl.configure(text=f"CUDA-детект упал: {exc}", text_color="#aa6"))

    def _refresh_gpu_status(self) -> None:
        """Когда пользователь снимает галку — показываем что CPU."""
        if not self.gpu_var.get():
            self.gpu_status_lbl.configure(text="GPU отключен пользователем", text_color="#888")
        else:
            self._detect_gpu_async()

    def _build_right_panel(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(0, weight=2)
        header.grid_columnconfigure(1, weight=1)
        header.grid_columnconfigure(2, weight=1)

        self.generate_btn = ctk.CTkButton(
            header,
            text="🚀 Сгенерировать клипы",
            command=self._start_generation,
            height=52,
            font=("Helvetica", 16, "bold"),
            fg_color="#7c3aed",
            hover_color="#9c54f0",
            corner_radius=12,
        )
        self.generate_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.preview_btn = ctk.CTkButton(
            header,
            text="🎬 Превью (15 сек)",
            command=self._start_preview,
            height=50,
            font=("Helvetica", 14, "bold"),
            fg_color="#7c3aed",
        )
        self.preview_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            header,
            text="📂 Папка с клипами",
            command=self._open_output_dir,
            height=50,
        ).grid(row=0, column=2, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(parent, height=14)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.progress_bar.set(0.0)

        log_frame = ctk.CTkFrame(parent)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 8))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_box = ctk.CTkTextbox(log_frame, wrap="word", font=("Consolas", 12))
        self.log_box.grid(row=0, column=0, sticky="nsew")

        self.results_frame = ctk.CTkScrollableFrame(parent, label_text="Результаты", height=180)
        self.results_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))

        self._log(
            "Готов к работе.\n"
            "• Выбери шаблон субтитров и нажми 'Превью (15 сек)' чтобы посмотреть как выглядит.\n"
            "• Когда всё устраивает — жми 'Сгенерировать клипы' на полное видео.\n"
            "• Можно сохранить настройки как пресет (кнопка вверху) и потом загружать одним кликом."
        )

    def _section_label(self, parent, text: str) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", pady=(14, 4))
        # Маленькая цветная плашка слева как акцент
        accent = ctk.CTkFrame(wrap, width=4, height=20, fg_color="#a78bfa", corner_radius=2)
        accent.pack(side="left", padx=(0, 8), pady=2)
        ctk.CTkLabel(
            wrap,
            text=text,
            font=("Helvetica", 14, "bold"),
            text_color="#e0e0f8",
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

    def _log(self, msg: str) -> None:
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    def _browse_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбери видеофайл",
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm *.avi"), ("All", "*.*")],
        )
        if path:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, path)

    def _browse_logo(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбери логотип",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")],
        )
        if path:
            self.overlay_path_var.set(path)

    def _browse_common_music(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбери музыкальный трек",
            filetypes=[("Audio", "*.mp3 *.m4a *.wav *.flac *.ogg *.aac"), ("All", "*.*")],
        )
        if path:
            self.music_common_path_var.set(path)
            # Авто-переключение режима: если пользователь выбрал файл, явно
            # хочет музыку — ставим mode="common" чтобы не забыть.
            if self.music_mode_var.get() == "none":
                self.music_mode_var.set("common")
                self._log("Музыка: режим переключён на 'common' (общий трек на все клипы)")

    def _browse_per_clip_music(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Выбери треки (по одному на клип, в нужном порядке)",
            filetypes=[("Audio", "*.mp3 *.m4a *.wav *.flac *.ogg *.aac"), ("All", "*.*")],
        )
        if paths:
            self.cfg.music.per_clip_tracks = list(paths)
            self.per_clip_tracks_label.configure(
                text=f"Per-clip треков: {len(self.cfg.music.per_clip_tracks)}"
            )
            # Авто-переключение режима на per_clip
            if self.music_mode_var.get() != "per_clip":
                self.music_mode_var.set("per_clip")
                self._log(f"Музыка: режим переключён на 'per_clip' ({len(paths)} треков)")

    def _on_template_change(self, display_name: str) -> None:
        tmpl_id = self._template_id_by_name.get(display_name)
        if not tmpl_id:
            return
        tmpl = TEMPLATES[tmpl_id]
        self.cfg.subtitle.template_id = tmpl_id
        self._apply_subtitle_template_to_form(tmpl)
        self.template_desc_label.configure(text=tmpl.description)
        self._log(f"Загружен шаблон субтитров: {tmpl.name}")

    def _on_title_template_change(self, display_name: str) -> None:
        tmpl_id = self._title_id_by_name.get(display_name)
        if not tmpl_id:
            return
        tmpl = TITLE_TEMPLATES[tmpl_id]
        self.cfg.title.template_id = tmpl_id
        self.cfg.title.font = tmpl.font
        self.cfg.title.font_size = tmpl.font_size
        self.cfg.title.primary_color = tmpl.primary_color
        self.cfg.title.outline_color = tmpl.outline_color
        self.cfg.title.outline_width = tmpl.outline_width
        self.cfg.title.back_color = tmpl.back_color
        self.cfg.title.back_alpha = tmpl.back_alpha
        self.cfg.title.back_padding = tmpl.back_padding
        self.cfg.title.uppercase = tmpl.uppercase
        self.cfg.title.variant = tmpl.variant
        self.cfg.title.margin_v_pct = tmpl.margin_v_pct
        self.cfg.title.duration = tmpl.duration
        self._log(f"Загружен шаблон заголовка: {tmpl.name}")

    def _apply_subtitle_template_to_form(self, tmpl: SubtitleTemplate) -> None:
        self.font_var.set(tmpl.font)
        self.font_size_entry.delete(0, "end"); self.font_size_entry.insert(0, str(tmpl.font_size))
        self.words_per_line_entry.delete(0, "end"); self.words_per_line_entry.insert(0, str(tmpl.max_words_per_line))
        self.primary_color_entry.delete(0, "end"); self.primary_color_entry.insert(0, tmpl.primary_color)
        self.highlight_color_entry.delete(0, "end"); self.highlight_color_entry.insert(0, tmpl.highlight_color)
        self.outline_color_entry.delete(0, "end"); self.outline_color_entry.insert(0, tmpl.outline_color)
        self.outline_width_entry.delete(0, "end"); self.outline_width_entry.insert(0, str(tmpl.outline_width))
        self.box_style_var.set(tmpl.box_style)
        self.back_color_entry.delete(0, "end"); self.back_color_entry.insert(0, tmpl.back_color)
        self.back_alpha_entry.delete(0, "end"); self.back_alpha_entry.insert(0, str(tmpl.back_alpha))
        self.uppercase_var.set(tmpl.uppercase)
        self.word_highlight_var.set(tmpl.word_highlight)
        self.position_var.set(tmpl.position_v)

    def _collect_config(self) -> AppConfig:
        cfg = self.cfg
        cfg.deepseek_api_key = self.api_key_entry.get().strip()
        cfg.use_ai = bool(self.use_ai_var.get())
        cfg.whisper_model_size = self.whisper_var.get()
        cfg.whisper_device = self.whisper_device_var.get() or "auto"
        # compute_type оставляем "auto" — transcriber.py сам выберет оптимальный
        cfg.whisper_compute_type = "auto"
        cfg.face_tracking = bool(self.face_tracking_var.get())
        cfg.face_zoom = bool(self.face_zoom_var.get())
        cfg.multi_face_mode = self.multi_face_var.get()
        cfg.speaker_detection = bool(self.speaker_var.get())
        cfg.use_gpu = bool(self.gpu_var.get())
        cfg.strip_subtitle_punct = bool(self.strip_punct_var.get())

        cfg.clip.length_preset = self.length_var.get()
        cfg.clip.min_clip_count = _safe_int(self.min_clips_entry.get(), 3)
        cfg.clip.max_clip_count = _safe_int(self.max_clips_entry.get(), 10)
        cfg.clip.language = self.language_var.get()

        cfg.subtitle.template_id = self._template_id_by_name.get(
            self.template_name_var.get(), cfg.subtitle.template_id
        )
        cfg.subtitle.font = self.font_var.get()
        cfg.subtitle.font_size = _safe_int(self.font_size_entry.get(), 78)
        cfg.subtitle.max_words_per_line = max(1, _safe_int(self.words_per_line_entry.get(), 4))
        cfg.subtitle.primary_color = self.primary_color_entry.get().strip() or "#FFFFFF"
        cfg.subtitle.highlight_color = self.highlight_color_entry.get().strip() or "#FFD800"
        cfg.subtitle.outline_color = self.outline_color_entry.get().strip() or "#000000"
        cfg.subtitle.outline_width = max(0, _safe_int(self.outline_width_entry.get(), 5))
        cfg.subtitle.box_style = self.box_style_var.get()
        cfg.subtitle.back_color = self.back_color_entry.get().strip() or "#000000"
        cfg.subtitle.back_alpha = max(0, min(255, _safe_int(self.back_alpha_entry.get(), 200)))
        cfg.subtitle.silence_gap_sec = max(0.05, _safe_float(self.silence_gap_entry.get(), 0.4))
        cfg.subtitle.uppercase = bool(self.uppercase_var.get())
        cfg.subtitle.word_highlight = bool(self.word_highlight_var.get())
        cfg.subtitle.position_v = self.position_var.get()

        cfg.title.enabled = bool(self.title_enabled_var.get())
        cfg.title.template_id = self._title_id_by_name.get(
            self.title_template_var.get(), cfg.title.template_id
        )
        cfg.title.text_mode = self.title_mode_var.get()
        cfg.title.custom_text = self.title_text_entry.get().strip()

        cfg.overlay.enabled = bool(self.overlay_enabled_var.get())
        cfg.overlay.image_path = self.overlay_path_var.get().strip()
        cfg.overlay.position = self.overlay_pos_var.get()
        cfg.overlay.size_pct = max(2.0, min(60.0, _safe_float(self.overlay_size_entry.get(), 12.0)))
        cfg.overlay.opacity = max(0.0, min(1.0, _safe_float(self.overlay_opacity_entry.get(), 0.85)))

        cfg.music.mode = self.music_mode_var.get()
        cfg.music.common_track = self.music_common_path_var.get().strip() or None
        cfg.music.volume = max(0.0, min(1.0, _safe_float(self.volume_entry.get(), 0.15)))
        cfg.music.sidechain = bool(self.sidechain_var.get())

        return cfg

    def _save_config(self) -> None:
        cfg = self._collect_config()
        cfg.save()
        self._log("Настройки сохранены.")

    def _save_preset(self) -> None:
        cfg = self._collect_config()
        path = filedialog.asksaveasfilename(
            title="Сохранить пресет как...",
            initialdir=str(PRESETS_DIR),
            defaultextension=".json",
            filetypes=[("JSON preset", "*.json")],
        )
        if not path:
            return
        cfg.export_preset(Path(path))
        self._log(f"Пресет сохранён: {path}")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Загрузить пресет",
            initialdir=str(PRESETS_DIR),
            filetypes=[("JSON preset", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        ok = self.cfg.import_preset(Path(path))
        if not ok:
            messagebox.showerror("Ошибка", "Не удалось загрузить пресет — неверный формат.")
            return
        self._reload_form_from_config()
        self._log(f"Пресет загружен: {path}")

    def _reload_form_from_config(self) -> None:
        """После загрузки пресета — обновить все поля формы."""
        c = self.cfg
        self.length_var.set(c.clip.length_preset)
        self.min_clips_entry.delete(0, "end"); self.min_clips_entry.insert(0, str(c.clip.min_clip_count))
        self.max_clips_entry.delete(0, "end"); self.max_clips_entry.insert(0, str(c.clip.max_clip_count))
        self.language_var.set(c.clip.language)

        if c.subtitle.template_id in self._template_name_by_id:
            self.template_name_var.set(self._template_name_by_id[c.subtitle.template_id])
            self.template_desc_label.configure(
                text=TEMPLATES[c.subtitle.template_id].description
            )
        self.font_var.set(c.subtitle.font)
        self.font_size_entry.delete(0, "end"); self.font_size_entry.insert(0, str(c.subtitle.font_size))
        self.words_per_line_entry.delete(0, "end"); self.words_per_line_entry.insert(0, str(c.subtitle.max_words_per_line))
        self.primary_color_entry.delete(0, "end"); self.primary_color_entry.insert(0, c.subtitle.primary_color)
        self.highlight_color_entry.delete(0, "end"); self.highlight_color_entry.insert(0, c.subtitle.highlight_color)
        self.outline_color_entry.delete(0, "end"); self.outline_color_entry.insert(0, c.subtitle.outline_color)
        self.outline_width_entry.delete(0, "end"); self.outline_width_entry.insert(0, str(c.subtitle.outline_width))
        self.box_style_var.set(c.subtitle.box_style)
        self.back_color_entry.delete(0, "end"); self.back_color_entry.insert(0, c.subtitle.back_color)
        self.back_alpha_entry.delete(0, "end"); self.back_alpha_entry.insert(0, str(c.subtitle.back_alpha))
        self.silence_gap_entry.delete(0, "end"); self.silence_gap_entry.insert(0, str(c.subtitle.silence_gap_sec))
        self.uppercase_var.set(c.subtitle.uppercase)
        self.word_highlight_var.set(c.subtitle.word_highlight)
        self.position_var.set(c.subtitle.position_v)

        self.title_enabled_var.set(c.title.enabled)
        if c.title.template_id in self._title_name_by_id:
            self.title_template_var.set(self._title_name_by_id[c.title.template_id])
        self.title_mode_var.set(c.title.text_mode)
        self.title_text_entry.delete(0, "end"); self.title_text_entry.insert(0, c.title.custom_text)

        self.overlay_enabled_var.set(c.overlay.enabled)
        self.overlay_path_var.set(c.overlay.image_path)
        self.overlay_pos_var.set(c.overlay.position)
        self.overlay_size_entry.delete(0, "end"); self.overlay_size_entry.insert(0, str(c.overlay.size_pct))
        self.overlay_opacity_entry.delete(0, "end"); self.overlay_opacity_entry.insert(0, str(c.overlay.opacity))

        self.music_mode_var.set(c.music.mode)
        self.music_common_path_var.set(c.music.common_track or "")
        self.volume_entry.delete(0, "end"); self.volume_entry.insert(0, str(c.music.volume))
        self.sidechain_var.set(c.music.sidechain)

        if hasattr(self, "face_tracking_var"):
            self.face_tracking_var.set(c.face_tracking)
        if hasattr(self, "face_zoom_var"):
            self.face_zoom_var.set(c.face_zoom)
        if hasattr(self, "multi_face_var"):
            self.multi_face_var.set(getattr(c, "multi_face_mode", "auto"))
        if hasattr(self, "speaker_var"):
            self.speaker_var.set(getattr(c, "speaker_detection", True))
        if hasattr(self, "gpu_var"):
            self.gpu_var.set(getattr(c, "use_gpu", True))
        if hasattr(self, "strip_punct_var"):
            self.strip_punct_var.set(getattr(c, "strip_subtitle_punct", True))

    def _open_template_editor(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning(
                "Сейчас нельзя",
                "Сейчас идёт полная обработка видео — редактор шаблонов недоступен." + MARK_WARNING,
            )
            return
        if self._preview_worker and self._preview_worker.is_alive():
            messagebox.showwarning(
                "Сейчас нельзя",
                "Сейчас идёт генерация превью — редактор шаблонов откроется когда оно закончится." + MARK_WARNING,
            )
            return
        cfg = self._collect_config()
        editor = TemplateEditor(
            self,
            initial_subtitle=cfg.subtitle,
            initial_title=cfg.title,
            initial_overlay=cfg.overlay,
            on_save=self._on_template_saved,
        )
        editor.transient(self)
        editor.grab_set()

    def _on_template_saved(self, name, subtitle, title_cfg, overlay) -> None:
        self.cfg.subtitle = subtitle
        self.cfg.title = title_cfg
        self.cfg.overlay = overlay
        self.cfg.save()
        self._reload_form_from_config()
        self._log(f"Шаблон '{name}' применён к текущим настройкам.")

    def _start_preview(self) -> None:
        if self._preview_worker and self._preview_worker.is_alive():
            messagebox.showinfo(
                "Сейчас нельзя",
                "Превью уже генерируется — подожди когда оно закончится." + MARK_WARNING,
            )
            return
        if self._worker and self._worker.is_alive():
            messagebox.showwarning(
                "Сейчас нельзя",
                "Сейчас идёт полная обработка видео — превью недоступно пока она не закончится." + MARK_WARNING,
            )
            return

        source = self.source_entry.get().strip()
        if not source:
            messagebox.showerror("Нет источника", "Сначала выбери видеофайл для превью.")
            return

        src_path = Path(source)
        if not src_path.exists():
            messagebox.showerror(
                "Файл не найден",
                f"Превью работает только с локальным файлом.\nНе найдено: {source}\n\n"
                f"Если у тебя URL — нажми сначала 'Сгенерировать клипы', оно скачает видео.",
            )
            return

        cfg = self._collect_config()
        cfg.save()

        self.preview_btn.configure(state="disabled", text="⏳ Превью...")
        self.progress_bar.set(0.1)
        self._log("\n--- ГЕНЕРАЦИЯ ПРЕВЬЮ (15 сек) ---")

        self._preview_worker = threading.Thread(
            target=self._run_preview_thread, args=(src_path, cfg), daemon=True
        )
        self._preview_worker.start()

    def _run_preview_thread(self, src_path: Path, cfg: AppConfig) -> None:
        def cb(msg: str) -> None:
            self.after(0, self._log, msg)

        try:
            output = generate_preview(src_path, cfg, duration=15.0, progress_cb=cb)
            self.after(0, self._on_preview_done, output, None)
        except Exception as e:
            self.after(0, self._on_preview_done, None, e)

    def _on_preview_done(self, output: Optional[Path], err: Optional[Exception]) -> None:
        self.preview_btn.configure(state="normal", text="🎬 Превью (15 сек)")
        self.progress_bar.set(0.0)
        if err is not None:
            self._log(f"Ошибка превью: {err}")
            messagebox.showerror("Ошибка превью", str(err))
            return
        if output:
            self._log(f"Превью готово: {output}")
            self._open_path(output)

    def _start_generation(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning(
                "Сейчас нельзя",
                "Уже идёт обработка видео — дождись окончания текущей задачи." + MARK_WARNING,
            )
            return
        if self._preview_worker and self._preview_worker.is_alive():
            messagebox.showwarning(
                "Сейчас нельзя",
                "Сейчас генерируется превью — дождись окончания, потом запускай полную обработку." + MARK_WARNING,
            )
            return

        source = self.source_entry.get().strip()
        if not source:
            messagebox.showerror("Нет источника", "Укажи URL или путь к видео.")
            return

        cfg = self._collect_config()
        cfg.save()

        if cfg.use_ai and not cfg.deepseek_api_key:
            ans = messagebox.askyesno(
                "Нет API ключа",
                "DeepSeek API ключ не задан. Продолжить без AI (равномерная нарезка)?",
            )
            if not ans:
                return
            cfg.use_ai = False

        self.generate_btn.configure(state="disabled", text="⏳ Обработка...")
        self.progress_bar.set(0.0)
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self._results = []

        self._worker = threading.Thread(
            target=self._run_in_thread, args=(source, cfg), daemon=True
        )
        self._worker.start()

    def _run_in_thread(self, source: str, cfg: AppConfig) -> None:
        def cb(msg: str, p: float) -> None:
            self.after(0, self._on_progress, msg, p)

        try:
            results = run_pipeline(source, cfg, progress_cb=cb)
            self.after(0, self._on_finished, results, None)
        except Exception as e:
            self.after(0, self._on_finished, [], e)

    def _on_progress(self, msg: str, p: float) -> None:
        self._log(msg)
        self.progress_bar.set(max(0.0, min(1.0, p)))

    def _on_finished(self, results: list[ClipResult], err: Optional[Exception]) -> None:
        self.generate_btn.configure(state="normal", text="🚀 Сгенерировать клипы")
        if err is not None:
            self._log(f"ОШИБКА: {err}")
            messagebox.showerror("Ошибка", str(err))
            return
        self._results = results
        self._log(f"\nГотово! Создано {len(results)} клипов.")
        for r in results:
            self._add_result_row(r)

    def _add_result_row(self, r: ClipResult) -> None:
        row = ctk.CTkFrame(self.results_frame)
        row.pack(fill="x", pady=4, padx=4)

        title = f"#{r.index}  {r.suggestion.title}  ({r.suggestion.duration:.1f}с, {r.size_mb:.1f} МБ)"
        ctk.CTkLabel(row, text=title, anchor="w").pack(side="left", padx=8, expand=True, fill="x")

        ctk.CTkButton(
            row, text="▶ Открыть", width=90,
            command=lambda p=r.output_path: self._open_path(p),
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            row, text="📂 В папке", width=90,
            command=lambda p=r.output_path: self._open_in_folder(p),
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            row, text="✏️ Редактор", width=110,
            fg_color="#7c3aed", hover_color="#9c54f0",
            command=lambda res=r: self._open_clip_editor(res),
        ).pack(side="right", padx=4)

    def _open_clip_editor(self, r: ClipResult) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning(
                "Подожди",
                "Сейчас идёт полная обработка видео — редактор клипа недоступен." + MARK_WARNING,
            )
            return
        if self._preview_worker and self._preview_worker.is_alive():
            messagebox.showwarning(
                "Подожди",
                "Сейчас рендерится превью — закрытие/правки могут поломать рендер." + MARK_WARNING,
            )
            return

        if r.source_video is None or r.transcript is None:
            messagebox.showerror(
                "Невозможно открыть редактор",
                "Этот клип был сгенерирован в старой сессии без сохранения "
                "исходника и транскрипта.\nПерезапусти генерацию из главного окна.",
            )
            return

        from .clip_editor import ClipEditorState, ClipEditorWindow
        from .pipeline import re_render_clip

        state = ClipEditorState(
            clip_index=r.index,
            source_video=r.source_video,
            output_video=r.output_path,
            suggestion=r.suggestion,
            multi_face_mode=self.cfg.multi_face_mode,
            sub_position_v=self.cfg.subtitle.position_v,
            sub_margin_v_pct=self.cfg.subtitle.margin_v_pct,
        )

        def _do_rerender(st):
            re_render_clip(
                st, self.cfg, r.transcript,
                progress_cb=lambda m: self.after(0, lambda mm=m: self._log(mm)),
            )

        ClipEditorWindow(self, state, self.cfg, r.transcript, on_rerender=_do_rerender)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror("Не открыть", str(e))

    def _open_in_folder(self, path: Path) -> None:
        folder = path.parent
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as e:
            messagebox.showerror("Не открыть", str(e))

    def _open_output_dir(self) -> None:
        self._open_path(Path(self.cfg.output_dir))


def main() -> None:
    if not FONTS_DIR.exists():
        print(f"Внимание: папка со шрифтами не найдена: {FONTS_DIR}")
    app = VizardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
