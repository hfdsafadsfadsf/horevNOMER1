"""
Окно редактора шаблонов (как в Vizard.ai).

Слева вкладки: Subtitle / Headline / Logo
В центре — сетка стилей с миниатюрами + кастомные настройки
Справа — живое 9:16 превью (PIL) которое обновляется на каждое изменение

Save → сохраняет шаблон в ~/.vizard_clone/presets/<name>.json
"""
from __future__ import annotations

import json
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from .config import (
    AppConfig,
    OverlayConfig,
    PRESETS_DIR,
    SubtitleStyle,
    TitleConfig,
)
from .preview_renderer import (
    render_preview_frame,
    render_template_thumbnail,
    render_title_thumbnail,
)
from .templates import TEMPLATES, TITLE_TEMPLATES


POSITION_LABELS = {
    "top_left": "↖", "top_center": "⬆", "top_right": "↗",
    "bottom_left": "↙", "bottom_center": "⬇", "bottom_right": "↘",
}


class TemplateEditor(ctk.CTkToplevel):
    def __init__(
        self,
        master,
        initial_subtitle: SubtitleStyle,
        initial_title: TitleConfig,
        initial_overlay: OverlayConfig,
        on_save: Optional[Callable[[str, SubtitleStyle, TitleConfig, OverlayConfig], None]] = None,
    ):
        super().__init__(master)
        self.title("Редактор шаблона")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        self._on_save = on_save
        self.subtitle = _clone_dataclass(initial_subtitle, SubtitleStyle)
        self.title_cfg = _clone_dataclass(initial_title, TitleConfig)
        self.overlay = _clone_dataclass(initial_overlay, OverlayConfig)

        self._thumb_refs: list[ctk.CTkImage] = []

        self._build_ui()
        self._refresh_preview()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=2)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        tabs = ctk.CTkFrame(self, width=170, fg_color="#1a1a25")
        tabs.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        tabs.grid_propagate(False)

        self._tab_buttons: dict[str, ctk.CTkButton] = {}
        self._current_tab = "subtitle"

        ctk.CTkLabel(tabs, text="Шаблон", font=("Helvetica", 16, "bold")).pack(pady=(14, 12), padx=12, anchor="w")

        for key, label, icon in [
            ("subtitle", "Субтитры", "💬"),
            ("headline", "Заголовок", "📝"),
            ("logo", "Логотип", "🖼"),
        ]:
            btn = ctk.CTkButton(
                tabs, text=f"  {icon}  {label}", anchor="w", height=40,
                fg_color="transparent", hover_color="#2a2a3a",
                command=lambda k=key: self._switch_tab(k),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._tab_buttons[key] = btn

        self.middle = ctk.CTkScrollableFrame(self)
        self.middle.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        right = ctk.CTkFrame(self, fg_color="#16161e")
        right.grid(row=0, column=2, sticky="nsew", padx=(0, 8), pady=8)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right, text="Превью 9:16  (тащи текст мышкой)",
            font=("Helvetica", 14, "bold")
        ).grid(row=0, column=0, pady=(12, 8))

        # Canvas размером 405x720 — превью видео + интерактивный drag&drop
        self.preview_w = 405
        self.preview_h = 720
        self.preview_canvas = tk.Canvas(
            right, width=self.preview_w, height=self.preview_h,
            bg="#101018", highlightthickness=0, bd=0, cursor="hand2",
        )
        self.preview_canvas.grid(row=1, column=0, padx=20, pady=(0, 16))

        # Drag state
        self._drag_target: Optional[str] = None  # "subtitle"|"title"|None
        self._drag_offset_y: int = 0
        # Хитбоксы (y_top, y_bottom) для каждого элемента — обновляются после render
        self._subtitle_hitbox: tuple[int, int] = (0, 0)
        self._title_hitbox: tuple[int, int] = (0, 0)
        # Текущая PIL картинка превью — её перерисуем при drag для скорости
        self._preview_pil: Optional[Image.Image] = None
        self._preview_tk: Optional[ImageTk.PhotoImage] = None

        self.preview_canvas.bind("<Button-1>", self._on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        ctk.CTkLabel(
            right,
            text="Тащи субтитры/заголовок мышкой для\nточной настройки положения",
            text_color="gray60",
            font=("Helvetica", 11),
            justify="center",
        ).grid(row=2, column=0, pady=(0, 12))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        footer.grid_columnconfigure(0, weight=1)

        name_row = ctk.CTkFrame(footer, fg_color="transparent")
        name_row.grid(row=0, column=0, sticky="ew", padx=8)
        ctk.CTkLabel(name_row, text="Название шаблона:").pack(side="left", padx=(0, 8))
        self.name_entry = ctk.CTkEntry(name_row, placeholder_text="мой_шаблон")
        self.name_entry.pack(side="left", expand=True, fill="x")

        btns = ctk.CTkFrame(footer, fg_color="transparent")
        btns.grid(row=0, column=1, sticky="e", padx=8)
        ctk.CTkButton(btns, text="Отмена", width=110, command=self.destroy,
                      fg_color="#444").pack(side="left", padx=4)
        ctk.CTkButton(btns, text="💾 Сохранить", width=140, command=self._save,
                      fg_color="#a78bfa", text_color="#1a1a25",
                      font=("Helvetica", 13, "bold")).pack(side="left", padx=4)

        self._switch_tab("subtitle")

    def _switch_tab(self, key: str) -> None:
        self._current_tab = key
        for k, btn in self._tab_buttons.items():
            btn.configure(
                fg_color="#a78bfa" if k == key else "transparent",
                text_color="#1a1a25" if k == key else "white",
            )
        for w in self.middle.winfo_children():
            w.destroy()
        if key == "subtitle":
            self._build_subtitle_tab()
        elif key == "headline":
            self._build_headline_tab()
        elif key == "logo":
            self._build_logo_tab()

    def _build_subtitle_tab(self) -> None:
        ctk.CTkLabel(
            self.middle, text="Выбери стиль субтитров",
            font=("Helvetica", 18, "bold")
        ).pack(anchor="w", pady=(8, 4))
        ctk.CTkLabel(
            self.middle,
            text="9 готовых шаблонов. Можно настроить вручную ниже.",
            text_color="gray70",
        ).pack(anchor="w", pady=(0, 12))

        grid = ctk.CTkFrame(self.middle, fg_color="transparent")
        grid.pack(fill="x", pady=(0, 16))
        templates = list(TEMPLATES.values())
        cols = 3
        for i, tmpl in enumerate(templates):
            row, col = divmod(i, cols)
            self._make_subtitle_card(grid, tmpl, row, col)

        ctk.CTkLabel(self.middle, text="Точные настройки",
                     font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(12, 4))

        adv = ctk.CTkFrame(self.middle, fg_color="#1f1f2e")
        adv.pack(fill="x", pady=(0, 8), padx=2)

        self._add_text_field(adv, "Цвет текста", "primary_color", self.subtitle.primary_color)
        self._add_text_field(adv, "Highlight (текущее слово)", "highlight_color", self.subtitle.highlight_color)
        self._add_text_field(adv, "Цвет обводки", "outline_color", self.subtitle.outline_color)
        self._add_int_field(adv, "Толщина обводки", "outline_width", self.subtitle.outline_width, 0, 20)
        self._add_int_field(adv, "Размер шрифта", "font_size", self.subtitle.font_size, 20, 160)

        self._add_segmented(adv, "Стиль рамки", "box_style",
                            ["outline", "shadow", "background"], self.subtitle.box_style)
        self._add_text_field(adv, "Цвет фона (для box_style=background)", "back_color", self.subtitle.back_color)
        self._add_int_field(adv, "Прозрачность фона (0-255)", "back_alpha", self.subtitle.back_alpha, 0, 255)
        self._add_segmented(adv, "Позиция", "position_v",
                            ["top", "center", "bottom"], self.subtitle.position_v)

        self._add_checkbox(adv, "ВЕРХНИЙ РЕГИСТР", "uppercase", self.subtitle.uppercase)
        self._add_checkbox(adv, "Подсветка текущего слова (karaoke)", "word_highlight", self.subtitle.word_highlight)

    def _make_subtitle_card(self, parent, tmpl, row: int, col: int) -> None:
        card = ctk.CTkFrame(parent, fg_color="#1a1a25",
                            border_width=2, border_color="#2a2a3a")
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)

        try:
            thumb_pil = render_template_thumbnail(tmpl, 240, 130)
            ctk_img = ctk.CTkImage(light_image=thumb_pil, dark_image=thumb_pil, size=(240, 130))
            self._thumb_refs.append(ctk_img)
            img_label = ctk.CTkLabel(card, image=ctk_img, text="")
            img_label.pack(padx=6, pady=6)
        except Exception as e:
            ctk.CTkLabel(card, text=tmpl.name, height=130).pack(padx=6, pady=6)

        ctk.CTkLabel(card, text=tmpl.name, font=("Helvetica", 12, "bold")).pack()
        ctk.CTkButton(
            card, text="Выбрать", height=28,
            command=lambda t=tmpl: self._apply_subtitle_template(t),
        ).pack(fill="x", padx=6, pady=(4, 6))

    def _apply_subtitle_template(self, tmpl) -> None:
        self.subtitle.template_id = tmpl.id
        self.subtitle.font = tmpl.font
        self.subtitle.font_size = tmpl.font_size
        self.subtitle.primary_color = tmpl.primary_color
        self.subtitle.highlight_color = tmpl.highlight_color
        self.subtitle.outline_color = tmpl.outline_color
        self.subtitle.outline_width = tmpl.outline_width
        self.subtitle.box_style = tmpl.box_style
        self.subtitle.shadow = tmpl.shadow
        self.subtitle.shadow_color = tmpl.shadow_color
        self.subtitle.back_color = tmpl.back_color
        self.subtitle.back_alpha = tmpl.back_alpha
        self.subtitle.back_padding = tmpl.back_padding
        self.subtitle.uppercase = tmpl.uppercase
        self.subtitle.word_highlight = tmpl.word_highlight
        self.subtitle.max_words_per_line = tmpl.max_words_per_line
        self.subtitle.position_v = tmpl.position_v
        self.subtitle.margin_v_pct = tmpl.margin_v_pct
        self._switch_tab("subtitle")
        self._refresh_preview()

    def _build_headline_tab(self) -> None:
        ctk.CTkLabel(self.middle, text="Заголовок поверх видео",
                     font=("Helvetica", 18, "bold")).pack(anchor="w", pady=(8, 4))

        en_row = ctk.CTkFrame(self.middle, fg_color="transparent")
        en_row.pack(fill="x", pady=(0, 8))
        self.title_enabled_var = ctk.BooleanVar(value=self.title_cfg.enabled)
        ctk.CTkCheckBox(
            en_row, text="Включить заголовок",
            variable=self.title_enabled_var,
            command=self._on_title_enabled_change,
        ).pack(side="left")

        ctk.CTkLabel(self.middle, text="Текст заголовка:").pack(anchor="w", pady=(4, 2))
        self.headline_text_var = ctk.StringVar(value=self.title_cfg.custom_text or "ВОТ ТАК БУДЕТ ВЫГЛЯДЕТЬ")
        entry = ctk.CTkEntry(self.middle, textvariable=self.headline_text_var)
        entry.pack(fill="x", pady=(0, 12))
        self.headline_text_var.trace_add("write", lambda *a: self._on_headline_text_change())

        grid = ctk.CTkFrame(self.middle, fg_color="transparent")
        grid.pack(fill="x", pady=(0, 16))
        templates = list(TITLE_TEMPLATES.values())
        cols = 3
        for i, tmpl in enumerate(templates):
            row, col = divmod(i, cols)
            self._make_title_card(grid, tmpl, row, col)

        ctk.CTkLabel(self.middle, text="Точные настройки",
                     font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(12, 4))

        adv = ctk.CTkFrame(self.middle, fg_color="#1f1f2e")
        adv.pack(fill="x", pady=(0, 8), padx=2)
        self._add_title_text_field(adv, "Цвет текста", "primary_color", self.title_cfg.primary_color)
        self._add_title_text_field(adv, "Цвет обводки", "outline_color", self.title_cfg.outline_color)
        self._add_title_text_field(adv, "Цвет фона", "back_color", self.title_cfg.back_color)
        self._add_title_int_field(adv, "Прозрачность фона", "back_alpha", self.title_cfg.back_alpha, 0, 255)
        self._add_title_int_field(adv, "Размер шрифта", "font_size", self.title_cfg.font_size, 20, 160)
        self._add_title_segmented(adv, "Вариант", "variant",
                                  ["top_banner", "top_centered", "lower_third"], self.title_cfg.variant)
        self._add_title_checkbox(adv, "ВЕРХНИЙ РЕГИСТР", "uppercase", self.title_cfg.uppercase)
        # 0 = показывать весь клип. >0 = скрывать через N секунд.
        self._add_title_float_field(adv, "Авто-скрытие (сек, 0=всегда)",
                                    "duration", self.title_cfg.duration, 0.0, 30.0)

    def _make_title_card(self, parent, tmpl, row: int, col: int) -> None:
        card = ctk.CTkFrame(parent, fg_color="#1a1a25",
                            border_width=2, border_color="#2a2a3a")
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        try:
            thumb_pil = render_title_thumbnail(tmpl, 240, 130)
            ctk_img = ctk.CTkImage(light_image=thumb_pil, dark_image=thumb_pil, size=(240, 130))
            self._thumb_refs.append(ctk_img)
            ctk.CTkLabel(card, image=ctk_img, text="").pack(padx=6, pady=6)
        except Exception:
            ctk.CTkLabel(card, text=tmpl.name, height=130).pack(padx=6, pady=6)
        ctk.CTkLabel(card, text=tmpl.name, font=("Helvetica", 12, "bold")).pack()
        ctk.CTkButton(
            card, text="Выбрать", height=28,
            command=lambda t=tmpl: self._apply_title_template(t),
        ).pack(fill="x", padx=6, pady=(4, 6))

    def _apply_title_template(self, tmpl) -> None:
        self.title_cfg.template_id = tmpl.id
        self.title_cfg.enabled = tmpl.variant != "none"
        self.title_cfg.font = tmpl.font
        self.title_cfg.font_size = tmpl.font_size
        self.title_cfg.primary_color = tmpl.primary_color
        self.title_cfg.outline_color = tmpl.outline_color
        self.title_cfg.outline_width = tmpl.outline_width
        self.title_cfg.back_color = tmpl.back_color
        self.title_cfg.back_alpha = tmpl.back_alpha
        self.title_cfg.back_padding = tmpl.back_padding
        self.title_cfg.uppercase = tmpl.uppercase
        self.title_cfg.variant = tmpl.variant
        self.title_cfg.margin_v_pct = tmpl.margin_v_pct
        self.title_cfg.duration = tmpl.duration
        self._switch_tab("headline")
        self._refresh_preview()

    def _build_logo_tab(self) -> None:
        ctk.CTkLabel(self.middle, text="Логотип / обложка",
                     font=("Helvetica", 18, "bold")).pack(anchor="w", pady=(8, 4))

        self.overlay_enabled_var = ctk.BooleanVar(value=self.overlay.enabled)
        ctk.CTkCheckBox(
            self.middle, text="Включить логотип",
            variable=self.overlay_enabled_var,
            command=self._on_overlay_enabled_change,
        ).pack(anchor="w", pady=(0, 12))

        path_row = ctk.CTkFrame(self.middle, fg_color="transparent")
        path_row.pack(fill="x", pady=(0, 8))
        self.overlay_path_var = ctk.StringVar(value=self.overlay.image_path)
        ctk.CTkEntry(path_row, textvariable=self.overlay_path_var,
                     placeholder_text="Путь к PNG/JPG").pack(side="left", expand=True, fill="x", padx=(0, 8))
        ctk.CTkButton(path_row, text="📁 Выбрать", width=110,
                      command=self._browse_logo).pack(side="left")
        self.overlay_path_var.trace_add("write", lambda *a: self._on_overlay_path_change())

        ctk.CTkLabel(self.middle, text="Позиция в кадре:").pack(anchor="w", pady=(8, 4))
        pos_frame = ctk.CTkFrame(self.middle, fg_color="transparent")
        pos_frame.pack(fill="x", pady=(0, 12))
        positions = [
            ["top_left", "top_center", "top_right"],
            ["bottom_left", "bottom_center", "bottom_right"],
        ]
        self._pos_buttons: dict[str, ctk.CTkButton] = {}
        for r_idx, row_pos in enumerate(positions):
            for c_idx, p in enumerate(row_pos):
                btn = ctk.CTkButton(
                    pos_frame, text=POSITION_LABELS[p], width=70, height=50,
                    font=("Helvetica", 20, "bold"),
                    command=lambda pp=p: self._set_overlay_position(pp),
                )
                btn.grid(row=r_idx, column=c_idx, padx=4, pady=4)
                self._pos_buttons[p] = btn
        self._update_pos_buttons()

        adv = ctk.CTkFrame(self.middle, fg_color="#1f1f2e")
        adv.pack(fill="x", pady=(8, 8), padx=2)

        self._add_overlay_float_field(adv, "Размер (% ширины видео)", "size_pct", self.overlay.size_pct, 2, 60)
        self._add_overlay_float_field(adv, "Прозрачность (0-1)", "opacity", self.overlay.opacity, 0, 1)
        self._add_overlay_float_field(adv, "Отступ от края (%)", "margin_pct", self.overlay.margin_pct, 0, 20)

    def _set_overlay_position(self, p: str) -> None:
        self.overlay.position = p
        self._update_pos_buttons()
        self._refresh_preview()

    def _update_pos_buttons(self) -> None:
        for p, btn in self._pos_buttons.items():
            btn.configure(fg_color="#a78bfa" if p == self.overlay.position else "#3a3a4a",
                          text_color="#1a1a25" if p == self.overlay.position else "white")

    def _browse_logo(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбери логотип",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")],
        )
        if path:
            self.overlay_path_var.set(path)

    def _on_overlay_enabled_change(self) -> None:
        self.overlay.enabled = bool(self.overlay_enabled_var.get())
        self._refresh_preview()

    def _on_overlay_path_change(self) -> None:
        self.overlay.image_path = self.overlay_path_var.get().strip()
        self._refresh_preview()

    def _on_title_enabled_change(self) -> None:
        self.title_cfg.enabled = bool(self.title_enabled_var.get())
        self._refresh_preview()

    def _on_headline_text_change(self) -> None:
        self.title_cfg.custom_text = self.headline_text_var.get()
        self._refresh_preview()

    def _add_text_field(self, parent, label: str, attr: str, val: str) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=val)
        e = ctk.CTkEntry(row, textvariable=var, width=140)
        e.pack(side="left")
        var.trace_add("write", lambda *a, n=attr, v=var: self._set_subtitle_attr(n, v.get()))

    def _add_int_field(self, parent, label: str, attr: str, val: int, lo: int, hi: int) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=str(val))
        e = ctk.CTkEntry(row, textvariable=var, width=80)
        e.pack(side="left")
        var.trace_add("write", lambda *a, n=attr, v=var: self._set_subtitle_attr_int(n, v.get(), lo, hi))

    def _add_segmented(self, parent, label: str, attr: str, options: list[str], val: str) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=val)
        seg = ctk.CTkSegmentedButton(row, values=options, variable=var,
                                     command=lambda v, n=attr: self._set_subtitle_attr(n, v))
        seg.pack(side="left")

    def _add_checkbox(self, parent, label: str, attr: str, val: bool) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        var = ctk.BooleanVar(value=val)
        ctk.CTkCheckBox(
            row, text=label, variable=var,
            command=lambda n=attr, v=var: self._set_subtitle_attr(n, bool(v.get())),
        ).pack(side="left", padx=(240, 0))

    def _set_subtitle_attr(self, name: str, value) -> None:
        setattr(self.subtitle, name, value)
        self._refresh_preview()

    def _set_subtitle_attr_int(self, name: str, value: str, lo: int, hi: int) -> None:
        try:
            v = max(lo, min(hi, int(value)))
        except ValueError:
            return
        setattr(self.subtitle, name, v)
        self._refresh_preview()

    def _add_title_text_field(self, parent, label: str, attr: str, val: str) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=val)
        ctk.CTkEntry(row, textvariable=var, width=140).pack(side="left")
        var.trace_add("write", lambda *a, n=attr, v=var: self._set_title_attr(n, v.get()))

    def _add_title_int_field(self, parent, label: str, attr: str, val: int, lo: int, hi: int) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=str(val))
        ctk.CTkEntry(row, textvariable=var, width=80).pack(side="left")
        var.trace_add("write", lambda *a, n=attr, v=var: self._set_title_attr_int(n, v.get(), lo, hi))

    def _add_title_float_field(self, parent, label: str, attr: str,
                               val: float, lo: float, hi: float) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=str(val))
        ctk.CTkEntry(row, textvariable=var, width=80).pack(side="left")
        var.trace_add(
            "write",
            lambda *a, n=attr, v=var: self._set_title_attr_float(n, v.get(), lo, hi),
        )

    def _set_title_attr_float(self, name: str, value: str, lo: float, hi: float) -> None:
        try:
            v = max(lo, min(hi, float(value)))
            setattr(self.title_cfg, name, v)
            self._refresh_preview()
        except (TypeError, ValueError):
            pass

    def _add_title_segmented(self, parent, label: str, attr: str, options: list[str], val: str) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=val)
        ctk.CTkSegmentedButton(row, values=options, variable=var,
                               command=lambda v, n=attr: self._set_title_attr(n, v)).pack(side="left")

    def _add_title_checkbox(self, parent, label: str, attr: str, val: bool) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        var = ctk.BooleanVar(value=val)
        ctk.CTkCheckBox(
            row, text=label, variable=var,
            command=lambda n=attr, v=var: self._set_title_attr(n, bool(v.get())),
        ).pack(side="left", padx=(240, 0))

    def _set_title_attr(self, name: str, value) -> None:
        setattr(self.title_cfg, name, value)
        self._refresh_preview()

    def _set_title_attr_int(self, name: str, value: str, lo: int, hi: int) -> None:
        try:
            v = max(lo, min(hi, int(value)))
        except ValueError:
            return
        setattr(self.title_cfg, name, v)
        self._refresh_preview()

    def _add_overlay_float_field(self, parent, label: str, attr: str, val: float, lo: float, hi: float) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=3)
        ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
        var = ctk.StringVar(value=str(val))
        ctk.CTkEntry(row, textvariable=var, width=80).pack(side="left")
        var.trace_add("write", lambda *a, n=attr, v=var: self._set_overlay_attr_float(n, v.get(), lo, hi))

    def _set_overlay_attr_float(self, name: str, value: str, lo: float, hi: float) -> None:
        try:
            v = max(lo, min(hi, float(value)))
        except ValueError:
            return
        setattr(self.overlay, name, v)
        self._refresh_preview()

    # ---------- Drag-to-position на канвасе ----------
    def _compute_subtitle_y(self) -> int:
        """Текущая центральная y субтитра в координатах превью 405x720."""
        h = self.preview_h
        # font_size в превью масштабируется к 405x720 (так же как в render_preview_frame)
        font_size_preview = int(self.subtitle.font_size * (h / 1920))
        if self.subtitle.position_v == "top":
            return int(h * self.subtitle.margin_v_pct + font_size_preview)
        elif self.subtitle.position_v == "center":
            return h // 2
        else:  # bottom
            return h - int(h * self.subtitle.margin_v_pct) - int(font_size_preview * 0.6)

    def _compute_title_y(self) -> int:
        h = self.preview_h
        font_size_preview = int(self.title_cfg.font_size * (h / 1920))
        if self.title_cfg.variant == "lower_third":
            return int(h - h * self.title_cfg.margin_v_pct - font_size_preview * 0.6)
        return int(h * self.title_cfg.margin_v_pct + font_size_preview * 0.6)

    def _refresh_hitboxes(self) -> None:
        # Hitbox субтитра — полоса ~120px вокруг центральной y субтитра
        sy = self._compute_subtitle_y()
        self._subtitle_hitbox = (max(0, sy - 70), min(self.preview_h, sy + 70))
        # Hitbox заголовка
        ty = self._compute_title_y()
        self._title_hitbox = (max(0, ty - 60), min(self.preview_h, ty + 60))

    def _on_canvas_press(self, event) -> None:
        y = event.y
        # 1) Если кликнули прямо в hitbox — грабим конкретный элемент
        if self.title_cfg.enabled:
            ty_top, ty_bot = self._title_hitbox
            if ty_top <= y <= ty_bot:
                self._drag_target = "title"
                self._drag_offset_y = y - self._compute_title_y()
                self._refresh_preview()
                return
        sy_top, sy_bot = self._subtitle_hitbox
        if sy_top <= y <= sy_bot:
            self._drag_target = "subtitle"
            self._drag_offset_y = y - self._compute_subtitle_y()
            self._refresh_preview()
            return
        # 2) Иначе по умолчанию: верхняя 40% области → заголовок, ниже → субтитр
        if y < self.preview_h * 0.40 and self.title_cfg.enabled:
            self._drag_target = "title"
            self._drag_offset_y = 0
        else:
            self._drag_target = "subtitle"
            self._drag_offset_y = 0
        # Сразу таскаем элемент в позицию клика
        self._on_canvas_drag(event)

    def _on_canvas_drag(self, event) -> None:
        if not self._drag_target:
            return
        new_y = max(20, min(self.preview_h - 20, event.y - self._drag_offset_y))
        h = self.preview_h
        if self._drag_target == "subtitle":
            # Автоматически выбираем position_v по зоне:
            # - верхняя треть → top
            # - средняя треть → center
            # - нижняя треть → bottom
            font_size_preview = int(self.subtitle.font_size * (h / 1920))
            if new_y < h * 0.30:
                self.subtitle.position_v = "top"
                # margin_v_pct = (cy - font_size) / h
                pct = max(0.01, (new_y - font_size_preview) / h)
                self.subtitle.margin_v_pct = min(0.5, pct)
            elif new_y > h * 0.65:
                self.subtitle.position_v = "bottom"
                pct = max(0.01, (h - new_y - font_size_preview * 0.6) / h)
                self.subtitle.margin_v_pct = min(0.5, pct)
            else:
                self.subtitle.position_v = "center"
                self.subtitle.margin_v_pct = 0.10
        elif self._drag_target == "title":
            font_size_preview = int(self.title_cfg.font_size * (h / 1920))
            # Если перетащили в нижнюю половину — переключаем variant в lower_third
            if new_y > h * 0.55 and self.title_cfg.variant != "lower_third":
                self.title_cfg.variant = "lower_third"
            elif new_y <= h * 0.55 and self.title_cfg.variant == "lower_third":
                # переключаем обратно на top_centered чтобы margin от верха работал
                self.title_cfg.variant = "top_centered"
            if self.title_cfg.variant == "lower_third":
                pct = max(0.01, (h - new_y - font_size_preview * 0.6) / h)
            else:
                pct = max(0.01, (new_y - font_size_preview * 0.6) / h)
            self.title_cfg.margin_v_pct = min(0.5, pct)
        self._refresh_preview()

    def _on_canvas_release(self, event) -> None:
        self._drag_target = None
    # ---------- /Drag ----------

    def _refresh_preview(self) -> None:
        try:
            img = render_preview_frame(
                self.subtitle, self.title_cfg, self.overlay,
                width=self.preview_w, height=self.preview_h,
                headline_text=(self.title_cfg.custom_text or None),
            )
            self._preview_pil = img
            self._preview_tk = ImageTk.PhotoImage(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor="nw", image=self._preview_tk)

            # Пересчитываем хитбоксы
            self._refresh_hitboxes()

            # Рамки safe-zone (slabs)
            safe_x = int(self.preview_w * 80 / 1080)  # 80px в координатах превью
            self.preview_canvas.create_line(
                safe_x, 0, safe_x, self.preview_h, fill="#ff5555", dash=(3, 3), width=1
            )
            self.preview_canvas.create_line(
                self.preview_w - safe_x, 0, self.preview_w - safe_x, self.preview_h,
                fill="#ff5555", dash=(3, 3), width=1,
            )

            # Индикаторы зон субтитра/заголовка при drag
            if self._drag_target == "subtitle":
                top, bot = self._subtitle_hitbox
                self.preview_canvas.create_rectangle(
                    2, top, self.preview_w - 2, bot,
                    outline="#22dd88", width=2, dash=(4, 2),
                )
            elif self._drag_target == "title":
                top, bot = self._title_hitbox
                self.preview_canvas.create_rectangle(
                    2, top, self.preview_w - 2, bot,
                    outline="#22dd88", width=2, dash=(4, 2),
                )
            else:
                # лёгкий hover-намёк: тонкая рамка вокруг hit-зон
                if self.title_cfg.enabled:
                    top, bot = self._title_hitbox
                    self.preview_canvas.create_rectangle(
                        2, top, self.preview_w - 2, bot,
                        outline="#444466", width=1, dash=(2, 4),
                    )
                top, bot = self._subtitle_hitbox
                self.preview_canvas.create_rectangle(
                    2, top, self.preview_w - 2, bot,
                    outline="#444466", width=1, dash=(2, 4),
                )
        except Exception as e:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                self.preview_w // 2, self.preview_h // 2,
                text=f"Ошибка превью: {e}", fill="red", width=self.preview_w - 40,
            )

    def _save(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Нет имени", "Введи название шаблона.")
            return

        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in " _-.").strip()
        if not safe_name:
            messagebox.showerror("Некорректное имя", "Используй буквы/цифры/пробелы.")
            return

        path = PRESETS_DIR / f"{safe_name}.json"

        preset = {
            "name": safe_name,
            "subtitle": asdict(self.subtitle),
            "title": asdict(self.title_cfg),
            "overlay": asdict(self.overlay),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2, ensure_ascii=False)

        if self._on_save:
            self._on_save(safe_name, self.subtitle, self.title_cfg, self.overlay)

        messagebox.showinfo("Сохранено", f"Шаблон '{safe_name}' сохранён.\n{path}")
        self.destroy()


def _clone_dataclass(src, cls):
    """Создаёт независимую копию dataclass — изменения в редакторе не трогают оригинал до Save."""
    return cls(**{k: v for k, v in asdict(src).items() if k in cls.__dataclass_fields__})
