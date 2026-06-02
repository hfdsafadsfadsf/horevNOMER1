"""
Библиотека встроенных шаблонов субтитров.

Каждый шаблон описывает полный визуальный стиль и может быть применён
одним кликом из GUI. Пользователь также может сохранить свой шаблон
в файл (JSON) и загружать его в будущих проектах.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SubtitleTemplate:
    """Полный визуальный стиль субтитров. Совместим с SubtitleStyle, но расширен."""

    id: str
    name: str
    description: str

    # Шрифт
    font: str = "Montserrat-Black.ttf"
    font_size: int = 78
    bold: bool = True
    italic: bool = False
    letter_spacing: float = 0.0

    # Цвета
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD800"
    outline_color: str = "#000000"
    outline_width: int = 5

    # Тень / задний фон
    box_style: str = "outline"
    shadow: int = 2
    shadow_color: str = "#000000"
    back_color: str = "#000000"
    back_alpha: int = 200
    back_padding: int = 20

    # Текст
    uppercase: bool = True
    word_highlight: bool = True
    karaoke_mode: str = "color"
    max_words_per_line: int = 4

    # Позиция
    position_v: str = "center"
    margin_v_pct: float = 0.10


TEMPLATES: dict[str, SubtitleTemplate] = {
    "tiktok_classic": SubtitleTemplate(
        id="tiktok_classic",
        name="TikTok Classic",
        description="Белый CAPS, толстая чёрная обводка, центр — самый узнаваемый стиль",
        font="Montserrat-Black.ttf",
        font_size=82,
        primary_color="#FFFFFF",
        highlight_color="#FFD800",
        outline_color="#000000",
        outline_width=6,
        box_style="outline",
        shadow=2,
        uppercase=True,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=4,
        position_v="center",
    ),
    "mrbeast": SubtitleTemplate(
        id="mrbeast",
        name="MrBeast",
        description="Огромный жирный текст, жёлтая подсветка ключевого слова",
        font="Anton-Regular.ttf",
        font_size=96,
        primary_color="#FFFFFF",
        highlight_color="#FFD60A",
        outline_color="#000000",
        outline_width=8,
        box_style="outline",
        shadow=3,
        uppercase=True,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=3,
        position_v="bottom",
        margin_v_pct=0.18,
    ),
    "hormozi": SubtitleTemplate(
        id="hormozi",
        name="Hormozi",
        description="CAPS с жёлтой подсветкой текущего слова, нижняя треть",
        font="Montserrat-Black.ttf",
        font_size=84,
        primary_color="#FFFFFF",
        highlight_color="#FFEB3B",
        outline_color="#000000",
        outline_width=7,
        box_style="outline",
        shadow=2,
        uppercase=True,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=3,
        position_v="center",
    ),
    "minimal": SubtitleTemplate(
        id="minimal",
        name="Minimal",
        description="Тонкий элегантный шрифт, мягкая тень, без обводки",
        font="Inter-Black.otf",
        font_size=64,
        primary_color="#FFFFFF",
        highlight_color="#A78BFA",
        outline_color="#000000",
        outline_width=1,
        box_style="shadow",
        shadow=4,
        shadow_color="#000000",
        uppercase=False,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=5,
        position_v="bottom",
        margin_v_pct=0.12,
    ),
    "cinema": SubtitleTemplate(
        id="cinema",
        name="Cinema",
        description="Кинематографичный, низ кадра, тонкая обводка + тень",
        font="Montserrat-Bold.ttf",
        font_size=58,
        primary_color="#F8F8F8",
        highlight_color="#FFD27A",
        outline_color="#000000",
        outline_width=2,
        box_style="outline",
        shadow=3,
        uppercase=False,
        word_highlight=False,
        max_words_per_line=6,
        position_v="bottom",
        margin_v_pct=0.10,
    ),
    "pop_box": SubtitleTemplate(
        id="pop_box",
        name="Pop Box",
        description="Текст на цветной плашке, очень читаемо",
        font="Montserrat-Black.ttf",
        font_size=68,
        primary_color="#FFFFFF",
        highlight_color="#FFE600",
        outline_color="#000000",
        outline_width=0,
        box_style="background",
        back_color="#000000",
        back_alpha=200,
        back_padding=24,
        uppercase=True,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=4,
        position_v="center",
    ),
    "karaoke_yellow": SubtitleTemplate(
        id="karaoke_yellow",
        name="Karaoke Yellow",
        description="Классическое подсвечивание слова по словам жёлтым",
        font="Rubik-Black.ttf",
        font_size=78,
        primary_color="#FFFFFF",
        highlight_color="#FFC400",
        outline_color="#000000",
        outline_width=5,
        box_style="outline",
        shadow=2,
        uppercase=True,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=4,
        position_v="center",
    ),
    "neon": SubtitleTemplate(
        id="neon",
        name="Neon Glow",
        description="Яркий неоновый стиль с цветной обводкой",
        font="Rubik-Black.ttf",
        font_size=76,
        primary_color="#FFFFFF",
        highlight_color="#22D3EE",
        outline_color="#A855F7",
        outline_width=6,
        box_style="outline",
        shadow=4,
        shadow_color="#A855F7",
        uppercase=True,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=4,
        position_v="center",
    ),
    "bebas_bottom": SubtitleTemplate(
        id="bebas_bottom",
        name="Bebas Bottom",
        description="Узкий жирный шрифт Bebas, тонкая обводка, низ",
        font="BebasNeue-Regular.ttf",
        font_size=92,
        primary_color="#FFFFFF",
        highlight_color="#EF4444",
        outline_color="#000000",
        outline_width=4,
        box_style="outline",
        shadow=2,
        letter_spacing=2.0,
        uppercase=True,
        word_highlight=True,
        karaoke_mode="color",
        max_words_per_line=4,
        position_v="bottom",
        margin_v_pct=0.14,
    ),
}


@dataclass
class TitleTemplate:
    """Стиль/вариация заголовка поверх видео."""

    id: str
    name: str
    description: str

    variant: str = "top_banner"
    font: str = "Montserrat-Black.ttf"
    font_size: int = 64
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 3
    back_color: str = "#000000"
    back_alpha: int = 220
    back_padding: int = 16
    uppercase: bool = True
    margin_v_pct: float = 0.04
    duration: float = 0.0


TITLE_TEMPLATES: dict[str, TitleTemplate] = {
    "none": TitleTemplate(
        id="none",
        name="Без заголовка",
        description="Не показывать заголовок",
        variant="none",
    ),
    "top_banner_dark": TitleTemplate(
        id="top_banner_dark",
        name="Top Banner (тёмный)",
        description="Заголовок сверху на тёмной плашке",
        variant="top_banner",
        font="Montserrat-Black.ttf",
        font_size=58,
        primary_color="#FFFFFF",
        outline_color="#000000",
        outline_width=2,
        back_color="#000000",
        back_alpha=220,
        back_padding=20,
        uppercase=True,
        margin_v_pct=0.03,
    ),
    "top_banner_red": TitleTemplate(
        id="top_banner_red",
        name="Top Banner (красный)",
        description="Привлекающий внимание заголовок на красной плашке",
        variant="top_banner",
        font="Montserrat-Black.ttf",
        font_size=56,
        primary_color="#FFFFFF",
        outline_color="#000000",
        outline_width=2,
        back_color="#DC2626",
        back_alpha=240,
        back_padding=20,
        uppercase=True,
        margin_v_pct=0.03,
    ),
    "top_centered_outline": TitleTemplate(
        id="top_centered_outline",
        name="Top Centered (только текст)",
        description="Большой заголовок сверху, только текст с обводкой",
        variant="top_centered",
        font="Anton-Regular.ttf",
        font_size=78,
        primary_color="#FFFFFF",
        outline_color="#000000",
        outline_width=6,
        back_alpha=0,
        uppercase=True,
        margin_v_pct=0.05,
    ),
    "lower_third": TitleTemplate(
        id="lower_third",
        name="Lower Third",
        description="Заголовок в нижней трети, как в новостях",
        variant="lower_third",
        font="Montserrat-Black.ttf",
        font_size=48,
        primary_color="#FFFFFF",
        outline_color="#000000",
        outline_width=2,
        back_color="#1E40AF",
        back_alpha=230,
        back_padding=18,
        uppercase=False,
        margin_v_pct=0.20,
    ),
    "intro_3s": TitleTemplate(
        id="intro_3s",
        name="Intro (3 сек)",
        description="Заголовок показывается только первые 3 секунды клипа",
        variant="top_centered",
        font="Montserrat-Black.ttf",
        font_size=84,
        primary_color="#FFFFFF",
        outline_color="#000000",
        outline_width=6,
        back_alpha=0,
        uppercase=True,
        margin_v_pct=0.10,
        duration=3.0,
    ),
}


def get_subtitle_template(template_id: str) -> SubtitleTemplate:
    return TEMPLATES.get(template_id, TEMPLATES["tiktok_classic"])


def get_title_template(template_id: str) -> TitleTemplate:
    return TITLE_TEMPLATES.get(template_id, TITLE_TEMPLATES["none"])


def list_subtitle_templates() -> list[SubtitleTemplate]:
    return list(TEMPLATES.values())


def list_title_templates() -> list[TitleTemplate]:
    return list(TITLE_TEMPLATES.values())


def save_template_to_file(tmpl: SubtitleTemplate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(tmpl), f, indent=2, ensure_ascii=False)


def load_template_from_file(path: Path) -> Optional[SubtitleTemplate]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return SubtitleTemplate(**data)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
