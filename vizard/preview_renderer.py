"""
Pillow-рендер кадра 9:16 для template editor.

Используется в модалке "Создать шаблон" — каждый раз когда юзер меняет
стиль, цвет или шрифт, этот модуль строит новое PIL.Image и показывает
в GUI. БЕЗ ffmpeg, БЕЗ перегона видео — мгновенно.

Текст рендерится с поддержкой:
- outline (Pillow stroke_width)
- shadow (отрисовка дубликата со смещением)
- background box (полупрозрачный прямоугольник за текстом)
- word_highlight (центральное слово другим цветом)
- uppercase
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import FONTS_DIR, OverlayConfig, SubtitleStyle, TitleConfig

DEMO_BG_PATH = Path(__file__).resolve().parent.parent / "resources" / "demo" / "demo_background.jpg"

DEMO_SUBTITLE_TEXT = "это твой стиль"
DEMO_HEADLINE_TEXT = "ВОТ ТАК БУДЕТ ВЫГЛЯДЕТЬ"


def _hex_rgb(c: str, default: str = "#FFFFFF") -> tuple[int, int, int]:
    c = (c or default).lstrip("#")
    if len(c) != 6:
        c = "FFFFFF"
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


CYRILLIC_FALLBACKS = [
    "Montserrat-Black.ttf",
    "Montserrat-Bold.ttf",
    "Rubik-Black.ttf",
    "Inter-Black.otf",
]

# Шрифты без кириллицы — для русского текста подменяем на Montserrat / Rubik.
LATIN_ONLY_FONTS = {
    "Anton-Regular.ttf",
    "Oswald-Bold.ttf",
    "BebasNeue-Regular.ttf",
    "Poppins-Black.ttf",
}


def _has_cyrillic(s: str) -> bool:
    return any("\u0400" <= ch <= "\u04FF" for ch in s)


def _load_font(filename: str, size: int, text: str = "") -> ImageFont.ImageFont:
    """Загружает шрифт. Если шрифт latin-only, а текст содержит кириллицу — fallback."""
    if filename in LATIN_ONLY_FONTS and _has_cyrillic(text):
        for fb in CYRILLIC_FALLBACKS:
            fp = FONTS_DIR / fb
            if fp.exists():
                try:
                    return ImageFont.truetype(str(fp), size)
                except OSError:
                    continue

    p = FONTS_DIR / filename
    if p.exists():
        try:
            return ImageFont.truetype(str(p), size)
        except OSError:
            pass

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_demo_background(w: int, h: int) -> Image.Image:
    if DEMO_BG_PATH.exists():
        bg = Image.open(DEMO_BG_PATH).convert("RGB")
        bg = bg.resize((w, h), Image.LANCZOS)
        return bg
    bg = Image.new("RGB", (w, h), (60, 80, 120))
    return bg


def _draw_text_with_style(
    base: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    primary_rgb: tuple[int, int, int],
    outline_rgb: tuple[int, int, int],
    outline_w: int,
    shadow_px: int,
    shadow_rgb: tuple[int, int, int] = (0, 0, 0),
    back_rgb: Optional[tuple[int, int, int]] = None,
    back_alpha: int = 0,
    back_padding: int = 0,
    anchor: str = "mm",
    highlight_word_idx: Optional[int] = None,
    highlight_rgb: Optional[tuple[int, int, int]] = None,
) -> None:
    """Рисует текст с обводкой/тенью/фоном на base изображении."""
    draw = ImageDraw.Draw(base, "RGBA")

    try:
        bbox = draw.textbbox(xy, text, font=font, anchor=anchor)
    except (TypeError, AttributeError):
        bbox = (xy[0] - 100, xy[1] - 20, xy[0] + 100, xy[1] + 20)

    if back_rgb is not None and back_alpha > 0:
        pad = back_padding
        rect = (
            bbox[0] - pad, bbox[1] - pad,
            bbox[2] + pad, bbox[3] + pad,
        )
        r = max(8, pad // 2)
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rounded_rectangle(
            rect, radius=r,
            fill=(back_rgb[0], back_rgb[1], back_rgb[2], back_alpha),
        )
        base.alpha_composite(overlay) if base.mode == "RGBA" else base.paste(overlay, (0, 0), overlay)

    if shadow_px > 0:
        draw.text(
            (xy[0] + shadow_px, xy[1] + shadow_px),
            text, font=font, fill=shadow_rgb + (200,), anchor=anchor,
        )

    if highlight_word_idx is not None and highlight_rgb is not None:
        _draw_text_with_word_highlight(
            draw, xy, text, font, primary_rgb, outline_rgb, outline_w,
            highlight_word_idx, highlight_rgb, anchor=anchor,
        )
    else:
        draw.text(
            xy, text, font=font, fill=primary_rgb, anchor=anchor,
            stroke_width=outline_w, stroke_fill=outline_rgb,
        )


def _draw_text_with_word_highlight(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    primary_rgb: tuple[int, int, int],
    outline_rgb: tuple[int, int, int],
    outline_w: int,
    highlight_idx: int,
    highlight_rgb: tuple[int, int, int],
    anchor: str = "mm",
) -> None:
    """Рисует строку, окрашивая указанное слово цветом highlight."""
    words = text.split()
    if not words:
        return
    space_w = draw.textlength(" ", font=font)
    word_widths = [draw.textlength(w, font=font) for w in words]
    total = sum(word_widths) + space_w * (len(words) - 1)
    cx, cy = xy

    if "m" in anchor:
        start_x = cx - total / 2
    elif "r" in anchor:
        start_x = cx - total
    else:
        start_x = cx

    if "m" in anchor:
        try:
            asc, desc = font.getmetrics()
            text_h = asc + desc
        except AttributeError:
            text_h = font.size
        y_top = cy - text_h / 2
    else:
        y_top = cy

    x = start_x
    for i, w in enumerate(words):
        color = highlight_rgb if i == highlight_idx else primary_rgb
        draw.text(
            (x, y_top), w, font=font, fill=color, anchor="lt",
            stroke_width=outline_w, stroke_fill=outline_rgb,
        )
        x += word_widths[i] + space_w


def render_preview_frame(
    subtitle: SubtitleStyle,
    title: TitleConfig,
    overlay: OverlayConfig,
    width: int = 405,
    height: int = 720,
    subtitle_text: Optional[str] = None,
    headline_text: Optional[str] = None,
) -> Image.Image:
    """
    Возвращает PIL.Image размером width×height с применёнными настройками.
    """
    img = _load_demo_background(width, height).convert("RGBA")

    sub_text = subtitle_text if subtitle_text is not None else DEMO_SUBTITLE_TEXT
    head_text = headline_text if headline_text is not None else DEMO_HEADLINE_TEXT

    if title.enabled and head_text.strip():
        _render_headline(img, title, head_text, width, height)

    if sub_text.strip():
        _render_subtitle(img, subtitle, sub_text, width, height)

    if overlay.enabled and overlay.image_path:
        _render_logo(img, overlay, width, height)

    return img.convert("RGB")


def _render_subtitle(
    img: Image.Image, style: SubtitleStyle, text: str, w: int, h: int
) -> None:
    if style.uppercase:
        text = text.upper()
    scale = w / 1080.0
    font_size = max(14, int(style.font_size * scale))
    font = _load_font(style.font, font_size, text)

    primary = _hex_rgb(style.primary_color, "#FFFFFF")
    outline = _hex_rgb(style.outline_color, "#000000")
    highlight = _hex_rgb(style.highlight_color, "#FFD800")
    back_rgb = _hex_rgb(style.back_color, "#000000")

    if style.position_v == "top":
        cy = int(h * style.margin_v_pct + font_size)
    elif style.position_v == "center":
        cy = h // 2
    else:
        cy = h - int(h * style.margin_v_pct) - int(font_size * 0.6)

    if style.box_style == "background":
        back_alpha = style.back_alpha
        outline_w = 0
        shadow_px = 0
        back_pad = max(6, int(style.back_padding * scale))
    elif style.box_style == "shadow":
        back_alpha = 0
        outline_w = 0
        shadow_px = max(1, int(style.shadow * scale * 1.2))
        back_pad = 0
    else:
        back_alpha = 0
        outline_w = max(1, int(style.outline_width * scale * 1.2))
        shadow_px = max(0, int(style.shadow * scale))
        back_pad = 0

    words = text.split()
    hi_idx = (len(words) // 2) if (style.word_highlight and len(words) >= 2) else None

    _draw_text_with_style(
        img,
        (w // 2, cy),
        text,
        font,
        primary,
        outline,
        outline_w,
        shadow_px,
        shadow_rgb=_hex_rgb(style.shadow_color, "#000000"),
        back_rgb=back_rgb if back_alpha > 0 else None,
        back_alpha=back_alpha,
        back_padding=back_pad,
        anchor="mm",
        highlight_word_idx=hi_idx,
        highlight_rgb=highlight,
    )


def _render_headline(
    img: Image.Image, title: TitleConfig, text: str, w: int, h: int
) -> None:
    if title.uppercase:
        text = text.upper()
    scale = w / 1080.0
    font_size = max(14, int(title.font_size * scale))
    font = _load_font(title.font, font_size, text)

    primary = _hex_rgb(title.primary_color, "#FFFFFF")
    outline = _hex_rgb(title.outline_color, "#000000")
    back_rgb = _hex_rgb(title.back_color, "#000000")

    if title.variant == "lower_third":
        cy = int(h - h * title.margin_v_pct - font_size * 0.6)
    else:
        cy = int(h * title.margin_v_pct + font_size * 0.6)

    if title.variant in ("top_banner", "lower_third"):
        back_alpha = title.back_alpha
        outline_w = 0
        back_pad = max(6, int(title.back_padding * scale))
    else:
        back_alpha = 0
        outline_w = max(1, int(title.outline_width * scale * 1.2))
        back_pad = 0

    _draw_text_with_style(
        img,
        (w // 2, cy),
        text,
        font,
        primary,
        outline,
        outline_w,
        shadow_px=0,
        back_rgb=back_rgb if back_alpha > 0 else None,
        back_alpha=back_alpha,
        back_padding=back_pad,
        anchor="mm",
    )


def _render_logo(
    img: Image.Image, overlay: OverlayConfig, w: int, h: int
) -> None:
    path = Path(overlay.image_path)
    if not path.exists():
        return
    try:
        logo = Image.open(path).convert("RGBA")
    except OSError:
        return

    logo_w = max(20, int(w * overlay.size_pct / 100))
    ratio = logo.height / max(1, logo.width)
    logo_h = max(20, int(logo_w * ratio))
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    if overlay.opacity < 1.0:
        alpha = logo.split()[3]
        alpha = alpha.point(lambda p: int(p * overlay.opacity))
        logo.putalpha(alpha)

    margin = max(0, int(w * overlay.margin_pct / 100))
    pos = overlay.position

    if pos == "top_left":
        x, y = margin, margin
    elif pos == "top_center":
        x, y = (w - logo_w) // 2, margin
    elif pos == "top_right":
        x, y = w - logo_w - margin, margin
    elif pos == "bottom_left":
        x, y = margin, h - logo_h - margin
    elif pos == "bottom_center":
        x, y = (w - logo_w) // 2, h - logo_h - margin
    else:
        x, y = w - logo_w - margin, h - logo_h - margin

    img.alpha_composite(logo, (x, y))


def render_template_thumbnail(
    template,
    width: int = 220,
    height: int = 140,
) -> Image.Image:
    """Маленькая превью-карточка для одного subtitle template (Vizard-style grid)."""
    style = SubtitleStyle(
        font=template.font,
        font_size=template.font_size,
        primary_color=template.primary_color,
        highlight_color=template.highlight_color,
        outline_color=template.outline_color,
        outline_width=template.outline_width,
        box_style=template.box_style,
        shadow=template.shadow,
        shadow_color=template.shadow_color,
        back_color=template.back_color,
        back_alpha=template.back_alpha,
        back_padding=template.back_padding,
        uppercase=template.uppercase,
        word_highlight=template.word_highlight,
        max_words_per_line=template.max_words_per_line,
        position_v="center",
        margin_v_pct=0.10,
    )
    img = Image.new("RGB", (width, height), (50, 50, 60))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / height
        c = (
            int(50 + 30 * t),
            int(50 + 25 * t),
            int(70 + 35 * t),
        )
        draw.line([(0, y), (width, y)], fill=c)
    img = img.convert("RGBA")
    _render_subtitle(img, style, "five box wizard", width, height)
    return img.convert("RGB")


def render_title_thumbnail(
    title_template,
    width: int = 220,
    height: int = 140,
) -> Image.Image:
    cfg = TitleConfig(
        enabled=True,
        font=title_template.font,
        font_size=title_template.font_size,
        primary_color=title_template.primary_color,
        outline_color=title_template.outline_color,
        outline_width=title_template.outline_width,
        back_color=title_template.back_color,
        back_alpha=title_template.back_alpha,
        back_padding=title_template.back_padding,
        uppercase=title_template.uppercase,
        variant=title_template.variant,
        margin_v_pct=0.15,
        duration=title_template.duration,
    )
    img = Image.new("RGB", (width, height), (50, 50, 60))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / height
        c = (int(60 + 20 * t), int(60 + 20 * t), int(80 + 30 * t))
        draw.line([(0, y), (width, y)], fill=c)
    img = img.convert("RGBA")
    if title_template.variant != "none":
        _render_headline(img, cfg, "HEADLINE", width, height)
    return img.convert("RGB")
