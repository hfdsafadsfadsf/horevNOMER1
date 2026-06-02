"""
Генерация и встраивание премиум-субтитров.

Использует формат ASS (Advanced SubStation Alpha) — поддерживается libass в ffmpeg.

Ключевые особенности:
- Слова группируются по N в строку, текущее подсвечивается karaoke {\\k}.
- При паузах > silence_gap_sec строка СБРАСЫВАЕТСЯ — никаких "висящих"
  субтитров в тишине.
- Внутри одного блока silence-таймеры тоже учитываются, чтобы подсветка
  слов шла строго по голосу, а не убегала вперёд.
- Поддерживаются три "box_style":
    * outline — обводка (по умолчанию)
    * shadow  — только тень
    * background — текст на полупрозрачной плашке (libass BorderStyle=3)
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from .config import FONTS_DIR, SubtitleStyle
from .transcriber import Transcript, Word

# Безопасные отступы (safe-zone) для 1080x1920 9:16 видео.
# TikTok/Reels UI занимает примерно по 80-100px по бокам — текст никогда
# не должен туда заходить, иначе обрезается.
SAFE_MARGIN_PX = 80
# Эмпирическая оценка средней ширины символа жирного капс-шрифта
# относительно font_size (для Montserrat-Black/Anton/Rubik-Black).
AVG_CHAR_WIDTH_RATIO = 0.55


def _color_to_ass(hex_color: str, alpha: int = 0) -> str:
    """#RRGGBB → &HAABBGGRR (ASS использует BGR + альфа). alpha: 0 = opaque, 255 = transparent."""
    h = (hex_color or "#FFFFFF").lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    a = max(0, min(255, alpha))
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def _format_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _estimate_width_px(text: str, font_size: int) -> float:
    """Грубая оценка ширины строки в пикселях (для safe-zone проверки)."""
    return len(text) * font_size * AVG_CHAR_WIDTH_RATIO


def _max_chars_per_line(font_size: int, video_w: int = 1080,
                       safe_margin: int = SAFE_MARGIN_PX) -> int:
    """Сколько символов влезает в одну строку чтобы не выйти за safe-zone."""
    usable_w = video_w - 2 * safe_margin
    return max(6, int(usable_w / (font_size * AVG_CHAR_WIDTH_RATIO)))


def _chunk_words(
    words: list[Word],
    max_per_chunk: int,
    silence_gap_sec: float,
    font_size: int = 78,
    uppercase: bool = True,
    video_w: int = 1080,
) -> list[list[Word]]:
    """
    Разбивает слова на группы по max_per_chunk + по фактической ширине строки
    (safe-zone). Дополнительно — закрывает группу при паузе > silence_gap_sec.
    """
    max_chars = _max_chars_per_line(font_size, video_w)
    chunks: list[list[Word]] = []
    cur: list[Word] = []

    def _line_chars(words_list: list[Word]) -> int:
        """Суммарная длина строки в символах (с пробелами)."""
        total = 0
        for i, w in enumerate(words_list):
            t = w.text.upper() if uppercase else w.text
            total += len(t)
            if i > 0:
                total += 1
        return total

    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            if gap > silence_gap_sec:
                chunks.append(cur)
                cur = []
        # Проверка safe-zone: если добавление слова превысит max_chars — закрываем
        if cur:
            cur_chars = _line_chars(cur + [w])
            if cur_chars > max_chars:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if len(cur) >= max_per_chunk:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


_STRIP_PUNCT = str.maketrans("", "", ".,!?;:…«»\"'")


def build_ass(
    transcript: Transcript,
    clip_start: float,
    clip_end: float,
    style: SubtitleStyle,
    video_w: int = 1080,
    video_h: int = 1920,
    strip_punct: bool = True,
) -> str:
    words_in_clip = transcript.words_in_range(clip_start, clip_end)
    if not words_in_clip:
        return _build_header(style, video_w, video_h)

    rel_words: list[Word] = []
    for w in words_in_clip:
        text = w.text.strip()
        if strip_punct:
            text = text.translate(_STRIP_PUNCT).strip()
        if not text:
            continue
        if style.uppercase:
            text = text.upper()
        rel_words.append(
            Word(
                start=max(0.0, w.start - clip_start),
                end=max(0.0, w.end - clip_start),
                text=text,
                probability=w.probability,
            )
        )

    chunks = _chunk_words(
        rel_words,
        max_per_chunk=style.max_words_per_line,
        silence_gap_sec=style.silence_gap_sec,
        font_size=style.font_size,
        uppercase=style.uppercase,
        video_w=video_w,
    )

    header = _build_header(style, video_w, video_h)
    highlight = _color_to_ass(style.highlight_color)
    primary = _color_to_ass(style.primary_color)

    events: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        c_start = chunk[0].start
        c_end = chunk[-1].end
        if c_end <= c_start:
            continue

        parts: list[str] = []
        for i, w in enumerate(chunk):
            if i > 0:
                gap_sec = w.start - chunk[i - 1].end
                if gap_sec > 0.02:
                    gap_cs = max(1, int(round(gap_sec * 100)))
                    parts.append(f"{{\\k{gap_cs}}}")

            word_dur_cs = max(1, int(round((w.end - w.start) * 100)))
            if style.word_highlight:
                parts.append(
                    f"{{\\1c{highlight}\\k{word_dur_cs}\\1c{primary}}}{w.text} "
                )
            else:
                parts.append(f"{{\\k{word_dur_cs}}}{w.text} ")

        line_text = "".join(parts).rstrip()
        events.append(
            f"Dialogue: 0,{_format_time(c_start)},{_format_time(c_end)},"
            f"Default,,0,0,0,,{line_text}"
        )

    return header + "\n".join(events) + "\n"


def _build_header(style: SubtitleStyle, video_w: int, video_h: int) -> str:
    font_name = Path(style.font).stem
    primary = _color_to_ass(style.primary_color)
    outline = _color_to_ass(style.outline_color)
    secondary = _color_to_ass(style.highlight_color)
    back = _color_to_ass(style.back_color, alpha=max(0, 255 - style.back_alpha))

    pos_map = {"bottom": 2, "center": 5, "top": 8}
    alignment = pos_map.get(style.position_v, 2)

    margin_v = 0
    if style.position_v == "bottom":
        margin_v = int(video_h * style.margin_v_pct)
    elif style.position_v == "top":
        margin_v = int(video_h * style.margin_v_pct)

    margin_h = SAFE_MARGIN_PX

    if style.box_style == "background":
        border_style = 3
        outline_w = max(2, style.back_padding // 4)
        shadow_w = 0
    elif style.box_style == "shadow":
        border_style = 1
        outline_w = 0
        shadow_w = max(1, style.shadow)
    else:
        border_style = 1
        outline_w = style.outline_width
        shadow_w = style.shadow

    bold_flag = -1 if style.bold else 0
    italic_flag = -1 if style.italic else 0
    spacing = max(0, int(style.letter_spacing))

    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_w}\n"
        f"PlayResY: {video_h}\n"
        "ScaledBorderAndShadow: yes\n"
        # WrapStyle=0 → smart auto-wrap по ширине (libass переносит слова если
        # они не влезают в (PlayResX - 2*MarginH)). Это критично для safe-zone.
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_name},{style.font_size},{primary},{secondary},{outline},"
        f"{back},{bold_flag},{italic_flag},0,0,100,100,{spacing},0,"
        f"{border_style},{outline_w},{shadow_w},{alignment},{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _escape_filter_path(p: Path) -> str:
    """Подготовка пути для -vf фильтра ffmpeg (Windows-safe)."""
    return str(p).replace("\\", "/").replace(":", r"\:")


def burn_subtitles(
    video_path: Path,
    transcript: Transcript,
    clip_start: float,
    clip_end: float,
    style: SubtitleStyle,
    output_path: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
    crf: int = 18,
    preset: str = "slow",
    audio_bitrate: str = "192k",
    use_gpu: bool = True,
    strip_punct: bool = True,
) -> Path:
    """Берёт переформатированный 9:16 клип + транскрипт, вжигает субтитры."""
    ass_content = build_ass(
        transcript, clip_start, clip_end, style,
        strip_punct=strip_punct,
    )
    tmp_ass = output_path.with_suffix(".ass")
    tmp_ass.write_text(ass_content, encoding="utf-8")

    fonts_dir_arg = _escape_filter_path(FONTS_DIR)
    ass_arg = _escape_filter_path(tmp_ass)
    vf = f"ass='{ass_arg}':fontsdir='{fonts_dir_arg}'"

    from .gpu import gpu_encode_args
    enc_args = gpu_encode_args(use_gpu=use_gpu, crf_or_cq=crf, preset=preset)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        *enc_args,
        "-r", "60",
        "-vsync", "cfr",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ]

    if progress_cb:
        progress_cb(f"Вжигание субтитров ({style.font}, размер {style.font_size})...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn failed:\n{result.stderr[-2000:]}")

    try:
        tmp_ass.unlink()
    except OSError:
        pass
    return output_path
