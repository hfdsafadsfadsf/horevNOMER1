"""
Рендеринг заголовка поверх видео.

Использует ASS subtitle filter ffmpeg — тот же механизм что и для субтитров,
но с другим стилем и единственным "Dialogue" событием на весь клип
(или на первые N секунд если duration > 0).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from .config import FONTS_DIR, TitleConfig
from .subtitle_renderer import (
    AVG_CHAR_WIDTH_RATIO,
    SAFE_MARGIN_PX,
    _color_to_ass,
    _escape_filter_path,
    _format_time,
)


def _wrap_title_text(text: str, font_size: int, video_w: int = 1080) -> str:
    """
    Переносит длинный заголовок по словам на несколько строк
    чтобы ни одна не превысила (video_w - 2*SAFE_MARGIN_PX).
    Разделитель в ASS — \\N.
    """
    usable_w = video_w - 2 * SAFE_MARGIN_PX
    max_chars = max(6, int(usable_w / (font_size * AVG_CHAR_WIDTH_RATIO)))

    words = text.split()
    if not words:
        return text

    lines: list[list[str]] = [[]]
    for w in words:
        cur = lines[-1]
        cur_len = sum(len(x) for x in cur) + max(0, len(cur) - 1)
        new_len = cur_len + (1 if cur else 0) + len(w)
        if cur and new_len > max_chars:
            lines.append([w])
        else:
            cur.append(w)
    return "\\N".join(" ".join(line) for line in lines)


def _build_title_ass(
    text: str,
    duration_total: float,
    cfg: TitleConfig,
    video_w: int = 1080,
    video_h: int = 1920,
) -> str:
    font_name = Path(cfg.font).stem
    primary = _color_to_ass(cfg.primary_color)
    outline = _color_to_ass(cfg.outline_color)
    back = _color_to_ass(cfg.back_color, alpha=max(0, 255 - cfg.back_alpha))

    if cfg.variant == "top_banner":
        alignment = 8
        margin_v = int(video_h * cfg.margin_v_pct)
        border_style = 3
        outline_w = max(2, cfg.back_padding // 4)
        shadow_w = 0
    elif cfg.variant == "top_centered":
        alignment = 8
        margin_v = int(video_h * cfg.margin_v_pct)
        border_style = 1
        outline_w = cfg.outline_width
        shadow_w = 2
    elif cfg.variant == "lower_third":
        alignment = 2
        margin_v = int(video_h * cfg.margin_v_pct)
        border_style = 3
        outline_w = max(2, cfg.back_padding // 4)
        shadow_w = 0
    else:
        alignment = 8
        margin_v = int(video_h * cfg.margin_v_pct)
        border_style = 1
        outline_w = cfg.outline_width
        shadow_w = 2

    display_text = text.upper() if cfg.uppercase else text
    display_text = display_text.replace("\n", "\\N")
    # Safe-zone: переносим длинный заголовок на несколько строк
    display_text = _wrap_title_text(display_text, cfg.font_size, video_w)

    show_dur = duration_total if cfg.duration <= 0 else min(cfg.duration, duration_total)
    # Fade-in 200мс + fade-out 300мс чтобы заголовок появлялся/исчезал плавно
    fade_tag = "{\\fad(200,300)}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_w}\n"
        f"PlayResY: {video_h}\n"
        "ScaledBorderAndShadow: yes\n"
        # WrapStyle=0 — smart auto-wrap в libass на случай если наш вручной
        # перенос не сработал.
        "WrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Title,{font_name},{cfg.font_size},{primary},{primary},{outline},"
        f"{back},-1,0,0,0,100,100,0,0,"
        f"{border_style},{outline_w},{shadow_w},{alignment},{SAFE_MARGIN_PX},{SAFE_MARGIN_PX},{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    event = (
        f"Dialogue: 0,{_format_time(0.0)},{_format_time(show_dur)},"
        f"Title,,0,0,0,,{fade_tag}{display_text}"
    )
    return header + event + "\n"


def burn_title(
    video_path: Path,
    output_path: Path,
    text: str,
    duration_total: float,
    cfg: TitleConfig,
    progress_cb: Optional[Callable[[str], None]] = None,
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "192k",
    video_w: int = 1080,
    video_h: int = 1920,
    use_gpu: bool = True,
) -> Path:
    """Вжигает заголовок поверх готового клипа."""
    if not cfg.enabled or not text.strip():
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path

    ass_content = _build_title_ass(text, duration_total, cfg, video_w, video_h)
    tmp_ass = output_path.with_suffix(".title.ass")
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
        progress_cb(f"Вжигание заголовка: \"{text[:40]}\"")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg title burn failed:\n{result.stderr[-2000:]}")

    try:
        tmp_ass.unlink()
    except OSError:
        pass
    return output_path
