"""
Наложение логотипа/обложки поверх готового клипа.

Использует ffmpeg overlay filter. Логотип масштабируется относительно
ширины видео (size_pct), позиционируется в одном из 6 углов с
настраиваемым отступом и прозрачностью.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .config import OverlayConfig


VALID_POSITIONS = (
    "top_left",
    "top_center",
    "top_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)


def _position_xy(position: str, margin_px: int) -> tuple[str, str]:
    """Возвращает (x, y) ffmpeg-выражения для overlay фильтра."""
    m = str(margin_px)
    if position == "top_left":
        return m, m
    if position == "top_center":
        return "(W-w)/2", m
    if position == "top_right":
        return f"W-w-{m}", m
    if position == "bottom_left":
        return m, f"H-h-{m}"
    if position == "bottom_center":
        return "(W-w)/2", f"H-h-{m}"
    return f"W-w-{m}", f"H-h-{m}"


def burn_overlay(
    video_path: Path,
    output_path: Path,
    cfg: OverlayConfig,
    progress_cb: Optional[Callable[[str], None]] = None,
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "192k",
    video_w: int = 1080,
    video_h: int = 1920,
    use_gpu: bool = True,
) -> Path:
    """Накладывает логотип на видео. Если overlay не включён — копирует файл."""
    if not cfg.enabled or not cfg.image_path:
        shutil.copy2(video_path, output_path)
        return output_path

    img_path = Path(cfg.image_path)
    if not img_path.exists():
        if progress_cb:
            progress_cb(f"Логотип не найден: {img_path} — пропускаю")
        shutil.copy2(video_path, output_path)
        return output_path

    logo_w = max(40, int(video_w * cfg.size_pct / 100))
    margin = max(0, int(video_w * cfg.margin_pct / 100))

    position = cfg.position if cfg.position in VALID_POSITIONS else "top_right"
    x_expr, y_expr = _position_xy(position, margin)

    opacity = max(0.0, min(1.0, cfg.opacity))

    filter_complex = (
        f"[1:v]scale={logo_w}:-1,format=rgba,"
        f"colorchannelmixer=aa={opacity}[logo];"
        f"[0:v][logo]overlay={x_expr}:{y_expr}[v]"
    )

    from .gpu import gpu_encode_args
    enc_args = gpu_encode_args(use_gpu=use_gpu, crf_or_cq=crf, preset=preset)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(img_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        *enc_args,
        "-r", "60",
        "-vsync", "cfr",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ]

    if progress_cb:
        progress_cb(f"Наложение логотипа ({position}, {cfg.size_pct:.0f}% ширины)...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg overlay failed:\n{result.stderr[-2000:]}")
    return output_path
