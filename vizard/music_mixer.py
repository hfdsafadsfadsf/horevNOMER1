"""
Подмешивание пользовательской музыки в готовый клип.

Особенности:
- Музыка ТОЛЬКО та, что загрузил пользователь (никаких встроенных треков).
- Sidechain compression: когда говорят, музыка автоматически приглушается.
- Если музыка короче клипа — зацикливается, длиннее — обрезается.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional


def mix_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    music_volume: float = 0.15,
    sidechain: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
    crf: int = 18,
    preset: str = "slow",
    audio_bitrate: str = "192k",
) -> Path:
    if not music_path.exists():
        raise FileNotFoundError(f"Музыкальный файл не найден: {music_path}")

    music_volume = max(0.0, min(1.0, music_volume))

    if sidechain:
        filter_complex = (
            f"[1:a]aloop=loop=-1:size=2e9,volume={music_volume}[music];"
            f"[0:a]asplit=2[orig][trigger];"
            f"[music][trigger]sidechaincompress="
            f"threshold=0.05:ratio=8:attack=5:release=200:makeup=1[ducked];"
            f"[orig][ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
    else:
        filter_complex = (
            f"[1:a]aloop=loop=-1:size=2e9,volume={music_volume}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(music_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    if progress_cb:
        progress_cb(
            f"Подмешивание музыки: {music_path.name} "
            f"(volume={music_volume}, sidechain={'on' if sidechain else 'off'})"
        )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg music mix failed:\n{result.stderr[-2000:]}")
    return output_path
