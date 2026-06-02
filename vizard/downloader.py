"""Скачивание видео по ссылке (YouTube, TikTok, Vimeo и т.д.) или валидация локального файла."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable, Optional

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


URL_REGEX = re.compile(r"^https?://", re.IGNORECASE)


def is_url(source: str) -> bool:
    return bool(URL_REGEX.match(source.strip()))


def download_video(
    source: str,
    output_dir: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Возвращает путь к скачанному (или существующему локальному) видео.

    - source: URL или путь к локальному файлу
    - output_dir: куда сохранить (используется только для URL)
    - progress_cb: функция для логирования прогресса (опционально)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Локальный файл не найден: {path}")
        if progress_cb:
            progress_cb(f"Используем локальный файл: {path.name}")
        return path

    if yt_dlp is None:
        raise RuntimeError("yt-dlp не установлен. Запусти: pip install yt-dlp")

    def hook(d: dict) -> None:
        if progress_cb is None:
            return
        if d.get("status") == "downloading":
            pct = d.get("_percent_str", "").strip()
            speed = d.get("_speed_str", "").strip()
            progress_cb(f"Скачивание... {pct} @ {speed}")
        elif d.get("status") == "finished":
            progress_cb("Скачивание завершено, обработка...")

    target_template = str(output_dir / "source_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": target_template,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "restrictfilenames": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        filename = ydl.prepare_filename(info)
        final_path = Path(filename)
        if final_path.suffix.lower() != ".mp4":
            mp4_candidate = final_path.with_suffix(".mp4")
            if mp4_candidate.exists():
                final_path = mp4_candidate

    if not final_path.exists():
        raise RuntimeError(f"yt-dlp скачал, но файл не найден: {final_path}")

    if progress_cb:
        progress_cb(f"Видео сохранено: {final_path.name}")
    return final_path


def ensure_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
