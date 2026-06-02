"""
Генерация короткого превью (15-30 сек) с текущими настройками.

Используется в GUI, чтобы пользователь мог увидеть как будут выглядеть
субтитры/заголовок/логотип ДО полной обработки часового видео.

Алгоритм:
1. Берёт небольшой отрезок исходного видео (по умолчанию середина).
2. Прогоняет через тот же pipeline: reframe → subs → title → overlay.
3. Транскрипт берётся быстрой моделью whisper-tiny (или передаётся готовый).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, TEMP_DIR
from .overlay import burn_overlay
from .reframer import reframe_to_9x16
from .subtitle_renderer import burn_subtitles
from .title_renderer import burn_title
from .transcriber import Transcript, transcribe_video


def generate_preview(
    source_path: Path,
    cfg: AppConfig,
    duration: float = 15.0,
    start_offset: Optional[float] = None,
    transcript: Optional[Transcript] = None,
    output_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Создаёт короткий превью-клип. Возвращает путь к итоговому MP4.

    Если transcript не передан — транскрибируется только нужный отрезок
    быстрой моделью (whisper-tiny на ~15 сек ≈ 5-10 сек на CPU).
    """
    if progress_cb is None:
        progress_cb = lambda _msg: None

    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    from .reframer import _probe_video
    info = _probe_video(source_path)

    if start_offset is None:
        start_offset = max(0.0, min(info.duration - duration, info.duration * 0.3))
    end = min(info.duration, start_offset + duration)
    if end - start_offset < 3:
        end = min(info.duration, start_offset + 3)

    if output_path is None:
        output_path = TEMP_DIR / "preview.mp4"

    work_dir = TEMP_DIR / "_preview_work"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if transcript is None:
        progress_cb(f"Транскрибирую отрезок {start_offset:.1f}–{end:.1f}с (это займёт ~10 сек)...")
        snippet_path = work_dir / "snippet.mp4"
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_offset),
            "-i", str(source_path),
            "-t", str(end - start_offset),
            "-c", "copy",
            str(snippet_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        if not snippet_path.exists() or snippet_path.stat().st_size < 1000:
            cmd2 = [
                "ffmpeg", "-y",
                "-ss", str(start_offset),
                "-i", str(source_path),
                "-t", str(end - start_offset),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                str(snippet_path),
            ]
            subprocess.run(cmd2, capture_output=True, text=True, check=True)

        transcript = transcribe_video(
            snippet_path,
            model_size="tiny",
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
            language=None if cfg.clip.language == "auto" else cfg.clip.language,
            progress_cb=progress_cb,
        )

        for w in transcript.words:
            w.start += start_offset
            w.end += start_offset
        for s in transcript.segments:
            s.start += start_offset
            s.end += start_offset
        transcript.duration = end

    reframed = work_dir / "reframed.mp4"
    progress_cb("Reframe 9:16 + face tracking...")
    reframe_to_9x16(
        source_path, reframed, start=start_offset, end=end,
        preset="fast", crf=20,
        enable_face_tracking=cfg.face_tracking,
        enable_face_zoom=cfg.face_zoom,
        progress_cb=progress_cb,
    )

    with_subs = work_dir / "with_subs.mp4"
    progress_cb("Вжигание субтитров...")
    burn_subtitles(
        reframed,
        transcript,
        clip_start=start_offset,
        clip_end=end,
        style=cfg.subtitle,
        output_path=with_subs,
        preset="fast", crf=20,
    )

    cur = with_subs
    if cfg.title.enabled:
        title_text = cfg.title.custom_text.strip() or "ПРЕВЬЮ ЗАГОЛОВКА"
        with_title = work_dir / "with_title.mp4"
        progress_cb("Вжигание заголовка...")
        burn_title(
            cur, with_title, title_text,
            duration_total=(end - start_offset),
            cfg=cfg.title, preset="fast", crf=20,
        )
        cur = with_title

    if cfg.overlay.enabled and cfg.overlay.image_path:
        with_logo = work_dir / "with_logo.mp4"
        progress_cb("Наложение логотипа...")
        burn_overlay(cur, with_logo, cfg.overlay, preset="fast", crf=20)
        cur = with_logo

    shutil.copy2(cur, output_path)
    shutil.rmtree(work_dir, ignore_errors=True)

    progress_cb(f"Превью готово: {output_path}")
    return output_path
