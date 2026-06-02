"""Оркестратор всего пайплайна: ссылка → клипы 9:16 с субтитрами."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .ai_analyzer import ClipSuggestion, analyze_transcript, _fallback_heuristic, LENGTH_PRESETS
from .config import AppConfig, TEMP_DIR
from .downloader import download_video, ensure_ffmpeg_available
from .music_mixer import mix_music
from .overlay import burn_overlay
from .reframer import reframe_to_9x16
from .subtitle_renderer import burn_subtitles
from .title_renderer import burn_title
from .transcriber import Transcript, transcribe_video


@dataclass
class ClipResult:
    index: int
    suggestion: ClipSuggestion
    output_path: Path
    size_mb: float
    # Сохраняем для редактора: исходник + transcript нужны для перерендера
    source_video: Optional[Path] = None
    transcript: Optional["Transcript"] = None


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    name = re.sub(r"[^\w\s\-А-Яа-яЁё]", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] or "clip"


ProgressCB = Optional[Callable[[str, float], None]]


def _wrap_cb(cb: ProgressCB, fraction_start: float, fraction_end: float):
    def inner(msg: str) -> None:
        if cb is None:
            return
        mid = (fraction_start + fraction_end) / 2
        cb(msg, mid)
    return inner


def run_pipeline(
    source: str,
    cfg: AppConfig,
    progress_cb: ProgressCB = None,
) -> list[ClipResult]:
    if not ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg/ffprobe не найдены в PATH. Установи ffmpeg "
            "(на Windows: winget install ffmpeg) и перезапусти."
        )

    def step(msg: str, p: float) -> None:
        if progress_cb:
            progress_cb(msg, p)

    step("Загрузка видео...", 0.02)
    src_path = download_video(
        source,
        output_dir=TEMP_DIR,
        progress_cb=_wrap_cb(progress_cb, 0.02, 0.10),
    )

    step("Транскрипция...", 0.12)
    transcript = transcribe_video(
        src_path,
        model_size=cfg.whisper_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
        language=None if cfg.clip.language == "auto" else cfg.clip.language,
        progress_cb=_wrap_cb(progress_cb, 0.12, 0.40),
    )

    step("Подбор клипов AI...", 0.42)
    suggestions = _select_clips(transcript, cfg, progress_cb)

    if not suggestions:
        raise RuntimeError("Не удалось выбрать ни одного клипа из видео.")

    # Snap границ к концу предложения/паузе — клипы не обрываются на полуслове
    step("Подгонка границ клипов к концу предложений...", 0.44)
    suggestions = _snap_to_sentence_boundaries(suggestions, transcript, cfg, max_shift=1.5)

    out_dir = Path(cfg.output_dir) / src_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[ClipResult] = []
    n = len(suggestions)
    for i, sugg in enumerate(suggestions):
        clip_idx = i + 1
        base_progress = 0.45 + (i / n) * 0.50
        step(
            f"[{clip_idx}/{n}] '{sugg.title}' ({sugg.duration:.1f}с)...",
            base_progress,
        )

        clip_name = f"{clip_idx:02d}_{_sanitize_filename(sugg.title)}"
        reframed_path = out_dir / f"{clip_name}_reframed.mp4"
        subtitled_path = out_dir / f"{clip_name}_subs.mp4"
        final_path = out_dir / f"{clip_name}.mp4"

        reframe_to_9x16(
            src_path,
            reframed_path,
            start=sugg.start,
            end=sugg.end,
            progress_cb=_wrap_cb(progress_cb, base_progress, base_progress + 0.15 / n),
            enable_face_tracking=cfg.face_tracking,
            enable_face_zoom=cfg.face_zoom,
            use_gpu=cfg.use_gpu,
            multi_face_mode=cfg.multi_face_mode,
            speaker_detection=cfg.speaker_detection,
        )

        burn_subtitles(
            reframed_path,
            transcript,
            clip_start=sugg.start,
            clip_end=sugg.end,
            style=cfg.subtitle,
            output_path=subtitled_path,
            progress_cb=_wrap_cb(progress_cb, base_progress + 0.15 / n, base_progress + 0.25 / n),
            use_gpu=cfg.use_gpu,
            strip_punct=cfg.strip_subtitle_punct,
        )

        cur_path = subtitled_path
        intermediates: list[Path] = [reframed_path]

        if cfg.title.enabled:
            title_text = cfg.title.custom_text.strip() or sugg.title
            with_title = out_dir / f"{clip_name}_title.mp4"
            burn_title(
                cur_path,
                with_title,
                title_text,
                duration_total=sugg.duration,
                cfg=cfg.title,
                progress_cb=_wrap_cb(progress_cb, base_progress + 0.25 / n, base_progress + 0.32 / n),
                use_gpu=cfg.use_gpu,
            )
            intermediates.append(cur_path)
            cur_path = with_title

        if cfg.overlay.enabled and cfg.overlay.image_path:
            with_logo = out_dir / f"{clip_name}_logo.mp4"
            burn_overlay(
                cur_path,
                with_logo,
                cfg.overlay,
                progress_cb=_wrap_cb(progress_cb, base_progress + 0.32 / n, base_progress + 0.38 / n),
                use_gpu=cfg.use_gpu,
            )
            intermediates.append(cur_path)
            cur_path = with_logo

        music_path = _pick_music_for_clip(cfg, clip_idx - 1)
        if music_path:
            step(f"[{clip_idx}/{n}] Подмешиваю музыку: {music_path.name}", base_progress + 0.36 / n)
            mixed_path = out_dir / f"{clip_name}_music.mp4"
            try:
                mix_music(
                    cur_path,
                    music_path,
                    mixed_path,
                    music_volume=cfg.music.volume,
                    sidechain=cfg.music.sidechain,
                    progress_cb=_wrap_cb(progress_cb, base_progress + 0.38 / n, base_progress + 0.45 / n),
                )
                intermediates.append(cur_path)
                cur_path = mixed_path
            except Exception as e:
                # Музыка не должна ронять весь клип — логируем и продолжаем без неё
                step(f"[{clip_idx}/{n}] Ошибка музыки ({e}) — клип сохранён без музыки.",
                     base_progress + 0.40 / n)
        elif cfg.music.mode != "none":
            step(
                f"[{clip_idx}/{n}] Музыка пропущена: mode='{cfg.music.mode}' но трек не найден.",
                base_progress + 0.40 / n,
            )

        # LUFS-нормализация финального аудио (TikTok/Reels стандарт: -14 LUFS)
        normed_path = out_dir / f"{clip_name}_normed.mp4"
        try:
            _normalize_loudness(
                cur_path, normed_path,
                progress_cb=_wrap_cb(progress_cb, base_progress + 0.45 / n, base_progress + 0.48 / n),
            )
            intermediates.append(cur_path)
            cur_path = normed_path
        except Exception as e:
            step(f"[{clip_idx}/{n}] LUFS-нормализация пропущена: {e}", base_progress + 0.46 / n)

        if cur_path != final_path:
            if final_path.exists():
                final_path.unlink()
            cur_path.rename(final_path)

        for p in intermediates:
            try:
                if p != final_path:
                    p.unlink()
            except OSError:
                pass

        size_mb = final_path.stat().st_size / (1024 * 1024)
        results.append(
            ClipResult(
                index=clip_idx, suggestion=sugg,
                output_path=final_path, size_mb=size_mb,
                source_video=src_path, transcript=transcript,
            )
        )
        step(f"[{clip_idx}/{n}] Готов: {final_path.name} ({size_mb:.1f} МБ)", base_progress + 0.50 / n)

    step(f"Готово! {len(results)} клипов сохранено в {out_dir}", 1.0)
    return results


def _select_clips(
    transcript: Transcript, cfg: AppConfig, progress_cb: ProgressCB
) -> list[ClipSuggestion]:
    lo, hi = LENGTH_PRESETS.get(cfg.clip.length_preset, LENGTH_PRESETS["auto"])

    if not cfg.use_ai or not cfg.deepseek_api_key:
        if progress_cb:
            progress_cb("AI отключён — используем эвристику равномерной нарезки.", 0.42)
        return _fallback_heuristic(transcript, lo, hi, cfg.clip.max_clip_count)

    try:
        return analyze_transcript(
            transcript,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            model=cfg.deepseek_model,
            length_preset=cfg.clip.length_preset,
            min_clips=cfg.clip.min_clip_count,
            max_clips=cfg.clip.max_clip_count,
            progress_cb=_wrap_cb(progress_cb, 0.42, 0.45),
        )
    except Exception as e:
        if progress_cb:
            progress_cb(f"AI-анализ не удался ({e}). Использую эвристику.", 0.42)
        return _fallback_heuristic(transcript, lo, hi, cfg.clip.max_clip_count)


def _normalize_loudness(
    input_path: Path,
    output_path: Path,
    target_i: float = -14.0,
    target_tp: float = -1.5,
    target_lra: float = 11.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Однопроходная LUFS-нормализация через ffmpeg loudnorm.
    Цель -14 LUFS = стандарт TikTok/Reels/YouTube Shorts.
    """
    import subprocess
    if progress_cb:
        progress_cb(f"  loudnorm I={target_i} TP={target_tp} LRA={target_lra}")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"loudnorm failed:\n{result.stderr[-1500:]}")
    return output_path


def _snap_to_sentence_boundaries(
    suggestions: list[ClipSuggestion],
    transcript: Transcript,
    cfg: AppConfig,
    max_shift: float = 1.5,
) -> list[ClipSuggestion]:
    """
    Подгоняет границы клипа к ближайшему концу предложения/паузе чтобы
    не было обрыва на полуслове.

    - end: расширяем до конца ближайшего слова перед которым следующая пауза
      >= 0.35s ИЛИ слово заканчивается на знак препинания. ±max_shift сек.
    - start: подгоняем к началу слова после паузы (тоже ±max_shift сек).
    """
    SENTENCE_END = ".!?…"
    PAUSE_MIN = 0.35
    words = transcript.words
    if not words:
        return suggestions

    # Найти "точки разреза": слова после которых пауза >= PAUSE_MIN или знак ! . ?
    cut_after: list[float] = []  # абсолютное время конца слова, после которого можно резать
    cut_before: list[float] = []  # абсолютное время начала слова, перед которым можно резать
    for i, w in enumerate(words):
        if i + 1 < len(words):
            gap = words[i + 1].start - w.end
            ends_punct = w.text.rstrip().endswith(tuple(SENTENCE_END))
            if gap >= PAUSE_MIN or ends_punct:
                cut_after.append(w.end)
                cut_before.append(words[i + 1].start)
        else:
            cut_after.append(w.end)

    def _nearest(arr: list[float], target: float, max_d: float) -> Optional[float]:
        best = None
        best_d = max_d
        for x in arr:
            d = abs(x - target)
            if d <= best_d:
                best_d = d
                best = x
        return best

    min_duration = 5.0
    max_duration = 60.0

    fixed: list[ClipSuggestion] = []
    for sugg in suggestions:
        # Конец критичнее — он обрывает речь. Snap'им в первую очередь его.
        new_end = _nearest(cut_after, sugg.end, max_shift)
        new_start = _nearest(cut_before, sugg.start, max_shift)
        s = new_start if new_start is not None else sugg.start
        e = new_end if new_end is not None else sugg.end
        # Если новые границы слишком сжали клип — сохраняем end (важнее) и
        # откатываем start
        if e - s < min_duration:
            s = sugg.start
        if e - s < min_duration:
            # Если и так не хватает — берём полный оригинал, лучше чем урезанный
            s, e = sugg.start, sugg.end
        if e - s > max_duration:
            e = s + max_duration - 1.0
        fixed.append(
            ClipSuggestion(
                start=s, end=e, title=sugg.title,
                viral_score=getattr(sugg, "viral_score", 70),
                hashtags=list(getattr(sugg, "hashtags", []) or []),
                reason=getattr(sugg, "reason", ""),
            )
        )
    return fixed


def _pick_music_for_clip(cfg: AppConfig, idx: int) -> Optional[Path]:
    if cfg.music.mode == "common" and cfg.music.common_track:
        p = Path(cfg.music.common_track)
        return p if p.exists() else None
    if cfg.music.mode == "per_clip" and cfg.music.per_clip_tracks:
        if idx < len(cfg.music.per_clip_tracks):
            p = Path(cfg.music.per_clip_tracks[idx])
            return p if p.exists() else None
    return None


def re_render_clip(
    state,  # ClipEditorState
    cfg: AppConfig,
    transcript: Transcript,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Перерендеривает один клип с отредактированными параметрами:
    - state.edited_words = отредактированные субтитры (текст + тайминги)
    - state.title_text   = новый текст заголовка
    - state.sub_position_v / sub_margin_v_pct = новое положение субтитров
    - state.multi_face_mode = режим кропа

    Создаёт временный transcript из edited_words и прогоняет тот же
    pipeline (reframe → subtitles → title → overlay → music → loudnorm).
    Финал затирает state.output_video.
    """
    from .transcriber import Segment, Transcript as TR

    def step(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if not state.edited_words:
        raise RuntimeError("Нет слов для перерендера")

    # Строим минимальный Transcript из edited_words
    edited_words = state.edited_words
    # edited_words timings — clip-local (от 0). Конвертируем обратно в абсолютные.
    abs_words: list[Word] = []
    for w in edited_words:
        abs_words.append(Word(
            start=w.start + state.suggestion.start,
            end=w.end + state.suggestion.start,
            text=w.text,
            probability=w.probability,
        ))
    fake_seg = Segment(start=state.suggestion.start, end=state.suggestion.end,
                       text=" ".join(w.text for w in abs_words),
                       words=abs_words)
    fake_transcript = TR(
        language=transcript.language,
        duration=transcript.duration,
        segments=[fake_seg],
    )

    # Cfg copy с подменёнными subtitle/title параметрами
    import copy
    cfg_local = copy.deepcopy(cfg)
    cfg_local.subtitle.position_v = state.sub_position_v
    cfg_local.subtitle.margin_v_pct = state.sub_margin_v_pct
    cfg_local.title.custom_text = state.title_text

    out_dir = state.output_video.parent
    base = state.output_video.stem  # без _final
    final_path = state.output_video
    src_path = state.source_video
    sugg = state.suggestion

    reframed_path = out_dir / f"{base}_redone_reframed.mp4"
    subtitled_path = out_dir / f"{base}_redone_subs.mp4"

    step(f"Re-render: reframe (mode={state.multi_face_mode})...")
    reframe_to_9x16(
        src_path, reframed_path,
        start=sugg.start, end=sugg.end,
        progress_cb=lambda m: progress_cb(m) if progress_cb else None,
        enable_face_tracking=cfg.face_tracking,
        enable_face_zoom=cfg.face_zoom,
        use_gpu=cfg.use_gpu,
        multi_face_mode=state.multi_face_mode,
        speaker_detection=cfg.speaker_detection,
    )

    step("Re-render: вжигаю обновлённые субтитры...")
    burn_subtitles(
        reframed_path,
        fake_transcript,
        clip_start=sugg.start, clip_end=sugg.end,
        style=cfg_local.subtitle,
        output_path=subtitled_path,
        progress_cb=lambda m: progress_cb(m) if progress_cb else None,
        use_gpu=cfg.use_gpu,
        strip_punct=cfg.strip_subtitle_punct,
    )

    cur = subtitled_path
    intermediates = [reframed_path]

    if cfg_local.title.enabled and state.title_text.strip():
        with_title = out_dir / f"{base}_redone_title.mp4"
        step("Re-render: заголовок...")
        burn_title(
            cur, with_title, state.title_text.strip(),
            duration_total=sugg.duration,
            cfg=cfg_local.title,
            progress_cb=lambda m: progress_cb(m) if progress_cb else None,
            use_gpu=cfg.use_gpu,
        )
        intermediates.append(cur)
        cur = with_title

    if cfg_local.overlay.enabled and cfg_local.overlay.image_path:
        with_logo = out_dir / f"{base}_redone_logo.mp4"
        step("Re-render: логотип...")
        burn_overlay(
            cur, with_logo, cfg_local.overlay,
            progress_cb=lambda m: progress_cb(m) if progress_cb else None,
            use_gpu=cfg.use_gpu,
        )
        intermediates.append(cur)
        cur = with_logo

    # Заменяем финал
    if final_path.exists():
        final_path.unlink()
    cur.rename(final_path)

    for p in intermediates:
        try:
            if p != final_path and p.exists():
                p.unlink()
        except OSError:
            pass

    step(f"Re-render готов: {final_path.name}")
    return final_path
