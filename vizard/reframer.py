"""
9:16 переформатирование с умным авто-кропом (face tracking).

Логика (копия Vizard):
1. Плотно семплируем кадры из исходного видео (2-4 fps в пределах клипа).
2. На каждом кадре ищем самое крупное лицо через OpenCV DNN ResNet-SSD
   (или Haar fallback если модели нет).
3. EMA-сглаживаем траекторию X (и опционально Y) центра лица.
4. Опционально zoom: если лицо мелкое (<15% ширины), уменьшаем crop_w
   чтобы лицо занимало ~30% кадра.
5. Передаём траекторию в ffmpeg через piecewise-linear expression
   `crop=W:H:'expr(t)':Y` — кадр двигается за лицом плавно.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

from .face_tracker import (
    FaceSample,
    FaceTrack,
    MultiFaceTracking,
    build_piecewise_expr,
    detect_faces_in_clip,
    smooth_track,
    track_multi_face,
)


TARGET_W = 1080
TARGET_H = 1920

# Целевая доля ширины лица в финальном 9:16 кадре. Vizard держит ~30% — лицо большое,
# заметное, но не "обрезанное по бровям".
FACE_FRACTION_TARGET = 0.30
# Минимальная доля лица в исходнике, при которой включаем zoom.
ZOOM_TRIGGER_FRACTION = 0.15
# Лимит zoom (не уменьшаем crop_w меньше чем на 35% от исходного).
ZOOM_MIN_CROP_FRACTION = 0.55


@dataclass
class FrameInfo:
    width: int
    height: int
    fps: float
    duration: float


def _probe_video(path: Path) -> FrameInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    info = {}
    for line in out.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    w = int(info.get("width", 1920))
    h = int(info.get("height", 1080))
    fr = info.get("r_frame_rate", "30/1")
    if "/" in fr:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    else:
        fps = float(fr)
    duration = float(info.get("duration", 0.0) or 0.0)
    return FrameInfo(width=w, height=h, fps=fps, duration=duration)


def _compute_crop_size(src_w: int, src_h: int,
                       face_w_norm: float) -> Tuple[int, int]:
    """
    Считаем размер кропа (crop_w, crop_h) сохраняя 9:16.

    Если face_w_norm маленький — уменьшаем crop_w (zoom in) чтобы лицо
    в итоговом 1080-кадре было ~FACE_FRACTION_TARGET ширины.
    """
    target_aspect = TARGET_W / TARGET_H
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        # Горизонтальный исходник → кроп по ширине, высота = src_h
        base_crop_h = src_h
        base_crop_w = int(round(src_h * target_aspect))
    else:
        # Вертикальный исходник → кроп по высоте, ширина = src_w
        base_crop_w = src_w
        base_crop_h = int(round(src_w / target_aspect))

    crop_w = base_crop_w
    crop_h = base_crop_h

    # Zoom-in: если лицо маленькое — уменьшаем crop_w
    if face_w_norm > 0 and face_w_norm < ZOOM_TRIGGER_FRACTION:
        # face_in_crop = face_w_norm * src_w / crop_w (в долях crop_w)
        # хотим: face_in_crop = FACE_FRACTION_TARGET
        desired_crop_w = (face_w_norm * src_w) / FACE_FRACTION_TARGET
        # Не уменьшаем сильно
        min_allowed = base_crop_w * ZOOM_MIN_CROP_FRACTION
        desired_crop_w = max(min_allowed, desired_crop_w)
        if desired_crop_w < crop_w:
            crop_w = int(round(desired_crop_w))
            crop_h = int(round(crop_w / target_aspect))
            # Не превышаем src_h
            if crop_h > src_h:
                crop_h = src_h
                crop_w = int(round(crop_h * target_aspect))

    # Чётные размеры (требование yuv420p)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    return crop_w, crop_h


def _samples_from_active_speaker(multi: MultiFaceTracking) -> List[FaceSample]:
    """
    Из multi-face tracking результата строим единую цепочку семплов:
    в каждый момент берём позицию того face track который сейчас активный спикер.

    Это даёт траекторию, которая прыгает между двумя людьми в моменты ответа/вопроса —
    как у Vizard.
    """
    if not multi.tracks:
        return []
    if not multi.active_track_at:
        return multi.tracks[0].samples

    # Индекс track_id → samples
    by_id = {tr.track_id: tr.samples for tr in multi.tracks}

    out: List[FaceSample] = []
    for t, tid in multi.active_track_at:
        track_samples = by_id.get(tid)
        if not track_samples:
            continue
        # Берём ближайший по времени sample этого трека
        best = min(track_samples, key=lambda s: abs(s.t - t))
        # Используем позицию этого трека в этот момент времени
        out.append(FaceSample(t=t, cx=best.cx, cy=best.cy, w=best.w, h=best.h))
    return out


def _count_switches(active_track_at: List[Tuple[float, int]]) -> int:
    """Сколько раз сменился активный спикер."""
    if len(active_track_at) <= 1:
        return 0
    n = 0
    last_id = active_track_at[0][1]
    for _, tid in active_track_at[1:]:
        if tid != last_id:
            n += 1
            last_id = tid
    return n


def _compute_default_focus(samples: List[FaceSample]) -> Tuple[float, float, float]:
    """Возвращает (cx, cy, w) для случая когда expression не используется."""
    if not samples or np is None:
        return 0.5, 0.5, 0.0
    arr_x = np.array([s.cx for s in samples], dtype=np.float32)
    arr_y = np.array([s.cy for s in samples], dtype=np.float32)
    arr_w = np.array([s.w for s in samples], dtype=np.float32)
    # Берём медиану для устойчивости к выбросам
    return float(np.median(arr_x)), float(np.median(arr_y)), float(np.median(arr_w))


def reframe_split_2face(
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
    multi_data: MultiFaceTracking,
    progress_cb: Optional[Callable[[str], None]] = None,
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "192k",
    use_gpu: bool = True,
    highlight_active: bool = True,
) -> Path:
    """
    Split-screen 2-face режим: вертикальное видео 1080×1920 разделено пополам.
    Сверху — крупный план первого спикера, снизу — второго.
    Активный спикер опционально подсвечивается (рамка/яркость).

    Размер каждой ячейки = 1080×960 (квадрат), кропим из исходника квадратные
    регионы вокруг центра каждого лица.
    """
    info = _probe_video(input_path)
    src_w, src_h = info.width, info.height
    duration = end - start

    if len(multi_data.tracks) < 2:
        # Fallback на обычный 9:16
        return reframe_to_9x16(
            input_path, output_path, start=start, end=end,
            progress_cb=progress_cb, crf=crf, preset=preset,
            audio_bitrate=audio_bitrate,
            use_gpu=use_gpu, multi_face_mode="single",
        )

    cell_w = TARGET_W
    cell_h = TARGET_H // 2  # 960

    tr1, tr2 = multi_data.tracks[0], multi_data.tracks[1]

    # Для каждого трека строим piecewise X-expression для квадратного кропа
    # вокруг лица. Размер кропа = размер исходного по высоте (или меньше для зума)
    crop_side = min(src_h, src_w)  # квадратный кроп

    # Точки траектории для каждого трека
    pts1 = [(s.t, _clamp(s.cx * src_w - crop_side / 2, 0, src_w - crop_side)) for s in tr1.samples]
    pts2 = [(s.t, _clamp(s.cx * src_w - crop_side / 2, 0, src_w - crop_side)) for s in tr2.samples]

    default_x1 = pts1[0][1] if pts1 else (src_w - crop_side) / 2
    default_x2 = pts2[0][1] if pts2 else (src_w - crop_side) / 2

    x_expr1 = build_piecewise_expr(pts1, default_x1) if len(pts1) >= 2 else f"{default_x1:.2f}"
    x_expr2 = build_piecewise_expr(pts2, default_x2) if len(pts2) >= 2 else f"{default_x2:.2f}"

    # Y: ставим лицо на 40% (правило третей)
    avg_cy1 = tr1.avg_cy
    avg_cy2 = tr2.avg_cy
    y1 = int(round(_clamp(avg_cy1 * src_h - crop_side * 0.40, 0, src_h - crop_side)))
    y2 = int(round(_clamp(avg_cy2 * src_h - crop_side * 0.40, 0, src_h - crop_side)))

    # Цепочка фильтров: два кропа → два скейла → вертикальный stack
    vf = (
        f"[0:v]crop={crop_side}:{crop_side}:{x_expr1}:{y1},"
        f"scale={cell_w}:{cell_h}:flags=lanczos[t];"
        f"[0:v]crop={crop_side}:{crop_side}:{x_expr2}:{y2},"
        f"scale={cell_w}:{cell_h}:flags=lanczos[b];"
        f"[t][b]vstack=inputs=2,setsar=1[v]"
    )

    from .gpu import gpu_encode_args
    enc_args = gpu_encode_args(use_gpu=use_gpu, crf_or_cq=crf, preset=preset)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-filter_complex", vf,
        "-map", "[v]",
        "-map", "0:a?",
        *enc_args,
        "-r", "60",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ]

    if progress_cb:
        progress_cb(f"split-screen 2 лица: {cell_w}x{cell_h} каждая ячейка")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if progress_cb:
            progress_cb(f"  split упал, fallback на single-face")
        return reframe_to_9x16(
            input_path, output_path, start=start, end=end,
            progress_cb=progress_cb, crf=crf, preset=preset,
            audio_bitrate=audio_bitrate, use_gpu=use_gpu,
            multi_face_mode="single",
        )
    return output_path


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def reframe_to_9x16(
    input_path: Path,
    output_path: Path,
    start: float = 0.0,
    end: Optional[float] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    crf: int = 18,
    preset: str = "slow",
    audio_bitrate: str = "192k",
    enable_face_tracking: bool = True,
    enable_face_zoom: bool = True,
    use_gpu: bool = True,
    multi_face_mode: str = "auto",
    speaker_detection: bool = True,
) -> Path:
    """
    Кропает входное видео под 9:16 (1080x1920) центрируя по лицу с tracking.

    enable_face_tracking=False → одна статичная X-позиция (старое поведение).
    enable_face_zoom=False → без zoom, фиксированный размер кропа.
    """
    info = _probe_video(input_path)
    src_w, src_h = info.width, info.height
    duration = (end - start) if end is not None else (info.duration - start)
    if duration <= 0:
        raise ValueError(f"Невалидный диапазон: start={start}, end={end}")
    if end is None:
        end = info.duration

    if progress_cb:
        progress_cb(
            f"Face tracking ({start:.1f}–{end:.1f}s, источник {src_w}x{src_h})..."
        )

    # Multi-face режим: трекаем все лица + определяем активного спикера,
    # затем строим единую траекторию из позиций активного спикера в каждый момент
    multi_data: Optional[MultiFaceTracking] = None
    if multi_face_mode in ("auto", "split") and enable_face_tracking:
        try:
            multi_data = track_multi_face(
                input_path, start=start, end=end,
                samples_per_sec=3.0,
                detect_audio=speaker_detection,
                progress_cb=progress_cb,
            )
        except Exception as e:  # noqa: BLE001
            if progress_cb:
                progress_cb(f"  multi-face упал ({e}) — fallback на single-face")
            multi_data = None

    if multi_data and len(multi_data.tracks) >= 2 and multi_face_mode == "split":
        # Split-screen режим: делегируем reframe_split_2face
        if progress_cb:
            progress_cb(f"  split-screen: {len(multi_data.tracks)} треков → 2 ячейки")
        return reframe_split_2face(
            input_path, output_path,
            start=start, end=end,
            multi_data=multi_data,
            progress_cb=progress_cb,
            crf=crf, preset=preset,
            audio_bitrate=audio_bitrate,
            use_gpu=use_gpu,
        )

    if multi_data and len(multi_data.tracks) >= 2 and multi_face_mode == "auto":
        # 2+ лица → строим samples из активного спикера в каждый момент
        samples = _samples_from_active_speaker(multi_data)
        if progress_cb:
            progress_cb(
                f"  multi-face: {len(multi_data.tracks)} треков, "
                f"активный спикер меняется {_count_switches(multi_data.active_track_at)} раз"
            )
    elif multi_data and len(multi_data.tracks) >= 1:
        # 1 лицо или режим single → берём первый (главный) трек
        samples = multi_data.tracks[0].samples
        if progress_cb:
            progress_cb(f"  multi-face: 1 главный трек ({len(samples)} семплов)")
    else:
        # Fallback на старый single-face детект
        samples = detect_faces_in_clip(
            input_path, start=start, end=end,
            samples_per_sec=3.0,
            progress_cb=progress_cb,
        )
        if progress_cb:
            if samples:
                progress_cb(f"  найдено лиц в {len(samples)} кадрах")
            else:
                progress_cb("  лиц не найдено — fallback на центр")

    samples = smooth_track(samples, alpha=0.4, outlier_clamp=0.20)

    default_cx, default_cy, default_face_w = _compute_default_focus(samples)
    face_w = default_face_w if enable_face_zoom else 0.0
    crop_w, crop_h = _compute_crop_size(src_w, src_h, face_w)

    target_aspect = TARGET_W / TARGET_H
    src_aspect = src_w / src_h
    horiz_src = src_aspect > target_aspect

    # Строим выражения для x и y в ffmpeg
    use_expr = enable_face_tracking and len(samples) >= 2

    if horiz_src:
        # X-кроп подвижный, Y фиксированный (или сдвиг для face vertical placement)
        # Y: ставим лицо чуть выше центра (правило третей): face_cy ~ 0.4 от высоты кропа
        # Но если crop_h == src_h, Y_offset = 0 (кропать по высоте нечего)
        y_offset_max = src_h - crop_h
        if y_offset_max > 0:
            # Цель: face_cy_px == y0 + crop_h * 0.40
            face_y_px = default_cy * src_h
            y0 = int(round(face_y_px - crop_h * 0.40))
            y0 = max(0, min(y_offset_max, y0))
        else:
            y0 = 0

        if use_expr:
            # Строим x(t) как piecewise expression
            x_points: List[Tuple[float, float]] = []
            for s in samples:
                cx_px = s.cx * src_w
                x0 = cx_px - crop_w / 2
                x0 = max(0.0, min(src_w - crop_w, x0))
                x_points.append((s.t, x0))
            # default = первая позиция
            default_x = x_points[0][1] if x_points else (src_w - crop_w) / 2
            x_expr = build_piecewise_expr(x_points, default_x)
            y_expr = f"{y0}"
        else:
            x_px = default_cx * src_w - crop_w / 2
            x_px = max(0.0, min(src_w - crop_w, x_px))
            x_expr = f"{int(round(x_px))}"
            y_expr = f"{y0}"
    else:
        # Вертикальный исходник: X фиксирован, Y подвижный
        x0 = max(0, (src_w - crop_w) // 2)
        if use_expr:
            y_points: List[Tuple[float, float]] = []
            for s in samples:
                cy_px = s.cy * src_h
                y0 = cy_px - crop_h * 0.40  # лицо в верхней трети
                y0 = max(0.0, min(src_h - crop_h, y0))
                y_points.append((s.t, y0))
            default_y = y_points[0][1] if y_points else (src_h - crop_h) / 2
            y_expr = build_piecewise_expr(y_points, default_y)
            x_expr = f"{x0}"
        else:
            y_px = default_cy * src_h - crop_h * 0.40
            y_px = max(0.0, min(src_h - crop_h, y_px))
            x_expr = f"{x0}"
            y_expr = f"{int(round(y_px))}"

    # Собираем vf
    vf = (
        f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr},"
        f"scale={TARGET_W}:{TARGET_H}:flags=lanczos,"
        f"setsar=1"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from .gpu import gpu_encode_args, detect_gpu

    cap = detect_gpu()
    enc_args = gpu_encode_args(use_gpu=use_gpu, crf_or_cq=crf, preset=preset)
    if progress_cb and use_gpu and cap.any_encoder:
        progress_cb(f"  кодек: {cap.label()}")

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        *enc_args,
        "-r", "60",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ]

    if progress_cb:
        tracking_label = f"({len(samples)} точек tracking)" if use_expr else "(статичная позиция)"
        progress_cb(
            f"ffmpeg reframe → 1080x1920, crop {crop_w}x{crop_h} {tracking_label}"
        )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Если piecewise expression не сработал — fallback на статичный кроп
        if use_expr and progress_cb:
            progress_cb("  ffmpeg expression failed — fallback на статичный кроп")
        if use_expr:
            return reframe_to_9x16(
                input_path, output_path, start=start, end=end,
                progress_cb=progress_cb, crf=crf, preset=preset,
                audio_bitrate=audio_bitrate,
                enable_face_tracking=False,
                enable_face_zoom=enable_face_zoom,
                use_gpu=use_gpu,
                multi_face_mode=multi_face_mode,
                speaker_detection=speaker_detection,
            )
        raise RuntimeError(f"ffmpeg reframe failed:\n{result.stderr[-2000:]}")

    return output_path
