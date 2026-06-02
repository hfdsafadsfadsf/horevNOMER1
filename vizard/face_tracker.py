"""
Face tracking для 9:16 кропа — копия логики Vizard.

Используем OpenCV DNN ResNet-SSD детектор (точнее Haar), плотный сэмплинг
(2-4 раза в секунду), EMA-сглаживание и опциональный zoom на лицо.

Возвращает массив FaceSample(t, cx, cy, w, h) в нормализованных координатах [0..1].
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


LOG = logging.getLogger(__name__)

# DNN модель для лиц: ResNet-10 SSD. ~10МБ + 28КБ proto.
# Качаем при первом запуске в ~/.vizard_clone/models/
MODEL_DIR = Path.home() / ".vizard_clone" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DNN_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/"
    "deploy.prototxt"
)
DNN_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20180205_fp16/"
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
DNN_PROTO_PATH = MODEL_DIR / "deploy.prototxt"
DNN_WEIGHTS_PATH = MODEL_DIR / "res10_300x300_ssd.caffemodel"

_dnn_net = None
_dnn_lock = threading.Lock()


@dataclass
class FaceSample:
    """Позиция лица в одном кадре."""

    t: float  # секунды от начала клипа (clip-local, 0-based)
    cx: float  # нормализованный X-центр лица [0..1]
    cy: float  # нормализованный Y-центр лица [0..1]
    w: float  # нормализованная ширина лица [0..1]
    h: float  # нормализованная высота лица [0..1]


@dataclass
class FaceTrack:
    """Один долгоживущий face track (ID одного человека через клип)."""

    track_id: int
    samples: List[FaceSample]
    # Доля времени когда этот человек 'активный спикер' (амплитуда мала когда он не виден)
    active_score: float = 0.0

    @property
    def avg_cx(self) -> float:
        if not self.samples:
            return 0.5
        return sum(s.cx for s in self.samples) / len(self.samples)

    @property
    def avg_cy(self) -> float:
        if not self.samples:
            return 0.5
        return sum(s.cy for s in self.samples) / len(self.samples)


@dataclass
class MultiFaceTracking:
    """Результат multi-face трекинга для целого клипа."""

    tracks: List[FaceTrack]            # все обнаруженные лица (по убыванию presence)
    active_track_at: List[Tuple[float, int]]  # (t, track_id) — кто активный в момент t
    fps_sampled: float                  # с какой частотой делали детект


def _download_if_missing(url: str, path: Path, label: str,
                        progress_cb: Optional[Callable[[str], None]] = None) -> bool:
    if path.exists() and path.stat().st_size > 1024:
        return True
    if progress_cb:
        progress_cb(f"  скачиваю {label} ({url.split('/')[-1]})...")
    try:
        urllib.request.urlretrieve(url, str(path))
        return path.exists() and path.stat().st_size > 1024
    except Exception as e:  # noqa: BLE001
        LOG.warning("Не удалось скачать %s: %s", label, e)
        return False


def _ensure_dnn_model(progress_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Гарантирует наличие весов DNN-детектора. Возвращает True если успешно."""
    if cv2 is None:
        return False
    ok_proto = _download_if_missing(DNN_PROTO_URL, DNN_PROTO_PATH, "face detector proto", progress_cb)
    ok_weights = _download_if_missing(DNN_WEIGHTS_URL, DNN_WEIGHTS_PATH, "face detector weights", progress_cb)
    return ok_proto and ok_weights


def _load_dnn(progress_cb: Optional[Callable[[str], None]] = None):
    global _dnn_net
    if cv2 is None:
        return None
    with _dnn_lock:
        if _dnn_net is not None:
            return _dnn_net
        if not _ensure_dnn_model(progress_cb):
            return None
        try:
            net = cv2.dnn.readNetFromCaffe(str(DNN_PROTO_PATH), str(DNN_WEIGHTS_PATH))
            _dnn_net = net
            return net
        except Exception as e:  # noqa: BLE001
            LOG.warning("DNN load failed: %s", e)
            return None


def _detect_face_dnn(frame, net, conf_threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
    """Детект самого крупного лица на BGR-кадре. Возвращает (x, y, w, h) в пикселях."""
    faces = _detect_all_faces_dnn(frame, net, conf_threshold)
    if not faces:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _detect_all_faces_dnn(frame, net, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Детект ВСЕХ лиц на BGR-кадре. Возвращает список (x, y, w, h) в пикселях."""
    if cv2 is None or net is None:
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()
    out: List[Tuple[int, int, int, int]] = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_threshold:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        out.append((x1, y1, bw, bh))
    # Удаляем сильно пересекающиеся (NMS-лайт)
    return _nms(out, iou_thr=0.4)


def _nms(boxes: List[Tuple[int, int, int, int]], iou_thr: float = 0.4) -> List[Tuple[int, int, int, int]]:
    """Простой non-max suppression: оставляем крупнейшее из пересекающихся лиц."""
    if not boxes:
        return []
    sorted_b = sorted(boxes, key=lambda b: -b[2] * b[3])
    kept: List[Tuple[int, int, int, int]] = []
    for b in sorted_b:
        skip = False
        for k in kept:
            if _iou(b, k) > iou_thr:
                skip = True
                break
        if not skip:
            kept.append(b)
    return kept


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def _detect_face_haar(frame, frontal, profile=None) -> Optional[Tuple[int, int, int, int]]:
    """Fallback-детект через Haar cascade (frontal + profile)."""
    if cv2 is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates: List[Tuple[int, int, int, int]] = []
    faces = frontal.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))
    for x, y, w, h in faces:
        candidates.append((int(x), int(y), int(w), int(h)))
    if profile is not None:
        for cas, do_flip in ((profile, False), (profile, True)):
            g = cv2.flip(gray, 1) if do_flip else gray
            faces_p = cas.detectMultiScale(g, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
            for x, y, w, h in faces_p:
                if do_flip:
                    x = g.shape[1] - x - w
                candidates.append((int(x), int(y), int(w), int(h)))
    if not candidates:
        return None
    return max(candidates, key=lambda f: f[2] * f[3])


def detect_faces_in_clip(
    video_path: Path,
    start: float,
    end: float,
    samples_per_sec: float = 3.0,
    detect_max_w: int = 480,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[FaceSample]:
    """
    Плотный face-tracking: 2-4 кадра в секунду от start до end.
    Возвращает список FaceSample с нормализованными координатами.
    """
    if cv2 is None:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080

    duration_total = (total_frames / fps) if fps > 0 else 0.0
    start = max(0.0, float(start))
    if end is None or end <= 0:
        end = duration_total if duration_total > 0 else (start + 60)
    end = min(float(end), duration_total) if duration_total > 0 else float(end)
    span = max(0.1, end - start)

    # Плотный сэмплинг: samples_per_sec кадров в секунду
    n_samples = max(2, int(round(span * samples_per_sec)))
    # Cap на разумный максимум (60с * 4fps = 240 — OK; больше — тяжело для ffmpeg выражения)
    n_samples = min(n_samples, 300)
    timestamps = [start + (span * i / (n_samples - 1)) for i in range(n_samples)]

    if progress_cb:
        progress_cb(f"  face-track: {n_samples} семплов на {span:.1f}с клип")

    net = _load_dnn(progress_cb)
    use_dnn = net is not None
    if use_dnn and progress_cb:
        progress_cb("  face-track: используем DNN детектор (точнее)")
    frontal = profile = None
    if not use_dnn:
        try:
            frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        except Exception:  # noqa: BLE001
            pass

    samples: List[FaceSample] = []
    progress_every = max(1, n_samples // 6)
    for i, t in enumerate(timestamps):
        t_rel = t - start  # clip-local time (для ffmpeg expression)
        frame_idx = int(round(t * fps))
        if 0 < total_frames and frame_idx >= total_frames:
            frame_idx = int(total_frames) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h0, w0 = frame.shape[:2]
        if w0 > detect_max_w:
            scale = detect_max_w / w0
            frame_small = cv2.resize(frame, (detect_max_w, int(round(h0 * scale))))
        else:
            scale = 1.0
            frame_small = frame
        ds_h, ds_w = frame_small.shape[:2]

        det = _detect_face_dnn(frame_small, net) if use_dnn else _detect_face_haar(frame_small, frontal, profile)
        if det is None:
            continue
        fx, fy, fw, fh = det
        cx = (fx + fw / 2) / ds_w
        cy = (fy + fh / 2) / ds_h
        nw = fw / ds_w
        nh = fh / ds_h
        samples.append(FaceSample(t=t_rel, cx=cx, cy=cy, w=nw, h=nh))

        if progress_cb and (i + 1) % progress_every == 0:
            progress_cb(f"  face-track: {i + 1}/{n_samples}")

    cap.release()
    return samples


def _ema_pass(values: List[float], alpha: float, outlier_clamp: float) -> List[float]:
    """Один EMA-проход с клипом резких выбросов."""
    out: List[float] = []
    last = values[0]
    for v in values:
        d = v - last
        if abs(d) > outlier_clamp:
            target = last + (outlier_clamp if d > 0 else -outlier_clamp)
        else:
            target = v
        new = alpha * target + (1 - alpha) * last
        out.append(new)
        last = new
    return out


def _bidirectional_ema(values: List[float], alpha: float = 0.4,
                        outlier_clamp: float = 0.20) -> List[float]:
    """
    Forward + backward EMA → почти без causal lag.
    Vizard-стиль: плавная траектория без отставания от лица.
    """
    if not values:
        return []
    if len(values) == 1:
        return list(values)
    fwd = _ema_pass(values, alpha, outlier_clamp)
    bwd = _ema_pass(list(reversed(fwd)), alpha, outlier_clamp)
    return list(reversed(bwd))


def smooth_track(samples: List[FaceSample], alpha: float = 0.4,
                 outlier_clamp: float = 0.20) -> List[FaceSample]:
    """
    Forward+backward EMA сглаживание — плавно без отставания.
    Также подавляет резкие выбросы (false positive detector).
    """
    if not samples:
        return []
    if len(samples) == 1:
        return list(samples)

    cxs = [s.cx for s in samples]
    cys = [s.cy for s in samples]
    ws = [s.w for s in samples]
    hs = [s.h for s in samples]

    smooth_cx = _bidirectional_ema(cxs, alpha, outlier_clamp)
    smooth_cy = _bidirectional_ema(cys, alpha, outlier_clamp)
    smooth_w = _bidirectional_ema(ws, alpha, outlier_clamp)
    smooth_h = _bidirectional_ema(hs, alpha, outlier_clamp)

    return [
        FaceSample(t=samples[i].t, cx=smooth_cx[i], cy=smooth_cy[i],
                   w=smooth_w[i], h=smooth_h[i])
        for i in range(len(samples))
    ]


def build_piecewise_expr(points: List[Tuple[float, float]],
                          default: float) -> str:
    """
    Строит ffmpeg expression для piecewise-linear интерполяции по точкам (t, value).
    Возвращает default до первой точки и после последней.

    Структура: if(lt(t,t1), v_lerp_01, if(lt(t,t2), v_lerp_12, ... default))
    """
    if not points:
        return f"{default:.2f}"
    if len(points) == 1:
        return f"{points[0][1]:.2f}"

    # Сортируем по t
    pts = sorted(points, key=lambda p: p[0])

    # Строим вложенный if справа налево
    expr = f"{pts[-1][1]:.2f}"  # для t >= последней точки
    for i in range(len(pts) - 1, 0, -1):
        t0, v0 = pts[i - 1]
        t1, v1 = pts[i]
        dt = t1 - t0
        if dt < 1e-3:
            continue
        slope = (v1 - v0) / dt
        # Между t0 и t1: v0 + slope*(t-t0)
        segment = f"({v0:.2f}+{slope:.5f}*(t-{t0:.3f}))"
        expr = f"if(lt(t\\,{t1:.3f})\\,{segment}\\,{expr})"
    # До первой точки — берём значение первой точки
    expr = f"if(lt(t\\,{pts[0][0]:.3f})\\,{pts[0][1]:.2f}\\,{expr})"
    return expr


# ---------------------------------------------------------------------------
# Multi-face tracking + speaker detection
# ---------------------------------------------------------------------------

def detect_multi_faces_in_clip(
    video_path: Path,
    start: float,
    end: float,
    samples_per_sec: float = 3.0,
    detect_max_w: int = 480,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[List[FaceSample]], float]:
    """
    Возвращает (per-frame face lists, fps_sampled).

    Каждый кадр → список всех найденных лиц как FaceSample (без ID).
    Дальше _associate_into_tracks() свяжет их в track'и.
    """
    if cv2 is None:
        return [], 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

    duration_total = (total_frames / fps) if fps > 0 else 0.0
    start = max(0.0, float(start))
    if end is None or end <= 0:
        end = duration_total if duration_total > 0 else (start + 60)
    end = min(float(end), duration_total) if duration_total > 0 else float(end)
    span = max(0.1, end - start)

    n_samples = max(2, int(round(span * samples_per_sec)))
    n_samples = min(n_samples, 300)
    timestamps = [start + (span * i / (n_samples - 1)) for i in range(n_samples)]

    if progress_cb:
        progress_cb(f"  multi-face: {n_samples} семплов на {span:.1f}с")

    net = _load_dnn(progress_cb)
    use_dnn = net is not None
    frontal = profile = None
    if not use_dnn:
        try:
            frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        except Exception:  # noqa: BLE001
            pass

    per_frame: List[List[FaceSample]] = []
    for i, t in enumerate(timestamps):
        t_rel = t - start
        frame_idx = int(round(t * fps))
        if 0 < total_frames and frame_idx >= total_frames:
            frame_idx = int(total_frames) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            per_frame.append([])
            continue

        h0, w0 = frame.shape[:2]
        if w0 > detect_max_w:
            scale = detect_max_w / w0
            frame_small = cv2.resize(frame, (detect_max_w, int(round(h0 * scale))))
        else:
            scale = 1.0
            frame_small = frame
        ds_h, ds_w = frame_small.shape[:2]

        if use_dnn:
            boxes = _detect_all_faces_dnn(frame_small, net)
        else:
            # Haar: одно лицо → wrap в список
            b = _detect_face_haar(frame_small, frontal, profile)
            boxes = [b] if b else []

        frame_samples: List[FaceSample] = []
        for fx, fy, fw, fh in boxes:
            cx = (fx + fw / 2) / ds_w
            cy = (fy + fh / 2) / ds_h
            nw = fw / ds_w
            nh = fh / ds_h
            frame_samples.append(FaceSample(t=t_rel, cx=cx, cy=cy, w=nw, h=nh))

        per_frame.append(frame_samples)

    cap.release()
    return per_frame, samples_per_sec


def _associate_into_tracks(
    per_frame: List[List[FaceSample]],
    max_distance: float = 0.18,
    min_track_len: int = 3,
) -> List[FaceTrack]:
    """
    Жадная ассоциация bbox'ов между кадрами по минимальной евклидовой дистанции
    в нормализованных координатах. Если ближайший прошлый трек дальше max_distance —
    создаём новый трек.
    """
    next_id = 0
    active_tracks: List[FaceTrack] = []

    for frame_idx, samples in enumerate(per_frame):
        # Для каждого лица в текущем кадре — ищем ближайший active track
        used_track_ids: set[int] = set()
        for s in samples:
            best_track: Optional[FaceTrack] = None
            best_dist = max_distance
            for tr in active_tracks:
                if tr.track_id in used_track_ids:
                    continue
                last = tr.samples[-1]
                d = ((s.cx - last.cx) ** 2 + (s.cy - last.cy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_track = tr
            if best_track is not None:
                best_track.samples.append(s)
                used_track_ids.add(best_track.track_id)
            else:
                new_tr = FaceTrack(track_id=next_id, samples=[s])
                next_id += 1
                active_tracks.append(new_tr)
                used_track_ids.add(new_tr.track_id)

    # Фильтр коротких треков (< min_track_len семплов)
    filtered = [tr for tr in active_tracks if len(tr.samples) >= min_track_len]
    # Сортируем по presence (длина трека, потом по размеру лица)
    filtered.sort(key=lambda tr: (-len(tr.samples), -tr.samples[0].w))
    return filtered


def detect_audio_amplitude(
    video_path: Path,
    start: float,
    end: float,
    samples_per_sec: float = 3.0,
) -> List[Tuple[float, float]]:
    """
    Извлекает огибающую амплитуды аудио по клипу — для speaker detection.

    Возвращает [(t_rel, rms), ...] с шагом 1/samples_per_sec.
    """
    import shutil as _shutil
    import subprocess
    import wave
    import struct
    import tempfile

    if _shutil.which("ffmpeg") is None:
        return []

    span = max(0.1, end - start)
    n_buckets = max(2, int(round(span * samples_per_sec)))
    bucket_dur = span / n_buckets

    # Извлекаем mono 8kHz PCM-сэмплы — этого хватит для амплитуды
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(video_path),
            "-t", f"{span:.3f}",
            "-ac", "1", "-ar", "8000",
            "-f", "wav", str(tmp_path),
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0:
            return []

        with wave.open(str(tmp_path), "rb") as w:
            n_frames = w.getnframes()
            sample_rate = w.getframerate()
            raw = w.readframes(n_frames)
        samples = struct.unpack(f"<{n_frames}h", raw)
    except Exception:  # noqa: BLE001
        return []
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    if not samples:
        return []

    out: List[Tuple[float, float]] = []
    bucket_size = max(1, int(sample_rate * bucket_dur))
    for i in range(n_buckets):
        seg = samples[i * bucket_size:(i + 1) * bucket_size]
        if not seg:
            out.append((i * bucket_dur, 0.0))
            continue
        rms = (sum(x * x for x in seg) / len(seg)) ** 0.5
        # Нормализуем к [0..1]: max 16-bit = 32768
        out.append((i * bucket_dur, rms / 32768.0))
    return out


def detect_active_speaker(
    tracks: List[FaceTrack],
    amplitude: List[Tuple[float, float]],
    silence_threshold: float = 0.015,
) -> List[Tuple[float, int]]:
    """
    Определяет активного спикера в каждый момент времени.

    Простая эвристика без аудиовизуальной корреляции движений губ:
    - Если в момент t слышен голос (RMS > silence_threshold) — активный = тот
      track который сейчас виден и больше по размеру лица в кадре.
    - Если 2 человека: чередуем кто говорит на интервалах разговора, выбирая
      каждый раз ближайшего к центру кадра (часто оператор фокусирует на нём).
    - Если тихо — активный = самый стабильный трек (с наибольшим presence).

    Это упрощение, но даёт намного лучший результат чем "одно главное лицо".
    Реальный mouth-movement detection требовал бы facial landmarks
    (mediapipe / dlib) — будем добавлять отдельно если этого мало.
    """
    if not tracks:
        return []
    if not amplitude:
        # Без аудио — всегда главный трек
        return [(0.0, tracks[0].track_id)]

    # Соберём индекс: для каждого t какие треки видны
    def visible_tracks_at(t: float) -> List[FaceTrack]:
        eps = 0.5  # допуск ±0.5с
        return [tr for tr in tracks
                if any(abs(s.t - t) <= eps for s in tr.samples)]

    # Строим решение для каждого временного бакета амплитуды
    out: List[Tuple[float, int]] = []
    last_speaker_id = tracks[0].track_id
    last_switch_t = -10.0
    min_switch_interval = 0.6  # не мечемся туда-сюда чаще раза в 0.6с

    for t, rms in amplitude:
        vis = visible_tracks_at(t)
        if not vis:
            out.append((t, last_speaker_id))
            continue

        if rms < silence_threshold:
            # Тихо — оставляем последнего спикера если он ещё видим, иначе главный
            if any(tr.track_id == last_speaker_id for tr in vis):
                out.append((t, last_speaker_id))
            else:
                # Главный среди видимых = самый крупный
                main = max(vis, key=lambda tr: max(s.w for s in tr.samples))
                out.append((t, main.track_id))
            continue

        # Голос есть → решаем кто говорит
        # Кандидаты: видимые треки. Без mouth-cues выбираем по приоритету:
        # 1) самое крупное лицо в кадре (часто оператор зумит на спикера),
        # 2) если несколько одинаковых — не переключаемся слишком часто
        candidates = sorted(vis, key=lambda tr:
                            -max(s.w for s in tr.samples if abs(s.t - t) <= 0.5))
        chosen = candidates[0]
        # Защита от дрожания: не переключаемся чаще min_switch_interval
        if chosen.track_id != last_speaker_id and (t - last_switch_t) < min_switch_interval:
            chosen_id = last_speaker_id
        else:
            chosen_id = chosen.track_id
            if chosen_id != last_speaker_id:
                last_switch_t = t
        out.append((t, chosen_id))
        last_speaker_id = chosen_id

    return out


def smooth_active_speaker(
    speaker_at: List[Tuple[float, int]],
    min_run_sec: float = 1.0,
) -> List[Tuple[float, int]]:
    """Подавляет короткие переключения (<min_run_sec)."""
    if len(speaker_at) <= 1:
        return speaker_at
    smoothed = list(speaker_at)
    # Если есть run длительностью <min_run_sec, объединяем с предыдущим
    i = 0
    while i < len(smoothed):
        j = i
        while j + 1 < len(smoothed) and smoothed[j + 1][1] == smoothed[i][1]:
            j += 1
        run_dur = smoothed[j][0] - smoothed[i][0]
        if run_dur < min_run_sec and i > 0:
            prev_id = smoothed[i - 1][1]
            for k in range(i, j + 1):
                smoothed[k] = (smoothed[k][0], prev_id)
        i = j + 1
    return smoothed


def track_multi_face(
    video_path: Path,
    start: float,
    end: float,
    samples_per_sec: float = 3.0,
    detect_audio: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> MultiFaceTracking:
    """
    Полный multi-face пайплайн:
    1. Детектируем все лица в каждом кадре
    2. Ассоциируем в треки (один человек = один track)
    3. Сглаживаем каждый трек (forward+backward EMA)
    4. Извлекаем огибающую аудио
    5. Определяем активного спикера в каждый момент
    """
    per_frame, fps_sampled = detect_multi_faces_in_clip(
        video_path, start=start, end=end,
        samples_per_sec=samples_per_sec,
        progress_cb=progress_cb,
    )
    if progress_cb:
        total = sum(len(f) for f in per_frame)
        progress_cb(f"  multi-face: {total} детектов на {len(per_frame)} кадрах")

    tracks = _associate_into_tracks(per_frame)
    if progress_cb:
        progress_cb(f"  multi-face: {len(tracks)} треков после ассоциации")

    # Сглаживаем каждый трек
    smoothed: List[FaceTrack] = []
    for tr in tracks:
        smoothed_samples = smooth_track(tr.samples)
        smoothed.append(FaceTrack(track_id=tr.track_id, samples=smoothed_samples))

    # Аудио (для speaker detection)
    amplitude: List[Tuple[float, float]] = []
    if detect_audio:
        amplitude = detect_audio_amplitude(video_path, start, end, samples_per_sec)
        if progress_cb:
            progress_cb(f"  amplitude: {len(amplitude)} семплов")

    active = detect_active_speaker(smoothed, amplitude)
    active = smooth_active_speaker(active)

    return MultiFaceTracking(
        tracks=smoothed,
        active_track_at=active,
        fps_sampled=fps_sampled,
    )
