"""Транскрипция видео в текст с word-level таймингами через faster-whisper.

Поддерживает CUDA-ускорение на NVIDIA GPU (5-15x быстрее CPU). Авто-детект:
    device="auto" → пробует cuda+float16 → cuda+int8_float16 → cpu+int8
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

LOG = logging.getLogger(__name__)


def _register_cuda_dll_dirs() -> None:
    """На Windows добавляем bin-папки nvidia-cublas-cu12 и nvidia-cudnn-cu12
    в DLL search path, чтобы ctranslate2 нашёл cudnn64_*.dll / cublas64_*.dll.

    Без этого faster-whisper на Windows падает с "Library cudnn_ops_infer not found".
    Вызывается до import faster_whisper.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore  # noqa: WPS433
    except ImportError:
        return

    nv_root = Path(nvidia.__file__).resolve().parent
    candidates = [
        nv_root / "cublas" / "bin",
        nv_root / "cudnn" / "bin",
        nv_root / "cuda_runtime" / "bin",
    ]
    for p in candidates:
        if p.is_dir():
            try:
                os.add_dll_directory(str(p))
                LOG.debug("Registered CUDA DLL dir: %s", p)
            except (OSError, AttributeError):
                # add_dll_directory не доступен на старых Python — попробуем PATH
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dll_dirs()

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

# Кешируем результат детекта CUDA чтобы не дёргать torch/ctranslate2 каждый раз
_cached_cuda_ok: Optional[bool] = None


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)

    @property
    def words(self) -> list[Word]:
        """Все слова из всех сегментов плоским списком (в порядке времени)."""
        out: list[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out

    def words_in_range(self, start: float, end: float) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            if seg.end < start or seg.start > end:
                continue
            for w in seg.words:
                if w.end >= start and w.start <= end:
                    out.append(w)
        return out


def _cuda_available() -> bool:
    """Проверяет доступность CUDA для faster-whisper.

    Не импортируем torch (он не обязателен) — пробуем напрямую ctranslate2 утилитой,
    либо просто загрузку модели на GPU в _try_load_model.
    """
    global _cached_cuda_ok
    if _cached_cuda_ok is not None:
        return _cached_cuda_ok

    # 1) Если стоит torch — это надёжный способ
    try:
        import torch  # noqa: WPS433
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _cached_cuda_ok = True
            return True
    except Exception:  # noqa: BLE001
        pass

    # 2) ctranslate2 (faster-whisper его юзает) умеет говорить про CUDA-устройства
    try:
        import ctranslate2  # noqa: WPS433
        try:
            count = ctranslate2.get_cuda_device_count()
            if count and count > 0:
                _cached_cuda_ok = True
                return True
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    # 3) nvidia-smi — последний fallback (GPU физически есть)
    try:
        import shutil
        import subprocess
        if shutil.which("nvidia-smi") is not None:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout.strip():
                # nvidia-smi видит GPU; ctranslate2 может всё равно не подцепиться
                # если нет cuBLAS/cuDNN — но пытаемся
                _cached_cuda_ok = True
                return True
    except Exception:  # noqa: BLE001
        pass

    _cached_cuda_ok = False
    return False


def _try_load_model(model_size: str, device: str, compute_type: str) -> Optional["WhisperModel"]:
    """Попытка загрузить модель. Возвращает None если упало (например libcudnn нет)."""
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:  # noqa: BLE001
        LOG.warning("Не удалось загрузить Whisper на %s/%s: %s", device, compute_type, e)
        return None


def _resolve_device_chain(
    requested_device: str,
    requested_compute: str,
) -> list[tuple[str, str]]:
    """Возвращает список пар (device, compute_type) для попыток загрузки.

    requested_device может быть: "auto" | "cuda" | "cpu".
    Если "auto" — детектим CUDA. Если есть — пробуем cuda сначала, затем CPU.
    Внутри cuda пробуем float16 → int8_float16 (на случай если нет cuDNN).
    """
    requested_device = (requested_device or "auto").lower().strip()
    chain: list[tuple[str, str]] = []

    if requested_device == "cpu":
        chain.append(("cpu", requested_compute or "int8"))
        return chain

    if requested_device == "cuda":
        # Если пользователь явно сказал cuda — пробуем cuda и только потом cpu
        chain.append(("cuda", requested_compute if requested_compute and "int8" not in requested_compute and requested_compute != "auto" else "float16"))
        chain.append(("cuda", "int8_float16"))
        chain.append(("cpu", "int8"))
        return chain

    # auto:
    if _cuda_available():
        chain.append(("cuda", "float16"))
        chain.append(("cuda", "int8_float16"))
    chain.append(("cpu", "int8"))
    return chain


def transcribe_video(
    video_path: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Transcript:
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper не установлен. Запусти: pip install faster-whisper"
        )

    chain = _resolve_device_chain(device, compute_type)

    model: Optional[WhisperModel] = None
    chosen_device = "cpu"
    chosen_compute = "int8"

    for dev, ct in chain:
        if progress_cb:
            progress_cb(f"Загружаем Whisper '{model_size}' на {dev.upper()}/{ct}...")
        model = _try_load_model(model_size, dev, ct)
        if model is not None:
            chosen_device, chosen_compute = dev, ct
            break

    if model is None:
        raise RuntimeError(
            "Не удалось загрузить Whisper ни на CUDA, ни на CPU. "
            "Установи: pip install faster-whisper, и для CUDA — cuBLAS/cuDNN."
        )

    if progress_cb:
        if chosen_device == "cuda":
            progress_cb(f"✓ Whisper на GPU (CUDA, {chosen_compute}) — будет быстро")
        else:
            progress_cb("Whisper на CPU — это самая медленная часть, можно поставить cuda")
        progress_cb("Транскрипция запущена...")

    # На GPU безопасно поднять beam_size — это даёт лучшее качество без огромного штрафа по скорости.
    # На CPU оставляем beam_size=1.
    beam = 5 if chosen_device == "cuda" else 1

    segments_iter, info = model.transcribe(
        str(video_path),
        language=None if language in (None, "", "auto") else language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=beam,
    )

    segments: list[Segment] = []
    total_words = 0
    last_t = 0.0
    src_dur = float(info.duration) if info.duration else 0.0
    for seg in segments_iter:
        words: list[Word] = []
        if seg.words:
            for w in seg.words:
                words.append(
                    Word(
                        start=float(w.start),
                        end=float(w.end),
                        text=w.word.strip(),
                        probability=float(w.probability or 1.0),
                    )
                )
        segments.append(
            Segment(start=float(seg.start), end=float(seg.end), text=seg.text.strip(), words=words)
        )
        total_words += len(words)
        # Прогресс по % от длительности
        if progress_cb:
            cur_t = float(seg.end or 0.0)
            if src_dur > 0 and cur_t - last_t >= max(2.0, src_dur * 0.05):
                pct = min(100, int(cur_t / src_dur * 100))
                progress_cb(f"  Whisper: {pct}% ({len(segments)} сегментов, {total_words} слов)")
                last_t = cur_t

    duration = float(info.duration) if info.duration else (segments[-1].end if segments else 0.0)
    if progress_cb:
        progress_cb(
            f"Транскрипция готова ({chosen_device.upper()}/{chosen_compute}): "
            f"язык={info.language}, длительность={duration:.1f}с, "
            f"{len(segments)} сегментов, {total_words} слов"
        )

    return Transcript(language=info.language, duration=duration, segments=segments)


def whisper_device_label() -> str:
    """Возвращает читаемую строку для GUI о доступности CUDA."""
    if _cuda_available():
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout.strip():
                name = r.stdout.strip().splitlines()[0].strip()
                return f"CUDA доступна: {name}"
        except Exception:  # noqa: BLE001
            pass
        return "CUDA доступна"
    return "CUDA не найдена — Whisper будет на CPU"
