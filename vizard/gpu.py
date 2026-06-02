"""
Авто-детекция GPU и подбор аргументов ffmpeg для NVENC/AMF/QSV.

Главная функция: gpu_ffmpeg_args() — возвращает список аргументов кодека/preset/qp,
которые подставляются вместо libx264 args когда GPU доступен.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

LOG = logging.getLogger(__name__)

# Кешируем результат детекта чтобы не дёргать ffmpeg/ Nvidia-smi каждый раз
_cached_gpu: Optional["GpuCapability"] = None


@dataclass
class GpuCapability:
    """Что доступно на этом железе."""
    nvenc: bool = False           # NVIDIA h264_nvenc
    nvdec: bool = False           # NVIDIA hardware decode (cuda)
    qsv: bool = False             # Intel QuickSync
    amf: bool = False             # AMD AMF
    nvidia_gpu_name: str = ""     # читаемое имя видеокарты для лога

    @property
    def any_encoder(self) -> bool:
        return self.nvenc or self.qsv or self.amf

    def label(self) -> str:
        parts: list[str] = []
        if self.nvenc:
            parts.append(f"NVENC ({self.nvidia_gpu_name})" if self.nvidia_gpu_name else "NVENC")
        if self.qsv:
            parts.append("Intel QSV")
        if self.amf:
            parts.append("AMD AMF")
        if not parts:
            return "CPU (libx264)"
        return " + ".join(parts)


def _run(cmd: List[str], timeout: float = 5.0) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        LOG.debug("Команда %s упала: %s", cmd[:2], e)
        return -1, "", str(e)


def detect_gpu(force: bool = False) -> GpuCapability:
    """Определяет доступные GPU-кодеки. Кешируется."""
    global _cached_gpu
    if _cached_gpu is not None and not force:
        return _cached_gpu

    cap = GpuCapability()

    # 1) ffmpeg должен быть в PATH
    if shutil.which("ffmpeg") is None:
        _cached_gpu = cap
        return cap

    # 2) Список доступных энкодеров
    rc, out, _ = _run(["ffmpeg", "-hide_banner", "-encoders"], timeout=5)
    if rc == 0:
        has_nvenc = "h264_nvenc" in out
        has_qsv = "h264_qsv" in out
        has_amf = "h264_amf" in out
    else:
        has_nvenc = has_qsv = has_amf = False

    # 3) NVENC: проверяем реальное наличие через nvidia-smi (на Linux/Windows)
    if has_nvenc:
        rc, out_smi, _ = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=3)
        if rc == 0 and out_smi.strip():
            cap.nvenc = True
            cap.nvdec = True
            cap.nvidia_gpu_name = out_smi.strip().splitlines()[0].strip()
        else:
            # ffmpeg видит NVENC но nvidia-smi нет → попробуем тест-кодирование
            cap.nvenc = _test_nvenc()
            cap.nvdec = cap.nvenc

    # 4) AMD AMF — на Windows доступен через AMD драйвера
    if has_amf:
        cap.amf = _test_amf()

    # 5) Intel QSV
    if has_qsv:
        cap.qsv = _test_qsv()

    _cached_gpu = cap
    LOG.info("Определён GPU: %s", cap.label())
    return cap


def _test_nvenc() -> bool:
    """Пробное микро-кодирование через h264_nvenc."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=320x240:r=30:d=0.1",
        "-c:v", "h264_nvenc", "-preset", "p1",
        "-f", "null", "-",
    ]
    rc, _, _ = _run(cmd, timeout=10)
    return rc == 0


def _test_amf() -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=320x240:r=30:d=0.1",
        "-c:v", "h264_amf",
        "-f", "null", "-",
    ]
    rc, _, _ = _run(cmd, timeout=10)
    return rc == 0


def _test_qsv() -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=320x240:r=30:d=0.1",
        "-c:v", "h264_qsv",
        "-f", "null", "-",
    ]
    rc, _, _ = _run(cmd, timeout=10)
    return rc == 0


def gpu_encode_args(
    use_gpu: bool,
    crf_or_cq: int = 20,
    preset: str = "slow",
) -> List[str]:
    """
    Возвращает ffmpeg-аргументы для кодека и качества.

    use_gpu=True → пробует NVENC → AMF → QSV → libx264 fallback.
    use_gpu=False → libx264.

    crf_or_cq: для libx264 это CRF (0..51, меньше=лучше), для NVENC — это CQ.
    preset: для libx264: ultrafast..veryslow; для NVENC: p1..p7 (мы маппим).
    """
    if not use_gpu:
        return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf_or_cq), "-pix_fmt", "yuv420p"]

    cap = detect_gpu()

    # NVENC: главный приоритет
    if cap.nvenc:
        nvenc_preset = _x264_to_nvenc_preset(preset)
        return [
            "-c:v", "h264_nvenc",
            "-preset", nvenc_preset,
            "-rc", "vbr",
            "-cq", str(crf_or_cq),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]

    # AMD AMF
    if cap.amf:
        return [
            "-c:v", "h264_amf",
            "-quality", _x264_to_amf_quality(preset),
            "-rc", "cqp",
            "-qp_i", str(crf_or_cq),
            "-qp_p", str(crf_or_cq),
            "-pix_fmt", "yuv420p",
        ]

    # Intel QSV
    if cap.qsv:
        return [
            "-c:v", "h264_qsv",
            "-preset", _x264_to_qsv_preset(preset),
            "-global_quality", str(crf_or_cq),
            "-pix_fmt", "yuv420p",
        ]

    # Fallback
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf_or_cq), "-pix_fmt", "yuv420p"]


def _x264_to_nvenc_preset(preset: str) -> str:
    """libx264 preset → NVENC p1-p7 (p1=fastest p7=slowest/best)."""
    mapping = {
        "ultrafast": "p1",
        "superfast": "p1",
        "veryfast": "p2",
        "faster": "p3",
        "fast": "p4",
        "medium": "p4",
        "slow": "p5",
        "slower": "p6",
        "veryslow": "p7",
    }
    return mapping.get(preset, "p5")


def _x264_to_amf_quality(preset: str) -> str:
    """libx264 → AMF: speed | balanced | quality."""
    if preset in ("ultrafast", "superfast", "veryfast", "faster", "fast"):
        return "speed"
    if preset in ("slower", "veryslow"):
        return "quality"
    return "balanced"


def _x264_to_qsv_preset(preset: str) -> str:
    """libx264 → QSV: veryfast..veryslow."""
    if preset in ("ultrafast", "superfast"):
        return "veryfast"
    if preset == "veryfast":
        return "faster"
    if preset == "faster":
        return "fast"
    if preset == "fast":
        return "medium"
    if preset in ("slower", "veryslow"):
        return "slower"
    return preset


def gpu_decode_args(use_gpu: bool) -> List[str]:
    """
    Аргументы для входного потока чтобы декодировать на GPU.
    Подставляются перед '-i input.mp4'.

    NB: NVDEC требует чтобы фильтры тоже были на GPU (или явная hwdownload),
    что усложняет цепочку. По умолчанию мы декодим на CPU а кодим на GPU —
    это даёт основной выигрыш и не ломает фильтры (crop/scale/ass/overlay).
    """
    # На текущей архитектуре фильтры все CPU — hardware decode тут не выиграет
    # (всё равно надо hwdownload). Поэтому возвращаем пусто.
    _ = use_gpu
    return []
