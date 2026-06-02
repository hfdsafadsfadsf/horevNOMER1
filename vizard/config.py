"""Конфигурация и хранение пользовательских настроек."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


APP_DIR = Path.home() / ".vizard_clone"
APP_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = APP_DIR / "config.json"
TEMP_DIR = APP_DIR / "temp"
OUTPUT_DIR = APP_DIR / "output"
PRESETS_DIR = APP_DIR / "presets"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = PROJECT_ROOT / "resources" / "fonts"


@dataclass
class SubtitleStyle:
    """Стиль субтитров. Можно задать вручную или загрузить из шаблона."""

    template_id: str = "tiktok_classic"

    font: str = "Montserrat-Black.ttf"
    font_size: int = 78
    bold: bool = True
    italic: bool = False
    letter_spacing: float = 0.0

    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD800"
    outline_color: str = "#000000"
    outline_width: int = 5

    box_style: str = "outline"
    shadow: int = 2
    shadow_color: str = "#000000"
    back_color: str = "#000000"
    back_alpha: int = 200
    back_padding: int = 20

    uppercase: bool = True
    word_highlight: bool = True
    karaoke_mode: str = "color"
    max_words_per_line: int = 4
    position_v: str = "center"
    margin_v_pct: float = 0.10

    silence_gap_sec: float = 0.4


@dataclass
class TitleConfig:
    """Заголовок поверх видео."""

    enabled: bool = False
    template_id: str = "top_banner_dark"
    text_mode: str = "ai"
    custom_text: str = ""

    font: str = "Montserrat-Black.ttf"
    font_size: int = 58
    primary_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 2
    back_color: str = "#000000"
    back_alpha: int = 220
    back_padding: int = 20
    uppercase: bool = True
    variant: str = "top_banner"
    margin_v_pct: float = 0.03
    duration: float = 0.0


@dataclass
class OverlayConfig:
    """Логотип/обложка поверх видео."""

    enabled: bool = False
    image_path: str = ""
    position: str = "top_right"
    size_pct: float = 12.0
    opacity: float = 0.85
    margin_pct: float = 3.0


@dataclass
class MusicConfig:
    mode: str = "none"
    common_track: Optional[str] = None
    per_clip_tracks: list[str] = field(default_factory=list)
    volume: float = 0.15
    sidechain: bool = True


VALID_LENGTH_PRESETS = ("0-15", "15-30", "30-59", "auto")


@dataclass
class ClipConfig:
    length_preset: str = "auto"
    min_clip_count: int = 3
    max_clip_count: int = 10
    language: str = "auto"


@dataclass
class AppConfig:
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    whisper_model_size: str = "small"
    whisper_device: str = "auto"  # auto | cuda | cpu — auto детектит NVIDIA GPU
    whisper_compute_type: str = "auto"  # auto подбирает float16 на cuda, int8 на cpu
    clip: ClipConfig = field(default_factory=ClipConfig)
    subtitle: SubtitleStyle = field(default_factory=SubtitleStyle)
    title: TitleConfig = field(default_factory=TitleConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    output_dir: str = str(OUTPUT_DIR)
    use_ai: bool = True

    # Face tracking настройки (9:16 кроп строго по лицу)
    face_tracking: bool = True   # двигать кроп вслед за лицом
    face_zoom: bool = True       # приближаться если лицо мелкое
    multi_face_mode: str = "auto"  # auto | single | split — что делать при 2 лицах
    speaker_detection: bool = True  # анализ амплитуды для определения говорящего

    # GPU ускорение через ffmpeg NVENC/AMF/QSV
    use_gpu: bool = True   # автодетект; если GPU не найдена — fallback на CPU
    encode_preset: str = "medium"  # для GPU маппится в p1..p7

    # Очистка субтитров (точки/запятые в whisper-аутпуте)
    strip_subtitle_punct: bool = True

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                with CONFIG_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg = cls()
                _apply_dict(cfg, data)
                if not cfg.deepseek_api_key:
                    cfg.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
                return cfg
            except (OSError, json.JSONDecodeError):
                pass
        cfg = cls()
        cfg.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        return cfg

    def save(self) -> None:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def export_preset(self, path: Path) -> None:
        """Сохраняет визуальные настройки (без API ключа) в файл-пресет."""
        preset = {
            "subtitle": asdict(self.subtitle),
            "title": asdict(self.title),
            "overlay": asdict(self.overlay),
            "music": {
                "mode": self.music.mode,
                "volume": self.music.volume,
                "sidechain": self.music.sidechain,
            },
            "clip": asdict(self.clip),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2, ensure_ascii=False)

    def import_preset(self, path: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        _apply_dict(self, data)
        return True


def _apply_dict(cfg: AppConfig, data: dict) -> None:
    for k, v in data.items():
        if k == "clip" and isinstance(v, dict):
            cfg.clip = _safe_dataclass(ClipConfig, v)
        elif k == "subtitle" and isinstance(v, dict):
            cfg.subtitle = _safe_dataclass(SubtitleStyle, v)
        elif k == "title" and isinstance(v, dict):
            cfg.title = _safe_dataclass(TitleConfig, v)
        elif k == "overlay" and isinstance(v, dict):
            cfg.overlay = _safe_dataclass(OverlayConfig, v)
        elif k == "music" and isinstance(v, dict):
            existing = asdict(cfg.music)
            existing.update(v)
            cfg.music = _safe_dataclass(MusicConfig, existing)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)


def _safe_dataclass(cls, data: dict):
    """Создаёт dataclass игнорируя неизвестные поля (для backward-compat)."""
    valid_fields = {f for f in cls.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)
