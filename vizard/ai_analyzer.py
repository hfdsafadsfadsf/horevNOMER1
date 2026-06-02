"""AI-анализ транскрипта через DeepSeek API: выбор лучших моментов для Reels."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .transcriber import Transcript


@dataclass
class ClipSuggestion:
    start: float
    end: float
    title: str
    viral_score: int = 70
    hashtags: list[str] | None = None
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


LENGTH_PRESETS = {
    "0-15": (5, 15),
    "15-30": (15, 30),
    "30-59": (30, 59),
    "auto": (10, 59),
}

MAX_CLIP_DURATION = 59.0


SYSTEM_PROMPT = """Ты — эксперт по созданию вирусных коротких видео для Instagram Reels / TikTok / YouTube Shorts.

Тебе дают транскрипт длинного видео с таймкодами. Выбери из него самые яркие, вирусные и самостоятельные фрагменты.

КРИТЕРИИ ВЫБОРА КЛИПОВ:
1. Сильное эмоциональное начало (hook) — первые 2 секунды должны зацепить
2. Завершённая мысль / законченная история / неожиданный поворот
3. Контраверсивные утверждения, лайфхаки, цифры, истории, мемы
4. НЕ выбирай: интро, аутро, благодарности, банальности
5. Сегменты в диапазоне длительности, указанном пользователем
6. Сегменты НЕ пересекаются
7. Границы клипа ДОЛЖНЫ быть на законченных предложениях — ни один клип не обрывается на полуслове

ТРЕБОВАНИЯ К ЗАГОЛОВКУ (title):
- ЭТО ХУК, НЕ ПЕРЕСКАЗ! МАКСИМУМ 5-7 слов, не больше 50 символов
- Это должен быть вопрос/провокация/цифра/шок-фраза
- Примеры ГОДНЫХ хуков:
  • "Я потерял $1М за день"
  • "3 правила миллионеров"
  • "Ты делаешь это неправильно"
  • "Никто об этом не говорит"
  • "95% людей проваливают"
  • "Секрет который скрывают"
- Примеры ПЛОХИХ заголовков (НЕ делай так):
  • "Спикер рассказывает историю про свой первый опыт в бизнесе" (длинно, пересказ)
  • "Обсуждение темы инвестирования" (скучно)
  • "Интервью с предпринимателем" (пересказ)

ФОРМАТ ОТВЕТА: только валидный JSON, без markdown-обрамления, в виде:
{
  "clips": [
    {
      "start": 12.3,
      "end": 45.7,
      "title": "3 правила миллионеров",
      "viral_score": 85,
      "reason": "Краткое объяснение почему это вирусно",
      "hashtags": ["#тег1", "#тег2", "#тег3"]
    }
  ]
}
"""


def _shorten_title(title: str, max_words: int = 7, max_chars: int = 50) -> str:
    """
    Жёсткое ограничение длины заголовка: hook, не пересказ.
    AI может проигнорировать инструкцию — обрежем сами.
    """
    title = title.strip().strip(".,;:")
    if not title:
        return "Clip"
    # Если AI вернул что-то с двоеточием в стиле "Тема: длинный пересказ" — берём
    # часть после двоеточия (обычно она и есть hook)
    if ":" in title and len(title.split(":", 1)[1].strip()) >= 5:
        title = title.split(":", 1)[1].strip()
    words = title.split()
    if len(words) > max_words:
        words = words[:max_words]
        title = " ".join(words)
    if len(title) > max_chars:
        title = title[: max_chars - 1].rstrip() + "…"
    return title


def _build_transcript_block(
    transcript: Transcript,
    seg_start: int = 0,
    seg_end: Optional[int] = None,
) -> str:
    segs = transcript.segments[seg_start:seg_end] if seg_end is not None else transcript.segments[seg_start:]
    lines: list[str] = []
    for seg in segs:
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
    return "\n".join(lines)


def _parse_json_response(text: str) -> dict:
    """Робастный JSON-парсер: пытается извлечь валидный JSON даже из оборванного ответа."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start_idx = text.find("{")
    if start_idx == -1:
        raise json.JSONDecodeError("Нет открывающей фигурной скобки", text, 0)

    try:
        return json.loads(text[start_idx:])
    except json.JSONDecodeError:
        pass

    candidate = text[start_idx:]
    candidate_fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
    try:
        return json.loads(candidate_fixed)
    except json.JSONDecodeError:
        pass

    clips_match = re.search(r'"clips"\s*:\s*\[', candidate)
    if clips_match:
        arr_start = clips_match.end()
        depth = 1
        i = arr_start
        while i < len(candidate) and depth > 0:
            ch = candidate[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        if depth > 0:
            partial = candidate[arr_start:i]
            last_obj_end = partial.rfind("}")
            if last_obj_end >= 0:
                cleaned = partial[: last_obj_end + 1]
                try:
                    items = json.loads("[" + cleaned + "]")
                    return {"clips": items}
                except json.JSONDecodeError:
                    pass

    raise json.JSONDecodeError("Не удалось спарсить JSON ответ AI", text, 0)


MAX_TRANSCRIPT_CHARS = 60000
CHUNK_OVERLAP_SEGS = 5


def _call_deepseek(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8000,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


def _split_transcript_chunks(transcript: Transcript) -> list[tuple[int, int]]:
    """Разбивает транскрипт на (seg_start, seg_end) чанки так, чтобы каждый был <= MAX_TRANSCRIPT_CHARS."""
    segs = transcript.segments
    if not segs:
        return []
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < len(segs):
        chars = 0
        j = i
        while j < len(segs):
            line_len = len(f"[{segs[j].start:.1f}-{segs[j].end:.1f}] {segs[j].text}") + 1
            if chars + line_len > MAX_TRANSCRIPT_CHARS and j > i:
                break
            chars += line_len
            j += 1
        chunks.append((i, j))
        if j >= len(segs):
            break
        i = max(i + 1, j - CHUNK_OVERLAP_SEGS)
    return chunks


def analyze_transcript(
    transcript: Transcript,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    length_preset: str = "auto",
    min_clips: int = 3,
    max_clips: int = 10,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[ClipSuggestion]:
    if OpenAI is None:
        raise RuntimeError("openai пакет не установлен. Запусти: pip install openai")
    if not api_key:
        raise RuntimeError(
            "Нет DEEPSEEK_API_KEY. Укажи его в GUI (Settings) или в переменной окружения."
        )

    lo, hi = LENGTH_PRESETS.get(length_preset, LENGTH_PRESETS["auto"])
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    chunks = _split_transcript_chunks(transcript)
    if not chunks:
        return []

    all_clips_raw: list[dict] = []

    for chunk_idx, (s_idx, e_idx) in enumerate(chunks):
        chunk_block = _build_transcript_block(transcript, s_idx, e_idx)
        clips_per_chunk = max(2, max_clips // len(chunks) + 2) if len(chunks) > 1 else max_clips

        user_prompt = (
            f"Язык видео: {transcript.language}\n"
            f"Общая длительность видео: {transcript.duration:.1f} секунд\n"
            f"Часть {chunk_idx + 1}/{len(chunks)} транскрипта.\n"
            f"Желаемая длина клипов: от {lo} до {hi} секунд (максимум {MAX_CLIP_DURATION:.0f} сек).\n"
            f"Нужно выбрать до {clips_per_chunk} лучших моментов из этой части.\n\n"
            f"Транскрипт (формат: [start-end] текст):\n{chunk_block}\n\n"
            f"Верни валидный JSON со списком клипов."
        )

        if progress_cb:
            progress_cb(
                f"DeepSeek: анализ части {chunk_idx + 1}/{len(chunks)} "
                f"({e_idx - s_idx} сегментов)..."
            )

        try:
            content = _call_deepseek(client, model, SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            if progress_cb:
                progress_cb(f"  DeepSeek API ошибка: {e}. Пропускаю часть.")
            continue

        try:
            data = _parse_json_response(content)
        except json.JSONDecodeError as e:
            if progress_cb:
                progress_cb(f"  Не спарсил JSON: {e}. Пропускаю часть.")
            continue

        chunk_clips = data.get("clips", [])
        all_clips_raw.extend(chunk_clips)
        if progress_cb:
            progress_cb(f"  Получено {len(chunk_clips)} клип(ов) из части {chunk_idx + 1}.")

    if not all_clips_raw:
        if progress_cb:
            progress_cb("AI не вернул ниодного клипа — fallback на эвристику.")
        return _fallback_heuristic(transcript, lo, hi, max_clips)

    suggestions: list[ClipSuggestion] = []
    seen_ranges: list[tuple[float, float]] = []
    for c in all_clips_raw:
        try:
            start = float(c["start"])
            end = float(c["end"])
            if end <= start or end - start < 3:
                continue
            if end - start > MAX_CLIP_DURATION:
                end = start + MAX_CLIP_DURATION

            overlap = False
            for s2, e2 in seen_ranges:
                if not (end <= s2 or start >= e2):
                    overlap = True
                    break
            if overlap:
                continue
            seen_ranges.append((start, end))

            suggestions.append(
                ClipSuggestion(
                    start=start,
                    end=end,
                    title=_shorten_title(str(c.get("title", "Clip"))),
                    viral_score=int(c.get("viral_score", 70)),
                    hashtags=list(c.get("hashtags") or []),
                    reason=str(c.get("reason", "")),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue

    if not suggestions:
        if progress_cb:
            progress_cb("AI не вернул валидных клипов — fallback на эвристику.")
        return _fallback_heuristic(transcript, lo, hi, max_clips)

    suggestions.sort(key=lambda s: s.viral_score, reverse=True)
    suggestions = suggestions[:max_clips]
    if progress_cb:
        progress_cb(f"AI выбрал {len(suggestions)} клип(ов) (из {len(all_clips_raw)} кандидатов).")
    return suggestions


def _fallback_heuristic(
    transcript: Transcript, lo: float, hi: float, max_clips: int
) -> list[ClipSuggestion]:
    """Простая эвристика без AI — нарезает видео равномерно."""
    if not transcript.segments:
        return []
    target_len = (lo + hi) / 2
    duration = transcript.duration
    n_clips = max(1, min(max_clips, int(duration // target_len)))
    step = duration / n_clips
    out: list[ClipSuggestion] = []
    for i in range(n_clips):
        start = i * step
        end = min(duration, start + target_len)
        if end - start < lo / 2:
            continue
        out.append(
            ClipSuggestion(
                start=start,
                end=end,
                title=f"Clip {i + 1}",
                viral_score=50,
                hashtags=[],
                reason="heuristic fallback",
            )
        )
    return out
