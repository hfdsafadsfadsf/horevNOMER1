#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/hfdsafadsfadsf/horevNOMER1.git"
INSTALL_DIR="/opt/horevNOMER1"

echo "=================================================="
echo "  🎬 Vizard Clone — Batch Server Installer"
echo "=================================================="

# ───── 1. Docker ─────
if ! command -v docker &>/dev/null; then
  echo "[1/7] Установка Docker..."
  curl -fsSL https://get.docker.com | sh
  apt-get install -y docker-compose-plugin git nano
else
  echo "[1/7] Docker уже установлен ✓"
fi

# ───── 2. Клонируем репо ─────
if [ ! -d "$INSTALL_DIR" ]; then
  echo "[2/7] Клонирую репозиторий в $INSTALL_DIR..."
  git clone "$REPO_URL" "$INSTALL_DIR"
else
  echo "[2/7] Обновляю репозиторий..."
  cd "$INSTALL_DIR" && git pull
fi
cd "$INSTALL_DIR"

# ───── 3. Создаём папки ─────
echo "[3/7] Создаю структуру..."
mkdir -p webapp/templates presets

# ───── 4. vizard/r2_uploader.py ─────
echo "[4/7] Генерирую файлы проекта..."

cat > vizard/r2_uploader.py <<'PYEOF'
"""Cloudflare R2 uploader."""
import os
import boto3
from botocore.client import Config
from pathlib import Path
from typing import Optional

class R2Uploader:
    def __init__(self):
        self.account_id = os.environ["R2_ACCOUNT_ID"]
        self.access_key = os.environ["R2_ACCESS_KEY_ID"]
        self.secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
        self.bucket = os.environ["R2_BUCKET"]
        self.prefix = os.environ.get("R2_PREFIX", "baget").strip("/")
        self.public_base = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")
        endpoint = f"https://{self.account_id}.r2.cloudflarestorage.com"
        self.client = boto3.client(
            "s3", endpoint_url=endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

    def upload_and_cleanup(self, local_path: str, subfolder: str = "") -> str:
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(local_path)
        key_parts = [self.prefix]
        if subfolder:
            key_parts.append(subfolder.strip("/"))
        key_parts.append(local.name)
        key = "/".join(p for p in key_parts if p)
        self.client.upload_file(str(local), self.bucket, key,
            ExtraArgs={"ContentType": "video/mp4"})
        try:
            local.unlink()
        except Exception as e:
            print(f"[R2] warning: couldn't delete {local}: {e}")
        return key

    def cleanup_source(self, source_video_path: str):
        try:
            p = Path(source_video_path)
            if p.exists():
                p.unlink()
            if p.parent.exists() and not any(p.parent.iterdir()):
                p.parent.rmdir()
        except Exception as e:
            print(f"[R2] cleanup_source warning: {e}")

    def public_url(self, key: str) -> Optional[str]:
        return f"{self.public_base}/{key}" if self.public_base else None
PYEOF

# ───── 5. vizard/queue_worker.py ─────
cat > vizard/queue_worker.py <<'PYEOF'
"""RQ job: process one URL with a random preset, upload to R2."""
import os
import random
import logging
from pathlib import Path
from vizard.config import AppConfig, PRESETS_DIR
from vizard.pipeline import run_pipeline
from vizard.r2_uploader import R2Uploader

logger = logging.getLogger("vizard.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _load_preset(preset_name: str) -> AppConfig:
    cfg = AppConfig.load()
    cfg.use_gpu = False
    cfg.whisper_device = "cpu"
    cfg.whisper_compute_type = "int8"
    if not cfg.deepseek_api_key:
        cfg.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    preset_path = Path(PRESETS_DIR) / f"{preset_name}.json"
    if preset_path.exists():
        cfg.import_preset(str(preset_path))
        logger.info(f"Loaded preset: {preset_name}")
    else:
        logger.warning(f"Preset not found: {preset_path}, using defaults")
    return cfg

def process_url(url: str, allowed_presets: list, batch_id: str = "") -> dict:
    chosen = random.choice(allowed_presets) if allowed_presets else "default"
    logger.info(f"[{batch_id}] Processing {url} with preset={chosen}")
    result = {"url": url, "preset": chosen, "clips": [], "status": "pending"}
    try:
        cfg = _load_preset(chosen)
        clips = run_pipeline(url, cfg)
        if not clips:
            result["status"] = "no_clips"
            return result
        uploader = R2Uploader()
        subfolder = f"{batch_id}/{chosen}" if batch_id else chosen
        source_path = None
        for clip in clips:
            try:
                source_path = clip.source_video
                key = uploader.upload_and_cleanup(clip.output_path, subfolder=subfolder)
                result["clips"].append({
                    "key": key,
                    "url": uploader.public_url(key),
                    "title": clip.suggestion.title if clip.suggestion else "",
                    "score": getattr(clip.suggestion, "viral_score", None) if clip.suggestion else None,
                    "size_mb": clip.size_mb,
                })
            except Exception as e:
                logger.exception(f"Failed to upload clip: {e}")
        if source_path:
            uploader.cleanup_source(str(source_path))
        result["status"] = "done"
        logger.info(f"[{batch_id}] DONE {url}: {len(result['clips'])} clips")
    except Exception as e:
        logger.exception(f"[{batch_id}] FAILED {url}: {e}")
        result["status"] = "error"
        result["error"] = str(e)
    return result
PYEOF

# ───── 6. webapp/main.py ─────
cat > webapp/main.py <<'PYEOF'
"""FastAPI web UI."""
import os
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from redis import Redis
from rq import Queue
from vizard.config import PRESETS_DIR
from vizard.queue_worker import process_url

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")
redis_conn = Redis(host=os.environ.get("REDIS_HOST", "redis"), port=6379)
queue = Queue("vizard", connection=redis_conn, default_timeout=3600)

def list_presets():
    p = Path(PRESETS_DIR)
    if not p.exists():
        return []
    return sorted([f.stem for f in p.glob("*.json")])

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html",
        {"request": request, "presets": list_presets()})

@app.post("/enqueue")
def enqueue(urls: str = Form(...), presets: list = Form(...)):
    url_list = [u.strip() for u in urls.replace(",", "\n").splitlines() if u.strip()]
    if not url_list:
        return JSONResponse({"error": "no urls"}, status_code=400)
    if not presets:
        return JSONResponse({"error": "no presets selected"}, status_code=400)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    for url in url_list:
        queue.enqueue(process_url, url, presets, batch_id, job_timeout=3600)
    return RedirectResponse(f"/batch/{batch_id}?count={len(url_list)}", status_code=303)

@app.get("/batch/{batch_id}", response_class=HTMLResponse)
def batch_status(request: Request, batch_id: str, count: int = 0):
    from rq.registry import FinishedJobRegistry, FailedJobRegistry, StartedJobRegistry
    started = StartedJobRegistry(queue=queue).count
    finished = FinishedJobRegistry(queue=queue).count
    failed = FailedJobRegistry(queue=queue).count
    pending = queue.count
    return templates.TemplateResponse("status.html", {
        "request": request, "batch_id": batch_id, "count": count,
        "pending": pending, "started": started,
        "finished": finished, "failed": failed,
    })
PYEOF

# ───── 7. HTML templates ─────
cat > webapp/templates/index.html <<'HTMLEOF'
<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><title>Vizard Batch</title>
<style>
body{font-family:-apple-system,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#0d1117;color:#c9d1d9}
h1{color:#58a6ff}
textarea{width:100%;min-height:300px;background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:12px;font-family:monospace;font-size:13px;border-radius:6px}
.presets{margin:20px 0;padding:16px;background:#161b22;border:1px solid #30363d;border-radius:6px}
.presets label{display:inline-block;margin:6px 12px 6px 0;padding:6px 12px;background:#21262d;border-radius:4px;cursor:pointer}
button{background:#238636;color:white;border:none;padding:12px 24px;font-size:16px;border-radius:6px;cursor:pointer;margin-top:12px}
.hint{color:#8b949e;font-size:13px;margin:8px 0}
</style></head><body>
<h1>🎬 Vizard Batch — кидай 500 ссылок</h1>
<form method="post" action="/enqueue">
  <label><b>YouTube ссылки</b> (по одной на строку):</label>
  <p class="hint">До 500 ссылок за раз. Каждое видео случайно получает один из выбранных пресетов.</p>
  <textarea name="urls" required placeholder="https://youtu.be/xxx&#10;https://youtu.be/yyy"></textarea>
  <div class="presets"><b>Пресеты для микса:</b>
    <p class="hint">Отметь пресеты, которые хочешь использовать.</p>
    {% for p in presets %}<label><input type="checkbox" name="presets" value="{{ p }}" checked> {{ p }}</label>{% endfor %}
    {% if not presets %}<p style="color:#f85149">⚠️ Нет пресетов в <code>presets/</code></p>{% endif %}
  </div>
  <button type="submit">🚀 В очередь</button>
</form></body></html>
HTMLEOF

cat > webapp/templates/status.html <<'HTMLEOF'
<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><title>Batch {{ batch_id }}</title>
<meta http-equiv="refresh" content="10">
<style>
body{font-family:-apple-system,sans-serif;max-width:800px;margin:40px auto;padding:0 20px;background:#0d1117;color:#c9d1d9}
h1{color:#58a6ff} .stat{padding:12px;background:#161b22;border:1px solid #30363d;border-radius:6px;margin:8px 0}
.num{font-size:32px;font-weight:bold;color:#58a6ff}
</style></head><body>
<h1>📊 Batch: {{ batch_id }}</h1>
<p>Закинуто задач: <b>{{ count }}</b></p>
<div class="stat">⏳ В очереди: <span class="num">{{ pending }}</span></div>
<div class="stat">🔄 В работе: <span class="num">{{ started }}</span></div>
<div class="stat">✅ Готово: <span class="num">{{ finished }}</span></div>
<div class="stat">❌ Ошибки: <span class="num">{{ failed }}</span></div>
<p><a href="/" style="color:#58a6ff">← Новая партия</a></p>
<p style="color:#8b949e">Обновится автоматически каждые 10 сек.</p>
</body></html>
HTMLEOF

# ───── 8. Dockerfile ─────
cat > Dockerfile <<'DOCKEREOF'
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /root/.vizard_clone/presets /root/.vizard_clone/temp /root/.vizard_clone/output
RUN cp -n presets/*.json /root/.vizard_clone/presets/ 2>/dev/null || true
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKEREOF

# ───── 9. docker-compose.yml ─────
cat > docker-compose.yml <<'COMPOSEEOF'
version: "3.9"
services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes: [redis_data:/data]
  web:
    build: .
    restart: unless-stopped
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [redis]
    volumes:
      - vizard_data:/root/.vizard_clone
      - ./presets:/root/.vizard_clone/presets:ro
  worker:
    build: .
    restart: unless-stopped
    command: rq worker vizard --url redis://redis:6379
    env_file: .env
    depends_on: [redis]
    volumes:
      - vizard_data:/root/.vizard_clone
      - ./presets:/root/.vizard_clone/presets:ro
    deploy:
      replicas: 2
volumes:
  redis_data:
  vizard_data:
COMPOSEEOF

# ───── 10. requirements.txt — добавляем недостающее ─────
for pkg in "fastapi==0.115.0" "uvicorn[standard]==0.30.6" "jinja2==3.1.4" \
           "python-multipart==0.0.9" "redis==5.0.8" "rq==1.16.2" "boto3==1.35.0"; do
  base="${pkg%%=*}"; base="${base%%[*}"
  if ! grep -qi "^${base}" requirements.txt 2>/dev/null; then
    echo "$pkg" >> requirements.txt
  fi
done

# ───── 11. дефолтный пресет ─────
if [ ! -f presets/tiktok_classic.json ]; then
cat > presets/tiktok_classic.json <<'JSONEOF'
{
  "subtitle": {"template_id": "tiktok_classic", "font": "Montserrat-Black.ttf",
    "font_size": 78, "bold": true, "primary_color": "#FFFFFF",
    "highlight_color": "#FFD800", "outline_color": "#000000",
    "outline_width": 5, "box_style": "outline", "uppercase": true,
    "word_highlight": true, "max_words_per_line": 4, "position_v": "center"},
  "title": {"enabled": false},
  "overlay": {"enabled": false},
  "music": {"mode": "none"},
  "clip": {"length_preset": "auto", "min_clip_count": 3, "max_clip_count": 10}
}
JSONEOF
fi

# ───── 12. .env интерактивно ─────
echo ""
echo "[5/7] Настройка .env"
if [ ! -f .env ]; then
  echo "Введи ключи (или нажми Enter и заполни позже через 'nano .env'):"
  read -p "R2_ACCOUNT_ID: " R2_ACCOUNT_ID
  read -p "R2_ACCESS_KEY_ID: " R2_ACCESS_KEY_ID
  read -p "R2_SECRET_ACCESS_KEY: " R2_SECRET_ACCESS_KEY
  read -p "R2_BUCKET [baget]: " R2_BUCKET; R2_BUCKET=${R2_BUCKET:-baget}
  read -p "R2_PUBLIC_BASE_URL (опционально, для публичных ссылок): " R2_PUBLIC_BASE_URL
  read -p "DEEPSEEK_API_KEY: " DEEPSEEK_API_KEY

  cat > .env <<ENVEOF
R2_ACCOUNT_ID=$R2_ACCOUNT_ID
R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY
R2_BUCKET=$R2_BUCKET
R2_PREFIX=baget
R2_PUBLIC_BASE_URL=$R2_PUBLIC_BASE_URL
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY
REDIS_HOST=redis
ENVEOF
  echo "✓ .env создан"
else
  echo "✓ .env уже существует, пропускаю"
fi

# ───── 13. Запуск ─────
echo "[6/7] Сборка контейнеров (5-10 минут в первый раз)..."
docker compose up -d --build

echo "[7/7] Готово!"
IP=$(curl -s ifconfig.me || hostname -I | awk '{print $1}')
echo ""
echo "=================================================="
echo "  ✅ ВСЁ ЗАПУЩЕНО"
echo "=================================================="
echo "  🌐 Открой:  http://$IP:8000"
echo "  📊 Логи:   docker compose logs -f worker"
echo "  🔄 Рестарт: cd $INSTALL_DIR && docker compose restart"
echo "=================================================="