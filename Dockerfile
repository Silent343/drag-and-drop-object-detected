# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Install Python deps in a separate layer so they're cached between deploys.
FROM python:3.11-slim AS builder

WORKDIR /build

# OpenCV needs libGL; install it first
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the YOLO model weights so they're baked into the image.
# This avoids a download on every cold start at Railway.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy runtime libraries from builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy the cached YOLO weights (in ~/.config/Ultralytics or similar)
COPY --from=builder /root /root

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/

# Railway injects $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE $PORT

# Healthcheck so Railway knows when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
