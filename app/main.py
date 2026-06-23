"""
FastAPI application for the YOLO object detection API.

Exposes two endpoints:
- POST /detect  — upload an image, get back detections + annotated image
- GET  /health  — liveness probe (used by Railway and load balancers)

The detector is loaded once at startup using FastAPI's lifespan pattern so
the model weights are not re-read on every request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from app.detector import ObjectDetectorAPI
from app.schemas import DetectResponse, HealthResponse

# ── Model configuration ───────────────────────────────────────────────────────

MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8s.pt")
GLOBAL_CONF = float(os.getenv("YOLO_CONF", "0.40"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

# Singleton detector, initialized in lifespan
detector: ObjectDetectorAPI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the YOLO model once at startup."""
    global detector
    print(f"Loading YOLO model: {MODEL_NAME} …")
    detector = ObjectDetectorAPI(model_name=MODEL_NAME, global_conf=GLOBAL_CONF)
    print("Model ready.")
    yield
    detector = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="YOLO Object Detection API",
    description=(
        "Smart object detector with per-class confidence thresholds, "
        "aspect-ratio validation, and area checks. "
        "Based on YOLOv8 + custom post-processing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """
    Liveness probe. Railway uses this to know the container is up.

    Returns:
        Status and model name.
    """
    return HealthResponse(status="ok", model=MODEL_NAME)


@app.post("/detect", response_model=DetectResponse, tags=["detection"])
async def detect(
    file: Annotated[UploadFile, File(description="Image to analyze (JPEG, PNG, WebP …)")]
) -> DetectResponse:
    """
    Detects objects in an uploaded image.

    Applies smart post-processing (per-class thresholds, aspect-ratio rules,
    area limits) to reduce false positives before returning results.

    Args:
        file: The uploaded image file.

    Returns:
        Detections sorted by confidence, plus the annotated image as base64.

    Raises:
        HTTPException 400: If the file is not an image or exceeds the size limit.
        HTTPException 503: If the model is not yet loaded.
    """
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Try again in a moment.",
        )

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File must be an image, got: {content_type}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image too large ({len(image_bytes) // 1024} KB). Max: 10 MB.",
        )

    try:
        detections, annotated_b64, width, height = detector.detect(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process image: {exc}",
        ) from exc

    return DetectResponse(
        detections=detections,
        total=len(detections),
        image_width=width,
        image_height=height,
        annotated_image_b64=annotated_b64,
    )


# ── Frontend (served at /) ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def frontend():
    """Serves the single-page frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
