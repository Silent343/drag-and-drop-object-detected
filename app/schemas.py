"""
Request and response schemas for the YOLO detection API.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """
    A single validated object detection.

    Attributes:
        class_name_en: YOLO class name in English.
        class_name_es: Translated class name in Spanish.
        confidence: Detection confidence score (0–1).
        bbox: Bounding box as [x1, y1, x2, y2] in pixels.
        color_hex: Display color for this class (HTML hex).
    """
    class_name_en: str
    class_name_es: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(min_length=4, max_length=4)
    color_hex: str


class DetectResponse(BaseModel):
    """
    Response body for POST /detect.

    Attributes:
        detections: List of validated detections, sorted by confidence.
        total: Total number of validated detections.
        image_width: Width of the processed image in pixels.
        image_height: Height of the processed image in pixels.
        annotated_image_b64: Base64-encoded JPEG with boxes drawn on it.
    """
    detections: list[Detection]
    total: int
    image_width: int
    image_height: int
    annotated_image_b64: str


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str
    model: str
