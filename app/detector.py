"""
Object detector: the same smart-validation logic as object_detector.py,
adapted to run on static images instead of webcam frames.

The per-class confidence thresholds, aspect-ratio rules, max-area checks
and Spanish translations are kept exactly as in the original script so the
API produces consistent results with the local demo.
"""

from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from app.schemas import Detection

# ── Per-class config (copied from object_detector.py) ────────────────────────

CONFIDENCE_PER_CLASS: dict[str, float] = {
    "cell phone": 0.72,
    "remote":     0.65,
    "keyboard":   0.60,
    "book":       0.55,
    "cup":        0.50,
    "bottle":     0.50,
    "laptop":     0.55,
    "mouse":      0.60,
    "person":     0.50,
    "clock":      0.55,
    "scissors":   0.58,
    "vase":       0.50,
    "teddy bear": 0.50,
    "chair":      0.50,
    "tv":         0.55,
    "dog":        0.50,
    "cat":        0.50,
}

ASPECT_RATIO_RULES: dict[str, tuple[float, float]] = {
    "cell phone": (0.15, 2.70),
    "remote":     (0.12, 0.82),
    "bottle":     (0.15, 0.60),
    "cup":        (0.55, 1.80),
    "laptop":     (1.10, 2.80),
    "tv":         (1.30, 3.00),
    "keyboard":   (2.20, 6.00),
    "book":       (0.40, 2.10),
    "mouse":      (0.55, 1.55),
    "clock":      (0.65, 1.55),
    "scissors":   (0.20, 1.10),
    "vase":       (0.20, 0.78),
}

MAX_AREA_FRACTION: dict[str, float] = {
    "cell phone": 0.30,
    "remote":     0.25,
    "cup":        0.25,
    "bottle":     0.20,
    "mouse":      0.15,
    "book":       0.40,
    "laptop":     0.70,
    "keyboard":   0.60,
    "scissors":   0.20,
    "clock":      0.25,
}

# BGR colors → converted to hex for the frontend
CLASS_COLORS_BGR: dict[str, tuple] = {
    "person":      (200, 200,  50),
    "cell phone":  ( 50, 120, 255),
    "laptop":      (255, 180,   0),
    "book":        ( 80, 200,  80),
    "cup":         (  0, 200, 230),
    "bottle":      ( 30, 160, 255),
    "mouse":       (200,  80, 200),
    "keyboard":    (200, 150,  50),
    "remote":      (100, 100, 255),
    "clock":       (255, 255,  80),
    "vase":        (  0, 230, 180),
    "scissors":    (180,  50, 180),
    "teddy bear":  (200, 140,  80),
    "chair":       (120, 180, 120),
    "bed":         (180, 120, 180),
    "tv":          (255, 100,  50),
    "dog":         ( 50, 200, 150),
    "cat":         (150, 220,  80),
}
DEFAULT_COLOR_BGR = (160, 160, 160)

TRADUCCIONES: dict[str, str] = {
    "cup": "Taza", "cell phone": "Celular", "laptop": "Laptop",
    "book": "Libro", "bottle": "Botella", "mouse": "Mouse",
    "keyboard": "Teclado", "remote": "Control remoto", "clock": "Reloj",
    "vase": "Florero", "scissors": "Tijeras", "teddy bear": "Peluche",
    "chair": "Silla", "bed": "Cama", "desk": "Escritorio", "tv": "TV",
    "person": "Persona", "dog": "Perro", "cat": "Gato",
    "backpack": "Mochila", "umbrella": "Paraguas", "handbag": "Bolso",
    "sports ball": "Pelota", "wine glass": "Copa", "fork": "Tenedor",
    "knife": "Cuchillo", "spoon": "Cuchara", "bowl": "Tazón",
    "banana": "Plátano", "apple": "Manzana", "orange": "Naranja",
    "sandwich": "Sándwich", "pizza": "Pizza", "donut": "Dona",
    "cake": "Pastel", "couch": "Sofá", "potted plant": "Planta",
    "dining table": "Mesa", "microwave": "Microondas", "oven": "Horno",
    "sink": "Lavabo", "refrigerator": "Refrigerador",
}


def _bgr_to_hex(bgr: tuple) -> str:
    """Converts a BGR tuple to an HTML hex color string."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _get_color(class_name: str) -> tuple:
    return CLASS_COLORS_BGR.get(class_name, DEFAULT_COLOR_BGR)


def _get_threshold(class_name: str, global_conf: float) -> float:
    return CONFIDENCE_PER_CLASS.get(class_name, global_conf)


def _valid_aspect_ratio(class_name: str, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Checks if the bounding box aspect ratio matches the expected shape."""
    if class_name not in ASPECT_RATIO_RULES:
        return True
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = w / h
    min_r, max_r = ASPECT_RATIO_RULES[class_name]
    if not (min_r <= ratio <= max_r):
        return False
    if class_name == "cell phone" and 0.72 <= ratio <= 1.30:
        return False  # anti-square rule (chargers)
    return True


def _valid_area(class_name: str, x1: int, y1: int,
                x2: int, y2: int, fw: int, fh: int) -> bool:
    """Checks the bounding box doesn't occupy an unrealistic area fraction."""
    if class_name not in MAX_AREA_FRACTION:
        return True
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = max(fw * fh, 1)
    return (box_area / frame_area) <= MAX_AREA_FRACTION[class_name]


def _is_valid(class_name: str, conf: float,
              x1: int, y1: int, x2: int, y2: int,
              fw: int, fh: int, global_conf: float) -> bool:
    """Runs all three validation layers on one detection."""
    if conf < _get_threshold(class_name, global_conf):
        return False
    if not _valid_aspect_ratio(class_name, x1, y1, x2, y2):
        return False
    if not _valid_area(class_name, x1, y1, x2, y2, fw, fh):
        return False
    return True


def _draw_boxes(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Draws bounding boxes and labels on a copy of the image.

    Args:
        image_bgr: The original image in BGR format.
        detections: The validated detections to draw.

    Returns:
        Annotated BGR image.
    """
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bgr = _get_color(det.class_name_en)
        label = f"{det.class_name_es}  {det.confidence:.0%}"

        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(y1 - th - 10, 0)
        cv2.rectangle(out, (x1, label_y1), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class ObjectDetectorAPI:
    """
    Wraps the YOLO model with smart validation for single-image inference.

    Loaded once at startup and reused across requests, so the model weights
    are not re-read on every call.
    """

    def __init__(self, model_name: str = "yolov8s.pt", global_conf: float = 0.40) -> None:
        """
        Args:
            model_name: YOLO weights file (downloaded automatically on first run).
            global_conf: Fallback confidence threshold for classes without a
                         specific threshold in CONFIDENCE_PER_CLASS.
        """
        self.model = YOLO(model_name)
        self.global_conf = global_conf
        self.model_name = model_name

    def detect(self, image_bytes: bytes) -> tuple[list[Detection], str, int, int]:
        """
        Runs detection on raw image bytes.

        Args:
            image_bytes: The uploaded image in any format PIL can read.

        Returns:
            A tuple of:
            - list of validated Detection objects (sorted by confidence desc)
            - base64-encoded JPEG of the annotated image
            - image width in pixels
            - image height in pixels
        """
        # Decode image → numpy BGR for OpenCV / YOLO
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        fh, fw = img_bgr.shape[:2]

        results = self.model(img_bgr, conf=self.global_conf, verbose=False)
        boxes = results[0].boxes

        detections: list[Detection] = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]

                if not _is_valid(class_name, conf, x1, y1, x2, y2, fw, fh, self.global_conf):
                    continue

                detections.append(Detection(
                    class_name_en=class_name,
                    class_name_es=TRADUCCIONES.get(class_name, class_name.capitalize()),
                    confidence=round(conf, 4),
                    bbox=[x1, y1, x2, y2],
                    color_hex=_bgr_to_hex(_get_color(class_name)),
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        annotated = _draw_boxes(img_bgr, detections)

        # Encode annotated image to base64 JPEG
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode()

        return detections, b64, fw, fh
