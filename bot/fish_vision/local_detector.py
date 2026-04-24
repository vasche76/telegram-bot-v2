"""
Local YOLO-based Stage A detector.

Uses a YOLOv8 model trained on fish/lure images to classify the image content
into one of the Stage A output types. Lazy-loads the model on first call
(singleton) to avoid startup cost.

Falls back to GPT (raises RuntimeError) if:
    - the model file is not found
    - ultralytics is not installed

YOLO class ID → object_type mapping (must match training labels):
    0 → whole_fish
    1 → lure
    2 → fish_part
    3 → fry
    4 → no_fish
"""

from __future__ import annotations

import asyncio
import tempfile
import urllib.request
from typing import Optional

from bot.fish_vision.detector import DetectionResult, OBJECT_TYPES, FILTER_CONFIDENCE_THRESHOLD
from bot.fish_vision.models.config import DETECTOR_MODEL_PATH, YOLO_DETECTION_CONF
from bot.utils.logging import get_logger

log = get_logger("fish_vision.local_detector")

# Maps YOLO class indices to the Stage A object_type strings.
# Must match the class order used during model training.
_YOLO_CLASS_MAP: dict[int, str] = {
    0: "whole_fish",
    1: "lure",
    2: "fish_part",
    3: "fry",
    4: "no_fish",
}


class LocalYOLODetector:
    """Singleton wrapper around a YOLOv8 fish detection model."""

    _instance: Optional["LocalYOLODetector"] = None
    _model = None  # ultralytics.YOLO or None if not yet loaded

    @classmethod
    def get_instance(cls) -> "LocalYOLODetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """
        Lazy-load the YOLO model from disk.

        Returns the loaded model object.
        Raises RuntimeError if the model file is missing or ultralytics
        is not installed — callers should catch this and fall back to GPT.
        """
        import os

        if not os.path.isfile(DETECTOR_MODEL_PATH):
            raise RuntimeError(
                f"YOLO model not found at {DETECTOR_MODEL_PATH}. "
                "Place detector_v1.pt in data/fish_models/ to enable local detection."
            )

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. "
                "Run `pip install ultralytics` to enable local YOLO detection."
            ) from exc

        log.info(f"Loading YOLO model from {DETECTOR_MODEL_PATH}")
        model = YOLO(DETECTOR_MODEL_PATH)
        log.info("YOLO model loaded successfully")
        return model

    async def detect(self, image_url: str, caption: str = "") -> DetectionResult:
        """
        Run YOLO inference on the image at image_url.

        Downloads the image to a temporary file, runs the model, maps the
        top detection to a DetectionResult, then deletes the temp file.

        Raises RuntimeError if the model is unavailable (file missing or
        ultralytics not installed) so the caller can fall back to GPT.
        """
        # Lazy-load on first call
        if self._model is None:
            self._model = self._load_model()

        # Download image to a temporary file (offloaded to thread pool to avoid blocking the
        # asyncio event loop — urllib is synchronous).
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name

            log.debug(f"Downloading image from {image_url} to {tmp_path}")
            loop = asyncio.get_running_loop()

            def _download() -> None:
                with urllib.request.urlopen(image_url, timeout=30) as resp:
                    with open(tmp_path, "wb") as f:
                        f.write(resp.read())

            await loop.run_in_executor(None, _download)

            # Run YOLO inference (also CPU-bound — offload so we don't block asyncio)
            results = await loop.run_in_executor(
                None,
                lambda: self._model(tmp_path, conf=YOLO_DETECTION_CONF, verbose=False),
            )
        finally:
            # Always delete the temp file even if inference fails
            if tmp_path is not None:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Parse YOLO results
        # results is a list; take the first (only one image was provided)
        result = results[0]
        boxes = result.boxes  # ultralytics Boxes object

        if boxes is None or len(boxes) == 0:
            # No detections at all — treat as no_fish
            log.info("YOLO: no detections above confidence threshold → no_fish")
            return DetectionResult(
                object_type="no_fish",
                confidence=1.0 - YOLO_DETECTION_CONF,  # inverse of threshold as proxy
                fish_count=0,
                estimated_length_cm=None,
                reasoning=(
                    f"Local YOLO detector (yolov8n), class=no_fish, "
                    f"conf={1.0 - YOLO_DETECTION_CONF:.2f} "
                    "(no detections above confidence threshold)"
                ),
                raw_description="YOLO detection: no_fish — no objects detected above threshold",
            )

        # Pick the detection with the highest confidence score
        confs = boxes.conf.tolist()
        cls_ids = boxes.cls.tolist()

        best_idx = int(max(range(len(confs)), key=lambda i: confs[i]))
        best_conf = float(confs[best_idx])
        best_cls_id = int(cls_ids[best_idx])

        obj_type = _YOLO_CLASS_MAP.get(best_cls_id, "no_fish")
        if obj_type not in OBJECT_TYPES:
            obj_type = "no_fish"

        # Count whole_fish detections (class 0) above the confidence threshold
        fish_count = sum(
            1
            for i, cid in enumerate(cls_ids)
            if int(cid) == 0 and float(confs[i]) >= YOLO_DETECTION_CONF
        )

        log.info(
            f"YOLO: {obj_type} (conf={best_conf:.2f}, fish_count={fish_count})"
        )

        return DetectionResult(
            object_type=obj_type,
            confidence=best_conf,
            fish_count=fish_count,
            estimated_length_cm=None,  # YOLO doesn't estimate size without calibration
            reasoning=(
                f"Local YOLO detector (yolov8n), "
                f"class={obj_type}, conf={best_conf:.2f}"
            ),
            raw_description=(
                f"YOLO detection: {obj_type} at {best_conf:.0%} confidence"
            ),
        )
