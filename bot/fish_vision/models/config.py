"""
Fish vision model backend configuration.

This file controls which backend is used for each pipeline stage.
Current production backend: "gpt" (OpenAI vision API).
Future production backend: "local" (YOLO + EfficientNet, trained on owner photos).

To enable a local model:
    1. Complete training (see training/README.md)
    2. Place model files in data/fish_models/
    3. Set FISH_DETECTOR_BACKEND=local and FISH_CLASSIFIER_BACKEND=local in .env
    4. pip install ultralytics for YOLO support

Directory layout for local models:
    data/fish_models/
        detector_v1.pt         # YOLO model for Stage A detection
        classifier_v1.pt       # EfficientNet for Stage B species classification
        class_names.json       # Maps class indices to species names
        model_card.md          # Training info, dataset used, accuracy metrics
"""

import os

# Stage A backend: "gpt" | "local"
DETECTOR_BACKEND: str = os.environ.get("FISH_DETECTOR_BACKEND", "gpt")

# Stage B backend: "gpt" | "local"
CLASSIFIER_BACKEND: str = os.environ.get("FISH_CLASSIFIER_BACKEND", "gpt")

# Paths for local models
MODEL_DIR: str = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "fish_models"
)
DETECTOR_MODEL_PATH: str = os.path.join(MODEL_DIR, "detector_v1.pt")
CLASSIFIER_MODEL_PATH: str = os.path.join(MODEL_DIR, "classifier_v1.pt")
CLASS_NAMES_PATH: str = os.path.join(MODEL_DIR, "class_names.json")

# Separate class-name files for Stage A (detector) and Stage B (classifier).
# Kept alongside CLASS_NAMES_PATH for backward compatibility.
CLASS_NAMES_A_PATH: str = os.path.join(MODEL_DIR, "class_names_a.json")
CLASS_NAMES_B_PATH: str = os.path.join(MODEL_DIR, "class_names_b.json")

# Minimum image size for reliable detection (pixels on shortest side)
MIN_IMAGE_SIZE_PX: int = 320

# YOLO detection confidence (for future local model)
YOLO_DETECTION_CONF: float = 0.45

# EfficientNet classification confidence (for future local model)
EFFICIENTNET_CONF: float = 0.65
