"""
Local EfficientNet-based Stage B classifier.

Uses a fine-tuned EfficientNet (B0 or B2) model to classify fish species from an image.
Lazy-loads the model on first call (singleton) to avoid startup cost.

Falls back to GPT (raises RuntimeError) if:
    - torch or torchvision is not installed
    - the model file is not found

Species class order must match the training dataset label mapping (15 classes v2):
    0  → pike
    1  → taimen
    2  → grayling
    3  → whitefish
    4  → perch
    5  → brown_trout
    6  → rainbow_trout
    7  → atlantic_salmon
    8  → common_carp
    9  → crucian_carp
    10 → bream
    11 → roach
    12 → ide
    13 → wels_catfish
    14 → unknown_fish

The actual class list is loaded dynamically from data/fish_models/class_names_b.json
so that re-training with a different class set requires no code changes.
The model variant (B0/B2) is detected from data/fish_models/metadata.json.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from io import BytesIO
from typing import Optional

from bot.fish_vision.classifier import (
    ClassificationResult,
    SPECIES_MAP,
    SPECIES_CONFIDENCE_THRESHOLD,
)
from bot.fish_vision.models.config import CLASSIFIER_MODEL_PATH, CLASS_NAMES_B_PATH
from bot.utils.logging import get_logger

log = get_logger("fish_vision.local_classifier")

# Fallback class list — used only when class_names_b.json is absent.
# The real source of truth is class_names_b.json written by train_stage_b.py
# so that class order always matches the trained model without code changes.
# Updated to v2 taxonomy (15 classes).
_DEFAULT_SPECIES_CLASSES = [
    "pike",
    "taimen",
    "grayling",
    "whitefish",
    "perch",
    "brown_trout",
    "rainbow_trout",
    "atlantic_salmon",
    "common_carp",
    "crucian_carp",
    "bream",
    "roach",
    "ide",
    "wels_catfish",
    "unknown_fish",
]


def _detect_model_variant() -> str:
    """
    Detect EfficientNet variant (b0 or b2) from metadata.json.
    Falls back to 'b0' if metadata is absent or unreadable.
    """
    import os
    meta_path = os.path.join(os.path.dirname(CLASS_NAMES_B_PATH), "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            backend = meta.get("classifier", {}).get("backend", "efficientnet_b0")
            if "b2" in backend.lower():
                return "b2"
        except Exception:
            pass
    return "b0"


MIN_TRAIN_IMAGES = 15
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class InactiveClassFallbackError(RuntimeError):
    """Raised when inference lands on a class with insufficient training data.

    Signals the caller (classifier.py) to try the GPT fallback before
    returning unknown_fish — gives a better result when local data is sparse.
    """
    def __init__(self, species_key: str, confidence: float) -> None:
        super().__init__(f"inactive class '{species_key}' (conf={confidence:.2f})")
        self.species_key = species_key
        self.confidence = confidence


def _compute_inactive_classes(species_classes: list[str]) -> set[str]:
    """Return set of class names that have fewer than MIN_TRAIN_IMAGES training images.

    These classes are present in the model taxonomy but lack sufficient data —
    any inference result pointing to them is unreliable and should fall back to
    unknown_fish rather than being returned as a confident prediction.
    """
    from pathlib import Path

    # Derive stage_b dir from the known models dir path
    stage_b_dir = Path(CLASS_NAMES_B_PATH).parent.parent / "fish_dataset" / "stage_b"

    inactive: set[str] = set()
    for name in species_classes:
        if name == "unknown_fish":
            continue
        folder = stage_b_dir / name
        if not folder.exists():
            count = 0
        else:
            count = sum(1 for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if count < MIN_TRAIN_IMAGES:
            log.warning(
                f"Inference safety: class '{name}' has only {count} training images "
                f"(< {MIN_TRAIN_IMAGES} minimum) — marked inactive; predictions will "
                "fall back to unknown_fish"
            )
            inactive.add(name)

    return inactive


def _load_species_classes() -> list[str]:
    """Load species class list from class_names_b.json (written by train_stage_b.py).

    Falls back to _DEFAULT_SPECIES_CLASSES if the file is absent, so that the
    classifier can still be used when the model was placed manually without running
    the training script.
    """
    import os

    if os.path.isfile(CLASS_NAMES_B_PATH):
        try:
            with open(CLASS_NAMES_B_PATH, encoding="utf-8") as f:
                mapping: dict[str, str] = json.load(f)
            # mapping is {"0": "pike", "1": "taimen", ...} — sort by int key
            return [mapping[k] for k in sorted(mapping, key=lambda x: int(x))]
        except Exception as exc:
            log.warning(
                f"Could not load {CLASS_NAMES_B_PATH}: {exc}. "
                "Falling back to default species list."
            )
    return list(_DEFAULT_SPECIES_CLASSES)


class LocalEfficientNetClassifier:
    """Singleton wrapper around a fine-tuned EfficientNet-B0 species classifier."""

    _instance: Optional["LocalEfficientNetClassifier"] = None
    _model = None   # torch.nn.Module or None if not yet loaded
    _transforms = None  # torchvision transforms or None if not yet loaded
    _species_classes: list[str] = []  # populated at load time from class_names_b.json
    _inactive_classes: set[str] = set()  # classes with < MIN_TRAIN_IMAGES — unsafe for inference

    @classmethod
    def get_instance(cls) -> "LocalEfficientNetClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """
        Lazy-load the EfficientNet-B0 model from disk.

        Returns the loaded (eval-mode) model.
        Raises RuntimeError if torch/torchvision is not installed or the
        model file is missing — callers should catch this and fall back to GPT.
        """
        import os

        # Guard: check dependencies before anything else
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "torch / torchvision is not installed. "
                "Run `pip install torch torchvision` to enable local classification."
            ) from exc

        if not os.path.isfile(CLASSIFIER_MODEL_PATH):
            raise RuntimeError(
                f"EfficientNet model not found at {CLASSIFIER_MODEL_PATH}. "
                "Place classifier_v1.pt in data/fish_models/ to enable local classification."
            )

        import torch
        import torch.nn as nn
        from torchvision import transforms  # type: ignore

        # Load class names from file so training can update them without code changes
        species_classes = _load_species_classes()

        # Auto-detect model variant from metadata.json (b0 or b2)
        model_variant = _detect_model_variant()
        log.info(
            f"Loading EfficientNet-{model_variant.upper()} model from {CLASSIFIER_MODEL_PATH} "
            f"({len(species_classes)} classes)"
        )

        if model_variant == "b2":
            from torchvision.models import efficientnet_b2  # type: ignore
            model = efficientnet_b2(weights=None)
            in_features = model.classifier[1].in_features  # 1408
        else:
            from torchvision.models import efficientnet_b0  # type: ignore
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features  # 1280

        model.classifier[1] = nn.Linear(in_features, len(species_classes))
        state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location="cpu", weights_only=True)

        # Compatibility guard: verify checkpoint output dimension matches class list.
        # Mismatch means the checkpoint was trained on a different taxonomy than
        # class_names_b.json currently describes (e.g. 5-class checkpoint vs 15-class list).
        ckpt_out_features: int | None = None
        for key in ("classifier.1.weight", "classifier.1.bias"):
            if key in state_dict:
                ckpt_out_features = state_dict[key].shape[0]
                break
        if ckpt_out_features is not None and ckpt_out_features != len(species_classes):
            log.warning(
                f"Checkpoint output size ({ckpt_out_features}) does not match "
                f"class_names_b.json ({len(species_classes)} classes). "
                f"The checkpoint may have been trained on a different taxonomy. "
                f"Rebuilding head to match checkpoint size to avoid load error."
            )
            model.classifier[1] = nn.Linear(in_features, ckpt_out_features)
            # Trim or pad species_classes to checkpoint size so index mapping is safe
            if ckpt_out_features < len(species_classes):
                species_classes = species_classes[:ckpt_out_features]
            # (if ckpt_out_features > len(species_classes), extra indices map to "unknown_fish")

        model.load_state_dict(state_dict)
        model.eval()

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        log.info(f"EfficientNet-{model_variant.upper()} model loaded successfully")
        # Store species_classes and inactive set on the singleton
        self.__class__._species_classes = species_classes
        self.__class__._inactive_classes = _compute_inactive_classes(species_classes)
        if self.__class__._inactive_classes:
            log.warning(
                f"Inactive classes (will fall back to unknown_fish during inference): "
                f"{sorted(self.__class__._inactive_classes)}"
            )
        return model, val_transforms

    async def classify(
        self,
        image_url: str,
        caption: str = "",
    ) -> ClassificationResult:
        """
        Run EfficientNet inference on the image at image_url.

        Downloads the image, applies val_transforms, runs the model, and
        returns a ClassificationResult using softmax probabilities.

        Raises RuntimeError if the model is unavailable (missing file or
        missing torch/torchvision) so the caller can fall back to GPT.
        """
        # Lazy-load on first call
        if self._model is None:
            self._model, self._transforms = self._load_model()

        # Import here — safe because _load_model already verified they exist
        import torch
        import torch.nn.functional as F
        from PIL import Image  # type: ignore

        # Download image into memory (offloaded to thread pool — urllib is synchronous)
        log.debug(f"Downloading image for classification: {image_url}")
        loop = asyncio.get_running_loop()

        def _fetch() -> bytes:
            with urllib.request.urlopen(image_url, timeout=30) as response:
                return response.read()

        image_bytes = await loop.run_in_executor(None, _fetch)

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Apply preprocessing transforms
        tensor = self._transforms(image)          # (3, 224, 224)
        tensor = tensor.unsqueeze(0)              # (1, 3, 224, 224)

        # Run inference (CPU-bound — offload so we don't block asyncio)
        def _infer():
            with torch.no_grad():
                logits = self._model(tensor)      # (1, num_classes)
                probs = F.softmax(logits, dim=1)  # (1, num_classes)
            return probs.squeeze(0).tolist()

        probs_list = await loop.run_in_executor(None, _infer)

        # Use class names loaded from class_names_b.json at model-load time
        species_classes = self._species_classes or _DEFAULT_SPECIES_CLASSES
        best_idx = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))
        confidence = float(probs_list[best_idx])
        species_key = species_classes[best_idx] if best_idx < len(species_classes) else "unknown_fish"

        # Enforce threshold: low-confidence predictions → unknown_fish
        if confidence < SPECIES_CONFIDENCE_THRESHOLD and species_key != "unknown_fish":
            log.info(
                f"Local classifier: {species_key} downgraded to unknown_fish "
                f"(conf={confidence:.2f} < threshold={SPECIES_CONFIDENCE_THRESHOLD})"
            )
            species_key = "unknown_fish"

        # Inference safety: inactive classes (insufficient training data) → GPT fallback
        if species_key in self._inactive_classes:
            log.warning(
                f"Local classifier: predicted class '{species_key}' is inactive "
                f"(insufficient training data) — raising InactiveClassFallbackError for GPT retry"
            )
            raise InactiveClassFallbackError(species_key, confidence)

        log.info(
            f"Local classifier: {species_key} (conf={confidence:.2f})"
        )

        return ClassificationResult(
            species_key=species_key,
            species_ru=SPECIES_MAP.get(species_key, "Рыба (вид не определён)"),
            confidence=confidence,
            weight_kg_estimate=None,      # local model doesn't estimate weight
            length_cm_estimate=None,      # local model doesn't estimate length
            fish_count=1,                 # EfficientNet classifies single-fish images
            person_name_in_photo=None,    # local model can't detect names
            distinguishing_features="Local EfficientNet-B0 prediction",
            reasoning=f"Local model: {species_key} (prob={confidence:.2f})",
        )
