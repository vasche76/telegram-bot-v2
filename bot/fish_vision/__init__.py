"""
Fish Vision Module — two-stage dedicated fish recognition pipeline.

Stage A (Detector): filters the image into one of:
    whole_fish | fish_part | lure | fry | no_fish

Stage B (Classifier): classifies whole-fish images into:
    pike | taimen | grayling | whitefish | perch | unknown_fish

Only catches that pass BOTH stages with sufficient confidence are
allowed into catch statistics. Bad detections are logged and rejected.
"""

from bot.fish_vision.pipeline import FishVisionPipeline, FishAnalysisResult

__all__ = ["FishVisionPipeline", "FishAnalysisResult"]

# Optional local-model backends — only available when torch/ultralytics are installed.
# Import them lazily so the bot starts normally even without those packages.
try:
    from bot.fish_vision.local_detector import LocalYOLODetector
    __all__ = __all__ + ["LocalYOLODetector"]
except Exception:
    pass

try:
    from bot.fish_vision.local_classifier import LocalEfficientNetClassifier
    __all__ = __all__ + ["LocalEfficientNetClassifier"]
except Exception:
    pass
