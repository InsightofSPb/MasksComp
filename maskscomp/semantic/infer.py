from __future__ import annotations

from pathlib import Path

import cv2

from .model_loader import FacadeSemanticPredictor
from .types import SemanticPrediction


def predict_image(predictor: FacadeSemanticPredictor, image_path: Path) -> tuple[SemanticPrediction, tuple[int, int]]:
    """Run semantic predictor on one image path."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    pred = predictor.predict(image)
    h, w = image.shape[:2]
    return pred, (h, w)
