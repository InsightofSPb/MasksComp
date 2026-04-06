from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemanticSample:
    """Input sample metadata for semantic inference/export."""

    sample_id: str
    image_path: Path
    meta: dict[str, Any]


@dataclass
class SemanticPrediction:
    """Facade semantic prediction outputs for a single image."""

    mask: np.ndarray
    features: np.ndarray
    probs_or_logits: np.ndarray | None = None
    class_ids: list[int] | None = None
    class_names: list[str] | None = None
    feature_grid_hw: tuple[int, int] | None = None
    source_model: str | None = None
