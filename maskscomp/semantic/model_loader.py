from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import SemanticPrediction


@dataclass
class FacadeSemanticPredictor:
    """Facade semantic predictor API wrapper."""

    predict_fn: Any
    source_model: str

    def predict(self, image_bgr: np.ndarray) -> SemanticPrediction:
        pred = self.predict_fn(image_bgr)
        if isinstance(pred, SemanticPrediction):
            if not pred.source_model:
                pred.source_model = self.source_model
            return pred
        if not isinstance(pred, dict):
            raise TypeError(f"Predictor must return dict|SemanticPrediction, got {type(pred)}")
        mask = np.asarray(pred["mask"])
        feat = np.asarray(pred["features"])
        probs = None if pred.get("probs_or_logits") is None else np.asarray(pred["probs_or_logits"])
        grid = pred.get("feature_grid_hw")
        if grid is None and feat.ndim == 3:
            grid = (int(feat.shape[0]), int(feat.shape[1]))
        return SemanticPrediction(
            mask=mask,
            features=feat,
            probs_or_logits=probs,
            feature_grid_hw=grid,
            class_ids=pred.get("class_ids"),
            class_names=pred.get("class_names"),
            source_model=self.source_model,
        )


def _read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return json.loads(text)


def _basic_color_predictor(image_bgr: np.ndarray) -> SemanticPrediction:
    """Fallback heuristic predictor for self-contained runs without LPOSS code."""
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[v < 60] = 1  # dark/window-like
    mask[(s < 30) & (v > 140)] = 2  # wall-like
    mask[(s > 80) & (v > 80)] = 3  # vegetation/signage-like

    probs = np.stack(
        [
            (mask == 0).astype(np.float32),
            (mask == 1).astype(np.float32),
            (mask == 2).astype(np.float32),
            (mask == 3).astype(np.float32),
        ],
        axis=0,
    )
    grid_h = max(1, h // 8)
    grid_w = max(1, w // 8)
    small = cv2.resize(image_bgr, (grid_w, grid_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    feat = np.concatenate([small, small.mean(axis=2, keepdims=True)], axis=2)
    return SemanticPrediction(
        mask=mask,
        probs_or_logits=probs,
        features=feat,
        feature_grid_hw=(grid_h, grid_w),
        class_ids=[0, 1, 2, 3],
        class_names=["background", "dark", "wall", "colorful"],
        source_model="fallback_color_facade_v1",
    )


def _load_python_predictor(spec: str) -> Any:
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Configured predictor is not callable: {spec}")
    return fn


def _load_lposs_adapter(cfg: dict[str, Any], checkpoint: Path, device: str) -> Any:
    """Load LPOSS helper module callable without adding runtime repo dependency.

    Expected config keys:
      backend.kind = "lposs_helper"
      backend.helper_path = "/path/to/lposs_inference.py"
    """
    backend = cfg.get("backend", {})
    helper_path = Path(str(backend.get("helper_path", "")))
    if not helper_path.exists():
        raise FileNotFoundError(f"LPOSS helper_path does not exist: {helper_path}")

    spec = importlib.util.spec_from_file_location("maskscomp_lposs_helper", str(helper_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import helper module: {helper_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    build = getattr(mod, "build_lposs_inferencer", None)
    pred_map = getattr(mod, "lposs_predict_map", None)
    if not callable(build) or not callable(pred_map):
        raise RuntimeError("LPOSS helper module must expose build_lposs_inferencer and lposs_predict_map")

    dataset_config = cfg.get("dataset_config")
    if not dataset_config:
        raise ValueError("Config must provide dataset_config for lposs_helper backend")

    inferencer = build(
        config_path=Path(cfg.get("lposs_config", cfg.get("config", ""))),
        checkpoint_path=checkpoint,
        dataset_config=Path(dataset_config),
        device=device,
    )

    def _predict(image_bgr: np.ndarray) -> SemanticPrediction:
        out = pred_map(inferencer=inferencer, image_bgr=image_bgr)
        arr = np.asarray(out)
        if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
            probs = arr
            mask = np.argmax(arr, axis=0).astype(np.uint8)
        elif arr.ndim == 3:
            probs = np.moveaxis(arr, -1, 0)
            mask = np.argmax(arr, axis=2).astype(np.uint8)
        else:
            probs = None
            mask = arr.astype(np.uint8)
        feat = cv2.resize(image_bgr, (max(1, image_bgr.shape[1] // 8), max(1, image_bgr.shape[0] // 8))).astype(np.float32)
        return SemanticPrediction(mask=mask, probs_or_logits=probs, features=feat, feature_grid_hw=feat.shape[:2])

    return _predict


def load_facade_semantic_predictor(config_path: Path, checkpoint: Path, device: str = "cpu") -> FacadeSemanticPredictor:
    """Build a predictor with a simple unified API."""
    cfg = _read_config(config_path)
    backend = cfg.get("backend", {}) if isinstance(cfg, dict) else {}
    kind = str(backend.get("kind", "fallback")).lower()

    if kind == "python_callable":
        fn = _load_python_predictor(str(backend["callable"]))
        return FacadeSemanticPredictor(predict_fn=fn, source_model=str(backend.get("name", "python_callable")))
    if kind == "lposs_helper":
        fn = _load_lposs_adapter(cfg, checkpoint=checkpoint, device=device)
        return FacadeSemanticPredictor(predict_fn=fn, source_model=str(backend.get("name", "lposs_helper")))

    return FacadeSemanticPredictor(predict_fn=_basic_color_predictor, source_model="fallback_color_facade_v1")
