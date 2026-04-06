from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .infer import predict_image
from .types import SemanticSample


PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 80, 80],
        [80, 255, 80],
        [80, 80, 255],
        [255, 255, 80],
        [255, 80, 255],
        [80, 255, 255],
    ],
    dtype=np.uint8,
)


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colors = PALETTE[mask.astype(np.int64) % len(PALETTE)]
    colors_bgr = colors[..., ::-1]
    mixed = (image_bgr.astype(np.float32) * (1.0 - alpha) + colors_bgr.astype(np.float32) * alpha).clip(0, 255)
    return mixed.astype(np.uint8)


def export_semantic_sample(
    sample: SemanticSample,
    predictor,
    output_root: Path,
    save_mask: bool,
    save_probs: bool,
    save_features: bool,
    save_overlay: bool,
    overwrite: bool,
) -> dict[str, object]:
    """Predict and export semantic artifacts for one sample."""
    mask_dir = output_root / "masks"
    probs_dir = output_root / "probs"
    feat_dir = output_root / "features"
    overlay_dir = output_root / "overlays"
    for d in (mask_dir, probs_dir, feat_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)

    mask_path = mask_dir / f"{sample.sample_id}.png"
    probs_path = probs_dir / f"{sample.sample_id}.npz"
    feat_path = feat_dir / f"{sample.sample_id}.npz"
    overlay_path = overlay_dir / f"{sample.sample_id}.jpg"

    if (not overwrite) and mask_path.exists() and feat_path.exists():
        return {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mask_path": str(mask_path),
            "probs_path": str(probs_path if probs_path.exists() else ""),
            "features_path": str(feat_path),
            "overlay_path": str(overlay_path if overlay_path.exists() else ""),
            "status": "skipped_exists",
            **sample.meta,
        }

    pred, (height, width) = predict_image(predictor, sample.image_path)
    mask = pred.mask.astype(np.uint16 if pred.mask.max(initial=0) > 255 else np.uint8)

    if save_mask:
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to write mask: {mask_path}")

    if save_probs and pred.probs_or_logits is not None:
        np.savez_compressed(
            probs_path,
            probs_or_logits=pred.probs_or_logits,
            sample_id=sample.sample_id,
            source_model=pred.source_model or "",
            height=height,
            width=width,
        )

    feat = pred.features
    grid_h, grid_w = pred.feature_grid_hw or feat.shape[:2]
    np.savez_compressed(
        feat_path,
        feat=feat,
        feat_h=int(grid_h),
        feat_w=int(grid_w),
        channels=int(feat.shape[-1] if feat.ndim == 3 else 1),
        source_model=pred.source_model or "",
        sample_id=sample.sample_id,
    )

    if save_overlay:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is not None:
            overlay = _overlay_mask(image, pred.mask)
            cv2.imwrite(str(overlay_path), overlay)

    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "mask_path": str(mask_path if save_mask else ""),
        "probs_path": str(probs_path if save_probs and pred.probs_or_logits is not None else ""),
        "features_path": str(feat_path),
        "overlay_path": str(overlay_path if save_overlay else ""),
        "status": "ok",
        "height": int(height),
        "width": int(width),
        "feature_h": int(grid_h),
        "feature_w": int(grid_w),
        "feature_channels": int(feat.shape[-1] if feat.ndim == 3 else 1),
        **sample.meta,
    }
