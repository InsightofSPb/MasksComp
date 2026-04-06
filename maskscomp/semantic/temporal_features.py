from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .tiling import Tile, generate_tiles


def mask_change_density(prev_mask: np.ndarray, cur_mask: np.ndarray) -> float:
    """Fraction of changed class-id pixels between two masks."""
    if prev_mask.shape != cur_mask.shape:
        raise ValueError(f"Mask shape mismatch: {prev_mask.shape} != {cur_mask.shape}")
    return float(np.mean(prev_mask != cur_mask))


def class_histogram(mask: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    if num_classes is None:
        num_classes = int(mask.max(initial=0)) + 1
    hist = np.bincount(mask.reshape(-1).astype(np.int64), minlength=num_classes).astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def class_histogram_drift_l1(prev_mask: np.ndarray, cur_mask: np.ndarray, num_classes: int | None = None) -> float:
    """L1 distance of normalized class histograms."""
    n = num_classes
    if n is None:
        n = int(max(prev_mask.max(initial=0), cur_mask.max(initial=0))) + 1
    hp = class_histogram(prev_mask, n)
    hc = class_histogram(cur_mask, n)
    return float(np.abs(hp - hc).sum())


def feature_cosine_distance(prev_vec: np.ndarray, cur_vec: np.ndarray) -> float:
    """Cosine distance in [0, 2], with exact 0 for identical vectors."""
    p = prev_vec.reshape(-1).astype(np.float64)
    c = cur_vec.reshape(-1).astype(np.float64)
    dn = np.linalg.norm(p) * np.linalg.norm(c)
    if dn == 0:
        return 0.0
    sim = float(np.dot(p, c) / dn)
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def tile_to_feature_slice(tile: Tile, image_hw: tuple[int, int], feat_hw: tuple[int, int]) -> tuple[slice, slice]:
    """Map image tile extents to feature-grid extents deterministically."""
    h, w = image_hw
    fh, fw = feat_hw
    y0 = int(math.floor(tile.y0 * fh / max(1, h)))
    y1 = int(math.ceil(tile.y1 * fh / max(1, h)))
    x0 = int(math.floor(tile.x0 * fw / max(1, w)))
    x1 = int(math.ceil(tile.x1 * fw / max(1, w)))
    y0, y1 = max(0, y0), min(fh, max(y0 + 1, y1))
    x0, x1 = max(0, x0), min(fw, max(x0 + 1, x1))
    return slice(y0, y1), slice(x0, x1)


def load_mask(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to load mask: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def load_feat_npz(path: str | Path, key: str = "feat") -> tuple[np.ndarray, tuple[int, int]]:
    d = np.load(path, allow_pickle=True)
    feat = np.asarray(d[key])
    fh = int(d["feat_h"]) if "feat_h" in d else int(feat.shape[0])
    fw = int(d["feat_w"]) if "feat_w" in d else int(feat.shape[1])
    return feat, (fh, fw)


def compute_pair_temporal_features(
    pair_id: str,
    prev_mask: np.ndarray,
    cur_mask: np.ndarray,
    prev_feat: np.ndarray,
    cur_feat: np.ndarray,
    prev_feat_hw: tuple[int, int],
    cur_feat_hw: tuple[int, int],
    tile_size: int,
    stride: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[list[dict[str, float | int | str]], np.ndarray, list[Tile]]:
    """Compute per-tile temporal semantic features and aggregated score heatmap."""
    if prev_mask.shape != cur_mask.shape:
        raise ValueError("prev_mask and cur_mask must share the same shape")
    h, w = prev_mask.shape[:2]
    tiles = generate_tiles(h, w, tile_size=tile_size, stride=stride)

    heat = np.zeros((h, w), dtype=np.float32)
    rows: list[dict[str, float | int | str]] = []

    for t in tiles:
        pm = prev_mask[t.y0 : t.y1, t.x0 : t.x1]
        cm = cur_mask[t.y0 : t.y1, t.x0 : t.x1]

        mcd = mask_change_density(pm, cm)
        chd = class_histogram_drift_l1(pm, cm)

        pys, pxs = tile_to_feature_slice(t, (h, w), prev_feat_hw)
        cys, cxs = tile_to_feature_slice(t, (h, w), cur_feat_hw)
        pv = prev_feat[pys, pxs].mean(axis=(0, 1))
        cv = cur_feat[cys, cxs].mean(axis=(0, 1))
        fcd = feature_cosine_distance(pv, cv)

        score = float(alpha * mcd + beta * chd + gamma * fcd)
        heat[t.y0 : t.y1, t.x0 : t.x1] = np.maximum(heat[t.y0 : t.y1, t.x0 : t.x1], score)

        rows.append(
            {
                "pair_id": pair_id,
                "tile_id": int(t.tile_id),
                "x0": int(t.x0),
                "y0": int(t.y0),
                "x1": int(t.x1),
                "y1": int(t.y1),
                "center_x": float(t.center_x),
                "center_y": float(t.center_y),
                "mask_change_density": float(mcd),
                "class_histogram_drift": float(chd),
                "feature_cosine_distance": float(fcd),
                "semantic_score": score,
            }
        )

    return rows, heat, tiles
