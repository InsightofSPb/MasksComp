from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from maskscomp.semantic.manifest import load_semantic_index, resolve_semantic_row
from maskscomp.semantic.temporal_features import (
    class_histogram_drift_l1,
    compute_pair_temporal_features,
    feature_cosine_distance,
    mask_change_density,
)
from maskscomp.semantic.tiling import generate_tiles


def test_semantic_index_resolution(tmp_path: Path) -> None:
    p = tmp_path / "index.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "mask_path", "features_path"])
        w.writeheader()
        w.writerow({"sample_id": "sid_1", "image_path": "a/facade_2001.jpg", "mask_path": "m1.png", "features_path": "f1.npz"})

    idx = load_semantic_index(p)
    assert resolve_semantic_row(idx, "sid_1")["mask_path"] == "m1.png"
    assert resolve_semantic_row(idx, "facade_2001.jpg")["features_path"] == "f1.npz"


def test_tile_indexing_consistency() -> None:
    t1 = generate_tiles(height=10, width=12, tile_size=4, stride=3)
    t2 = generate_tiles(height=10, width=12, tile_size=4, stride=3)
    assert t1 == t2
    assert t1[0].x0 == 0 and t1[0].y0 == 0
    assert t1[-1].x1 == 12 and t1[-1].y1 == 10


def test_mask_change_density_correctness() -> None:
    a = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    b = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert np.isclose(mask_change_density(a, b), 0.5)


def test_class_histogram_drift_correctness() -> None:
    a = np.array([[0, 0], [0, 1]], dtype=np.uint8)  # p=[0.75, 0.25]
    b = np.array([[0, 1], [1, 1]], dtype=np.uint8)  # q=[0.25, 0.75]
    drift = class_histogram_drift_l1(a, b, num_classes=2)
    assert np.isclose(drift, 1.0)


def test_feature_cosine_distance_sanity() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.isclose(feature_cosine_distance(v, v), 0.0)


def test_temporal_feature_shape_consistency() -> None:
    prev_mask = np.zeros((8, 8), dtype=np.uint8)
    cur_mask = np.zeros((8, 8), dtype=np.uint8)
    cur_mask[0:4, 0:4] = 1

    prev_feat = np.ones((4, 4, 2), dtype=np.float32)
    cur_feat = np.ones((4, 4, 2), dtype=np.float32)
    rows, heat, tiles = compute_pair_temporal_features(
        pair_id="p1",
        prev_mask=prev_mask,
        cur_mask=cur_mask,
        prev_feat=prev_feat,
        cur_feat=cur_feat,
        prev_feat_hw=(4, 4),
        cur_feat_hw=(4, 4),
        tile_size=4,
        stride=4,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )
    assert len(tiles) == 4
    assert len(rows) == 4
    assert heat.shape == (8, 8)
    for r in rows:
        assert 0.0 <= float(r["mask_change_density"]) <= 1.0
        assert 0.0 <= float(r["feature_cosine_distance"]) <= 2.0
