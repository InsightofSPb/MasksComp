#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maskscomp.semantic.manifest import infer_pair_columns, load_csv_rows, load_semantic_index, resolve_semantic_row
from maskscomp.semantic.temporal_features import compute_pair_temporal_features, load_feat_npz, load_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build temporal semantic tile features for aligned facade pairs")
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--semantic-index-prev", type=Path, default=None)
    p.add_argument("--semantic-index-curr", type=Path, default=None)
    p.add_argument("--semantic-index", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tile-size", type=int, required=True)
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--feature-key", type=str, default="feat")
    p.add_argument("--score-config", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    return p.parse_args()


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def _render_preview(image_path: str | Path, heat: np.ndarray, out_path: Path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    heat_n = heat
    if heat_n.max(initial=0) > 0:
        heat_n = heat_n / (heat_n.max() + 1e-8)
    heat_u8 = (255.0 * heat_n).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, color, 0.45, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main() -> None:
    args = parse_args()
    rows = load_csv_rows(args.pairs_csv)
    if not rows:
        raise SystemExit("pairs-csv is empty")

    pair_col, prev_col, curr_col = infer_pair_columns(list(rows[0].keys()))

    if args.semantic_index is not None:
        prev_index = curr_index = load_semantic_index(args.semantic_index)
    else:
        if args.semantic_index_prev is None or args.semantic_index_curr is None:
            raise SystemExit("Provide either --semantic-index or both --semantic-index-prev/--semantic-index-curr")
        prev_index = load_semantic_index(args.semantic_index_prev)
        curr_index = load_semantic_index(args.semantic_index_curr)

    alpha, beta, gamma = args.alpha, args.beta, args.gamma

    out_rows: list[dict[str, object]] = []
    sem_heat_dir = args.output_root / "heatmaps_semantic"
    sem_prev_dir = args.output_root / "previews_semantic"
    sem_heat_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows):
        if args.limit is not None and i >= args.limit:
            break
        pair_id = str(row[pair_col])
        prev_key = str(row[prev_col])
        curr_key = str(row[curr_col])

        prev_art = resolve_semantic_row(prev_index, prev_key)
        curr_art = resolve_semantic_row(curr_index, curr_key)

        prev_mask = load_mask(prev_art["mask_path"])
        curr_mask = load_mask(curr_art["mask_path"])
        prev_feat, prev_hw = load_feat_npz(prev_art["features_path"], key=args.feature_key)
        curr_feat, curr_hw = load_feat_npz(curr_art["features_path"], key=args.feature_key)

        tile_rows, heat, _tiles = compute_pair_temporal_features(
            pair_id=pair_id,
            prev_mask=prev_mask,
            cur_mask=curr_mask,
            prev_feat=prev_feat,
            cur_feat=curr_feat,
            prev_feat_hw=prev_hw,
            cur_feat_hw=curr_hw,
            tile_size=args.tile_size,
            stride=args.stride,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        out_rows.extend(tile_rows)

        np.savez_compressed(
            sem_heat_dir / f"{pair_id}.npz",
            semantic_score=heat,
            pair_id=pair_id,
            tile_size=args.tile_size,
            stride=args.stride,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        _render_preview(curr_art.get("image_path", curr_key), heat, sem_prev_dir / f"{pair_id}.jpg")

    _write_csv(out_rows, args.output_root / "temporal_semantic_features.csv")
    print(f"pairs_processed={len({r['pair_id'] for r in out_rows})}")
    print(f"tile_rows={len(out_rows)}")
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
