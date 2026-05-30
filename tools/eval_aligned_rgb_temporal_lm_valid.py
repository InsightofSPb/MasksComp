#!/usr/bin/env python3
"""Score aligned RGB residual tiles with a valid-target ByteMSDZipLM checkpoint.

The output CSV follows the tile-score format consumed by
``Compress_to_prevent/tools/evaluate_temporal_tile_scores.py``. For every
retained tile, the score is ideal codelength per valid RGB byte; higher scores
mean a less expected residual under the train distribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maskscomp.raw_temporal import ByteMSDZipLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate valid-target RGB temporal LM tile scores.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--method-name", default="rgb_msdzip_valid_bpb")
    parser.add_argument("--write-overlays", action="store_true")
    parser.add_argument("--overlay-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{str(key): (value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError("Missing split file: {}".format(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.no_grad()
def compute_valid_sequence_bits(
    model: ByteMSDZipLM,
    sequence: np.ndarray,
    valid_sequence: np.ndarray,
    device: torch.device,
    timesteps: int,
    batch_size: int,
) -> Tuple[float, int]:
    if sequence.shape != valid_sequence.shape:
        raise ValueError("Sequence and validity sequence shapes differ")
    positions = np.flatnonzero(valid_sequence[1:] > 0) + 1
    if positions.size == 0:
        return 0.0, 0
    contexts: List[np.ndarray] = []
    targets: List[int] = []
    for position in positions.tolist():
        start = max(0, int(position) - timesteps)
        context = np.zeros((timesteps,), dtype=np.int64)
        window = sequence[start:int(position)].astype(np.int64)
        context[-window.size:] = window
        contexts.append(context)
        targets.append(int(sequence[int(position)]))
    total_bits = 0.0
    for start in range(0, len(contexts), batch_size):
        x = torch.as_tensor(np.stack(contexts[start:start + batch_size]), dtype=torch.long, device=device)
        y = torch.as_tensor(np.asarray(targets[start:start + batch_size]), dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]
        nll = F.cross_entropy(logits, y, reduction="sum")
        total_bits += float(nll.item() / math.log(2.0))
    return total_bits, int(positions.size)


def save_overlay(dataset_root: Path, pair_id: str, rows: List[Dict[str, object]], out_dir: Path) -> None:
    current_path = dataset_root / "pairs" / pair_id / "cur_rgb.png"
    current = cv2.imread(str(current_path), cv2.IMREAD_COLOR)
    if current is None or not rows:
        return
    height, width = current.shape[:2]
    score = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for row in rows:
        y = int(row["tile_origin_y"])
        x = int(row["tile_origin_x"])
        tile_size = int(row["tile_size"])
        y1, x1 = min(height, y + tile_size), min(width, x + tile_size)
        score[y:y1, x:x1] += float(row["tile_score"])
        count[y:y1, x:x1] += 1
    valid = count > 0
    heat = np.zeros_like(score)
    heat[valid] = score[valid] / count[valid]
    vals = heat[valid]
    if vals.size and float(vals.max()) > float(vals.min()):
        low, high = np.percentile(vals, [5, 95])
        heat = np.clip((heat - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    else:
        heat[:] = 0.0
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(current, 0.55, colour, 0.45, 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / (pair_id + "_rgb_msdzip_overlay.png")), overlay)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = ByteMSDZipLM(
        timesteps=int(config["timesteps"]),
        hidden_dim=int(config["hidden_dim"]),
        vocab_dim=int(config["vocab_dim"]),
        ffn_dim=int(config["ffn_dim"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pair_metadata = {row["pair_id"]: row for row in read_csv(args.dataset_root / "pair_metadata.csv")}
    split_ids = load_ids(args.dataset_root / "splits" / ("facade_" + args.split + ".txt"))
    output_rows: List[Dict[str, object]] = []
    rows_by_pair: Dict[str, List[Dict[str, object]]] = {}
    for pair_id in split_ids:
        tile_path = args.dataset_root / "tiles" / (pair_id + ".npz")
        if not tile_path.exists():
            continue
        with np.load(tile_path, allow_pickle=False) as payload:
            sequences = payload["sequences"]
            valid_sequences = payload["valid_sequences"]
            tile_yx = payload["tile_yx"]
            valid_fraction = payload["valid_fraction"]
            tile_size = int(payload["tile_size"])
        pair_meta = pair_metadata[pair_id]
        pair_rows: List[Dict[str, object]] = []
        for tile_index in range(sequences.shape[0]):
            bits, valid_byte_count = compute_valid_sequence_bits(
                model, sequences[tile_index], valid_sequences[tile_index], device,
                timesteps=int(config["timesteps"]), batch_size=args.batch_size,
            )
            if valid_byte_count == 0:
                continue
            origin_y, origin_x = int(tile_yx[tile_index][0]), int(tile_yx[tile_index][1])
            row: Dict[str, object] = {
                "pair_id": pair_id,
                "facade_id": pair_meta["facade_id"],
                "split": args.split,
                "method": args.method_name,
                "score_type": "learned_codelength_per_valid_byte",
                "tile_origin_y": origin_y,
                "tile_origin_x": origin_x,
                "tile_y": origin_y // tile_size,
                "tile_x": origin_x // tile_size,
                "tile_size": tile_size,
                "tile_score": bits / max(valid_byte_count, 1),
                "bit_length": bits,
                "valid_byte_count": valid_byte_count,
                "valid_pixel_count": valid_byte_count // 3,
                "valid_ratio": float(valid_fraction[tile_index]),
            }
            output_rows.append(row)
            pair_rows.append(row)
        rows_by_pair[pair_id] = pair_rows

    fields = [
        "pair_id", "facade_id", "split", "method", "score_type", "tile_origin_y", "tile_origin_x",
        "tile_y", "tile_x", "tile_size", "tile_score", "bit_length", "valid_byte_count",
        "valid_pixel_count", "valid_ratio",
    ]
    write_csv(args.out_csv, fields, output_rows)
    if args.write_overlays:
        overlay_dir = args.overlay_dir or (args.out_csv.parent / ("heatmap_overlays_" + args.split))
        for pair_id, rows in rows_by_pair.items():
            save_overlay(args.dataset_root, pair_id, rows, overlay_dir)
    report = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "method": args.method_name,
        "n_pairs_with_scores": sum(1 for rows in rows_by_pair.values() if rows),
        "n_tile_scores": len(output_rows),
        "score_definition": "ideal model bits divided by valid predicted RGB bytes per residual tile",
        "validity_policy": "invalid residual and padding bytes excluded from codelength targets",
    }
    report_path = args.out_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Scored {} {} tiles for {} pairs".format(len(output_rows), args.split, report["n_pairs_with_scores"]))
    print("Scores:", args.out_csv)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
