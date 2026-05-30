#!/usr/bin/env python3
"""Score aligned RGB residual tiles with a valid-target ByteMSDZipLM checkpoint.

The output CSV follows the tile-score format consumed by
``Compress_to_prevent/tools/evaluate_temporal_tile_scores.py``. For every
retained tile, the score is ideal codelength per valid RGB byte; higher scores
mean a less expected residual under the train distribution.

Scoring uses all retained validation/test tiles by default. Context generation
is vectorized per tile and model inference is batched across tile contexts
within each pair. Progress and partial CSV files are written per completed pair
so a long run is observable and recoverable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maskscomp.raw_temporal import ByteMSDZipLM  # noqa: E402


FIELDS = [
    "pair_id", "facade_id", "split", "method", "score_type", "tile_origin_y", "tile_origin_x",
    "tile_y", "tile_x", "tile_size", "tile_score", "bit_length", "valid_byte_count",
    "valid_pixel_count", "valid_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate valid-target RGB temporal LM tile scores.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Target contexts per GPU/CPU model forward pass.")
    parser.add_argument("--buffer-batches", type=int, default=4,
                        help="Number of inference batches buffered before scoring; controls CPU RAM use.")
    parser.add_argument("--method-name", default="rgb_msdzip_valid_bpb")
    parser.add_argument("--write-overlays", action="store_true")
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--save-every-pairs", type=int, default=1,
                        help="Write current score CSV after this many completed pairs; 0 writes only at the end.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Debug-only limit. Do not use for final reported val/test metrics.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
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


def build_valid_contexts(sequence: np.ndarray, valid_sequence: np.ndarray, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct fixed-length contexts and targets for valid predicted bytes of one tile."""
    if sequence.shape != valid_sequence.shape:
        raise ValueError("Sequence and validity sequence shapes differ")
    seq = np.asarray(sequence, dtype=np.int64)
    positions = np.flatnonzero(valid_sequence[1:] > 0).astype(np.int64) + 1
    if positions.size == 0:
        return np.empty((0, timesteps), dtype=np.int64), np.empty((0,), dtype=np.int64)
    offsets = np.arange(-timesteps, 0, dtype=np.int64)
    source_positions = positions[:, None] + offsets[None, :]
    contexts = np.zeros((positions.size, timesteps), dtype=np.int64)
    usable = source_positions >= 0
    contexts[usable] = seq[source_positions[usable]]
    targets = seq[positions]
    return contexts, targets


@torch.no_grad()
def score_buffer(
    model: ByteMSDZipLM,
    context_parts: List[np.ndarray],
    target_parts: List[np.ndarray],
    tile_index_parts: List[np.ndarray],
    tile_bits: np.ndarray,
    tile_counts: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, int]:
    if not context_parts:
        return 0.0, 0
    contexts = np.concatenate(context_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    tile_indices = np.concatenate(tile_index_parts, axis=0)
    scored_bits = 0.0
    scored_count = 0
    for start in range(0, targets.size, batch_size):
        end = min(start + batch_size, targets.size)
        x = torch.as_tensor(contexts[start:end], dtype=torch.long, device=device)
        y = torch.as_tensor(targets[start:end], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]
        per_target_bits = F.cross_entropy(logits, y, reduction="none") / math.log(2.0)
        bits_np = per_target_bits.detach().cpu().numpy().astype(np.float64)
        ids_np = tile_indices[start:end]
        np.add.at(tile_bits, ids_np, bits_np)
        np.add.at(tile_counts, ids_np, 1)
        scored_bits += float(bits_np.sum())
        scored_count += int(bits_np.size)
    return scored_bits, scored_count


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
    if args.batch_size <= 0 or args.buffer_batches <= 0:
        raise ValueError("batch-size and buffer-batches must be positive")
    if args.save_every_pairs < 0:
        raise ValueError("save-every-pairs must be non-negative")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("max-pairs must be positive when provided")

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
    if args.max_pairs is not None:
        split_ids = split_ids[:args.max_pairs]
        print("WARNING: --max-pairs is active; outputs are for debugging only, not final metrics.", flush=True)

    output_rows: List[Dict[str, object]] = []
    rows_by_pair: Dict[str, List[Dict[str, object]]] = {}
    global_bits = 0.0
    global_valid_bytes = 0
    total_tiles = 0
    for pair_id in split_ids:
        tile_path = args.dataset_root / "tiles" / (pair_id + ".npz")
        if tile_path.exists():
            with np.load(tile_path, allow_pickle=False) as payload:
                total_tiles += int(payload["sequences"].shape[0])

    overlay_dir = args.overlay_dir or (args.out_csv.parent / ("heatmap_overlays_" + args.split))
    progress = tqdm(total=total_tiles, desc="Scoring {} RGB residual tiles".format(args.split), unit="tile", disable=args.no_progress)
    buffer_limit = args.batch_size * args.buffer_batches

    for pair_number, pair_id in enumerate(split_ids, start=1):
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
        n_tiles = int(sequences.shape[0])
        tile_bits = np.zeros((n_tiles,), dtype=np.float64)
        tile_counts = np.zeros((n_tiles,), dtype=np.int64)
        context_parts: List[np.ndarray] = []
        target_parts: List[np.ndarray] = []
        tile_index_parts: List[np.ndarray] = []
        buffered_targets = 0
        pair_bits = 0.0
        pair_valid_bytes = 0

        for tile_index in range(n_tiles):
            contexts, targets = build_valid_contexts(
                sequences[tile_index], valid_sequences[tile_index], timesteps=int(config["timesteps"])
            )
            if targets.size:
                context_parts.append(contexts)
                target_parts.append(targets)
                tile_index_parts.append(np.full((targets.size,), tile_index, dtype=np.int64))
                buffered_targets += int(targets.size)
            if buffered_targets >= buffer_limit:
                bits, count = score_buffer(
                    model, context_parts, target_parts, tile_index_parts, tile_bits, tile_counts,
                    device=device, batch_size=args.batch_size,
                )
                pair_bits += bits
                pair_valid_bytes += count
                context_parts, target_parts, tile_index_parts = [], [], []
                buffered_targets = 0
            progress.update(1)

        bits, count = score_buffer(
            model, context_parts, target_parts, tile_index_parts, tile_bits, tile_counts,
            device=device, batch_size=args.batch_size,
        )
        pair_bits += bits
        pair_valid_bytes += count
        global_bits += pair_bits
        global_valid_bytes += pair_valid_bytes

        pair_rows: List[Dict[str, object]] = []
        for tile_index in range(n_tiles):
            valid_byte_count = int(tile_counts[tile_index])
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
                "tile_score": float(tile_bits[tile_index] / valid_byte_count),
                "bit_length": float(tile_bits[tile_index]),
                "valid_byte_count": valid_byte_count,
                "valid_pixel_count": valid_byte_count // 3,
                "valid_ratio": float(valid_fraction[tile_index]),
            }
            output_rows.append(row)
            pair_rows.append(row)
        rows_by_pair[pair_id] = pair_rows
        if args.write_overlays:
            save_overlay(args.dataset_root, pair_id, pair_rows, overlay_dir)
        if args.save_every_pairs and pair_number % args.save_every_pairs == 0:
            write_csv(args.out_csv, FIELDS, output_rows)
        progress.set_postfix(
            pairs="{}/{}".format(pair_number, len(split_ids)),
            scored_tiles=len(output_rows),
            avg_bpb="{:.4f}".format(global_bits / max(global_valid_bytes, 1)),
        )

    progress.close()
    write_csv(args.out_csv, FIELDS, output_rows)
    report = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "method": args.method_name,
        "n_pairs_requested": len(split_ids),
        "n_pairs_with_scores": sum(1 for rows in rows_by_pair.values() if rows),
        "n_tile_scores": len(output_rows),
        "n_valid_predicted_bytes": int(global_valid_bytes),
        "mean_bits_per_valid_byte": float(global_bits / max(global_valid_bytes, 1)),
        "score_definition": "ideal model bits divided by valid predicted RGB bytes per residual tile",
        "validity_policy": "invalid residual and padding bytes excluded from codelength targets",
        "debug_max_pairs": args.max_pairs,
    }
    report_path = args.out_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Scored {} {} tiles for {} pairs".format(len(output_rows), args.split, report["n_pairs_with_scores"]))
    print("Mean bits per valid byte: {:.6f}".format(report["mean_bits_per_valid_byte"]))
    print("Scores:", args.out_csv)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
