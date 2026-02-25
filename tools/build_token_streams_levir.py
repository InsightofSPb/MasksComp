#!/usr/bin/env python3
"""Build LEVIR-CD+ token streams (int32) from change masks.

This script scans train/test mask directories, encodes each PNG mask into
scanline RLE tokens (contract-v0), writes a contiguous int32 stream,
and saves per-mask index and aggregate stats.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from mask_tokens import decode_mask_rle_scanline, encode_mask_rle_scanline


DEFAULT_SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build token-stream artifacts from LEVIR-CD+ masks. "
            "Supports both 'mask/' and 'label/' directories (prefers 'mask/')."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help=(
            "Dataset root. Supports either <root>/LEVIR-CD+/{train,test}/{mask|label} "
            "or <root>/{train,test}/{mask|label}."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for artifacts.",
    )
    parser.add_argument(
        "--splits",
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated splits to process (e.g. train,test).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of masks per split; applies seeded shuffle first.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--verify",
        type=int,
        default=50,
        help="Number of random masks per split to verify via stream round-trip.",
    )
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be a positive integer")
    if args.verify < 0:
        parser.error("--verify must be >= 0")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        parser.error("--splits cannot be empty")
    args.splits = splits

    return args


def resolve_dataset_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    nested = root / "LEVIR-CD+"
    if nested.is_dir():
        return nested
    return root


def find_split_mask_dir(dataset_root: Path, split: str) -> Path:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    for candidate in ("mask", "label"):
        d = split_dir / candidate
        if d.is_dir():
            return d

    raise FileNotFoundError(
        f"Neither 'mask' nor 'label' found for split '{split}' under: {split_dir}"
    )


def list_pngs(mask_dir: Path) -> list[Path]:
    files = sorted(p for p in mask_dir.glob("*.png") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No PNG masks found in: {mask_dir}")
    return files


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        # Accept RGB-like masks by taking the first channel.
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after loading, got shape {arr.shape} for {path}")
    return arr


def select_samples(paths: list[Path], max_samples: int | None, seed: int) -> list[Path]:
    if max_samples is None or max_samples >= len(paths):
        return paths
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    return shuffled[:max_samples]


def compute_stats(index_rows: list[dict]) -> dict:
    num_masks = len(index_rows)
    total_pixels = int(sum(int(r["H"]) * int(r["W"]) for r in index_rows))
    total_tokens = int(sum(int(r["token_len"]) for r in index_rows))

    mean_tokens_per_pixel = float(total_tokens / total_pixels) if total_pixels else 0.0
    mean_token_len = float(total_tokens / num_masks) if num_masks else 0.0
    mean_runs_per_mask = (
        float(sum((int(r["token_len"]) - 4) / 2 for r in index_rows) / num_masks)
        if num_masks
        else 0.0
    )

    return {
        "num_masks": num_masks,
        "total_pixels": total_pixels,
        "total_tokens": total_tokens,
        "mean_tokens_per_pixel": mean_tokens_per_pixel,
        "mean_token_len": mean_token_len,
        "mean_runs_per_mask": mean_runs_per_mask,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_round_trip(
    stream_path: Path,
    index_rows: list[dict],
    seed: int,
    verify_n: int,
) -> None:
    if verify_n <= 0 or not index_rows:
        return

    rng = random.Random(seed)
    k = min(verify_n, len(index_rows))
    chosen = rng.sample(index_rows, k=k)

    with stream_path.open("rb") as f:
        for row in chosen:
            token_offset = int(row["token_offset"])
            token_len = int(row["token_len"])
            f.seek(token_offset * 4)
            blob = f.read(token_len * 4)
            if len(blob) != token_len * 4:
                raise ValueError(
                    f"Short read in stream for {row['id']}: "
                    f"expected {token_len * 4} bytes, got {len(blob)}"
                )
            tokens = np.frombuffer(blob, dtype=np.int32)

            decoded = decode_mask_rle_scanline(tokens)
            original = (read_mask(Path(row["path"])) > 0).astype(np.uint8)
            if decoded.shape != original.shape or not np.array_equal(decoded, original):
                raise ValueError(f"Round-trip verification failed for mask: {row['path']}")


def process_split(
    split: str,
    dataset_root: Path,
    out_streams: Path,
    max_samples: int | None,
    seed: int,
    verify_n: int,
) -> None:
    mask_dir = find_split_mask_dir(dataset_root, split)
    pngs = select_samples(list_pngs(mask_dir), max_samples=max_samples, seed=seed)

    stream_path = out_streams / f"levir_{split}.int32.bin"
    index_path = out_streams / f"levir_{split}.index.jsonl"
    stats_path = out_streams / f"levir_{split}.stats.json"

    # Avoid appending to old runs.
    if stream_path.exists():
        stream_path.unlink()

    index_rows: list[dict] = []
    token_offset = 0

    for p in pngs:
        mask = read_mask(p)
        tokens = encode_mask_rle_scanline(mask).astype(np.int32, copy=False)

        with stream_path.open("ab") as f:
            tokens.tofile(f)

        row = {
            "id": p.name,
            "path": str(p.resolve()),
            "H": int(mask.shape[0]),
            "W": int(mask.shape[1]),
            "token_offset": int(token_offset),
            "token_len": int(tokens.size),
        }
        index_rows.append(row)
        token_offset += int(tokens.size)

    verify_round_trip(stream_path, index_rows, seed=seed, verify_n=verify_n)

    write_jsonl(index_path, index_rows)
    stats = compute_stats(index_rows)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"[{split}] masks={stats['num_masks']} pixels={stats['total_pixels']} "
        f"tokens={stats['total_tokens']} -> {stream_path}"
    )


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.root)

    out_streams = args.out_dir.expanduser().resolve() / "streams"
    out_streams.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        process_split(
            split=split,
            dataset_root=dataset_root,
            out_streams=out_streams,
            max_samples=args.max_samples,
            seed=args.seed,
            verify_n=args.verify,
        )


if __name__ == "__main__":
    main()
