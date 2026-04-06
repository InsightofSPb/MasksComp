#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maskscomp.semantic.export import export_semantic_sample
from maskscomp.semantic.io import load_samples, write_index_csv
from maskscomp.semantic.model_loader import load_facade_semantic_predictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export facade semantic artifacts (mask/probs/features/overlay)")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--manifest-csv", type=Path, default=None)
    p.add_argument("--image-col", type=str, default="image_path")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--save-mask", action="store_true")
    p.add_argument("--save-probs", action="store_true")
    p.add_argument("--save-features", action="store_true")
    p.add_argument("--save-overlay", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(
        input_root=args.input_root,
        manifest_csv=args.manifest_csv,
        image_col=args.image_col,
        limit=args.limit,
    )
    predictor = load_facade_semantic_predictor(args.config, args.checkpoint, device=args.device)

    rows: list[dict[str, object]] = []
    processed = 0
    for s in samples:
        row = export_semantic_sample(
            sample=s,
            predictor=predictor,
            output_root=args.output_root,
            save_mask=args.save_mask,
            save_probs=args.save_probs,
            save_features=args.save_features,
            save_overlay=args.save_overlay,
            overwrite=args.overwrite,
        )
        if args.tile_size is not None:
            row["tile_size"] = int(args.tile_size)
        if args.stride is not None:
            row["stride"] = int(args.stride)
        rows.append(row)
        if row.get("status") == "ok":
            processed += 1

    write_index_csv(rows, args.output_root / "index.csv")
    print(f"images_discovered={len(samples)}")
    print(f"images_processed={processed}")
    print(f"output_root={args.output_root}")
    print(
        "outputs="
        f"mask:{bool(args.save_mask)} "
        f"probs:{bool(args.save_probs)} "
        f"features:{True} "
        f"overlay:{bool(args.save_overlay)}"
    )


if __name__ == "__main__":
    main()
