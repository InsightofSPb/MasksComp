#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from maskscomp.raw_temporal import ByteMSDZipLM, compute_sequence_bits, overlay_heatmap_on_rgb, read_pairs_csv, score_to_heatmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate raw temporal residual LM and PNG tile baseline")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None, help="Base path for optional LPOSS/superpixel paths from pairs CSV")
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--write-heatmaps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overlay-lposs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--superpixel-aggregate", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _load_lposs(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        key = "labels" if "labels" in z else list(z.keys())[0]
        arr = z[key]
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to read LPOSS predictions: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != shape_hw:
        arr = cv2.resize(arr.astype(np.int32), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.int32)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = ByteMSDZipLM(
        timesteps=int(cfg["timesteps"]),
        hidden_dim=int(cfg["hidden_dim"]),
        vocab_dim=int(cfg["vocab_dim"]),
        ffn_dim=int(cfg["ffn_dim"]),
        layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)
    model.eval()

    pairs = [p for p in read_pairs_csv(args.pairs_csv) if p.split == args.split]
    tile_csv = args.out_dir / f"tile_scores_{args.split}.csv"
    image_csv = args.out_dir / f"image_scores_{args.split}.csv"

    tile_rows = []
    image_rows = []
    super_rows = []

    for rec in pairs:
        npz_path = args.dataset_root / "tiles" / f"{rec.pair_id}.npz"
        if not npz_path.exists():
            continue
        z = np.load(npz_path)
        seqs = z["sequences"]
        yx = z["tile_yx"]
        png_bytes = z["png_bytes"]
        tile_size = int(z["tile_size"])
        H = int(z["H"])
        W = int(z["W"])

        per_tile = []
        lm_bits_sum = 0.0
        png_bits_sum = 0.0
        for i in range(seqs.shape[0]):
            bits = compute_sequence_bits(model, seqs[i], device=device, timesteps=int(cfg["timesteps"]), batch_size=args.batch_size)
            pb = float(png_bytes[i]) * 8.0
            y, x = int(yx[i, 0]), int(yx[i, 1])
            row = {
                "pair_id": rec.pair_id,
                "sample_id": rec.sample_id,
                "tile_idx": i,
                "tile_y": y,
                "tile_x": x,
                "tile_size": tile_size,
                "lm_bits": bits,
                "lm_bpp": bits / max(1.0, tile_size * tile_size * 3),
                "png_bits": pb,
                "png_bpp": pb / max(1.0, tile_size * tile_size * 3),
                "delta_bits": bits - pb,
            }
            tile_rows.append(row)
            per_tile.append(row)
            lm_bits_sum += bits
            png_bits_sum += pb

        image_rows.append(
            {
                "pair_id": rec.pair_id,
                "sample_id": rec.sample_id,
                "n_tiles": len(per_tile),
                "lm_bits_total": lm_bits_sum,
                "png_bits_total": png_bits_sum,
                "lm_bpp": lm_bits_sum / max(1.0, H * W * 3),
                "png_bpp": png_bits_sum / max(1.0, H * W * 3),
            }
        )

        if args.write_heatmaps and per_tile:
            cur = cv2.imread(str(args.dataset_root / "pairs" / rec.pair_id / "cur_rgb.png"), cv2.IMREAD_COLOR)
            if cur is not None:
                cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
                hm = score_to_heatmap(per_tile, H, W, tile_size=tile_size)
                ov = overlay_heatmap_on_rgb(cur, hm)
                out_hm = args.out_dir / "heatmaps" / f"{rec.pair_id}_lm_heatmap.png"
                out_hm.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_hm), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

                if args.overlay_lposs and rec.lposs_cur_path:
                    lp_path = Path(rec.lposs_cur_path)
                    if not lp_path.is_absolute() and args.data_root is not None:
                        lp_path = args.data_root / lp_path
                    lp = _load_lposs(lp_path, (H, W))
                    hot = hm > np.percentile(hm[np.isfinite(hm)], 90.0)
                    classes, counts = np.unique(lp[hot], return_counts=True)
                    for c, n in zip(classes.tolist(), counts.tolist()):
                        super_rows.append({"pair_id": rec.pair_id, "kind": "lposs_hot", "region_id": int(c), "count": int(n)})

        if args.superpixel_aggregate and rec.superpixel_labels_path:
            sp_path = Path(rec.superpixel_labels_path)
            if not sp_path.is_absolute() and args.data_root is not None:
                sp_path = args.data_root / sp_path
            sp = _load_lposs(sp_path, (H, W))
            hm = score_to_heatmap(per_tile, H, W, tile_size=tile_size)
            for sp_id in np.unique(sp):
                m = sp == sp_id
                if not np.any(m):
                    continue
                super_rows.append(
                    {
                        "pair_id": rec.pair_id,
                        "kind": "superpixel_mean",
                        "region_id": int(sp_id),
                        "score": float(np.mean(hm[m])),
                        "score_max": float(np.max(hm[m])),
                        "pixels": int(np.sum(m)),
                    }
                )

    with tile_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys()) if tile_rows else ["pair_id"])
        wr.writeheader()
        if tile_rows:
            wr.writerows(tile_rows)
    with image_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()) if image_rows else ["pair_id"])
        wr.writeheader()
        if image_rows:
            wr.writerows(image_rows)
    if super_rows:
        with (args.out_dir / f"semantic_regions_{args.split}.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(super_rows[0].keys()))
            wr.writeheader()
            wr.writerows(super_rows)


if __name__ == "__main__":
    main()
