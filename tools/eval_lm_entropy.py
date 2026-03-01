#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from maskscomp.lm_entropy import (
    CachedRowTokenDataset,
    RowTokenDataset,
    build_model_from_checkpoint_config,
    build_row_items,
    collect_images,
    compute_msdzip_shifted_token_bits,
    compute_shifted_token_bits,
    load_split_list,
    make_loader,
    load_cache_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LM entropy model as ideal code length")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--wmax", type=int, default=None)
    p.add_argument("--arch", choices=["transformer", "msdzip"], default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--max-items", type=int, default=None, help="Limit number of images for quick smoke tests")
    p.add_argument("--use-2d-context", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--row-cache-dir", type=Path, default=None)
    p.add_argument("--row-cache-manifest", type=Path, default=None)
    p.add_argument("--row-cache-lru-size", type=int, default=8)
    return p.parse_args()


def describe(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p90": float(np.percentile(vals, 90)),
    }


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    labels = cfg["labels"]
    label_to_idx = {int(k): int(v) for k, v in cfg["label_to_idx"].items()}
    no_above_idx = len(labels)
    wmax = int(args.wmax) if args.wmax is not None else int(cfg["wmax"])

    ckpt_use_2d = bool(cfg.get("use_2d_context", False))
    if bool(args.use_2d_context) != ckpt_use_2d:
        raise ValueError(f"use_2d_context mismatch: checkpoint={ckpt_use_2d} cli={bool(args.use_2d_context)}")
    use_2d_context = ckpt_use_2d

    ckpt_arch = str(cfg.get("arch", "transformer")).lower()
    arch = str(args.arch or ckpt_arch).lower()
    if arch != ckpt_arch:
        raise ValueError(f"arch mismatch: checkpoint={ckpt_arch} cli={arch}")

    split_facades = set(load_split_list(args.splits_dir / f"facade_{args.split}.txt"))
    if args.row_cache_dir is not None:
        manifest_path = args.row_cache_manifest or (args.row_cache_dir / "manifest.csv")
        manifest_rows = load_cache_manifest(manifest_path)
        max_w_manifest = max(int(r["W"]) for r in manifest_rows)
        if wmax < max_w_manifest:
            raise ValueError(f"wmax={wmax} is smaller than cache max width={max_w_manifest}")
        ds = CachedRowTokenDataset(
            cache_root=args.row_cache_dir,
            manifest_csv=manifest_path,
            allowed_facade_ids=split_facades,
            label_bos_idx=no_above_idx,
            no_above_idx=no_above_idx,
            label_to_idx=label_to_idx,
            use_2d_context=use_2d_context,
            lru_cache_size=args.row_cache_lru_size,
        )
    else:
        records = collect_images(args.data_root, subdir=args.subdir)
        rec_split = [r for r in records if r.facade_id in split_facades]
        if args.max_items is not None:
            rec_split = rec_split[: int(args.max_items)]

        rows = build_row_items(rec_split, label_to_idx=label_to_idx, no_above_idx=no_above_idx)
        for row in rows:
            if np.any((row.token_types == 1) & (row.tokens > wmax)):
                raise ValueError(f"Encountered run length > wmax ({wmax}) in {row.rel_path}")

        ds = RowTokenDataset(rows, label_bos_idx=no_above_idx, no_above_idx=no_above_idx)
    loader = make_loader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_model_from_checkpoint_config(cfg, use_2d_context=use_2d_context)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    is_msdzip = bool(ckpt_arch == "msdzip") or hasattr(model, "timesteps")
    timesteps = int(args.timesteps) if args.timesteps is not None else int(getattr(model, "timesteps", cfg.get("timesteps", 16)))

    per_image = defaultdict(lambda: {"bits_label": 0.0, "bits_len": 0.0, "H": 0, "W": 0, "facade_id": "", "rel_path": ""})
    with torch.no_grad():
        for batch in loader:
            if is_msdzip:
                out = compute_msdzip_shifted_token_bits(model, batch, device, use_2d_context, timesteps=timesteps)
            else:
                out = compute_shifted_token_bits(model, batch, device, use_2d_context)

            row_bits_label = out["label_bits"].sum(dim=1).cpu().numpy()
            row_bits_len = out["len_bits"].sum(dim=1).cpu().numpy()

            for i, meta in enumerate(batch["meta"]):
                key = (meta.facade_id, meta.rel_path)
                per_image[key]["bits_label"] += float(row_bits_label[i])
                per_image[key]["bits_len"] += float(row_bits_len[i])
                per_image[key]["H"] = meta.H
                per_image[key]["W"] = meta.W
                per_image[key]["facade_id"] = meta.facade_id
                per_image[key]["rel_path"] = meta.rel_path

    rows_csv = []
    bbps_total = []
    bbps_len = []
    for (_fid, _rp), m in sorted(per_image.items()):
        pixels = max(1, int(m["H"] * m["W"]))
        bits_total = m["bits_label"] + m["bits_len"]
        bbp_total = bits_total / pixels
        bbp_label = m["bits_label"] / pixels
        bbp_len = m["bits_len"] / pixels
        len_frac = m["bits_len"] / bits_total if bits_total > 0 else 0.0
        bbps_total.append(bbp_total)
        bbps_len.append(bbp_len)
        rows_csv.append(
            {
                "facade_id": m["facade_id"],
                "rel_path": m["rel_path"],
                "H": m["H"],
                "W": m["W"],
                "C": len(labels),
                "bits_total": bits_total,
                "bits_label": m["bits_label"],
                "bits_len": m["bits_len"],
                "bbp_total": bbp_total,
                "bbp_label": bbp_label,
                "bbp_len": bbp_len,
                "len_frac": len_frac,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "facade_id",
                "rel_path",
                "H",
                "W",
                "C",
                "bits_total",
                "bits_label",
                "bits_len",
                "bbp_total",
                "bbp_label",
                "bbp_len",
                "len_frac",
            ],
        )
        wr.writeheader()
        wr.writerows(rows_csv)

    summary = {
        "split": args.split,
        "arch": arch,
        "bbp_total": describe(np.asarray(bbps_total, dtype=np.float64)),
        "bbp_len": describe(np.asarray(bbps_len, dtype=np.float64)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
