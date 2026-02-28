#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from maskscomp.lm_entropy import (
    LMEntropyModel,
    RowTokenDataset,
    build_row_items,
    collect_images,
    collate_rows,
    load_split_list,
    masked_length_nll_bits,
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
    p.add_argument("--use-2d-context", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
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
    cfg = ckpt.get("config", {})

    labels = cfg["labels"]
    label_to_idx = {int(k): int(v) for k, v in cfg["label_to_idx"].items()}
    no_above_idx = len(labels)
    wmax = int(args.wmax) if args.wmax is not None else int(cfg["wmax"])
    use_2d_context = bool(args.use_2d_context or cfg.get("use_2d_context", False))

    records = collect_images(args.data_root, subdir=args.subdir)
    facade_list = load_split_list(args.splits_dir / f"facade_{args.split}.txt")
    rec_split = [r for r in records if r.facade_id in set(facade_list)]

    rows = build_row_items(rec_split, label_to_idx=label_to_idx, no_above_idx=no_above_idx)
    ds = RowTokenDataset(rows, label_bos_idx=no_above_idx)

    model = LMEntropyModel(
        num_labels=len(labels),
        wmax=wmax,
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        dropout=float(cfg["dropout"]),
        use_2d_context=use_2d_context,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device(args.device))
    model.eval()

    per_image = defaultdict(lambda: {"bits_label": 0.0, "bits_len": 0.0, "H": 0, "W": 0, "facade_id": "", "rel_path": ""})

    with torch.no_grad():
        for i in range(len(ds)):
            batch = collate_rows([ds[i]])
            inp = batch["input_tokens"].to(args.device)
            tgt = batch["target_tokens"].to(args.device)
            ttypes = batch["token_types"].to(args.device)
            rem = batch["rem_width"].to(args.device)
            pad = batch["pad_mask"].to(args.device)
            above_label = batch["above_label"].to(args.device)
            above_same = batch["above_same"].to(args.device)

            if torch.any((ttypes == 1) & (tgt > wmax)):
                raise ValueError("Encountered target length greater than wmax")

            label_logits, len_logits = model(
                inp,
                ttypes,
                pad,
                above_label=above_label if use_2d_context else None,
                above_same=above_same if use_2d_context else None,
            )
            label_pos = (ttypes == 0) & (tgt >= 0) & (~pad)
            len_pos = (ttypes == 1) & (tgt >= 1) & (~pad)

            bits_label = 0.0
            if label_pos.any():
                lp = F.log_softmax(label_logits, dim=-1)
                g = torch.gather(lp, -1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                bits_label = float((-g[label_pos] / np.log(2.0)).sum().item())

            bits_len = 0.0
            if len_pos.any():
                nll_bits = masked_length_nll_bits(len_logits, tgt, rem)
                bits_len = float(nll_bits[len_pos].sum().item())

            meta = batch["meta"][0]
            key = (meta.facade_id, meta.rel_path)
            per_image[key]["bits_label"] += bits_label
            per_image[key]["bits_len"] += bits_len
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
    import csv

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

    summary = {"bbp_total": describe(np.asarray(bbps_total, dtype=np.float64)), "bbp_len": describe(np.asarray(bbps_len, dtype=np.float64))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
