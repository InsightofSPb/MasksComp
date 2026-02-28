#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from maskscomp.lm_entropy import (
    LMEntropyModel,
    RowTokenDataset,
    build_row_items,
    collect_images,
    compute_batch_losses,
    compute_dataset_stats,
    dump_json,
    make_loader,
    save_split_lists,
    split_facades,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LM entropy model on row-wise RLE token stream")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--wmax", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use-2d-context", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = collect_images(args.data_root, subdir=args.subdir)
    max_w_data, labels = compute_dataset_stats(records)
    if not labels:
        raise RuntimeError("No valid masks found")

    wmax = int(args.wmax) if args.wmax is not None else int(max_w_data)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    train_facades, val_facades = split_facades([r.facade_id for r in records], args.val_ratio, args.seed)
    split_dir = args.out_dir / "splits"
    save_split_lists(train_facades, val_facades, split_dir)

    train_records = [r for r in records if r.facade_id in set(train_facades)]
    val_records = [r for r in records if r.facade_id in set(val_facades)]

    no_above_idx = len(labels)
    train_rows = build_row_items(train_records, label_to_idx=label_to_idx, no_above_idx=no_above_idx)
    val_rows = build_row_items(val_records, label_to_idx=label_to_idx, no_above_idx=no_above_idx)
    for row in train_rows + val_rows:
        if np.any((row.token_types == 1) & (row.tokens > wmax)):
            raise ValueError(f"Encountered run length > wmax ({wmax}) in {row.rel_path}")

    train_ds = RowTokenDataset(train_rows, label_bos_idx=no_above_idx, no_above_idx=no_above_idx)
    val_ds = RowTokenDataset(val_rows, label_bos_idx=no_above_idx, no_above_idx=no_above_idx)

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)

    max_seq_len = 2 * wmax + 2
    device = torch.device(args.device)
    model = LMEntropyModel(
        num_labels=len(labels),
        wmax=wmax,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
        use_2d_context=args.use_2d_context,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["out_dir"] = str(args.out_dir)
    config["num_labels"] = len(labels)
    config["labels"] = labels
    config["label_to_idx"] = label_to_idx
    config["wmax"] = wmax
    config["max_seq_len"] = max_seq_len
    dump_json(args.out_dir / "config.json", config)

    log_rows = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_bits = 0.0
        train_pixels = 0
        train_loss_label = 0.0
        train_loss_len = 0.0
        n_batches = 0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            loss_label, loss_len, stats = compute_batch_losses(model, batch, device, args.use_2d_context)
            loss = loss_label + loss_len
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            train_loss_label += float(loss_label.item())
            train_loss_len += float(loss_len.item())
            train_bits += stats["bits_label"] + stats["bits_len"]
            train_pixels += sum(int(m.W) for m in batch["meta"])
            n_batches += 1

        model.eval()
        val_bits = 0.0
        val_pixels = 0
        val_loss_label = 0.0
        val_loss_len = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss_label, loss_len, stats = compute_batch_losses(model, batch, device, args.use_2d_context)
                val_loss_label += float(loss_label.item())
                val_loss_len += float(loss_len.item())
                val_bits += stats["bits_label"] + stats["bits_len"]
                val_pixels += sum(int(m.W) for m in batch["meta"])
                val_batches += 1

        train_bbp = train_bits / max(1, train_pixels)
        val_bbp = val_bits / max(1, val_pixels)
        row = {
            "epoch": epoch,
            "train_loss_label": train_loss_label / max(1, n_batches),
            "train_loss_len": train_loss_len / max(1, n_batches),
            "train_bbp": train_bbp,
            "val_loss_label": val_loss_label / max(1, val_batches),
            "val_loss_len": val_loss_len / max(1, val_batches),
            "val_bbp": val_bbp,
        }
        log_rows.append(row)
        print(json.dumps(row))

        ckpt = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch:02d}.pt")
        if val_bbp < best_val:
            best_val = val_bbp
            torch.save(ckpt, ckpt_dir / "best.pt")

    write_csv(
        args.out_dir / "train_log.csv",
        fieldnames=["epoch", "train_loss_label", "train_loss_len", "train_bbp", "val_loss_label", "val_loss_len", "val_bbp"],
        rows=log_rows,
    )


if __name__ == "__main__":
    main()
