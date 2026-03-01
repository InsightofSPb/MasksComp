#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maskscomp.lm_entropy import (
    RowTokenDataset,
    build_lm_model,
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
from maskscomp.utils.msdzip_windows import compute_msdzip_window_loss_stats


def _read_id_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Split list not found: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _pick_existing(dir_path: Path, preferred_name: Optional[str], candidates: Sequence[str]) -> Path:
    if preferred_name:
        p = dir_path / preferred_name
        if p.exists():
            return p
        raise FileNotFoundError(f"Requested split file does not exist: {p}")

    for name in candidates:
        p = dir_path / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find split files in splits-dir. Looked for: "
        + ", ".join(str(dir_path / c) for c in candidates)
    )


def load_splits(
    splits_dir: Path,
    train_split_file: Optional[str] = None,
    val_split_file: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    splits_dir = Path(splits_dir)
    train_path = _pick_existing(splits_dir, train_split_file, ["facade_train.txt", "train.txt", "split_train.txt"])
    val_path = _pick_existing(splits_dir, val_split_file, ["facade_val.txt", "val.txt", "split_val.txt"])

    train_ids = _read_id_list(train_path)
    val_ids = _read_id_list(val_path)

    # If there is overlap, keep val as-is and remove duplicates from train (safer for model selection).
    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        train_ids = [x for x in train_ids if x not in overlap]

    return train_ids, val_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LM entropy model on row-wise RLE token stream")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--out-dir", type=Path, required=True)

    # Splits
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Use precomputed split lists (facade_train.txt / facade_val.txt). "
        "If set, --val-ratio/--seed splitting is ignored.",
    )
    p.add_argument(
        "--train-split-file",
        type=str,
        default=None,
        help="Train split filename inside --splits-dir (default: auto-detect).",
    )
    p.add_argument(
        "--val-split-file",
        type=str,
        default=None,
        help="Val split filename inside --splits-dir (default: auto-detect).",
    )

    # Fallback split (only when --splits-dir is not provided)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)

    # Training
    p.add_argument("--wmax", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--arch", choices=["transformer", "msdzip"], default="transformer")

    # Transformer/MSDZip params
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--timesteps", type=int, default=16)
    p.add_argument("--vocab-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)

    p.add_argument("--use-2d-context", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    t_total0 = time.perf_counter()
    log("Starting run")
    log("Command: " + " ".join(sys.argv))
    log("Args:\n" + json.dumps(vars(args), indent=2, ensure_ascii=False, default=str))

    t = time.perf_counter()
    records = collect_images(args.data_root, subdir=args.subdir)
    log(
        f"collect_images: {len(records)} records from {args.data_root}/{args.subdir}, "
        f"took {(time.perf_counter()-t):.2f}s"
    )

    t = time.perf_counter()
    max_w_data, labels = compute_dataset_stats(records)
    log(
        f"compute_dataset_stats: max_w_data={max_w_data} labels={len(labels)}, "
        f"took {(time.perf_counter()-t):.2f}s"
    )
    if not labels:
        raise RuntimeError("No valid masks found")

    wmax = int(args.wmax) if args.wmax is not None else int(max_w_data)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    log(f"Derived: wmax={wmax} num_labels={len(labels)}")

    # -----------------------------
    # SPLITS (either fixed from --splits-dir, or random split)
    # -----------------------------
    if args.splits_dir is not None:
        t = time.perf_counter()
        train_facades, val_facades = load_splits(
            args.splits_dir, train_split_file=args.train_split_file, val_split_file=args.val_split_file
        )
        log(
            f"Loaded splits from {args.splits_dir} "
            f"(train={len(train_facades)} val={len(val_facades)}), "
            f"took {(time.perf_counter()-t):.2f}s"
        )
    else:
        train_facades, val_facades = split_facades([r.facade_id for r in records], args.val_ratio, args.seed)
        log(
            f"Random split: train_facades={len(train_facades)} val_facades={len(val_facades)} "
            f"(val_ratio={args.val_ratio})"
        )

    # Save the actual lists used for this run (provenance)
    split_dir = args.out_dir / "splits"
    save_split_lists(train_facades, val_facades, split_dir)
    log(f"Saved splits to: {split_dir}")

    train_set = set(train_facades)
    val_set = set(val_facades)

    train_records = [r for r in records if r.facade_id in train_set]
    val_records = [r for r in records if r.facade_id in val_set]
    log(f"Split records: train_records={len(train_records)} val_records={len(val_records)}")

    if len(train_records) == 0:
        raise RuntimeError("Empty train split after filtering records. Check --splits-dir and --subdir.")
    if len(val_records) == 0:
        log("WARNING: Empty val split after filtering records (val_bbp will be meaningless).")

    no_above_idx = len(labels)
    log("Building row-wise token items (this can take a while on large masks)...")

    t = time.perf_counter()
    train_rows = build_row_items(train_records, label_to_idx=label_to_idx, no_above_idx=no_above_idx)
    log(f"build_row_items(train): rows={len(train_rows)}, took {(time.perf_counter()-t):.2f}s")

    t = time.perf_counter()
    val_rows = build_row_items(val_records, label_to_idx=label_to_idx, no_above_idx=no_above_idx)
    log(f"build_row_items(val): rows={len(val_rows)}, took {(time.perf_counter()-t):.2f}s")

    for row in train_rows + val_rows:
        if np.any((row.token_types == 1) & (row.tokens > wmax)):
            raise ValueError(f"Encountered run length > wmax ({wmax}) in {row.rel_path}")

    train_ds = RowTokenDataset(train_rows, label_bos_idx=no_above_idx, no_above_idx=no_above_idx)
    val_ds = RowTokenDataset(val_rows, label_bos_idx=no_above_idx, no_above_idx=no_above_idx)

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    log(f"Dataloaders: train_ds={len(train_ds)} val_ds={len(val_ds)} batch_size={args.batch_size}")

    max_seq_len = 2 * wmax + 2
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log("WARNING: device=cuda requested but CUDA is not available; training will likely fail.")

    track_cuda_memory = device.type == "cuda" and torch.cuda.is_available()
    if track_cuda_memory:
        torch.cuda.reset_peak_memory_stats(device)

    model = build_lm_model(
        arch=args.arch,
        num_labels=len(labels),
        wmax=wmax,
        max_seq_len=max_seq_len,
        use_2d_context=args.use_2d_context,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        timesteps=args.timesteps,
        vocab_dim=args.vocab_dim,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        layers=args.layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"Model: arch={args.arch} params={n_params/1e6:.2f}M "
        f"use_2d_context={args.use_2d_context} max_seq_len={max_seq_len} device={device}"
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["out_dir"] = str(args.out_dir)
    config["num_labels"] = len(labels)
    config["labels"] = labels
    config["label_to_idx"] = label_to_idx
    config["wmax"] = wmax
    config["max_seq_len"] = max_seq_len
    config["arch"] = args.arch
    config["splits_used_dir"] = str(split_dir)
    dump_json(args.out_dir / "config.json", config)

    log("Saved config.json to: " + str(args.out_dir / "config.json"))
    log(f"Preprocessing total: {(time.perf_counter()-t_total0):.2f}s. Entering training loop...")

    log_rows = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        if track_cuda_memory:
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        train_bits = 0.0
        train_pixels = 0
        train_loss_label = 0.0
        train_loss_len = 0.0
        n_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for batch in train_pbar:
            optim.zero_grad(set_to_none=True)
            if args.arch == "msdzip":
                loss_label, loss_len, stats = compute_msdzip_window_loss_stats(
                    model, batch, device, args.use_2d_context, args.timesteps
                )
            else:
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

            avg_train_bbp = train_bits / max(1, train_pixels)
            train_pbar.set_postfix(
                train_bbp=f"{avg_train_bbp:.4f}",
                loss_lbl=f"{train_loss_label / max(1, n_batches):.4f}",
                loss_len=f"{train_loss_len / max(1, n_batches):.4f}",
            )

        model.eval()
        val_bits = 0.0
        val_pixels = 0
        val_loss_label = 0.0
        val_loss_len = 0.0
        val_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False)
            for batch in val_pbar:
                if args.arch == "msdzip":
                    loss_label, loss_len, stats = compute_msdzip_window_loss_stats(
                        model, batch, device, args.use_2d_context, args.timesteps
                    )
                else:
                    loss_label, loss_len, stats = compute_batch_losses(model, batch, device, args.use_2d_context)

                val_loss_label += float(loss_label.item())
                val_loss_len += float(loss_len.item())
                val_bits += stats["bits_label"] + stats["bits_len"]
                val_pixels += sum(int(m.W) for m in batch["meta"])
                val_batches += 1

                avg_val_bbp = val_bits / max(1, val_pixels)
                val_pbar.set_postfix(
                    val_bbp=f"{avg_val_bbp:.4f}",
                    loss_lbl=f"{val_loss_label / max(1, val_batches):.4f}",
                    loss_len=f"{val_loss_len / max(1, val_batches):.4f}",
                )

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
        row["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2) if track_cuda_memory else None
        row["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2) if track_cuda_memory else None
        row["gpu_mem_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2) if track_cuda_memory else None

        log_rows.append(row)
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_bbp={train_bbp:.4f} val_bbp={val_bbp:.4f} "
            f"train_loss=({row['train_loss_label']:.4f}, {row['train_loss_len']:.4f}) "
            f"val_loss=({row['val_loss_label']:.4f}, {row['val_loss_len']:.4f}) "
            f"gpu_reserved_mb={row['gpu_mem_reserved_mb']} "
            f"gpu_peak_reserved_mb={row['gpu_mem_peak_reserved_mb']}"
        )
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
        fieldnames=[
            "epoch",
            "train_loss_label",
            "train_loss_len",
            "train_bbp",
            "val_loss_label",
            "val_loss_len",
            "val_bbp",
            "gpu_mem_allocated_mb",
            "gpu_mem_reserved_mb",
            "gpu_mem_peak_reserved_mb",
        ],
        rows=log_rows,
    )


if __name__ == "__main__":
    main()