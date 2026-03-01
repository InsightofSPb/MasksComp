#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LM entropy model on row-wise RLE token stream")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--wmax", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--arch", choices=["transformer", "msdzip"], default="transformer")
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


def compute_msdzip_window_losses(model: torch.nn.Module, batch: dict, device: torch.device, use_2d_context: bool, timesteps: int):
    input_tokens = batch["input_tokens"]
    token_types = batch["token_types"]
    rem_width = batch["rem_width"]
    pad_mask = batch["pad_mask"]
    above_label = batch["above_label"]
    above_same = batch["above_same"]

    bsz, seqlen = input_tokens.shape
    bos_vals = input_tokens[:, :1]
    no_above_vals = above_label[:, :1]

    win_inp = []
    win_types = []
    win_al = []
    win_as = []
    tgt_ids = []
    tgt_types = []
    tgt_rem = []

    for b in range(bsz):
        valid = (~pad_mask[b]).nonzero(as_tuple=False).flatten()
        if valid.numel() <= 1:
            continue
        n = int(valid[-1].item()) + 1
        for t in range(1, n):
            start = max(0, t - timesteps)
            ctx_tokens = input_tokens[b, start:t]
            ctx_types = token_types[b, start:t]
            ctx_al = above_label[b, start:t]
            ctx_as = above_same[b, start:t]
            ctx_len = int(ctx_tokens.numel())

            x_tok = bos_vals[b].repeat(timesteps)
            x_typ = torch.full((timesteps,), fill_value=-1, dtype=torch.long)
            x_al = no_above_vals[b].repeat(timesteps)
            x_as = torch.zeros((timesteps,), dtype=torch.long)
            if ctx_len > 0:
                x_tok[-ctx_len:] = ctx_tokens
                x_typ[-ctx_len:] = ctx_types
                x_al[-ctx_len:] = ctx_al
                x_as[-ctx_len:] = ctx_as

            win_inp.append(x_tok)
            win_types.append(x_typ)
            win_al.append(x_al)
            win_as.append(x_as)
            tgt_ids.append(input_tokens[b, t])
            tgt_types.append(token_types[b, t])
            tgt_rem.append(rem_width[b, t])

    if not win_inp:
        z = torch.tensor(0.0, device=device)
        return z, z, {"bits_label": 0.0, "bits_len": 0.0, "n_label": 0, "n_len": 0}

    inp = torch.stack(win_inp, dim=0).to(device)
    ttypes = torch.stack(win_types, dim=0).to(device)
    al = torch.stack(win_al, dim=0).to(device)
    aeq = torch.stack(win_as, dim=0).to(device)
    target_ids = torch.stack(tgt_ids, dim=0).to(device)
    target_types = torch.stack(tgt_types, dim=0).to(device)
    target_rem = torch.stack(tgt_rem, dim=0).to(device)
    pad = torch.zeros_like(inp, dtype=torch.bool)

    label_logits, len_logits = model(
        inp,
        ttypes,
        pad,
        above_label=al if use_2d_context else None,
        above_same=aeq if use_2d_context else None,
    )
    pred_label = label_logits[:, -1, :]
    pred_len = len_logits[:, -1, :]

    label_pos = target_types == 0
    len_pos = target_types == 1

    label_bits = torch.zeros((target_ids.shape[0],), device=device)
    if label_pos.any():
        lp = torch.log_softmax(pred_label[label_pos], dim=-1)
        tg = target_ids[label_pos].clamp(min=0, max=pred_label.shape[-1] - 1)
        label_bits[label_pos] = -lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1) / np.log(2.0)

    len_bits = torch.zeros((target_ids.shape[0],), device=device)
    if len_pos.any():
        lp_len = pred_len[len_pos]
        rem = target_rem[len_pos]
        idx = torch.arange(lp_len.shape[-1], device=device).view(1, -1)
        allowed = (idx >= 1) & (idx <= rem.unsqueeze(-1))
        masked_logits = torch.where(allowed, lp_len, torch.full_like(lp_len, -1e9))
        logp = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
        tg = target_ids[len_pos].clamp(min=1, max=lp_len.shape[-1] - 1)
        len_bits[len_pos] = -logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1) / np.log(2.0)

    loss_label = torch.tensor(0.0, device=device)
    if label_pos.any():
        loss_label = (label_bits[label_pos] * np.log(2.0)).mean()
    loss_len = torch.tensor(0.0, device=device)
    if len_pos.any():
        loss_len = (len_bits[len_pos] * np.log(2.0)).mean()
    return loss_label, loss_len, {
        "bits_label": float(label_bits.sum().item()),
        "bits_len": float(len_bits.sum().item()),
        "n_label": int(label_pos.sum().item()),
        "n_len": int(len_pos.sum().item()),
    }


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
    dump_json(args.out_dir / "config.json", config)

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
                loss_label, loss_len, stats = compute_msdzip_window_losses(
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
                    loss_label, loss_len, stats = compute_msdzip_window_losses(
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
        row["gpu_mem_allocated_mb"] = (
            torch.cuda.memory_allocated(device) / (1024**2) if track_cuda_memory else None
        )
        row["gpu_mem_reserved_mb"] = (
            torch.cuda.memory_reserved(device) / (1024**2) if track_cuda_memory else None
        )
        row["gpu_mem_peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / (1024**2) if track_cuda_memory else None
        )
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
