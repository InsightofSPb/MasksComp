#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from maskscomp.raw_temporal import ByteMSDZipLM


class WindowDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], timesteps: int, max_samples: int | None = None, seed: int = 42):
        self.sequences = sequences
        self.timesteps = int(timesteps)
        index: List[Tuple[int, int]] = []
        for i, s in enumerate(self.sequences):
            if s.size <= 1:
                continue
            index.extend((i, t) for t in range(1, int(s.size)))
        if max_samples is not None and len(index) > int(max_samples):
            rng = random.Random(seed)
            index = rng.sample(index, int(max_samples))
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        seq_i, t = self.index[idx]
        s = self.sequences[seq_i]
        start = max(0, t - self.timesteps)
        ctx = np.zeros((self.timesteps,), dtype=np.int64)
        w = s[start:t].astype(np.int64)
        ctx[-w.size :] = w
        y = int(s[t])
        return torch.as_tensor(ctx, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def _load_split_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_sequences(dataset_root: Path, split_ids: list[str]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for pid in split_ids:
        p = dataset_root / "tiles" / f"{pid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        arr = z["sequences"]
        out.extend([arr[i] for i in range(arr.shape[0])])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MSDZip-style byte LM on raw temporal residual tiles")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--timesteps", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=200000)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--vocab-dim", type=int, default=32)
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def evaluate(model: ByteMSDZipLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)[:, -1, :]
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_n += int(y.numel())
    return total_loss / max(1, total_n)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tr_ids = _load_split_ids(args.splits_dir / "facade_train.txt")
    va_ids = _load_split_ids(args.splits_dir / "facade_val.txt")
    tr_seq = _load_sequences(args.dataset_root, tr_ids)
    va_seq = _load_sequences(args.dataset_root, va_ids)

    ds_tr = WindowDataset(tr_seq, timesteps=args.timesteps, max_samples=args.max_train_samples, seed=args.seed)
    ds_va = WindowDataset(va_seq, timesteps=args.timesteps, max_samples=args.max_val_samples, seed=args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = ByteMSDZipLM(
        timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        vocab_dim=args.vocab_dim,
        ffn_dim=args.ffn_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    log_rows = []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)[:, -1, :]
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(y.numel())
            n += int(y.numel())
        tr_nll = total / max(1, n)
        va_nll = evaluate(model, dl_va, device)
        row = {"epoch": ep, "train_nll": tr_nll, "val_nll": va_nll, "train_bpb": tr_nll / np.log(2.0), "val_bpb": va_nll / np.log(2.0)}
        log_rows.append(row)
        if va_nll < best:
            best = va_nll
            torch.save({"model_state": model.state_dict(), "config": vars(args), "arch": "msdzip-byte"}, args.out_dir / "best.pt")
        print(json.dumps(row))

    torch.save({"model_state": model.state_dict(), "config": vars(args), "arch": "msdzip-byte"}, args.out_dir / "last.pt")
    import csv

    with (args.out_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        wr.writeheader()
        wr.writerows(log_rows)


if __name__ == "__main__":
    main()
