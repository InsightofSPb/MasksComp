#!/usr/bin/env python3
"""Train ByteMSDZipLM on valid targets from aligned RGB residual tiles.

Invalid residual bytes are retained only as zero-valued context placeholders and
are never selected as prediction targets. This preserves tile geometry while
excluding warp-invalid regions from the learned codelength objective.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maskscomp.raw_temporal import ByteMSDZipLM  # noqa: E402


class ValidWindowDataset(Dataset):
    def __init__(
        self,
        sequences: List[np.ndarray],
        valid_sequences: List[np.ndarray],
        timesteps: int,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if len(sequences) != len(valid_sequences):
            raise ValueError("sequences and valid_sequences lengths differ")
        self.sequences = sequences
        self.timesteps = int(timesteps)
        index: List[Tuple[int, int]] = []
        for seq_index, (sequence, valid) in enumerate(zip(sequences, valid_sequences)):
            if sequence.shape != valid.shape:
                raise ValueError("sequence/valid mask shape mismatch")
            # Position zero has no preceding symbol under the inherited LM protocol.
            valid_target_positions = np.flatnonzero(valid[1:] > 0) + 1
            index.extend((seq_index, int(position)) for position in valid_target_positions.tolist())
        if max_samples is not None and len(index) > int(max_samples):
            rng = random.Random(seed)
            index = rng.sample(index, int(max_samples))
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        seq_index, position = self.index[idx]
        sequence = self.sequences[seq_index]
        start = max(0, position - self.timesteps)
        context = np.zeros((self.timesteps,), dtype=np.int64)
        window = sequence[start:position].astype(np.int64)
        context[-window.size:] = window
        target = int(sequence[position])
        return torch.as_tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train valid-target byte MSDZip model on aligned RGB residual tiles.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-train-samples", type=int, default=2000000)
    parser.add_argument("--max-val-samples", type=int, default=200000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--vocab-dim", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError("Missing split list: {}".format(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sequences(dataset_root: Path, pair_ids: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    sequences: List[np.ndarray] = []
    valid_sequences: List[np.ndarray] = []
    for pair_id in pair_ids:
        path = dataset_root / "tiles" / (pair_id + ".npz")
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            tile_sequences = payload["sequences"]
            tile_valid = payload["valid_sequences"]
        if tile_sequences.shape != tile_valid.shape:
            raise ValueError("Sequence validity array mismatch for {}".format(pair_id))
        sequences.extend([tile_sequences[index] for index in range(tile_sequences.shape[0])])
        valid_sequences.extend([tile_valid[index] for index in range(tile_valid.shape[0])])
    return sequences, valid_sequences


@torch.no_grad()
def evaluate(model: ByteMSDZipLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for context, target in loader:
        context = context.to(device)
        target = target.to(device)
        logits = model(context)[:, -1, :]
        loss = F.cross_entropy(logits, target, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(target.numel())
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_ids(args.splits_dir / "facade_train.txt")
    val_ids = load_ids(args.splits_dir / "facade_val.txt")
    train_sequences, train_valid = load_sequences(args.dataset_root, train_ids)
    val_sequences, val_valid = load_sequences(args.dataset_root, val_ids)
    if not train_sequences or not val_sequences:
        raise ValueError("No train/val residual tile sequences loaded")

    train_dataset = ValidWindowDataset(
        train_sequences, train_valid, args.timesteps,
        max_samples=args.max_train_samples, seed=args.seed,
    )
    val_dataset = ValidWindowDataset(
        val_sequences, val_valid, args.timesteps,
        max_samples=args.max_val_samples, seed=args.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    requested_device = args.device
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    model = ByteMSDZipLM(
        timesteps=args.timesteps,
        hidden_dim=args.hidden_dim,
        vocab_dim=args.vocab_dim,
        ffn_dim=args.ffn_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = vars(args).copy()
    run_config.update({
        "n_train_pairs": len(train_ids),
        "n_val_pairs": len(val_ids),
        "n_train_tiles": len(train_sequences),
        "n_val_tiles": len(val_sequences),
        "n_train_target_samples": len(train_dataset),
        "n_val_target_samples": len(val_dataset),
        "target_validity_policy": "invalid residual bytes excluded from loss; retained as zero context placeholders",
    })
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(run_config, indent=2, ensure_ascii=False, default=str))

    best_val_nll = float("inf")
    log_rows: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for context, target in train_loader:
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(context)[:, -1, :]
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(target.numel())
            total_count += int(target.numel())
        train_nll = total_loss / max(total_count, 1)
        val_nll = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_nll": train_nll,
            "val_nll": val_nll,
            "train_bpb": train_nll / np.log(2.0),
            "val_bpb": val_nll / np.log(2.0),
        }
        log_rows.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {"model_state": model.state_dict(), "config": vars(args), "arch": "msdzip-byte-valid"}
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save(checkpoint, args.out_dir / "best.pt")
        torch.save(checkpoint, args.out_dir / "last.pt")

    with (args.out_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print("Best validation bpb: {:.6f}".format(best_val_nll / np.log(2.0)))
    print("Checkpoint: {}".format(args.out_dir / "best.pt"))


if __name__ == "__main__":
    main()
