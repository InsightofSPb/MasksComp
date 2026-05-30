#!/usr/bin/env python3
"""Train ByteMSDZipLM on valid targets from aligned RGB residual tiles.

Invalid residual bytes are retained only as zero-valued context placeholders and
are never selected as prediction targets. This preserves tile geometry while
excluding warp-invalid regions from the learned codelength objective.

The valid target index is sampled before materialization. This is important for
RGB residual datasets, where tens of thousands of tiles can yield hundreds of
millions of valid byte targets even when only a controlled training subset is
required for a first experiment.
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
from tqdm.auto import tqdm

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
        progress_desc: Optional[str] = None,
        disable_progress: bool = False,
    ) -> None:
        if len(sequences) != len(valid_sequences):
            raise ValueError("sequences and valid_sequences lengths differ")
        self.sequences = sequences
        self.timesteps = int(timesteps)

        counts = np.zeros((len(sequences),), dtype=np.int64)
        counting_iter = enumerate(zip(sequences, valid_sequences))
        counting_iter = tqdm(
            counting_iter,
            total=len(sequences),
            desc=progress_desc or "Indexing valid targets",
            unit="tile",
            leave=False,
            disable=disable_progress,
        )
        for sequence_index, (sequence, valid) in counting_iter:
            if sequence.shape != valid.shape:
                raise ValueError("sequence/valid mask shape mismatch")
            # Position zero has no preceding symbol under the inherited LM protocol.
            counts[sequence_index] = int(np.count_nonzero(valid[1:] > 0))
        cumulative = np.cumsum(counts)
        self.n_available_targets = int(cumulative[-1]) if cumulative.size else 0
        if self.n_available_targets == 0:
            self.sequence_indices = np.empty((0,), dtype=np.int32)
            self.positions = np.empty((0,), dtype=np.int32)
            return

        n_selected = self.n_available_targets
        if max_samples is not None:
            n_selected = min(int(max_samples), self.n_available_targets)
        elif self.n_available_targets > 10_000_000:
            raise ValueError(
                "The dataset contains {} valid byte targets. Pass --max-train-samples/--max-val-samples "
                "to avoid materializing an excessively large full index.".format(self.n_available_targets)
            )

        if n_selected < self.n_available_targets:
            # random.sample(range(...)) selects only requested flat ids without creating
            # a list for the full available-target population.
            selected_flat = np.asarray(
                sorted(random.Random(seed).sample(range(self.n_available_targets), n_selected)),
                dtype=np.int64,
            )
        else:
            selected_flat = np.arange(self.n_available_targets, dtype=np.int64)

        sequence_indices = np.searchsorted(cumulative, selected_flat, side="right").astype(np.int32)
        previous_cumulative = np.concatenate((np.asarray([0], dtype=np.int64), cumulative[:-1]))
        local_ranks = selected_flat - previous_cumulative[sequence_indices]
        positions = np.empty((n_selected,), dtype=np.int32)

        unique_sequences, starts, selected_counts = np.unique(
            sequence_indices, return_index=True, return_counts=True
        )
        positions_iter = zip(unique_sequences.tolist(), starts.tolist(), selected_counts.tolist())
        positions_iter = tqdm(
            positions_iter,
            total=len(unique_sequences),
            desc=(progress_desc or "Indexing valid targets") + " sampled positions",
            unit="tile",
            leave=False,
            disable=disable_progress,
        )
        for sequence_index, start, selected_count in positions_iter:
            end = start + selected_count
            available_positions = np.flatnonzero(valid_sequences[sequence_index][1:] > 0).astype(np.int32) + 1
            positions[start:end] = available_positions[local_ranks[start:end]]

        self.sequence_indices = sequence_indices
        self.positions = positions

    def __len__(self) -> int:
        return int(self.positions.size)

    def __getitem__(self, idx: int):
        sequence_index = int(self.sequence_indices[idx])
        position = int(self.positions[idx])
        sequence = self.sequences[sequence_index]
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
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def load_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError("Missing split list: {}".format(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sequences(
    dataset_root: Path,
    pair_ids: List[str],
    desc: str,
    disable_progress: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    sequences: List[np.ndarray] = []
    valid_sequences: List[np.ndarray] = []
    for pair_id in tqdm(pair_ids, desc=desc, unit="pair", disable=disable_progress):
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
def evaluate(
    model: ByteMSDZipLM,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    disable_progress: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    progress = tqdm(loader, desc="Epoch {} validation".format(epoch), unit="batch", disable=disable_progress)
    for context, target in progress:
        context = context.to(device)
        target = target.to(device)
        logits = model(context)[:, -1, :]
        loss = F.cross_entropy(logits, target, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(target.numel())
        running_nll = total_loss / max(total_count, 1)
        progress.set_postfix(val_bpb="{:.4f}".format(running_nll / np.log(2.0)))
    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_ids(args.splits_dir / "facade_train.txt")
    val_ids = load_ids(args.splits_dir / "facade_val.txt")
    print("Loading train RGB residual tile sequences...", flush=True)
    train_sequences, train_valid = load_sequences(
        args.dataset_root, train_ids, "Loading train tile files", disable_progress=args.no_progress
    )
    print("Loading validation RGB residual tile sequences...", flush=True)
    val_sequences, val_valid = load_sequences(
        args.dataset_root, val_ids, "Loading val tile files", disable_progress=args.no_progress
    )
    if not train_sequences or not val_sequences:
        raise ValueError("No train/val residual tile sequences loaded")

    print("Building sampled valid-target indices...", flush=True)
    train_dataset = ValidWindowDataset(
        train_sequences, train_valid, args.timesteps,
        max_samples=args.max_train_samples, seed=args.seed,
        progress_desc="Indexing train targets", disable_progress=args.no_progress,
    )
    val_dataset = ValidWindowDataset(
        val_sequences, val_valid, args.timesteps,
        max_samples=args.max_val_samples, seed=args.seed,
        progress_desc="Indexing val targets", disable_progress=args.no_progress,
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
        "n_train_available_valid_targets": train_dataset.n_available_targets,
        "n_val_available_valid_targets": val_dataset.n_available_targets,
        "n_train_target_samples": len(train_dataset),
        "n_val_target_samples": len(val_dataset),
        "target_validity_policy": "invalid residual bytes excluded from loss; retained as zero context placeholders",
        "index_sampling_policy": "valid target positions sampled before materializing bounded training/validation index",
    })
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(run_config, indent=2, ensure_ascii=False, default=str), flush=True)

    best_val_nll = float("inf")
    log_rows: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        progress = tqdm(
            train_loader,
            desc="Epoch {}/{} train".format(epoch, args.epochs),
            unit="batch",
            disable=args.no_progress,
        )
        for context, target in progress:
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(context)[:, -1, :]
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(target.numel())
            total_count += int(target.numel())
            running_nll = total_loss / max(total_count, 1)
            progress.set_postfix(
                loss="{:.4f}".format(float(loss.item())),
                avg_bpb="{:.4f}".format(running_nll / np.log(2.0)),
                lr="{:.2e}".format(optimizer.param_groups[0]["lr"]),
            )
        train_nll = total_loss / max(total_count, 1)
        val_nll = evaluate(model, val_loader, device, epoch=epoch, disable_progress=args.no_progress)
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
