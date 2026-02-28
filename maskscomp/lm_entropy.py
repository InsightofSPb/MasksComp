from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from maskscomp.rle_tokenizer import RowSample, encode_mask_to_row_tokens

IGNORE_INDEX = -100


@dataclass
class ImageRecord:
    facade_id: str
    path: Path
    rel_path: str


@dataclass
class RowItem:
    facade_id: str
    rel_path: str
    y: int
    H: int
    W: int
    tokens: np.ndarray
    token_types: np.ndarray
    rem_width: np.ndarray
    above_label: np.ndarray
    above_same: np.ndarray


def read_mask_png(path: Path) -> Optional[np.ndarray]:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3:
        if arr.shape[2] != 3:
            return None
        if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
            arr = arr[..., 0]
        else:
            return None
    if arr.ndim != 2:
        return None
    if arr.dtype not in (np.uint8, np.uint16):
        return None
    return arr


def collect_images(data_root: Path, subdir: str = "warped_masks") -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for facade_dir in sorted(data_root.iterdir()):
        if not facade_dir.is_dir():
            continue
        d = facade_dir / subdir
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png")):
            records.append(ImageRecord(facade_id=facade_dir.name, path=p, rel_path=str(p.relative_to(data_root))))
    return records


def split_facades(facade_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    unique = sorted(set(facade_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    if len(unique) <= 1:
        return unique, []
    n_val = max(1, int(round(len(unique) * float(val_ratio))))
    n_val = min(n_val, len(unique) - 1)
    val = sorted(unique[:n_val])
    train = sorted(unique[n_val:])
    return train, val


def save_split_lists(train_facades: Sequence[str], val_facades: Sequence[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "facade_train.txt").write_text("\n".join(train_facades) + "\n", encoding="utf-8")
    (out_dir / "facade_val.txt").write_text("\n".join(val_facades) + "\n", encoding="utf-8")


def load_split_list(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def compute_dataset_stats(records: Sequence[ImageRecord]) -> Tuple[int, List[int]]:
    max_w = 1
    labels = set()
    for rec in records:
        mask = read_mask_png(rec.path)
        if mask is None:
            continue
        max_w = max(max_w, int(mask.shape[1]))
        labels.update(int(x) for x in np.unique(mask))
    return max_w, sorted(labels)


def build_row_items(records: Sequence[ImageRecord], label_to_idx: Dict[int, int], no_above_idx: int) -> List[RowItem]:
    rows: List[RowItem] = []
    for rec in records:
        mask = read_mask_png(rec.path)
        if mask is None:
            print(f"[WARN] skipping non-mask image: {rec.path}")
            continue
        h, w = mask.shape
        encoded_rows = encode_mask_to_row_tokens(mask)
        for row in encoded_rows:
            n_runs = row.labels.shape[0]
            above_label = np.full((2 * n_runs,), fill_value=no_above_idx, dtype=np.int64)
            above_same = np.zeros((2 * n_runs,), dtype=np.int64)
            for j in range(n_runs):
                t_label = 2 * j
                start_x = int(row.run_starts[j])
                cur_label = int(row.labels[j])
                if rec.path is not None and row.y > 0:
                    raw_above = int(mask[row.y - 1, start_x])
                    above_idx = label_to_idx[raw_above]
                    same = 1 if raw_above == cur_label else 0
                else:
                    above_idx = no_above_idx
                    same = 0
                above_label[t_label] = above_idx
                above_same[t_label] = same
            mapped_tokens = row.tokens.copy()
            mapped_tokens[row.token_types == 0] = np.vectorize(label_to_idx.__getitem__)(
                mapped_tokens[row.token_types == 0]
            )
            rows.append(
                RowItem(
                    facade_id=rec.facade_id,
                    rel_path=rec.rel_path,
                    y=row.y,
                    H=h,
                    W=w,
                    tokens=mapped_tokens.astype(np.int64, copy=False),
                    token_types=row.token_types.astype(np.int64, copy=False),
                    rem_width=row.rem_width.astype(np.int64, copy=False),
                    above_label=above_label,
                    above_same=above_same,
                )
            )
    return rows


class RowTokenDataset(Dataset):
    def __init__(self, rows: Sequence[RowItem], label_bos_idx: int):
        self.rows = list(rows)
        self.label_bos_idx = int(label_bos_idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        inp = np.concatenate(([self.label_bos_idx], row.tokens)).astype(np.int64)
        tgt = np.concatenate(([IGNORE_INDEX], row.tokens)).astype(np.int64)
        ttypes = np.concatenate(([-1], row.token_types)).astype(np.int64)
        rem = np.concatenate(([0], row.rem_width)).astype(np.int64)
        above_label = np.concatenate(([0], row.above_label)).astype(np.int64)
        above_same = np.concatenate(([0], row.above_same)).astype(np.int64)
        return {
            "input_tokens": inp,
            "target_tokens": tgt,
            "token_types": ttypes,
            "rem_width": rem,
            "above_label": above_label,
            "above_same": above_same,
            "meta": row,
        }


class LMEntropyModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
        wmax: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_2d_context: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.wmax = int(wmax)
        self.use_2d_context = bool(use_2d_context)

        self.label_embed = nn.Embedding(self.num_labels + 1, d_model)
        self.len_embed = nn.Embedding(self.wmax + 1, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.above_label_embed = nn.Embedding(self.num_labels + 1, d_model)
        self.above_same_embed = nn.Embedding(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.label_head = nn.Linear(d_model, self.num_labels)
        self.len_head = nn.Linear(d_model, self.wmax + 1)

    def forward(
        self,
        input_tokens: torch.Tensor,
        token_types: torch.Tensor,
        pad_mask: torch.Tensor,
        above_label: Optional[torch.Tensor] = None,
        above_same: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen = input_tokens.shape
        device = input_tokens.device

        clamped_types = torch.where(token_types < 0, torch.zeros_like(token_types), token_types)
        label_vals = input_tokens.clamp(0, self.num_labels)
        len_vals = input_tokens.clamp(0, self.wmax)
        val_emb = torch.where(
            clamped_types.unsqueeze(-1) == 0,
            self.label_embed(label_vals),
            self.len_embed(len_vals),
        )
        pos_idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
        x = val_emb + self.pos_embed(pos_idx) + self.type_embed(clamped_types.clamp(0, 1))

        if self.use_2d_context and above_label is not None and above_same is not None:
            lbl_ctx = self.above_label_embed(above_label.clamp(0, self.num_labels))
            same_ctx = self.above_same_embed(above_same.clamp(0, 1))
            label_mask = (clamped_types == 0).unsqueeze(-1)
            x = x + torch.where(label_mask, lbl_ctx + same_ctx, torch.zeros_like(x))

        x = self.dropout(x)

        causal = torch.triu(torch.ones(seqlen, seqlen, device=device, dtype=torch.bool), diagonal=0)
        hidden = self.backbone(x, mask=causal, src_key_padding_mask=pad_mask)
        label_logits = self.label_head(hidden)
        len_logits = self.len_head(hidden)
        return label_logits, len_logits


def collate_rows(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    max_len = max(int(len(item["input_tokens"])) for item in batch)
    bsz = len(batch)

    inp = torch.zeros((bsz, max_len), dtype=torch.long)
    tgt = torch.full((bsz, max_len), fill_value=IGNORE_INDEX, dtype=torch.long)
    types = torch.full((bsz, max_len), fill_value=-1, dtype=torch.long)
    rem = torch.zeros((bsz, max_len), dtype=torch.long)
    above_label = torch.zeros((bsz, max_len), dtype=torch.long)
    above_same = torch.zeros((bsz, max_len), dtype=torch.long)
    pad_mask = torch.ones((bsz, max_len), dtype=torch.bool)

    meta: List[RowItem] = []
    for i, item in enumerate(batch):
        n = len(item["input_tokens"])
        inp[i, :n] = torch.from_numpy(item["input_tokens"])
        tgt[i, :n] = torch.from_numpy(item["target_tokens"])
        types[i, :n] = torch.from_numpy(item["token_types"])
        rem[i, :n] = torch.from_numpy(item["rem_width"])
        above_label[i, :n] = torch.from_numpy(item["above_label"])
        above_same[i, :n] = torch.from_numpy(item["above_same"])
        pad_mask[i, :n] = False
        meta.append(item["meta"])

    return {
        "input_tokens": inp,
        "target_tokens": tgt,
        "token_types": types,
        "rem_width": rem,
        "above_label": above_label,
        "above_same": above_same,
        "pad_mask": pad_mask,
        "meta": meta,
    }


def masked_length_nll_bits(len_logits: torch.Tensor, targets: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
    device = len_logits.device
    idx = torch.arange(len_logits.shape[-1], device=device).view(1, 1, -1)
    allowed = (idx >= 1) & (idx <= remaining.unsqueeze(-1))
    neg_inf = torch.full_like(len_logits, fill_value=-1e9)
    masked_logits = torch.where(allowed, len_logits, neg_inf)
    log_probs = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
    gather = torch.gather(log_probs, dim=-1, index=targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    valid = targets >= 1
    nll_nats = torch.where(valid, -gather, torch.zeros_like(gather))
    return nll_nats / math.log(2.0)


def compute_batch_losses(
    model: LMEntropyModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_2d_context: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    inp = batch["input_tokens"].to(device)
    tgt = batch["target_tokens"].to(device)
    ttypes = batch["token_types"].to(device)
    rem = batch["rem_width"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    above_label = batch["above_label"].to(device)
    above_same = batch["above_same"].to(device)

    label_logits, len_logits = model(
        inp,
        ttypes,
        pad_mask,
        above_label=above_label if use_2d_context else None,
        above_same=above_same if use_2d_context else None,
    )

    label_pos = (ttypes == 0) & (tgt >= 0) & (~pad_mask)
    len_pos = (ttypes == 1) & (tgt >= 1) & (~pad_mask)

    loss_label = torch.tensor(0.0, device=device)
    if label_pos.any():
        loss_label = F.cross_entropy(label_logits[label_pos], tgt[label_pos], reduction="mean")

    loss_len = torch.tensor(0.0, device=device)
    if len_pos.any():
        bits = masked_length_nll_bits(len_logits, tgt, rem)
        loss_len = (bits[len_pos] * math.log(2.0)).mean()

    with torch.no_grad():
        bits_label = 0.0
        bits_len = 0.0
        if label_pos.any():
            lps = F.log_softmax(label_logits, dim=-1)
            gathered = torch.gather(lps, -1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            bits_label = float((-gathered[label_pos] / math.log(2.0)).sum().item())
        if len_pos.any():
            bits = masked_length_nll_bits(len_logits, tgt, rem)
            bits_len = float(bits[len_pos].sum().item())

    stats = {
        "bits_label": bits_label,
        "bits_len": bits_len,
        "n_label": int(label_pos.sum().item()),
        "n_len": int(len_pos.sum().item()),
    }
    return loss_label, loss_len, stats


def make_loader(dataset: RowTokenDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_rows)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def dump_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
