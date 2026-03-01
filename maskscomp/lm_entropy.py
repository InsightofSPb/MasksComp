from __future__ import annotations

import csv
from collections import OrderedDict
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from maskscomp.models import MSDZipMaskLM
from maskscomp.rle_tokenizer import encode_mask_to_row_tokens

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
    if arr.ndim != 2 or arr.dtype not in (np.uint8, np.uint16):
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
    return sorted(unique[n_val:]), sorted(unique[:n_val])


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
        h, _w = mask.shape
        for row in encode_mask_to_row_tokens(mask):
            n_runs = row.labels.shape[0]
            above_label = np.full((2 * n_runs,), fill_value=no_above_idx, dtype=np.int64)
            above_same = np.zeros((2 * n_runs,), dtype=np.int64)
            for j in range(n_runs):
                t_label = 2 * j
                x = int(row.run_starts[j])
                cur_lab = int(row.labels[j])
                if row.y > 0:
                    raw_above = int(mask[row.y - 1, x])
                    above_label[t_label] = label_to_idx[raw_above]
                    above_same[t_label] = 1 if raw_above == cur_lab else 0
            mapped_tokens = row.tokens.copy()
            is_label = row.token_types == 0
            mapped_tokens[is_label] = np.vectorize(label_to_idx.__getitem__)(mapped_tokens[is_label])
            rows.append(
                RowItem(
                    facade_id=rec.facade_id,
                    rel_path=rec.rel_path,
                    y=row.y,
                    H=h,
                    W=row.width,
                    tokens=mapped_tokens.astype(np.int64, copy=False),
                    token_types=row.token_types.astype(np.int64, copy=False),
                    rem_width=row.rem_width.astype(np.int64, copy=False),
                    above_label=above_label,
                    above_same=above_same,
                )
            )
    return rows


class RowTokenDataset(Dataset):
    def __init__(self, rows: Sequence[RowItem], label_bos_idx: int, no_above_idx: int):
        self.rows = list(rows)
        self.label_bos_idx = int(label_bos_idx)
        self.no_above_idx = int(no_above_idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        input_tokens = np.concatenate(([self.label_bos_idx], row.tokens)).astype(np.int64)
        target_tokens = np.concatenate(([IGNORE_INDEX], row.tokens)).astype(np.int64)
        token_types = np.concatenate(([-1], row.token_types)).astype(np.int64)
        rem_width = np.concatenate(([0], row.rem_width)).astype(np.int64)
        above_label = np.concatenate(([self.no_above_idx], row.above_label)).astype(np.int64)
        above_same = np.concatenate(([0], row.above_same)).astype(np.int64)
        return {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "token_types": token_types,
            "rem_width": rem_width,
            "above_label": above_label,
            "above_same": above_same,
            "meta": row,
        }


def _load_cache_bundle(cache_path: Path) -> Dict[str, np.ndarray]:
    cache_path = Path(cache_path)
    if cache_path.suffix == ".npz":
        with np.load(cache_path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    if cache_path.suffix == ".npy":
        obj = np.load(cache_path, allow_pickle=True)
        if not isinstance(obj, np.ndarray) or obj.dtype != object or obj.size != 1:
            raise ValueError(f"Unsupported cache .npy structure: {cache_path}")
        payload = obj.item()
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported cache .npy payload: {cache_path}")
        return payload
    raise ValueError(f"Unsupported cache extension for {cache_path}")


def load_cache_manifest(manifest_csv: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(manifest_csv).open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            if not row:
                continue
            rows.append(dict(row))
    return rows


class CachedRowTokenDataset(Dataset):
    def __init__(
        self,
        cache_root: Path,
        manifest_csv: Path,
        allowed_facade_ids: Optional[set[str]],
        label_bos_idx: int,
        no_above_idx: int,
        label_to_idx: Optional[Dict[int, int]] = None,
        use_2d_context: bool = False,
        lru_cache_size: int = 8,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.entries = load_cache_manifest(manifest_csv)
        if allowed_facade_ids is not None:
            allowed = set(allowed_facade_ids)
            self.entries = [e for e in self.entries if e["facade_id"] in allowed]
        self.label_bos_idx = int(label_bos_idx)
        self.no_above_idx = int(no_above_idx)
        self.use_2d_context = bool(use_2d_context)
        self.lru_cache_size = max(1, int(lru_cache_size))

        max_label = 0
        for e in self.entries:
            uniq_s = e.get("unique_vals", "")
            if uniq_s:
                max_label = max(max_label, max(int(x) for x in uniq_s.split("|") if x != ""))
        if label_to_idx is None:
            self.label_to_idx = {i: i for i in range(max_label + 1)}
        else:
            self.label_to_idx = {int(k): int(v) for k, v in label_to_idx.items()}

        self.file_meta: List[Tuple[str, str, int, int, Path]] = []
        heights: List[int] = []
        for e in self.entries:
            h = int(e["H"])
            w = int(e["W"])
            cache_rel = e["cache_path"]
            self.file_meta.append((e["facade_id"], e["rel_path"], h, w, self.cache_root / cache_rel))
            heights.append(h)

        self.file_heights = np.asarray(heights, dtype=np.int64)
        self.file_row_offsets = np.zeros((len(self.file_heights) + 1,), dtype=np.int64)
        if self.file_heights.size > 0:
            self.file_row_offsets[1:] = np.cumsum(self.file_heights, dtype=np.int64)
        self.n_files = int(len(self.file_meta))
        self.total_rows = int(self.file_row_offsets[-1])

        self._lru: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.total_rows

    def _locate(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        if idx < 0:
            idx += self.total_rows
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(f"index {idx} out of range for dataset of size {self.total_rows}")
        file_i = int(np.searchsorted(self.file_row_offsets, idx, side="right") - 1)
        row_y = int(idx - int(self.file_row_offsets[file_i]))
        return file_i, row_y

    def _get_cached_file(self, file_i: int) -> Dict[str, np.ndarray]:
        if file_i in self._lru:
            v = self._lru.pop(file_i)
            self._lru[file_i] = v
            return v
        _facade_id, _rel_path, _h, _w, p = self.file_meta[file_i]
        arrs = _load_cache_bundle(p)
        self._lru[file_i] = arrs
        while len(self._lru) > self.lru_cache_size:
            self._lru.popitem(last=False)
        return arrs

    def __getitem__(self, idx: int) -> Dict[str, object]:
        file_i, y = self._locate(int(idx))
        facade_id, rel_path, h, w, _path = self.file_meta[file_i]
        arrs = self._get_cached_file(file_i)

        row_ptr = arrs["row_ptr"]
        starts = arrs["starts"]
        lens = arrs["lens"]
        labels = arrs["labels"]
        a = int(row_ptr[y])
        b = int(row_ptr[y + 1])
        n_runs = b - a

        tokens = np.empty((2 * n_runs,), dtype=np.int64)
        token_types = np.empty((2 * n_runs,), dtype=np.int64)
        rem_width = np.zeros((2 * n_runs,), dtype=np.int64)

        lbl = labels[a:b].astype(np.int64, copy=False)
        mapped_labels = np.array([self.label_to_idx[int(v)] for v in lbl], dtype=np.int64)
        tokens[0::2] = mapped_labels
        tokens[1::2] = lens[a:b].astype(np.int64, copy=False)
        token_types[0::2] = 0
        token_types[1::2] = 1
        rem_width[1::2] = int(w) - starts[a:b].astype(np.int64, copy=False)

        above_label = np.full((2 * n_runs,), fill_value=self.no_above_idx, dtype=np.int64)
        above_same = np.zeros((2 * n_runs,), dtype=np.int64)
        if self.use_2d_context:
            if "above_label" not in arrs or "above_same" not in arrs:
                raise ValueError(
                    "Cache does not include above features. Rebuild cache with --include-above-features for --use-2d-context."
                )
            raw_above = arrs["above_label"][a:b].astype(np.int64, copy=False)
            mapped_above = np.empty_like(raw_above)
            for i, v in enumerate(raw_above.tolist()):
                mapped_above[i] = self.no_above_idx if int(v) == 65535 else self.label_to_idx[int(v)]
            above_label[0::2] = mapped_above
            above_same[0::2] = arrs["above_same"][a:b].astype(np.int64, copy=False)

        row_meta = RowItem(
            facade_id=facade_id,
            rel_path=rel_path,
            y=int(y),
            H=int(h),
            W=int(w),
            tokens=tokens,
            token_types=token_types,
            rem_width=rem_width,
            above_label=above_label,
            above_same=above_same,
        )

        input_tokens = np.concatenate(([self.label_bos_idx], tokens)).astype(np.int64)
        target_tokens = np.concatenate(([IGNORE_INDEX], tokens)).astype(np.int64)
        token_types_i = np.concatenate(([-1], token_types)).astype(np.int64)
        rem_width_i = np.concatenate(([0], rem_width)).astype(np.int64)
        above_label_i = np.concatenate(([self.no_above_idx], above_label)).astype(np.int64)
        above_same_i = np.concatenate(([0], above_same)).astype(np.int64)
        return {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "token_types": token_types_i,
            "rem_width": rem_width_i,
            "above_label": above_label_i,
            "above_same": above_same_i,
            "meta": row_meta,
        }


class FileShuffleRowSampler(Sampler[int]):
    def __init__(self, dataset: CachedRowTokenDataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        n_files = int(self.dataset.n_files)
        file_indices = list(range(n_files))
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(file_indices)
        offsets = self.dataset.file_row_offsets
        for file_i in file_indices:
            start = int(offsets[file_i])
            end = int(offsets[file_i + 1])
            for global_idx in range(start, end):
                yield global_idx

    def __len__(self) -> int:
        return len(self.dataset)


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
        self.max_seq_len = int(max_seq_len)
        self.use_2d_context = bool(use_2d_context)

        self.label_embed = nn.Embedding(self.num_labels + 1, d_model)
        self.len_embed = nn.Embedding(self.wmax + 1, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Embedding(self.max_seq_len, d_model)

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
        if seqlen > self.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}")

        clamped_types = torch.where(token_types < 0, torch.zeros_like(token_types), token_types)
        label_vals = input_tokens.clamp(0, self.num_labels)
        len_vals = input_tokens.clamp(0, self.wmax)
        val_emb = torch.where(
            clamped_types.unsqueeze(-1) == 0,
            self.label_embed(label_vals),
            self.len_embed(len_vals),
        )
        pos_idx = torch.arange(seqlen, device=input_tokens.device).unsqueeze(0).expand(bsz, seqlen)
        x = val_emb + self.pos_embed(pos_idx) + self.type_embed(clamped_types.clamp(0, 1))

        if self.use_2d_context and above_label is not None and above_same is not None:
            ctx = self.above_label_embed(above_label.clamp(0, self.num_labels)) + self.above_same_embed(
                above_same.clamp(0, 1)
            )
            x = x + torch.where((clamped_types == 0).unsqueeze(-1), ctx, torch.zeros_like(x))

        x = self.dropout(x)
        causal = torch.triu(torch.ones(seqlen, seqlen, device=input_tokens.device, dtype=torch.bool), diagonal=1)
        hidden = self.backbone(x, mask=causal, src_key_padding_mask=pad_mask)
        return self.label_head(hidden), self.len_head(hidden)


def build_lm_model(
    arch: str,
    num_labels: int,
    wmax: int,
    max_seq_len: int,
    use_2d_context: bool,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    dropout: float = 0.1,
    timesteps: int = 16,
    vocab_dim: int = 16,
    hidden_dim: int = 128,
    ffn_dim: int = 256,
    layers: int = 4,
) -> nn.Module:
    arch = str(arch).lower()
    if arch == "transformer":
        return LMEntropyModel(
            num_labels=num_labels,
            wmax=wmax,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_2d_context=use_2d_context,
        )
    if arch == "msdzip":
        return MSDZipMaskLM(
            num_labels=num_labels,
            wmax=wmax,
            timesteps=timesteps,
            vocab_dim=vocab_dim,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            layers=layers,
            dropout=dropout,
            use_2d_context=use_2d_context,
        )
    raise ValueError(f"Unknown arch={arch!r}")


def build_model_from_checkpoint_config(cfg: Dict[str, object], use_2d_context: bool) -> nn.Module:
    arch = str(cfg.get("arch", "transformer")).lower()
    wmax = int(cfg["wmax"])
    return build_lm_model(
        arch=arch,
        num_labels=len(cfg["labels"]),
        wmax=wmax,
        max_seq_len=int(cfg.get("max_seq_len", 2 * wmax + 2)),
        use_2d_context=use_2d_context,
        d_model=int(cfg.get("d_model", 256)),
        n_layers=int(cfg.get("n_layers", 4)),
        n_heads=int(cfg.get("n_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        timesteps=int(cfg.get("timesteps", 16)),
        vocab_dim=int(cfg.get("vocab_dim", 16)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        ffn_dim=int(cfg.get("ffn_dim", 256)),
        layers=int(cfg.get("layers", cfg.get("n_layers", 4))),
    )


def collate_rows(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    max_len = max(len(item["input_tokens"]) for item in batch)
    bsz = len(batch)

    input_tokens = torch.zeros((bsz, max_len), dtype=torch.long)
    target_tokens = torch.full((bsz, max_len), fill_value=IGNORE_INDEX, dtype=torch.long)
    token_types = torch.full((bsz, max_len), fill_value=-1, dtype=torch.long)
    rem_width = torch.zeros((bsz, max_len), dtype=torch.long)
    above_label = torch.zeros((bsz, max_len), dtype=torch.long)
    above_same = torch.zeros((bsz, max_len), dtype=torch.long)
    pad_mask = torch.ones((bsz, max_len), dtype=torch.bool)

    meta: List[RowItem] = []
    for i, item in enumerate(batch):
        n = len(item["input_tokens"])
        input_tokens[i, :n] = torch.from_numpy(item["input_tokens"])
        target_tokens[i, :n] = torch.from_numpy(item["target_tokens"])
        token_types[i, :n] = torch.from_numpy(item["token_types"])
        rem_width[i, :n] = torch.from_numpy(item["rem_width"])
        above_label[i, :n] = torch.from_numpy(item["above_label"])
        above_same[i, :n] = torch.from_numpy(item["above_same"])
        pad_mask[i, :n] = False
        meta.append(item["meta"])

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "token_types": token_types,
        "rem_width": rem_width,
        "above_label": above_label,
        "above_same": above_same,
        "pad_mask": pad_mask,
        "meta": meta,
    }


def masked_length_nll_bits(len_logits: torch.Tensor, targets: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(len_logits.shape[-1], device=len_logits.device).view(1, 1, -1)
    allowed = (idx >= 1) & (idx <= remaining.unsqueeze(-1))
    masked_logits = torch.where(allowed, len_logits, torch.full_like(len_logits, -1e9))
    log_probs = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
    gather_idx = targets.clamp(min=0, max=len_logits.shape[-1] - 1)
    gathered = torch.gather(log_probs, dim=-1, index=gather_idx.unsqueeze(-1)).squeeze(-1)
    valid = targets >= 1
    nll_nats = torch.where(valid, -gathered, torch.zeros_like(gathered))
    return nll_nats / math.log(2.0)


def compute_shifted_token_bits(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_2d_context: bool,
) -> Dict[str, torch.Tensor]:
    input_tokens = batch["input_tokens"].to(device)
    token_types = batch["token_types"].to(device)
    rem_width = batch["rem_width"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    above_label = batch["above_label"].to(device)
    above_same = batch["above_same"].to(device)

    label_logits, len_logits = model(
        input_tokens,
        token_types,
        pad_mask,
        above_label=above_label if use_2d_context else None,
        above_same=above_same if use_2d_context else None,
    )

    tgt_ids = input_tokens[:, 1:]
    tgt_types = token_types[:, 1:]
    tgt_remw = rem_width[:, 1:]
    tgt_valid = ~pad_mask[:, 1:]
    pred_label = label_logits[:, :-1, :]
    pred_len = len_logits[:, :-1, :]

    label_pos = tgt_valid & (tgt_types == 0)
    len_pos = tgt_valid & (tgt_types == 1)

    label_log_probs = F.log_softmax(pred_label, dim=-1)
    label_idx = tgt_ids.clamp(min=0, max=pred_label.shape[-1] - 1)
    label_bits = -torch.gather(label_log_probs, -1, label_idx.unsqueeze(-1)).squeeze(-1) / math.log(2.0)
    label_bits = torch.where(label_pos, label_bits, torch.zeros_like(label_bits))

    len_bits = masked_length_nll_bits(pred_len, tgt_ids, tgt_remw)
    len_bits = torch.where(len_pos, len_bits, torch.zeros_like(len_bits))

    return {
        "label_bits": label_bits,
        "len_bits": len_bits,
        "label_pos": label_pos,
        "len_pos": len_pos,
    }


def _last_step_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits[:, -1, :]
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def compute_msdzip_shifted_token_bits(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_2d_context: bool,
    timesteps: int,
) -> Dict[str, torch.Tensor]:
    """Compute per-position next-token bits using fixed MSDZip context windows."""
    input_tokens = batch["input_tokens"].to(device)
    token_types = batch["token_types"].to(device)
    rem_width = batch["rem_width"].to(device)
    pad_mask = batch["pad_mask"].to(device)
    above_label = batch["above_label"].to(device)
    above_same = batch["above_same"].to(device)

    bsz, seqlen = input_tokens.shape
    out_len = max(0, seqlen - 1)
    label_bits = torch.zeros((bsz, out_len), device=device)
    len_bits = torch.zeros((bsz, out_len), device=device)

    if out_len == 0:
        zmask = torch.zeros((bsz, 0), dtype=torch.bool, device=device)
        return {"label_bits": label_bits, "len_bits": len_bits, "label_pos": zmask, "len_pos": zmask}

    bos_vals = input_tokens[:, :1]
    no_above_vals = above_label[:, :1]

    for t in range(out_len):
        start = max(0, t - int(timesteps) + 1)
        ctx_tokens = input_tokens[:, start : t + 1]
        ctx_types = token_types[:, start : t + 1]
        ctx_above_label = above_label[:, start : t + 1]
        ctx_above_same = above_same[:, start : t + 1]
        ctx_len = ctx_tokens.shape[1]

        win_tokens = bos_vals.repeat(1, int(timesteps))
        win_types = torch.full((bsz, int(timesteps)), fill_value=-1, dtype=torch.long, device=device)
        win_above_label = no_above_vals.repeat(1, int(timesteps))
        win_above_same = torch.zeros((bsz, int(timesteps)), dtype=torch.long, device=device)

        win_tokens[:, -ctx_len:] = ctx_tokens
        win_types[:, -ctx_len:] = ctx_types
        win_above_label[:, -ctx_len:] = ctx_above_label
        win_above_same[:, -ctx_len:] = ctx_above_same

        win_pad = torch.zeros((bsz, int(timesteps)), dtype=torch.bool, device=device)
        outputs = model(
            win_tokens,
            win_types,
            win_pad,
            above_label=win_above_label if use_2d_context else None,
            above_same=win_above_same if use_2d_context else None,
        )

        tgt_ids = input_tokens[:, t + 1]
        tgt_types = token_types[:, t + 1]
        tgt_remw = rem_width[:, t + 1]
        tgt_valid = ~pad_mask[:, t + 1]
        label_pos_t = tgt_valid & (tgt_types == 0)
        len_pos_t = tgt_valid & (tgt_types == 1)

        if isinstance(outputs, tuple):
            label_logits = _last_step_logits(outputs[0])
            len_logits = _last_step_logits(outputs[1])

            if label_pos_t.any():
                label_log_probs = F.log_softmax(label_logits[label_pos_t], dim=-1)
                label_idx = tgt_ids[label_pos_t].clamp(min=0, max=label_logits.shape[-1] - 1)
                bits = -torch.gather(label_log_probs, -1, label_idx.unsqueeze(-1)).squeeze(-1) / math.log(2.0)
                label_bits[label_pos_t, t] = bits

            if len_pos_t.any():
                len_bits_t = masked_length_nll_bits(
                    len_logits[len_pos_t].unsqueeze(1),
                    tgt_ids[len_pos_t].unsqueeze(1),
                    tgt_remw[len_pos_t].unsqueeze(1),
                ).squeeze(1)
                len_bits[len_pos_t, t] = len_bits_t
        else:
            logits = _last_step_logits(outputs)
            log_probs = F.log_softmax(logits, dim=-1)
            idx = tgt_ids.clamp(min=0, max=logits.shape[-1] - 1)
            bits_all = -torch.gather(log_probs, -1, idx.unsqueeze(-1)).squeeze(-1) / math.log(2.0)
            bits_all = torch.where(tgt_valid, bits_all, torch.zeros_like(bits_all))
            label_bits[:, t] = torch.where(label_pos_t, bits_all, torch.zeros_like(bits_all))
            len_bits[:, t] = torch.where(len_pos_t, bits_all, torch.zeros_like(bits_all))

    label_pos = (~pad_mask[:, 1:]) & (token_types[:, 1:] == 0)
    len_pos = (~pad_mask[:, 1:]) & (token_types[:, 1:] == 1)
    return {
        "label_bits": label_bits,
        "len_bits": len_bits,
        "label_pos": label_pos,
        "len_pos": len_pos,
    }


def compute_batch_losses(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_2d_context: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    out = compute_shifted_token_bits(model, batch, device, use_2d_context)

    label_pos = out["label_pos"]
    len_pos = out["len_pos"]
    label_bits = out["label_bits"]
    len_bits = out["len_bits"]

    loss_label = torch.tensor(0.0, device=device)
    if label_pos.any():
        loss_label = (label_bits[label_pos] * math.log(2.0)).mean()

    loss_len = torch.tensor(0.0, device=device)
    if len_pos.any():
        loss_len = (len_bits[len_pos] * math.log(2.0)).mean()

    stats = {
        "bits_label": float(label_bits.sum().item()),
        "bits_len": float(len_bits.sum().item()),
        "n_label": int(label_pos.sum().item()),
        "n_len": int(len_pos.sum().item()),
    }
    return loss_label, loss_len, stats


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    sampler: Optional[Sampler[int]] = None,
) -> DataLoader:
    if sampler is not None and shuffle:
        raise ValueError("shuffle must be False when sampler is provided")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=0, collate_fn=collate_rows)


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
