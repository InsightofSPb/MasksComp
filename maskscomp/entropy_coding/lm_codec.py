from __future__ import annotations

import hashlib
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from maskscomp.entropy_coding.arith import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitInputStream,
    BitOutputStream,
)
from maskscomp.lm_entropy import LMEntropyModel, read_mask_png
from maskscomp.rle_tokenizer import decode_row_runs, encode_mask_to_row_tokens


@dataclass
class OnlineConfig:
    mode: str = "pretrained"
    optimizer: str = "sgd"
    lr: float = 1e-3
    steps_per_row: int = 1
    clip: float = 1.0
    online_after: str = "row"


def set_deterministic(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def probs_to_cdf(
    probs: np.ndarray,
    allowed_mask: Optional[np.ndarray] = None,
    total_freq: int = 1 << 16,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probs must be 1D")

    v = probs.shape[0]
    allowed = (
        np.ones((v,), dtype=bool)
        if allowed_mask is None
        else np.asarray(allowed_mask, dtype=bool)
    )
    if allowed.shape != (v,):
        raise ValueError("allowed_mask has wrong shape")
    if allowed.sum() == 0:
        raise ValueError("No allowed symbols")
    if int(allowed.sum()) > int(total_freq):
        raise ValueError("allowed symbols exceed total_freq")

    masked = np.where(allowed, np.maximum(probs, 0.0), 0.0)
    total = float(masked.sum())
    if total <= 0.0:
        masked = allowed.astype(np.float64)
        total = float(masked.sum())

    norm = masked / total
    raw = norm * float(total_freq)
    freq = np.zeros((v,), dtype=np.int64)
    freq[allowed] = np.floor(raw[allowed]).astype(np.int64)
    freq[allowed & (freq == 0)] = 1

    delta = int(total_freq - int(freq.sum()))
    if delta > 0:
        rema = raw - np.floor(raw)
        idxs = np.where(allowed)[0]
        order = idxs[np.argsort(-rema[idxs])]
        for i in range(delta):
            freq[order[i % len(order)]] += 1
    elif delta < 0:
        rema = raw - np.floor(raw)
        idxs = np.where((allowed) & (freq > 1))[0]
        if len(idxs) == 0:
            raise ValueError("Cannot adjust frequencies")
        order = idxs[np.argsort(rema[idxs])]
        for i in range(-delta):
            j = order[i % len(order)]
            if freq[j] > 1:
                freq[j] -= 1

    if int(freq.sum()) != int(total_freq):
        raise ValueError("Frequency sum mismatch")
    if np.any(freq[~allowed] != 0):
        raise ValueError("Forbidden symbol received non-zero frequency")

    cumul = np.zeros((v + 1,), dtype=np.uint32)
    cumul[1:] = np.cumsum(freq, dtype=np.uint64)
    return cumul


def checkpoint_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _build_model(
    ckpt: dict, use_2d_context: bool, device: torch.device
) -> LMEntropyModel:
    cfg = ckpt["config"]
    wmax = int(cfg["wmax"])
    model = LMEntropyModel(
        num_labels=len(cfg["labels"]),
        wmax=wmax,
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        dropout=float(cfg["dropout"]),
        max_seq_len=int(cfg.get("max_seq_len", 2 * wmax + 2)),
        use_2d_context=use_2d_context,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _next_probs(
    model: LMEntropyModel,
    prefix_tokens: List[int],
    prefix_types: List[int],
    above_label: List[int],
    above_same: List[int],
    use_2d_context: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    inp = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(0)
    ttypes = torch.tensor(prefix_types, dtype=torch.long, device=device).unsqueeze(0)
    pad = torch.zeros_like(ttypes, dtype=torch.bool)
    al = torch.tensor(above_label, dtype=torch.long, device=device).unsqueeze(0)
    aeq = torch.tensor(above_same, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        label_logits, len_logits = model(
            inp,
            ttypes,
            pad,
            above_label=al if use_2d_context else None,
            above_same=aeq if use_2d_context else None,
        )
    return (
        F.softmax(label_logits[0, -1], dim=-1).detach().cpu().numpy(),
        F.softmax(len_logits[0, -1], dim=-1).detach().cpu().numpy(),
    )


def _online_update_row(
    model: LMEntropyModel,
    optimizer: torch.optim.Optimizer,
    tokens: np.ndarray,
    token_types: np.ndarray,
    rem_width: np.ndarray,
    above_label: np.ndarray,
    above_same: np.ndarray,
    no_above_idx: int,
    clip: float,
    use_2d_context: bool,
    device: torch.device,
) -> None:
    input_tokens = np.concatenate(([no_above_idx], tokens)).astype(np.int64)
    types = np.concatenate(([-1], token_types)).astype(np.int64)
    rem = np.concatenate(([0], rem_width)).astype(np.int64)
    al = np.concatenate(([no_above_idx], above_label)).astype(np.int64)
    aeq = np.concatenate(([0], above_same)).astype(np.int64)

    inp = torch.from_numpy(input_tokens).to(device).unsqueeze(0)
    ttypes = torch.from_numpy(types).to(device).unsqueeze(0)
    rem_t = torch.from_numpy(rem).to(device).unsqueeze(0)
    al_t = torch.from_numpy(al).to(device).unsqueeze(0)
    aeq_t = torch.from_numpy(aeq).to(device).unsqueeze(0)
    pad = torch.zeros_like(ttypes, dtype=torch.bool)

    with torch.enable_grad():
        label_logits, len_logits = model(
            inp,
            ttypes,
            pad,
            above_label=al_t if use_2d_context else None,
            above_same=aeq_t if use_2d_context else None,
        )
        target = inp[:, 1:]
        target_types = ttypes[:, 1:]
        remw = rem_t[:, 1:]

        pred_label = label_logits[:, :-1, :]
        pred_len = len_logits[:, :-1, :]

        loss = torch.tensor(0.0, device=device)
        label_pos = target_types == 0
        if label_pos.any():
            logp = F.log_softmax(pred_label, dim=-1)
            nll = -torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1)
            loss = loss + nll[label_pos].mean()

        len_pos = target_types == 1
        if len_pos.any():
            idx = torch.arange(pred_len.shape[-1], device=device).view(1, 1, -1)
            allowed = (idx >= 1) & (idx <= remw.unsqueeze(-1))
            masked_logits = torch.where(
                allowed, pred_len, torch.full_like(pred_len, -1e9)
            )
            logp = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
            nll = -torch.gather(logp, -1, target.unsqueeze(-1)).squeeze(-1)
            loss = loss + nll[len_pos].mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()


def encode_mask(
    mask: np.ndarray,
    model: LMEntropyModel,
    labels: Sequence[int],
    label_to_idx: Dict[int, int],
    use_2d_context: bool,
    online: OnlineConfig,
    device: torch.device,
) -> Tuple[bytes, float, float]:
    rows = encode_mask_to_row_tokens(mask)
    no_above_idx = len(labels)
    optimizer = None
    if online.mode == "online":
        if online.optimizer != "sgd":
            raise ValueError("Only SGD online optimizer is supported")
        optimizer = torch.optim.SGD(model.parameters(), lr=online.lr, momentum=0.0)

    sink = io.BytesIO()
    enc = ArithmeticEncoder(BitOutputStream(sink))
    ideal_bits = 0.0
    ideal_len_bits = 0.0
    deferred_rows = []

    for row in rows:
        mapped = row.tokens.copy()
        mapped[row.token_types == 0] = np.vectorize(label_to_idx.__getitem__)(
            mapped[row.token_types == 0]
        )
        above_label = np.full((len(mapped),), fill_value=no_above_idx, dtype=np.int64)
        above_same = np.zeros((len(mapped),), dtype=np.int64)
        for j in range(len(row.labels)):
            if row.y == 0:
                continue
            x = int(row.run_starts[j])
            cur_lab = int(row.labels[j])
            ab = int(mask[row.y - 1, x])
            above_label[2 * j] = label_to_idx[ab]
            above_same[2 * j] = 1 if ab == cur_lab else 0

        prefix_tokens = [no_above_idx]
        prefix_types = [-1]
        prefix_al = [no_above_idx]
        prefix_as = [0]
        for t, sym in enumerate(mapped.tolist()):
            next_type = 0 if (t % 2 == 0) else 1
            label_probs, len_probs = _next_probs(
                model,
                prefix_tokens,
                prefix_types,
                prefix_al,
                prefix_as,
                use_2d_context,
                device,
            )
            if next_type == 0:
                cdf = probs_to_cdf(label_probs)
                p = max(float(label_probs[sym]), 1e-12)
            else:
                rem = int(row.rem_width[t])
                allowed = np.zeros((model.wmax + 1,), dtype=bool)
                allowed[1 : rem + 1] = True
                masked = len_probs * allowed
                masked /= max(float(masked.sum()), 1e-12)
                cdf = probs_to_cdf(len_probs, allowed_mask=allowed)
                p = max(float(masked[sym]), 1e-12)

            token_bits = -math.log2(p)
            ideal_bits += token_bits
            if next_type == 1:
                ideal_len_bits += token_bits

            enc.write(cdf, int(sym))
            prefix_tokens.append(int(sym))
            prefix_types.append(next_type)
            prefix_al.append(int(above_label[t]))
            prefix_as.append(int(above_same[t]))

        if online.mode == "online":
            row_tup = (mapped, row.token_types, row.rem_width, above_label, above_same)
            deferred_rows.append(row_tup)
            if online.online_after == "row":
                for _ in range(online.steps_per_row):
                    _online_update_row(
                        model,
                        optimizer,
                        *row_tup,
                        no_above_idx,
                        online.clip,
                        use_2d_context,
                        device,
                    )

    if online.mode == "online" and online.online_after == "image":
        for row_tup in deferred_rows:
            for _ in range(online.steps_per_row):
                _online_update_row(
                    model,
                    optimizer,
                    *row_tup,
                    no_above_idx,
                    online.clip,
                    use_2d_context,
                    device,
                )

    enc.finish()
    return sink.getvalue(), ideal_bits, ideal_len_bits


def decode_mask(
    payload: bytes,
    header: dict,
    model: LMEntropyModel,
    use_2d_context: bool,
    device: torch.device,
) -> np.ndarray:
    h, w = int(header["H"]), int(header["W"])
    labels = [int(x) for x in header["labels"]]
    label_to_raw = {i: lab for i, lab in enumerate(labels)}
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    no_above_idx = len(labels)

    online = OnlineConfig(**header["online"])
    optimizer = None
    if online.mode == "online":
        optimizer = torch.optim.SGD(model.parameters(), lr=online.lr, momentum=0.0)

    out = np.zeros((h, w), dtype=np.dtype(header["dtype"]))
    dec = ArithmeticDecoder(BitInputStream(io.BytesIO(payload)))
    deferred_rows = []

    with torch.set_grad_enabled(False):
        for y in range(h):
            x = 0
            labels_row: List[int] = []
            lens_row: List[int] = []
            mapped: List[int] = []
            types: List[int] = []
            rems: List[int] = []
            above_label_arr: List[int] = []
            above_same_arr: List[int] = []

            prefix_tokens = [no_above_idx]
            prefix_types = [-1]
            prefix_al = [no_above_idx]
            prefix_as = [0]

            while x < w:
                label_probs, _ = _next_probs(
                    model,
                    prefix_tokens,
                    prefix_types,
                    prefix_al,
                    prefix_as,
                    use_2d_context,
                    device,
                )
                cdf_lab = probs_to_cdf(label_probs)
                lab_idx = dec.read(cdf_lab)
                raw_label = label_to_raw[lab_idx]

                ab_idx = no_above_idx
                ab_same = 0
                if y > 0:
                    ab_raw = int(out[y - 1, x])
                    ab_idx = label_to_idx.get(ab_raw, no_above_idx)
                    ab_same = 1 if ab_raw == raw_label else 0

                prefix_tokens.append(int(lab_idx))
                prefix_types.append(0)
                prefix_al.append(ab_idx)
                prefix_as.append(ab_same)
                mapped.append(int(lab_idx))
                types.append(0)
                rems.append(0)
                above_label_arr.append(ab_idx)
                above_same_arr.append(ab_same)

                _, len_probs = _next_probs(
                    model,
                    prefix_tokens,
                    prefix_types,
                    prefix_al,
                    prefix_as,
                    use_2d_context,
                    device,
                )
                allowed = np.zeros((model.wmax + 1,), dtype=bool)
                allowed[1 : (w - x) + 1] = True
                cdf_len = probs_to_cdf(len_probs, allowed_mask=allowed)
                run_len = dec.read(cdf_len)
                if run_len < 1 or run_len > (w - x):
                    raise ValueError(
                        f"Decoded invalid run length {run_len} at row={y}, x={x}"
                    )

                prefix_tokens.append(int(run_len))
                prefix_types.append(1)
                prefix_al.append(no_above_idx)
                prefix_as.append(0)
                mapped.append(int(run_len))
                types.append(1)
                rems.append(w - x)
                above_label_arr.append(no_above_idx)
                above_same_arr.append(0)

                labels_row.append(raw_label)
                lens_row.append(run_len)
                x += run_len

            out[y] = decode_row_runs(
                list(zip(labels_row, lens_row)), w, dtype=out.dtype
            )

            if online.mode == "online":
                row_tup = (
                    np.asarray(mapped, dtype=np.int64),
                    np.asarray(types, dtype=np.int64),
                    np.asarray(rems, dtype=np.int64),
                    np.asarray(above_label_arr, dtype=np.int64),
                    np.asarray(above_same_arr, dtype=np.int64),
                )
                deferred_rows.append(row_tup)
                if online.online_after == "row":
                    for _ in range(online.steps_per_row):
                        _online_update_row(
                            model,
                            optimizer,
                            *row_tup,
                            no_above_idx,
                            online.clip,
                            use_2d_context,
                            device,
                        )

    if online.mode == "online" and online.online_after == "image":
        for row_tup in deferred_rows:
            for _ in range(online.steps_per_row):
                _online_update_row(
                    model,
                    optimizer,
                    *row_tup,
                    no_above_idx,
                    online.clip,
                    use_2d_context,
                    device,
                )

    return out


def load_checkpoint_and_model(
    checkpoint_path: Path, use_2d_context: bool, device: torch.device
) -> Tuple[dict, LMEntropyModel]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    ckpt_use_2d = bool(cfg.get("use_2d_context", False))
    if use_2d_context != ckpt_use_2d:
        raise ValueError(
            f"use_2d_context mismatch: checkpoint={ckpt_use_2d} cli={use_2d_context}"
        )
    return ckpt, _build_model(ckpt, use_2d_context, device)


def build_header(
    mask: np.ndarray,
    ckpt: dict,
    checkpoint_path: Path,
    use_2d_context: bool,
    online: OnlineConfig,
) -> dict:
    cfg = ckpt["config"]
    return {
        "H": int(mask.shape[0]),
        "W": int(mask.shape[1]),
        "dtype": str(mask.dtype),
        "wmax": int(cfg["wmax"]),
        "num_labels": len(cfg["labels"]),
        "labels": [int(x) for x in cfg["labels"]],
        "use_2d_context": bool(use_2d_context),
        "mode": online.mode,
        "online": {
            "mode": online.mode,
            "optimizer": online.optimizer,
            "lr": online.lr,
            "steps_per_row": online.steps_per_row,
            "clip": online.clip,
            "online_after": online.online_after,
        },
        "checkpoint_sha256": checkpoint_sha(checkpoint_path),
    }


def encode_file(
    mask_path: Path,
    checkpoint_path: Path,
    out_bin: Path,
    use_2d_context: bool,
    online: OnlineConfig,
    device: torch.device,
    dump_stats: Optional[Path] = None,
) -> dict:
    set_deterministic(0)
    mask = read_mask_png(mask_path)
    if mask is None:
        raise ValueError(f"Invalid mask: {mask_path}")

    ckpt, model = load_checkpoint_and_model(checkpoint_path, use_2d_context, device)
    cfg = ckpt["config"]
    labels = [int(x) for x in cfg["labels"]]
    label_to_idx = {int(k): int(v) for k, v in cfg["label_to_idx"].items()}

    payload, ideal_bits, ideal_len_bits = encode_mask(
        mask,
        model,
        labels,
        label_to_idx,
        use_2d_context,
        online,
        device,
    )
    header = build_header(mask, ckpt, checkpoint_path, use_2d_context, online)

    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    n = len(header_bytes)
    uv = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            uv.append(b | 0x80)
        else:
            uv.append(b)
            break

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as f:
        f.write(bytes(uv))
        f.write(header_bytes)
        f.write(payload)

    stats = {
        "ideal_bits": ideal_bits,
        "ideal_len_bits": ideal_len_bits,
        "actual_bits_payload": 8 * len(payload),
        "actual_bits_with_header": 8 * (len(payload) + len(header_bytes) + len(uv)),
    }
    stats["overhead_bits"] = stats["actual_bits_payload"] - stats["ideal_bits"]
    stats["overhead_pct"] = (
        (stats["overhead_bits"] / stats["ideal_bits"] * 100.0)
        if stats["ideal_bits"] > 0
        else 0.0
    )
    stats["header_overhead_bits"] = (
        stats["actual_bits_with_header"] - stats["actual_bits_payload"]
    )
    stats["actual_header_overhead_pct"] = (
        (stats["header_overhead_bits"] / stats["actual_bits_payload"] * 100.0)
        if stats["actual_bits_payload"] > 0
        else 0.0
    )

    if dump_stats is not None:
        dump_stats.parent.mkdir(parents=True, exist_ok=True)
        dump_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def read_bin(path: Path) -> Tuple[dict, bytes]:
    blob = path.read_bytes()
    shift = 0
    idx = 0
    length = 0
    while True:
        x = blob[idx]
        idx += 1
        length |= (x & 0x7F) << shift
        if (x & 0x80) == 0:
            break
        shift += 7
    header = json.loads(blob[idx : idx + length].decode("utf-8"))
    payload = blob[idx + length :]
    return header, payload
