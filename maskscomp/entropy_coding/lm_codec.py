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
from maskscomp.lm_entropy import build_model_from_checkpoint_config, read_mask_png
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

    idxs = np.where(allowed)[0]
    k = int(idxs.size)
    if k == 0:
        raise ValueError("No allowed symbols")
    if k > int(total_freq):
        raise ValueError("allowed symbols exceed total_freq")

    masked = np.where(allowed, np.maximum(probs, 0.0), 0.0)
    s = float(masked.sum())
    if s <= 0.0:
        masked = allowed.astype(np.float64)
        s = float(masked.sum())
    masked /= s

    # Allocate at least 1 for each allowed symbol
    freq = np.zeros((v,), dtype=np.int64)
    freq[idxs] = 1

    remaining = int(total_freq) - k
    if remaining < 0:
        raise ValueError("total_freq too small for min-1 allocation")

    if remaining > 0:
        raw = masked[idxs] * float(remaining)
        add = np.floor(raw).astype(np.int64)
        freq[idxs] += add

        leftover = remaining - int(add.sum())
        if leftover > 0:
            frac = raw - np.floor(raw)
            order = idxs[np.argsort(-frac)]
            # leftover < k typically, but cycle just in case
            for i in range(leftover):
                freq[order[i % k]] += 1

    if int(freq.sum()) != int(total_freq):
        # Hard safety fallback: fix on the largest allowed freq
        j = idxs[int(np.argmax(freq[idxs]))]
        freq[j] += int(total_freq) - int(freq.sum())

    if int(freq.sum()) != int(total_freq):
        raise ValueError("Frequency sum mismatch (after fallback)")
    if np.any(freq[~allowed] != 0):
        raise ValueError("Forbidden symbol received non-zero frequency")
    if np.any(freq[idxs] <= 0):
        raise ValueError("Non-positive freq for allowed symbol")

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
) -> torch.nn.Module:
    cfg = ckpt["config"]
    model = build_model_from_checkpoint_config(cfg, use_2d_context=use_2d_context)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _next_probs(
    model: torch.nn.Module,
    prefix_tokens: List[int],
    prefix_types: List[int],
    above_label: List[int],
    above_same: List[int],
    use_2d_context: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "timesteps"):
        t = int(model.timesteps)
        prefix_tokens = prefix_tokens[-t:]
        prefix_types = prefix_types[-t:]
        above_label = above_label[-t:]
        above_same = above_same[-t:]

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
    model: torch.nn.Module,
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

    inp = torch.from_numpy(input_tokens).to(device).unsqueeze(0)   # [1, L]
    ttypes = torch.from_numpy(types).to(device).unsqueeze(0)       # [1, L]
    rem_t = torch.from_numpy(rem).to(device).unsqueeze(0)          # [1, L]
    al_t = torch.from_numpy(al).to(device).unsqueeze(0)            # [1, L]
    aeq_t = torch.from_numpy(aeq).to(device).unsqueeze(0)          # [1, L]
    pad = torch.zeros_like(ttypes, dtype=torch.bool)

    with torch.enable_grad():
        label_logits, len_logits = model(
            inp,
            ttypes,
            pad,
            above_label=al_t if use_2d_context else None,
            above_same=aeq_t if use_2d_context else None,
        )

        # next-token targets
        target = inp[:, 1:]          # [1, L-1] contains BOTH labels and lengths
        target_types = ttypes[:, 1:] # [1, L-1] 0=label, 1=len
        remw = rem_t[:, 1:]          # [1, L-1]

        pred_label = label_logits[:, :-1, :]  # [1, L-1, C]
        pred_len = len_logits[:, :-1, :]      # [1, L-1, Wmax+1]

        loss = torch.tensor(0.0, device=device)

        # ---- label loss: gather ONLY at label positions
        label_pos = target_types == 0
        if label_pos.any():
            logp_lab = F.log_softmax(pred_label, dim=-1)  # [1, L-1, C]
            lp = logp_lab[label_pos]                      # [N, C]
            tg = target[label_pos]                        # [N]
            nll = -lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            loss = loss + nll.mean()

        # ---- length loss: masked normalization, gather ONLY at len positions
        len_pos = target_types == 1
        if len_pos.any():
            idx = torch.arange(pred_len.shape[-1], device=device).view(1, 1, -1)
            allowed = (idx >= 1) & (idx <= remw.unsqueeze(-1))
            masked_logits = torch.where(allowed, pred_len, torch.full_like(pred_len, -1e9))
            logp_len = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)  # [1, L-1, Vlen]
            lp = logp_len[len_pos]   # [N, Vlen]
            tg = target[len_pos]     # [N]
            nll = -lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            loss = loss + nll.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()


def encode_mask(
    mask: np.ndarray,
    model: torch.nn.Module,
    labels: Sequence[int],
    label_to_idx: Dict[int, int],
    use_2d_context: bool,
    online: OnlineConfig,
    device: torch.device,
) -> Tuple[bytes, float, float]:
    """Encode a mask using arithmetic coding with LM probabilities."""
    rows = encode_mask_to_row_tokens(mask)
    no_above_idx = len(labels)
    is_msdzip = hasattr(model, "timesteps")

    optimizer = None
    if online.mode == "online":
        if is_msdzip:
            raise ValueError("Online adaptation is not supported for arch=msdzip")
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
        lab_mask = row.token_types == 0
        if np.any(lab_mask):
            mapped_labels = mapped[lab_mask]
            mapped[lab_mask] = np.array([label_to_idx[int(x)] for x in mapped_labels], dtype=mapped.dtype)

        above_label = np.full((len(mapped),), fill_value=no_above_idx, dtype=np.int64)
        above_same = np.zeros((len(mapped),), dtype=np.int64)
        if row.y > 0:
            for j in range(len(row.labels)):
                x = int(row.run_starts[j])
                cur_lab = int(row.labels[j])
                ab_raw = int(mask[row.y - 1, x])
                ab_idx = label_to_idx.get(ab_raw, no_above_idx)
                above_label[2 * j] = ab_idx
                above_same[2 * j] = 1 if ab_raw == cur_lab else 0

        mapped_list = mapped.tolist()
        types_list = row.token_types.tolist()

        if not is_msdzip:
            input_tokens = np.concatenate(([no_above_idx], mapped.astype(np.int64, copy=False))).astype(np.int64)
            input_types = np.concatenate(([-1], row.token_types.astype(np.int64, copy=False))).astype(np.int64)
            al_full = np.concatenate(([no_above_idx], above_label)).astype(np.int64)
            as_full = np.concatenate(([0], above_same)).astype(np.int64)

            inp = torch.from_numpy(input_tokens).to(device).unsqueeze(0)
            tt = torch.from_numpy(input_types).to(device).unsqueeze(0)
            pad = torch.zeros_like(tt, dtype=torch.bool)

            if use_2d_context:
                al_t = torch.from_numpy(al_full).to(device).unsqueeze(0)
                as_t = torch.from_numpy(as_full).to(device).unsqueeze(0)
            else:
                al_t = None
                as_t = None

            with torch.inference_mode():
                label_logits, len_logits = model(inp, tt, pad, above_label=al_t, above_same=as_t)

            pred_label = label_logits[0, :-1].detach().cpu()
            pred_len = len_logits[0, :-1].detach().cpu()

            for t, sym in enumerate(mapped_list):
                next_type = int(types_list[t])
                if next_type == 0:
                    probs = torch.softmax(pred_label[t], dim=-1).numpy()
                    cdf = probs_to_cdf(probs)
                    p = max(float(probs[int(sym)]), 1e-12)
                else:
                    rem = int(row.rem_width[t])
                    probs = torch.softmax(pred_len[t], dim=-1).numpy()
                    allowed = np.zeros((model.wmax + 1,), dtype=bool)
                    if rem >= 1:
                        allowed[1 : rem + 1] = True
                    denom = float(probs[allowed].sum())
                    p = max(float(probs[int(sym)]) / max(denom, 1e-12), 1e-12)
                    cdf = probs_to_cdf(probs, allowed_mask=allowed)

                token_bits = -math.log2(p)
                ideal_bits += token_bits
                if next_type == 1:
                    ideal_len_bits += token_bits
                enc.write(cdf, int(sym))
        else:
            prefix_tokens = [no_above_idx]
            prefix_types = [-1]
            prefix_al = [no_above_idx]
            prefix_as = [0]

            for t, sym in enumerate(mapped_list):
                next_type = int(types_list[t])
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
                    probs = label_probs
                    cdf = probs_to_cdf(probs)
                    p = max(float(probs[int(sym)]), 1e-12)
                    prefix_al.append(int(above_label[t]))
                    prefix_as.append(int(above_same[t]))
                else:
                    probs = len_probs
                    rem = int(row.rem_width[t])
                    allowed = np.zeros((model.wmax + 1,), dtype=bool)
                    if rem >= 1:
                        allowed[1 : rem + 1] = True
                    denom = float(probs[allowed].sum())
                    p = max(float(probs[int(sym)]) / max(denom, 1e-12), 1e-12)
                    cdf = probs_to_cdf(probs, allowed_mask=allowed)
                    prefix_al.append(no_above_idx)
                    prefix_as.append(0)

                token_bits = -math.log2(p)
                ideal_bits += token_bits
                if next_type == 1:
                    ideal_len_bits += token_bits
                enc.write(cdf, int(sym))

                prefix_tokens.append(int(sym))
                prefix_types.append(next_type)

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
    model: torch.nn.Module,
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
        if hasattr(model, "timesteps"):
            raise ValueError("Online adaptation is not supported for arch=msdzip")
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
) -> Tuple[dict, torch.nn.Module]:
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
