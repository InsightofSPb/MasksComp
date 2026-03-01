from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _normalize_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3 and logits.shape[1] == 1:
        return logits[:, 0, :]
    if logits.ndim == 2:
        return logits
    if logits.ndim == 3:
        return logits[:, -1, :]
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def _build_window_batch(batch: Dict[str, torch.Tensor], timesteps: int) -> Optional[Dict[str, torch.Tensor]]:
    input_tokens = batch["input_tokens"]
    token_types = batch["token_types"]
    rem_width = batch["rem_width"]
    pad_mask = batch["pad_mask"]
    above_label = batch["above_label"]
    above_same = batch["above_same"]

    bsz = input_tokens.shape[0]
    bos_vals = input_tokens[:, :1]
    no_above_vals = above_label[:, :1]

    win_inp = []
    win_types = []
    win_al = []
    win_as = []
    tgt_ids = []
    tgt_types = []
    tgt_rem = []
    row_ids = []

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
            row_ids.append(b)

    if not win_inp:
        return None

    return {
        "inp": torch.stack(win_inp, dim=0),
        "ttypes": torch.stack(win_types, dim=0),
        "al": torch.stack(win_al, dim=0),
        "aeq": torch.stack(win_as, dim=0),
        "target_ids": torch.stack(tgt_ids, dim=0),
        "target_types": torch.stack(tgt_types, dim=0),
        "target_rem": torch.stack(tgt_rem, dim=0),
        "row_ids": torch.tensor(row_ids, dtype=torch.long),
        "batch_size": torch.tensor(bsz, dtype=torch.long),
    }


def compute_msdzip_window_loss_stats(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_2d_context: bool,
    timesteps: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Compute MSDZip fixed-context next-token losses/bit statistics.

    This matches training semantics: every valid token at position `t` is scored using
    a left-padded context window built from positions `[max(0, t-timesteps), t)`.
    """

    prepared = _build_window_batch(batch, timesteps=int(timesteps))
    if prepared is None:
        z = torch.tensor(0.0, device=device)
        bsz = len(batch["meta"])
        return z, z, {
            "bits_label": 0.0,
            "bits_len": 0.0,
            "n_label": 0,
            "n_len": 0,
            "row_bits_label": torch.zeros((bsz,), dtype=torch.float32),
            "row_bits_len": torch.zeros((bsz,), dtype=torch.float32),
        }

    inp = prepared["inp"].to(device)
    ttypes = prepared["ttypes"].to(device)
    al = prepared["al"].to(device)
    aeq = prepared["aeq"].to(device)
    target_ids = prepared["target_ids"].to(device)
    target_types = prepared["target_types"].to(device)
    target_rem = prepared["target_rem"].to(device)
    row_ids = prepared["row_ids"].to(device)
    bsz = int(prepared["batch_size"].item())
    pad = torch.zeros_like(inp, dtype=torch.bool)

    outputs = model(
        inp,
        ttypes,
        pad,
        above_label=al if use_2d_context else None,
        above_same=aeq if use_2d_context else None,
    )

    log2e = 1.0 / math.log(2.0)
    row_bits_label = torch.zeros((bsz,), dtype=torch.float32, device=device)
    row_bits_len = torch.zeros((bsz,), dtype=torch.float32, device=device)

    if isinstance(outputs, tuple):
        label_logits, len_logits = outputs
        pred_label = _normalize_logits_shape(label_logits)
        pred_len = _normalize_logits_shape(len_logits)

        label_pos = target_types == 0
        len_pos = target_types == 1

        label_bits = torch.zeros((target_ids.shape[0],), device=device)
        if label_pos.any():
            lp = F.log_softmax(pred_label[label_pos], dim=-1)
            tg = target_ids[label_pos].clamp(min=0, max=pred_label.shape[-1] - 1)
            label_bits[label_pos] = -lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1) * log2e

        len_bits = torch.zeros((target_ids.shape[0],), device=device)
        if len_pos.any():
            len_sel = pred_len[len_pos]
            rem = target_rem[len_pos]
            idx = torch.arange(len_sel.shape[-1], device=device).view(1, -1)
            allowed = (idx >= 1) & (idx <= rem.unsqueeze(-1))
            masked_logits = torch.where(allowed, len_sel, torch.full_like(len_sel, -1e9))
            logp = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
            tg = target_ids[len_pos].clamp(min=1, max=len_sel.shape[-1] - 1)
            len_bits[len_pos] = -logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1) * log2e

        row_bits_label.scatter_add_(0, row_ids, label_bits)
        row_bits_len.scatter_add_(0, row_ids, len_bits)

        loss_label = torch.tensor(0.0, device=device)
        if label_pos.any():
            loss_label = (label_bits[label_pos] / log2e).mean()
        loss_len = torch.tensor(0.0, device=device)
        if len_pos.any():
            loss_len = (len_bits[len_pos] / log2e).mean()

        stats = {
            "bits_label": float(label_bits.sum().item()),
            "bits_len": float(len_bits.sum().item()),
            "n_label": int(label_pos.sum().item()),
            "n_len": int(len_pos.sum().item()),
            "row_bits_label": row_bits_label.detach().cpu(),
            "row_bits_len": row_bits_len.detach().cpu(),
        }
        return loss_label, loss_len, stats

    logits = _normalize_logits_shape(outputs)
    logp = F.log_softmax(logits, dim=-1)
    tg = target_ids.clamp(min=0, max=logits.shape[-1] - 1)
    bits = -logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1) * log2e

    row_bits_label.scatter_add_(0, row_ids, bits)
    loss = (bits / log2e).mean() if bits.numel() > 0 else torch.tensor(0.0, device=device)
    stats = {
        "bits_label": float(bits.sum().item()),
        "bits_len": 0.0,
        "n_label": int(bits.numel()),
        "n_len": 0,
        "row_bits_label": row_bits_label.detach().cpu(),
        "row_bits_len": row_bits_len.detach().cpu(),
    }
    return loss, torch.tensor(0.0, device=device), stats
