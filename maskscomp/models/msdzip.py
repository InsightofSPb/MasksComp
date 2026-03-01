from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MSDZipConfig:
    num_labels: int
    wmax: int
    timesteps: int = 16
    vocab_dim: int = 16
    hidden_dim: int = 128
    ffn_dim: int = 256
    layers: int = 4
    dropout: float = 0.1
    use_2d_context: bool = False


class _MixBlock(nn.Module):
    """Temporal mixing + feed-forward block inspired by MSDZip MixedModel."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dwconv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=True,
        )
        self.pwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        y = self.pwconv(self.dwconv(y)).transpose(1, 2)
        x = x + self.dropout(y)
        z = self.ffn(self.norm2(x))
        return x + self.dropout(z)


class MixedModel(nn.Module):
    """
    MSDZip-style predictor backbone without online cache side effects.

    Cache is disabled by default for training/inference reproducibility. A no-op
    reset_cache() API is provided for compatibility with online execution loops.
    """

    def __init__(
        self,
        timesteps: int,
        hidden_dim: int,
        ffn_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        self.pos_embed = nn.Embedding(self.timesteps, hidden_dim)
        self.blocks = nn.ModuleList(
            [_MixBlock(hidden_dim=hidden_dim, ffn_dim=ffn_dim, dropout=dropout) for _ in range(int(layers))]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self._cache_enabled = False
        self._last: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        self._last = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        bsz, seqlen, _ = x.shape
        if seqlen > self.timesteps:
            raise ValueError(f"Sequence length {seqlen} exceeds timesteps={self.timesteps}")
        pos = torch.arange(seqlen, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, seqlen)
        h = x + self.pos_embed(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        if self._cache_enabled:
            self._last = h[:, -1:].detach()
        return h


class MSDZipMaskLM(nn.Module):
    def __init__(
        self,
        num_labels: int,
        wmax: int,
        timesteps: int = 16,
        vocab_dim: int = 16,
        hidden_dim: int = 128,
        ffn_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.1,
        use_2d_context: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.wmax = int(wmax)
        self.timesteps = int(timesteps)
        self.use_2d_context = bool(use_2d_context)

        self.label_embed = nn.Embedding(self.num_labels + 1, vocab_dim)
        self.len_embed = nn.Embedding(self.wmax + 1, vocab_dim)
        self.type_embed = nn.Embedding(2, vocab_dim)

        self.above_label_embed = nn.Embedding(self.num_labels + 1, vocab_dim)
        self.above_same_embed = nn.Embedding(2, vocab_dim)

        self.in_proj = nn.Linear(vocab_dim, hidden_dim)
        self.backbone = MixedModel(
            timesteps=self.timesteps,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            layers=layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.label_head = nn.Linear(hidden_dim, self.num_labels)
        self.len_head = nn.Linear(hidden_dim, self.wmax + 1)

    def reset_cache(self) -> None:
        self.backbone.reset_cache()

    def forward(
        self,
        input_tokens: torch.Tensor,
        token_types: torch.Tensor,
        pad_mask: torch.Tensor,
        above_label: Optional[torch.Tensor] = None,
        above_same: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del pad_mask  # explicit: fixed-context model does not use padding mask internally
        bsz, seqlen = input_tokens.shape
        if seqlen > self.timesteps:
            raise ValueError(f"Sequence length {seqlen} exceeds timesteps={self.timesteps}")

        clamped_types = torch.where(token_types < 0, torch.zeros_like(token_types), token_types)
        label_vals = input_tokens.clamp(0, self.num_labels)
        len_vals = input_tokens.clamp(0, self.wmax)

        token_emb = torch.where(
            clamped_types.unsqueeze(-1) == 0,
            self.label_embed(label_vals),
            self.len_embed(len_vals),
        )
        x = token_emb + self.type_embed(clamped_types.clamp(0, 1))

        if self.use_2d_context and above_label is not None and above_same is not None:
            ctx = self.above_label_embed(above_label.clamp(0, self.num_labels)) + self.above_same_embed(
                above_same.clamp(0, 1)
            )
            x = x + torch.where((clamped_types == 0).unsqueeze(-1), ctx, torch.zeros_like(x))

        x = self.in_proj(x)
        self.reset_cache()
        hidden = self.backbone(self.dropout(x))
        return self.label_head(hidden), self.len_head(hidden)
