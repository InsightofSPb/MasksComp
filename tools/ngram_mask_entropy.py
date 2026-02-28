#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import pickle
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


PAIR_RE = re.compile(r"(?P<y1>\d{4})_(?P<y2>\d{4})_.*\.png$", re.IGNORECASE)


def parse_years(name: str) -> Tuple[Optional[int], Optional[int]]:
    m = PAIR_RE.match(name)
    if not m:
        return None, None
    return int(m.group("y1")), int(m.group("y2"))


def log2_safe(x: float) -> float:
    return math.log(x, 2) if x > 0.0 else float("inf")


def iter_facade_masks(data_root: Path, subdir: str = "warped_masks") -> Iterable[Tuple[str, Path]]:
    for facade_dir in sorted(data_root.iterdir()):
        if not facade_dir.is_dir():
            continue
        d = facade_dir / subdir
        if not d.exists():
            continue
        for p in sorted(d.glob("*.png")):
            yield facade_dir.name, p


def read_mask_png(path: Path) -> Optional[np.ndarray]:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim != 2:
        return None
    if arr.dtype not in (np.uint8, np.uint16):
        return None
    return arr


def row_rle(mask_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = mask_row.shape[0]
    if w == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=mask_row.dtype),
            np.zeros((0,), dtype=np.int32),
        )
    changes = np.nonzero(mask_row[1:] != mask_row[:-1])[0] + 1
    starts = np.concatenate(([0], changes)).astype(np.int32, copy=False)
    ends = np.concatenate((changes, [w])).astype(np.int32, copy=False)
    lengths = (ends - starts).astype(np.int32, copy=False)
    labels = mask_row[starts]
    return starts, labels, lengths


@dataclass
class Split:
    train_facades: List[str]
    val_facades: List[str]


def split_facades(facade_ids: List[str], val_ratio: float, seed: int) -> Split:
    rng = random.Random(seed)
    ids = sorted(set(facade_ids))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:])
    return Split(train_facades=train, val_facades=val)


class NgramLabelModel:
    def __init__(self, order: int = 2, alpha: float = 0.5, bos_token: int = -1):
        assert order >= 1
        self.order = int(order)
        self.alpha = float(alpha)
        self.bos = int(bos_token)

        self.vocab: List[int] = []
        self.vocab_set: set = set()
        self.V: int = 1

        self.counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self.context_totals: Counter = Counter()

    def fit(self, sequences: Iterable[List[int]]) -> None:
        vocab = set()
        for seq in sequences:
            for x in seq:
                vocab.add(int(x))
        self.vocab = sorted(vocab)
        self.vocab_set = set(self.vocab)
        self.V = len(self.vocab) if self.vocab else 1

        for seq in sequences:
            ctx: Deque[int] = deque([self.bos] * (self.order - 1), maxlen=self.order - 1)
            for x in seq:
                x = int(x)
                context = tuple(ctx) if self.order > 1 else tuple()
                self.counts[context][x] += 1
                self.context_totals[context] += 1
                if self.order > 1:
                    ctx.append(x)

    def neglog2_prob(self, x: int, context: Tuple[int, ...]) -> float:
        x = int(x)
        if x not in self.vocab_set:
            return log2_safe(self.V)
        cnt = self.counts.get(context, None)
        c_xy = cnt[x] if cnt is not None else 0
        c_ctx = self.context_totals.get(context, 0)
        p = (c_xy + self.alpha) / (c_ctx + self.alpha * self.V)
        return -log2_safe(p)


class ExactLenBackoffModel:
    """
    Lossless run-length model with truncation by row remainder.

    Modes:
      - label:  P(len|label) -> P(len)
      - bigram: P(len|prev,label) -> P(len|label) -> P(len)
      - xbin:   P(len|label,xbin) -> P(len|label) -> P(len)
      - above:  P(len|label,above_ctx) -> P(len|label) -> P(len)
               where above_ctx is label in previous row at start_x, optionally with a coarse bin of remaining
               length in prev-row run at start_x (above_run_rem).

    All are categorical on {1..max_width} with add-beta smoothing and exact truncation:
      bits = -log2 p_mix(len) + log2 Z_mix(max_len)
    """

    def __init__(
        self,
        max_width: int,
        beta: float = 0.5,
        mode: str = "above",
        w_pair: float = 0.9,
        w_label: Optional[float] = None,
        w_xbin: float = 0.8,
        xbins: int = 8,
        w_above: float = 0.8,
        above_lenbins: int = 0,
        bos_label: int = -1,
        bos_above: int = -2,
    ):
        if max_width < 1:
            raise ValueError("max_width must be >= 1")
        if mode not in ("label", "bigram", "xbin", "above"):
            raise ValueError("mode must be one of: label, bigram, xbin, above")
        if xbins < 1:
            raise ValueError("xbins must be >= 1")
        if above_lenbins < 0:
            raise ValueError("above_lenbins must be >= 0")

        self.max_width = int(max_width)
        self.beta = float(beta)
        self.mode = str(mode)

        self.w_pair = float(w_pair)
        self.w_label = float(w_label) if w_label is not None else float(w_pair)

        self.w_xbin = float(w_xbin)
        self.xbins = int(xbins)

        self.w_above = float(w_above)
        self.above_lenbins = int(above_lenbins)

        self.bos_label = int(bos_label)
        self.bos_above = int(bos_above)

        # global
        self.global_counts = np.zeros((self.max_width + 1,), dtype=np.int64)
        self.global_total: int = 0
        self.p_global: Optional[np.ndarray] = None
        self.Z_global: Optional[np.ndarray] = None

        # per-label
        self.label_counts: Dict[int, np.ndarray] = {}
        self.label_total: Dict[int, int] = {}
        self.p_label: Dict[int, np.ndarray] = {}
        self.Z_label: Dict[int, np.ndarray] = {}

        # per-pair
        self.pair_counts: Dict[Tuple[int, int], np.ndarray] = {}
        self.pair_total: Dict[Tuple[int, int], int] = {}
        self.p_pair: Dict[Tuple[int, int], np.ndarray] = {}
        self.Z_pair: Dict[Tuple[int, int], np.ndarray] = {}

        # per-(label,xbin)
        self.lx_counts: Dict[Tuple[int, int], np.ndarray] = {}
        self.lx_total: Dict[Tuple[int, int], int] = {}
        self.p_lx: Dict[Tuple[int, int], np.ndarray] = {}
        self.Z_lx: Dict[Tuple[int, int], np.ndarray] = {}

        # per-(label, above_label) or per-(label, above_label, above_lenbin)
        self.la_counts: Dict[Tuple[int, ...], np.ndarray] = {}
        self.la_total: Dict[Tuple[int, ...], int] = {}
        self.p_la: Dict[Tuple[int, ...], np.ndarray] = {}
        self.Z_la: Dict[Tuple[int, ...], np.ndarray] = {}

    def calc_xbin(self, start_x: int, row_w: int) -> int:
        if self.xbins <= 1 or row_w <= 0:
            return 0
        xb = int(self.xbins * float(start_x) / float(row_w))
        if xb >= self.xbins:
            xb = self.xbins - 1
        if xb < 0:
            xb = 0
        return xb

    def calc_lenbin(self, rem: int) -> int:
        """
        Coarse bin for remaining length (>=1). If above_lenbins==0, never used.
        Default: log2 bins clipped into [0, above_lenbins-1].
        """
        if self.above_lenbins <= 1:
            return 0
        rem = max(1, int(rem))
        b = int(math.log2(rem))
        if b < 0:
            b = 0
        if b >= self.above_lenbins:
            b = self.above_lenbins - 1
        return b

    def _ensure_label(self, lab: int) -> None:
        if lab not in self.label_counts:
            self.label_counts[lab] = np.zeros((self.max_width + 1,), dtype=np.int64)
            self.label_total[lab] = 0

    def _ensure_pair(self, prev_lab: int, lab: int) -> None:
        key = (prev_lab, lab)
        if key not in self.pair_counts:
            self.pair_counts[key] = np.zeros((self.max_width + 1,), dtype=np.int64)
            self.pair_total[key] = 0

    def _ensure_lx(self, lab: int, xb: int) -> None:
        key = (lab, xb)
        if key not in self.lx_counts:
            self.lx_counts[key] = np.zeros((self.max_width + 1,), dtype=np.int64)
            self.lx_total[key] = 0

    def _ensure_la(self, key: Tuple[int, ...]) -> None:
        if key not in self.la_counts:
            self.la_counts[key] = np.zeros((self.max_width + 1,), dtype=np.int64)
            self.la_total[key] = 0

    def fit(self, masks: Iterable[np.ndarray]) -> None:
        for mask in masks:
            H, W = mask.shape
            prev_starts = prev_labs = prev_lens = None

            for r in range(H):
                starts, labs, lens = row_rle(mask[r])

                # prepare previous-row RLE to allow O(#runs) pointer lookup for above context
                if self.mode == "above":
                    if r == 0:
                        prev_starts = prev_labs = prev_lens = None
                    else:
                        prev_starts, prev_labs, prev_lens = row_rle(mask[r - 1])

                prev = self.bos_label
                j_prev = 0  # pointer for prev-row runs

                for s, lab, ln in zip(starts.tolist(), labs.tolist(), lens.tolist()):
                    s = int(s)
                    lab = int(lab)
                    ln = int(ln)
                    if ln < 1:
                        continue
                    if ln > self.max_width:
                        ln = self.max_width

                    # global
                    self.global_counts[ln] += 1
                    self.global_total += 1

                    # label
                    self._ensure_label(lab)
                    self.label_counts[lab][ln] += 1
                    self.label_total[lab] += 1

                    if self.mode == "bigram":
                        self._ensure_pair(prev, lab)
                        self.pair_counts[(prev, lab)][ln] += 1
                        self.pair_total[(prev, lab)] += 1

                    if self.mode == "xbin":
                        xb = self.calc_xbin(s, W)
                        self._ensure_lx(lab, xb)
                        self.lx_counts[(lab, xb)][ln] += 1
                        self.lx_total[(lab, xb)] += 1

                    if self.mode == "above":
                        if r == 0 or prev_starts is None:
                            above_lab = self.bos_above
                            above_rem = 1
                        else:
                            # advance pointer to run containing s
                            while j_prev + 1 < len(prev_starts) and s >= int(prev_starts[j_prev] + prev_lens[j_prev]):
                                j_prev += 1
                            above_lab = int(prev_labs[j_prev])
                            above_end = int(prev_starts[j_prev] + prev_lens[j_prev])
                            above_rem = max(1, above_end - s)

                        if self.above_lenbins > 0:
                            key = (lab, above_lab, self.calc_lenbin(above_rem))
                        else:
                            key = (lab, above_lab)
                        self._ensure_la(key)
                        self.la_counts[key][ln] += 1
                        self.la_total[key] += 1

                    prev = lab

        # global probs
        denom_g = self.global_total + self.beta * self.max_width
        p_g = (self.global_counts[1:] + self.beta) / denom_g
        self.p_global = p_g.astype(np.float64, copy=False)
        self.Z_global = np.cumsum(self.p_global, dtype=np.float64)

        # label probs
        for lab, cnts in self.label_counts.items():
            t = self.label_total.get(lab, 0)
            denom_l = t + self.beta * self.max_width
            p_l = (cnts[1:] + self.beta) / denom_l
            self.p_label[lab] = p_l.astype(np.float64, copy=False)
            self.Z_label[lab] = np.cumsum(self.p_label[lab], dtype=np.float64)

        # pair probs
        if self.mode == "bigram":
            for key, cnts in self.pair_counts.items():
                t = self.pair_total.get(key, 0)
                denom_p = t + self.beta * self.max_width
                p_p = (cnts[1:] + self.beta) / denom_p
                self.p_pair[key] = p_p.astype(np.float64, copy=False)
                self.Z_pair[key] = np.cumsum(self.p_pair[key], dtype=np.float64)

        # (label,xbin) probs
        if self.mode == "xbin":
            for key, cnts in self.lx_counts.items():
                t = self.lx_total.get(key, 0)
                denom = t + self.beta * self.max_width
                p = (cnts[1:] + self.beta) / denom
                self.p_lx[key] = p.astype(np.float64, copy=False)
                self.Z_lx[key] = np.cumsum(self.p_lx[key], dtype=np.float64)

        # (label,above) probs
        if self.mode == "above":
            for key, cnts in self.la_counts.items():
                t = self.la_total.get(key, 0)
                denom = t + self.beta * self.max_width
                p = (cnts[1:] + self.beta) / denom
                self.p_la[key] = p.astype(np.float64, copy=False)
                self.Z_la[key] = np.cumsum(self.p_la[key], dtype=np.float64)

    def _pZ_global(self, ln: int, k: int) -> Tuple[float, float]:
        assert self.p_global is not None and self.Z_global is not None
        return float(self.p_global[ln - 1]), float(self.Z_global[k - 1])

    def _pZ_label(self, lab: int, ln: int, k: int) -> Tuple[float, float, bool]:
        arr = self.p_label.get(lab, None)
        Zarr = self.Z_label.get(lab, None)
        if arr is None or Zarr is None:
            return (1.0 / float(self.max_width), float(k) / float(self.max_width), False)
        return (float(arr[ln - 1]), float(Zarr[k - 1]), True)

    def _pZ_pair(self, prev: int, lab: int, ln: int, k: int) -> Tuple[float, float, bool]:
        arr = self.p_pair.get((prev, lab), None)
        Zarr = self.Z_pair.get((prev, lab), None)
        if arr is None or Zarr is None:
            return (1.0 / float(self.max_width), float(k) / float(self.max_width), False)
        return (float(arr[ln - 1]), float(Zarr[k - 1]), True)

    def _pZ_lx(self, lab: int, xb: int, ln: int, k: int) -> Tuple[float, float, bool]:
        key = (lab, xb)
        arr = self.p_lx.get(key, None)
        Zarr = self.Z_lx.get(key, None)
        if arr is None or Zarr is None:
            return (1.0 / float(self.max_width), float(k) / float(self.max_width), False)
        return (float(arr[ln - 1]), float(Zarr[k - 1]), True)

    def _pZ_la(self, key: Tuple[int, ...], ln: int, k: int) -> Tuple[float, float, bool]:
        arr = self.p_la.get(key, None)
        Zarr = self.Z_la.get(key, None)
        if arr is None or Zarr is None:
            return (1.0 / float(self.max_width), float(k) / float(self.max_width), False)
        return (float(arr[ln - 1]), float(Zarr[k - 1]), True)

    def neglog2_prob(
        self,
        run_len: int,
        lab: int,
        max_len: int,
        prev_lab: Optional[int] = None,
        start_x: Optional[int] = None,
        row_w: Optional[int] = None,
        above_lab: Optional[int] = None,
        above_rem: Optional[int] = None,
    ) -> float:
        if self.p_global is None or self.Z_global is None:
            raise RuntimeError("len model not fitted")

        ln = int(run_len)
        k = int(max_len)
        lab = int(lab)
        prev = self.bos_label if prev_lab is None else int(prev_lab)

        if k < 1:
            k = 1
        if k > self.max_width:
            k = self.max_width
        if ln < 1:
            ln = 1
        if ln > k:
            ln = k

        p_g, Z_g = self._pZ_global(ln, k)
        p_l, Z_l, has_l = self._pZ_label(lab, ln, k)
        wL = self.w_label if has_l else 0.0

        p_base = wL * p_l + (1.0 - wL) * p_g
        Z_base = wL * Z_l + (1.0 - wL) * Z_g

        if self.mode == "label":
            p_mix, Z_mix = p_base, Z_base

        elif self.mode == "bigram":
            p_p, Z_p, has_p = self._pZ_pair(prev, lab, ln, k)
            wP = self.w_pair if has_p else 0.0
            p_mix = wP * p_p + (1.0 - wP) * p_base
            Z_mix = wP * Z_p + (1.0 - wP) * Z_base

        elif self.mode == "xbin":
            sx = 0 if start_x is None else int(start_x)
            rw = self.max_width if row_w is None else int(row_w)
            xb = self.calc_xbin(sx, rw)
            p_x, Z_x, has_x = self._pZ_lx(lab, xb, ln, k)
            wX = self.w_xbin if has_x else 0.0
            p_mix = wX * p_x + (1.0 - wX) * p_base
            Z_mix = wX * Z_x + (1.0 - wX) * Z_base

        else:  # above
            a_lab = self.bos_above if above_lab is None else int(above_lab)
            a_rem = 1 if above_rem is None else int(above_rem)
            if self.above_lenbins > 0:
                key = (lab, a_lab, self.calc_lenbin(a_rem))
            else:
                key = (lab, a_lab)

            p_a, Z_a, has_a = self._pZ_la(key, ln, k)
            wA = self.w_above if has_a else 0.0
            p_mix = wA * p_a + (1.0 - wA) * p_base
            Z_mix = wA * Z_a + (1.0 - wA) * Z_base

        return -log2_safe(p_mix) + log2_safe(Z_mix)


def extract_label_seqs_per_row(mask: np.ndarray) -> List[List[int]]:
    H, _W = mask.shape
    out: List[List[int]] = []
    for r in range(H):
        _, labs, _lens = row_rle(mask[r])
        out.append([int(x) for x in labs.tolist()])
    return out


def extract_run_labels_full(mask: np.ndarray) -> List[int]:
    H, _W = mask.shape
    labels_seq: List[int] = []
    for r in range(H):
        _, labs, _lens = row_rle(mask[r])
        labels_seq.extend([int(x) for x in labs.tolist()])
    return labels_seq


def compute_bits_and_heatmap(
    mask: np.ndarray,
    label_model: NgramLabelModel,
    len_model: ExactLenBackoffModel,
    tile: int = 0,
    reset_ctx_per_row: bool = False,
) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
    H, W = mask.shape
    order = label_model.order

    bpp_map = np.zeros((H, W), dtype=np.float32) if (tile and tile > 0) else None

    total_bits = 0.0
    bits_label_sum = 0.0
    bits_len_sum = 0.0
    total_pixels = float(H * W)

    if not reset_ctx_per_row:
        ctx: Deque[int] = deque([label_model.bos] * (order - 1), maxlen=order - 1)

    for r in range(H):
        if reset_ctx_per_row:
            ctx = deque([label_model.bos] * (order - 1), maxlen=order - 1)

        starts, labs, lens = row_rle(mask[r])

        # prev_label for within-row bigram length (if used)
        prev_len_lab = len_model.bos_label

        # prev-row RLE (for above mode)
        if len_model.mode == "above" and r > 0:
            prev_starts, prev_labs, prev_lens = row_rle(mask[r - 1])
            j_prev = 0
        else:
            prev_starts = prev_labs = prev_lens = None
            j_prev = 0

        for s, lab, ln in zip(starts.tolist(), labs.tolist(), lens.tolist()):
            s_i = int(s)
            lab_i = int(lab)
            ln_i = int(ln)

            context = tuple(ctx) if order > 1 else tuple()
            b_lab = label_model.neglog2_prob(lab_i, context)

            # compute above context if needed
            if len_model.mode == "above":
                if r == 0 or prev_starts is None:
                    a_lab = len_model.bos_above
                    a_rem = 1
                else:
                    while j_prev + 1 < len(prev_starts) and s_i >= int(prev_starts[j_prev] + prev_lens[j_prev]):
                        j_prev += 1
                    a_lab = int(prev_labs[j_prev])
                    a_end = int(prev_starts[j_prev] + prev_lens[j_prev])
                    a_rem = max(1, a_end - s_i)
            else:
                a_lab = None
                a_rem = None

            b_len = len_model.neglog2_prob(
                ln_i,
                lab_i,
                max_len=(W - s_i),
                prev_lab=prev_len_lab,
                start_x=s_i,
                row_w=W,
                above_lab=a_lab,
                above_rem=a_rem,
            )

            b_run = b_lab + b_len
            total_bits += b_run
            bits_label_sum += b_lab
            bits_len_sum += b_len

            if bpp_map is not None:
                bpp = float(b_run) / max(1.0, float(ln_i))
                bpp_map[r, s_i : s_i + ln_i] = bpp

            if order > 1:
                ctx.append(lab_i)
            prev_len_lab = lab_i

    bbp = total_bits / total_pixels

    if bpp_map is None:
        return total_bits, bbp, bits_label_sum, bits_len_sum, None

    th = int(math.ceil(H / tile))
    tw = int(math.ceil(W / tile))
    heat = np.zeros((th, tw), dtype=np.float32)
    for ty in range(th):
        y0 = ty * tile
        y1 = min(H, (ty + 1) * tile)
        for tx in range(tw):
            x0 = tx * tile
            x1 = min(W, (tx + 1) * tile)
            heat[ty, tx] = float(bpp_map[y0:y1, x0:x1].mean())
    return total_bits, bbp, bits_label_sum, bits_len_sum, heat


def save_heatmap_png(heat: np.ndarray, out_png: Path) -> None:
    h = heat.astype(np.float32)
    mn = float(np.min(h))
    mx = float(np.max(h))
    if mx <= mn + 1e-12:
        img = np.zeros_like(h, dtype=np.uint8)
    else:
        img = ((h - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def main() -> None:
    ap = argparse.ArgumentParser("Train n-gram entropy model on facade masks and compute bbp/heatmaps")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subdir", default="warped_masks")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--reset-ctx-per-row", action="store_true")

    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--len-model", choices=["label", "bigram", "xbin", "above"], default="above")
    ap.add_argument("--len-backoff", type=float, default=0.95, help="w_pair for bigram")
    ap.add_argument("--len-backoff-label", type=float, default=None, help="w_label (if None, uses len-backoff)")
    ap.add_argument("--len-backoff-xbin", type=float, default=0.8, help="w_xbin for xbin mode")
    ap.add_argument("--len-xbins", type=int, default=8, help="number of x-position bins for xbin mode")

    ap.add_argument("--len-backoff-above", type=float, default=0.8, help="w_above for above mode")
    ap.add_argument("--len-above-lenbins", type=int, default=0, help="0 disables, >0 enables log2 bins for above-run remaining length")

    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--save-heatmaps", action="store_true")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(iter_facade_masks(data_root, subdir=args.subdir))
    if args.max_files and args.max_files > 0:
        items = items[: args.max_files]
    if not items:
        raise SystemExit("No masks found")

    # width support
    max_width_all = 0
    for _fid, p in items:
        m = read_mask_png(p)
        if m is not None:
            max_width_all = max(max_width_all, int(m.shape[1]))
    if max_width_all < 1:
        raise SystemExit("Failed to read any masks")

    facade_ids = [fid for fid, _ in items]
    split = split_facades(facade_ids, val_ratio=args.val_ratio, seed=args.seed)
    train_set = set(split.train_facades)
    val_set = set(split.val_facades)
    train_items = [(fid, p) for fid, p in items if fid in train_set]
    val_items = [(fid, p) for fid, p in items if fid in val_set]

    print(f"[Split] train facades={len(train_set)} val facades={len(val_set)}")
    print(f"[Split] train files={len(train_items)} val files={len(val_items)}")
    print(f"[Data] max_width_all={max_width_all}")

    train_masks: List[np.ndarray] = []
    train_label_seqs: List[List[int]] = []
    for _fid, p in train_items:
        mask = read_mask_png(p)
        if mask is None:
            continue
        train_masks.append(mask)
        if args.reset_ctx_per_row:
            train_label_seqs.extend([row for row in extract_label_seqs_per_row(mask) if row])
        else:
            seq = extract_run_labels_full(mask)
            if seq:
                train_label_seqs.append(seq)

    label_model = NgramLabelModel(order=args.order, alpha=args.alpha, bos_token=-1)
    label_model.fit(train_label_seqs)

    len_model = ExactLenBackoffModel(
        max_width=max_width_all,
        beta=args.beta,
        mode=args.len_model,
        w_pair=args.len_backoff,
        w_label=args.len_backoff_label,
        w_xbin=args.len_backoff_xbin,
        xbins=args.len_xbins,
        w_above=args.len_backoff_above,
        above_lenbins=args.len_above_lenbins,
        bos_label=-1,
        bos_above=-2,
    )
    len_model.fit(train_masks)

    if args.save_model:
        with (out_dir / "model.pkl").open("wb") as f:
            pickle.dump({"label_model": label_model, "len_model": len_model, "args": vars(args)}, f)

    csv_path = out_dir / "ngram_bbp.csv"
    fieldnames = [
        "split","facade_id","filename","y1","y2","H","W","dtype","unique_vals",
        "order","alpha","reset_ctx_per_row",
        "beta","len_model",
        "len_backoff_pair","len_backoff_label","len_backoff_xbin","len_xbins",
        "len_backoff_above","len_above_lenbins",
        "max_width_support",
        "total_bits","bits_label","bits_len","bbp","bbp_label","bbp_len","len_frac_bits",
        "tile","tiles_h","tiles_w","heat_min","heat_p50","heat_p90","heat_max",
    ]

    def eval_items(split_name: str, subset: List[Tuple[str, Path]], heat_dir: Optional[Path]) -> List[dict]:
        rows = []
        for fid, p in subset:
            mask = read_mask_png(p)
            if mask is None:
                continue
            H, W = mask.shape
            u = int(len(np.unique(mask)))
            y1, y2 = parse_years(p.name)

            total_bits, bbp, bits_label, bits_len, heat = compute_bits_and_heatmap(
                mask, label_model, len_model,
                tile=(args.tile if args.save_heatmaps else 0),
                reset_ctx_per_row=args.reset_ctx_per_row,
            )

            total_pixels = float(H * W)
            bbp_label = float(bits_label) / total_pixels
            bbp_len = float(bits_len) / total_pixels
            len_frac = float(bits_len) / max(1e-12, float(total_bits))

            tile_sz = ""
            tiles_h = ""
            tiles_w = ""
            heat_min = ""
            heat_p50 = ""
            heat_p90 = ""
            heat_max = ""

            if heat is not None:
                h = heat.astype(np.float32).ravel()
                tile_sz = args.tile
                tiles_h = heat.shape[0]
                tiles_w = heat.shape[1]
                heat_min = float(np.min(h))
                heat_p50 = float(np.quantile(h, 0.50))
                heat_p90 = float(np.quantile(h, 0.90))
                heat_max = float(np.max(h))

                if heat_dir is not None:
                    rel = f"{fid}/{p.stem}"
                    out_npy = heat_dir / f"{rel}.npy"
                    out_png = heat_dir / f"{rel}.png"
                    out_npy.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_npy, heat)
                    save_heatmap_png(heat, out_png)

            rows.append(dict(
                split=split_name, facade_id=fid, filename=p.name,
                y1=y1 if y1 is not None else "", y2=y2 if y2 is not None else "",
                H=H, W=W, dtype=str(mask.dtype), unique_vals=u,
                order=args.order, alpha=args.alpha, reset_ctx_per_row=bool(args.reset_ctx_per_row),
                beta=args.beta, len_model=args.len_model,
                len_backoff_pair=float(len_model.w_pair),
                len_backoff_label=float(len_model.w_label),
                len_backoff_xbin=float(len_model.w_xbin),
                len_xbins=int(len_model.xbins),
                len_backoff_above=float(len_model.w_above),
                len_above_lenbins=int(len_model.above_lenbins),
                max_width_support=int(len_model.max_width),
                total_bits=float(total_bits), bits_label=float(bits_label), bits_len=float(bits_len),
                bbp=float(bbp), bbp_label=float(bbp_label), bbp_len=float(bbp_len),
                len_frac_bits=float(len_frac),
                tile=tile_sz, tiles_h=tiles_h, tiles_w=tiles_w,
                heat_min=heat_min, heat_p50=heat_p50, heat_p90=heat_p90, heat_max=heat_max,
            ))
        return rows

    heat_dir = out_dir / "heatmaps_tiles" if args.save_heatmaps else None
    rows = eval_items("train", train_items, heat_dir) + eval_items("val", val_items, heat_dir)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote CSV: {csv_path}")
    if args.save_heatmaps:
        print(f"[OK] saved heatmaps to: {heat_dir}")


if __name__ == "__main__":
    main()