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
    w = int(mask_row.shape[0])
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
    if len(ids) <= 1:
        return Split(train_facades=ids, val_facades=[])
    n_val = max(1, int(round(len(ids) * float(val_ratio))))
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:])
    return Split(train_facades=train, val_facades=val)


def scan_dataset_stats(items: List[Tuple[str, Path]]) -> Tuple[int, int]:
    max_w = 0
    max_lab = 0
    for _fid, p in items:
        m = read_mask_png(p)
        if m is None:
            continue
        max_w = max(max_w, int(m.shape[1]))
        mx = int(np.max(m))
        max_lab = max(max_lab, mx)
    return max_w, max_lab


def make_tokens_for_row(
    starts: np.ndarray,
    labs: np.ndarray,
    lens: np.ndarray,
    label_vocab: int,
    max_len_support: int,
) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for s, lab, ln in zip(starts.tolist(), labs.tolist(), lens.tolist()):
        s_i = int(s)
        lab_i = int(lab)
        ln_i = int(ln)
        if ln_i < 1:
            continue
        if lab_i < 0 or lab_i >= label_vocab:
            continue
        if ln_i > max_len_support:
            ln_i = max_len_support  # lossy if max_len_support < real widths
        t_lab = lab_i
        t_len = label_vocab + (ln_i - 1)
        out.append((s_i, ln_i, t_lab, t_len))
    return out


class SeqModel:
    def neglog2_prob(self, sym: int, history: Deque[int]) -> float:
        raise NotImplementedError

    def prob(self, sym: int, history: Deque[int]) -> float:
        b = self.neglog2_prob(sym, history)
        if not math.isfinite(b):
            return 0.0
        return float(2.0 ** (-b))

    def mass_over_range(self, history: Deque[int], lo: int, hi: int) -> float:
        # Fallback: exact sum by enumeration (can be very slow for big ranges).
        lo = int(lo)
        hi = int(hi)
        if hi < lo:
            return 0.0
        Z = 0.0
        for s in range(lo, hi + 1):
            Z += float(self.prob(s, history))
        return max(Z, 0.0)


class PPMMethodC(SeqModel):
    """
    PPM (Method C + exclusion) over alphabet size A.
    Also implements mass_over_range efficiently (no enumeration over all symbols).
    """

    def __init__(self, max_order: int, alphabet_size: int):
        if max_order < 0:
            raise ValueError("max_order must be >= 0")
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be >= 2")
        self.K = int(max_order)
        self.A = int(alphabet_size)

        self.counts: List[Dict[Tuple[int, ...], Counter]] = [defaultdict(Counter) for _ in range(self.K + 1)]
        self.totals: List[Dict[Tuple[int, ...], int]] = [defaultdict(int) for _ in range(self.K + 1)]

    @staticmethod
    def _ctx(history: Deque[int], k: int) -> Tuple[int, ...]:
        if k <= 0:
            return tuple()
        h = tuple(history)
        return h[-k:]

    def fit_on_token_streams(self, token_rows: Iterable[List[int]], reset_ctx_per_row: bool = True) -> None:
        hist: Deque[int] = deque(maxlen=self.K)
        for row_tokens in token_rows:
            if reset_ctx_per_row:
                hist.clear()
            for sym in row_tokens:
                sym = int(sym)
                max_k = min(self.K, len(hist))
                for k in range(0, max_k + 1):
                    ctx = self._ctx(hist, k)
                    self.counts[k][ctx][sym] += 1
                    self.totals[k][ctx] += 1
                hist.append(sym)

    def prob(self, sym: int, history: Deque[int]) -> float:
        sym = int(sym)
        if sym < 0 or sym >= self.A:
            return max(1.0 / float(self.A), 1e-12)

        excluded = set()
        p_prefix = 1.0

        max_k = min(self.K, len(history))
        for k in range(max_k, -1, -1):
            ctx = self._ctx(history, k)
            cnt = self.counts[k].get(ctx, None)
            if cnt is None or len(cnt) == 0:
                continue

            if excluded:
                N = 0
                U = 0
                c_sym = 0
                for a, c in cnt.items():
                    if a in excluded:
                        continue
                    N += int(c)
                    U += 1
                    if a == sym:
                        c_sym = int(c)
            else:
                N = int(self.totals[k].get(ctx, 0))
                U = int(len(cnt))
                c_sym = int(cnt.get(sym, 0))

            if N <= 0 or U <= 0:
                continue

            denom = float(N + U)

            if sym not in excluded and c_sym > 0:
                return max(p_prefix * (float(c_sym) / denom), 1e-12)

            p_prefix *= float(U) / denom

            for a in cnt.keys():
                if a not in excluded:
                    excluded.add(a)

            if p_prefix < 1e-300:
                p_prefix = 1e-300

        remaining = self.A - len(excluded)
        if remaining <= 0:
            return 1e-12
        return max(p_prefix * (1.0 / float(remaining)), 1e-12)

    def neglog2_prob(self, sym: int, history: Deque[int]) -> float:
        return -log2_safe(self.prob(sym, history))

    def mass_over_range(self, history: Deque[int], lo: int, hi: int) -> float:
        """
        Efficient sum of probabilities over a symbol subset [lo..hi] under PPM Method C + exclusion.
        No enumeration over all symbols in the range; iterates only over observed continuations.
        """
        lo = max(0, int(lo))
        hi = min(self.A - 1, int(hi))
        if hi < lo:
            return 0.0

        subset_size = hi - lo + 1

        excluded = set()
        excluded_in_subset = 0
        p_prefix = 1.0
        mass = 0.0

        max_k = min(self.K, len(history))
        for k in range(max_k, -1, -1):
            ctx = self._ctx(history, k)
            cnt = self.counts[k].get(ctx, None)
            if cnt is None or len(cnt) == 0:
                continue

            # Compute N, U, and sum of counts for subset (all excluding excluded symbols)
            N = 0
            U = 0
            sum_subset = 0
            for a, c in cnt.items():
                if a in excluded:
                    continue
                cc = int(c)
                N += cc
                U += 1
                if lo <= a <= hi:
                    sum_subset += cc

            if N <= 0 or U <= 0:
                continue

            denom = float(N + U)

            # Add probability mass assigned (at this context) to observed subset symbols
            mass += p_prefix * (float(sum_subset) / denom)

            # Escape
            p_prefix *= float(U) / denom

            # Update exclusion set and excluded_in_subset
            for a in cnt.keys():
                if a not in excluded:
                    excluded.add(a)
                    if lo <= a <= hi:
                        excluded_in_subset += 1

            if p_prefix < 1e-300:
                p_prefix = 1e-300
                break

        # Base uniform over remaining symbols (not excluded)
        remaining = self.A - len(excluded)
        if remaining <= 0:
            return max(mass, 1e-12)

        subset_remaining = subset_size - excluded_in_subset
        if subset_remaining > 0:
            mass += p_prefix * (float(subset_remaining) / float(remaining))

        return max(mass, 1e-12)


class KTPredictor:
    """
    Fixed-order KT predictor for categorical alphabet size A:
      p(sym|ctx) = (c + 0.5) / (N + 0.5*A)
    Also supports mass_over_range computed from sparse counts (no enumeration).
    """

    def __init__(self, order: int, alphabet_size: int):
        self.k = int(order)
        self.A = int(alphabet_size)
        self.counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self.totals: Dict[Tuple[int, ...], int] = defaultdict(int)

    @staticmethod
    def _ctx(history: Deque[int], k: int) -> Tuple[int, ...]:
        if k <= 0:
            return tuple()
        h = tuple(history)
        return h[-k:]

    def fit(self, token_rows: Iterable[List[int]], reset_ctx_per_row: bool = True) -> None:
        hist: Deque[int] = deque(maxlen=self.k)
        for row_tokens in token_rows:
            if reset_ctx_per_row:
                hist.clear()
            for sym in row_tokens:
                sym = int(sym)
                ctx = self._ctx(hist, min(self.k, len(hist)))
                self.counts[ctx][sym] += 1
                self.totals[ctx] += 1
                hist.append(sym)

    def prob(self, sym: int, history: Deque[int]) -> float:
        sym = int(sym)
        if sym < 0 or sym >= self.A:
            return max(1.0 / float(self.A), 1e-12)
        ctx = self._ctx(history, min(self.k, len(history)))
        N = int(self.totals.get(ctx, 0))
        c = int(self.counts.get(ctx, Counter()).get(sym, 0))
        p = (c + 0.5) / (N + 0.5 * self.A)
        return max(float(p), 1e-12)

    def mass_over_range(self, history: Deque[int], lo: int, hi: int) -> float:
        lo = max(0, int(lo))
        hi = min(self.A - 1, int(hi))
        if hi < lo:
            return 0.0
        ctx = self._ctx(history, min(self.k, len(history)))
        N = int(self.totals.get(ctx, 0))
        cnt = self.counts.get(ctx, Counter())

        # sum counts only for symbols in subset (sparse iteration)
        sum_c = 0
        for a, c in cnt.items():
            if lo <= a <= hi:
                sum_c += int(c)

        subset_size = hi - lo + 1
        mass = (sum_c + 0.5 * subset_size) / (N + 0.5 * self.A)
        return max(float(mass), 1e-12)


class KTMix(SeqModel):
    """
    CTW-like-in-spirit mixture of KT predictors for orders 0..K:
      p = sum_k w_k * p_k
    Supports mass_over_range via mixture of per-order masses.
    """

    def __init__(self, max_order: int, alphabet_size: int, weights: str = "uniform"):
        if max_order < 0:
            raise ValueError("max_order must be >= 0")
        if alphabet_size < 2:
            raise ValueError("alphabet_size must be >= 2")
        if weights not in ("uniform", "train_evidence"):
            raise ValueError("weights must be one of: uniform, train_evidence")

        self.K = int(max_order)
        self.A = int(alphabet_size)
        self.weights_mode = str(weights)

        self.models: List[KTPredictor] = [KTPredictor(order=k, alphabet_size=self.A) for k in range(self.K + 1)]
        self.w: np.ndarray = np.ones((self.K + 1,), dtype=np.float64) / float(self.K + 1)

    def _estimate_weights_from_train(self, token_rows: List[List[int]], reset_ctx_per_row: bool) -> None:
        counts = [defaultdict(Counter) for _ in range(self.K + 1)]
        totals = [defaultdict(int) for _ in range(self.K + 1)]
        hists = [deque(maxlen=k) for k in range(self.K + 1)]
        logev = np.zeros((self.K + 1,), dtype=np.float64)

        for row_tokens in token_rows:
            if reset_ctx_per_row:
                for h in hists:
                    h.clear()
            for sym in row_tokens:
                sym = int(sym)
                for k in range(self.K + 1):
                    ctx = tuple(hists[k]) if k > 0 else tuple()
                    N = totals[k].get(ctx, 0)
                    c = counts[k][ctx].get(sym, 0)
                    p = (c + 0.5) / (N + 0.5 * self.A)
                    logev[k] += math.log(max(p, 1e-300))
                for k in range(self.K + 1):
                    ctx = tuple(hists[k]) if k > 0 else tuple()
                    counts[k][ctx][sym] += 1
                    totals[k][ctx] += 1
                    if k > 0:
                        hists[k].append(sym)

        m = float(np.max(logev))
        w = np.exp(logev - m)
        w = w / max(1e-12, float(np.sum(w)))
        self.w = w.astype(np.float64)

    def fit_on_token_streams(self, token_rows: Iterable[List[int]], reset_ctx_per_row: bool = True) -> None:
        token_rows = list(token_rows)
        if self.weights_mode == "train_evidence":
            self._estimate_weights_from_train(token_rows, reset_ctx_per_row)
        for m in self.models:
            m.fit(token_rows, reset_ctx_per_row=reset_ctx_per_row)

    def prob(self, sym: int, history: Deque[int]) -> float:
        ps = 0.0
        for k, m in enumerate(self.models):
            ps += float(self.w[k]) * float(m.prob(sym, history))
        return max(ps, 1e-12)

    def neglog2_prob(self, sym: int, history: Deque[int]) -> float:
        return -log2_safe(self.prob(sym, history))

    def mass_over_range(self, history: Deque[int], lo: int, hi: int) -> float:
        Z = 0.0
        for k, m in enumerate(self.models):
            Z += float(self.w[k]) * float(m.mass_over_range(history, lo, hi))
        return max(Z, 1e-12)


def _hist_key(history: Deque[int]) -> Tuple[int, ...]:
    return tuple(history)


def _cache_put(cache: Dict, key, value, max_size: int) -> None:
    cache[key] = value
    if max_size > 0 and len(cache) > max_size:
        cache.clear()


def constrained_prob_label(
    model: SeqModel,
    history: Deque[int],
    sym_label: int,
    label_vocab: int,
    z_cache: Dict[Tuple[int, ...], float],
    cache_max: int,
) -> float:
    p_raw = float(model.prob(sym_label, history))
    key = _hist_key(history)
    Z = z_cache.get(key)
    if Z is None:
        Z = float(model.mass_over_range(history, 0, label_vocab - 1))
        Z = max(Z, 1e-12)
        _cache_put(z_cache, key, Z, cache_max)
    return max(p_raw / Z, 1e-12)


def constrained_prob_len(
    model: SeqModel,
    history: Deque[int],
    sym_len: int,
    label_vocab: int,
    max_len_support: int,
    allowed_max_len: int,
    z_cache: Dict[Tuple[Tuple[int, ...], int], float],
    cache_max: int,
) -> float:
    p_raw = float(model.prob(sym_len, history))
    Lmax = int(min(max_len_support, max(1, allowed_max_len)))
    key = (_hist_key(history), Lmax)
    Z = z_cache.get(key)
    if Z is None:
        lo = int(label_vocab)
        hi = int(label_vocab + (Lmax - 1))
        Z = float(model.mass_over_range(history, lo, hi))
        Z = max(Z, 1e-12)
        _cache_put(z_cache, key, Z, cache_max)
    return max(p_raw / Z, 1e-12)


def iter_token_rows_from_masks(
    items: List[Tuple[str, Path]],
    label_vocab: int,
    max_len_support: int,
) -> Iterable[List[int]]:
    for _fid, p in items:
        mask = read_mask_png(p)
        if mask is None:
            continue
        H, _W = mask.shape
        for r in range(H):
            starts, labs, lens = row_rle(mask[r])
            runs = make_tokens_for_row(starts, labs, lens, label_vocab, max_len_support)
            row_tokens: List[int] = []
            for _sx, _ln, t_lab, t_len in runs:
                row_tokens.append(int(t_lab))
                row_tokens.append(int(t_len))
            if row_tokens:
                yield row_tokens


def compute_bits_and_heatmap(
    mask: np.ndarray,
    model: SeqModel,
    label_vocab: int,
    max_order: int,
    max_len_support: int,
    tile: int = 0,
    reset_ctx_per_row: bool = True,
    type_renorm: bool = False,
    len_remainder_renorm: bool = False,
    renorm_cache_max: int = 200000,
) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
    H, W = mask.shape
    bpp_map = np.zeros((H, W), dtype=np.float32) if (tile and tile > 0) else None

    total_bits = 0.0
    bits_label = 0.0
    bits_len = 0.0
    total_pixels = float(H * W)

    hist: Deque[int] = deque(maxlen=max_order)

    z_label_cache: Dict[Tuple[int, ...], float] = {}
    z_len_cache: Dict[Tuple[Tuple[int, ...], int], float] = {}

    for r in range(H):
        if reset_ctx_per_row:
            hist.clear()

        starts, labs, lens = row_rle(mask[r])
        for s, lab, ln in zip(starts.tolist(), labs.tolist(), lens.tolist()):
            s_i = int(s)
            lab_i = int(lab)
            ln_i = int(ln)
            if ln_i < 1:
                continue
            if lab_i < 0 or lab_i >= label_vocab:
                continue
            if ln_i > max_len_support:
                ln_i = max_len_support

            t_lab = lab_i
            t_len = label_vocab + (ln_i - 1)

            if type_renorm:
                p1 = constrained_prob_label(
                    model=model,
                    history=hist,
                    sym_label=t_lab,
                    label_vocab=label_vocab,
                    z_cache=z_label_cache,
                    cache_max=renorm_cache_max,
                )
                b1 = -log2_safe(p1)
            else:
                b1 = model.neglog2_prob(t_lab, hist)
            hist.append(t_lab)

            if type_renorm or len_remainder_renorm:
                if len_remainder_renorm:
                    allowed_max = int(W - s_i)
                else:
                    allowed_max = int(max_len_support)
                p2 = constrained_prob_len(
                    model=model,
                    history=hist,
                    sym_len=t_len,
                    label_vocab=label_vocab,
                    max_len_support=max_len_support,
                    allowed_max_len=allowed_max,
                    z_cache=z_len_cache,
                    cache_max=renorm_cache_max,
                )
                b2 = -log2_safe(p2)
            else:
                b2 = model.neglog2_prob(t_len, hist)
            hist.append(t_len)

            b_run = b1 + b2
            total_bits += b_run
            bits_label += b1
            bits_len += b2

            if bpp_map is not None:
                bpp = float(b_run) / max(1.0, float(ln_i))
                bpp_map[r, s_i : s_i + ln_i] = bpp

    bbp = total_bits / total_pixels

    if bpp_map is None:
        return total_bits, bbp, bits_label, bits_len, None

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
    return total_bits, bbp, bits_label, bits_len, heat


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
    ap = argparse.ArgumentParser("Variable-order entropy models (PPM / KT-mixture) on scanline-RLE token stream")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subdir", default="warped_masks")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-files", type=int, default=0)

    ap.add_argument("--model", choices=["ppm", "ktmix"], default="ppm")
    ap.add_argument("--max-order", type=int, default=6, help="max context length K")
    ap.add_argument("--ktmix-weights", choices=["uniform", "train_evidence"], default="uniform")

    ap.add_argument("--label-vocab", type=int, default=0, help="If 0, inferred as max_label+1 from all masks")
    ap.add_argument(
        "--max-len-support",
        type=int,
        default=0,
        help="If 0, inferred as max mask width. Setting smaller makes length tokenization lossy.",
    )

    ap.add_argument("--reset-ctx-per-row", action="store_true", help="reset history at each row (recommended)")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--save-heatmaps", action="store_true")
    ap.add_argument("--save-model", action="store_true")

    ap.add_argument("--type-renorm", action="store_true", help="Constrain coding to token type (labels vs lengths)")
    ap.add_argument("--len-remainder-renorm", action="store_true", help="Constrain len tokens by row remainder len<=W-start_x")
    ap.add_argument(
        "--renorm-cache-max",
        type=int,
        default=200000,
        help="Max Z-cache size (0 disables limit). Cache clears on overflow.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(iter_facade_masks(data_root, subdir=args.subdir))
    if args.max_files and args.max_files > 0:
        items = items[: args.max_files]
    if not items:
        raise SystemExit("No masks found")

    max_width_all, max_label_all = scan_dataset_stats(items)
    if max_width_all < 1:
        raise SystemExit("Failed to read any masks")

    label_vocab = int(args.label_vocab) if int(args.label_vocab) > 0 else int(max_label_all + 1)
    max_len_support = int(args.max_len_support) if int(args.max_len_support) > 0 else int(max_width_all)

    alphabet_size = label_vocab + max_len_support

    facade_ids = [fid for fid, _ in items]
    split = split_facades(facade_ids, val_ratio=args.val_ratio, seed=args.seed)
    train_set = set(split.train_facades)
    val_set = set(split.val_facades)

    train_items = [(fid, p) for fid, p in items if fid in train_set]
    val_items = [(fid, p) for fid, p in items if fid in val_set]

    print(f"[Split] train facades={len(train_set)} val facades={len(val_set)}")
    print(f"[Split] train files={len(train_items)} val files={len(val_items)}")
    print(f"[Data] max_width_all={max_width_all} max_label_all={max_label_all}")
    print(f"[Vocab] label_vocab={label_vocab} max_len_support={max_len_support} alphabet_size={alphabet_size}")

    train_token_rows = list(iter_token_rows_from_masks(train_items, label_vocab=label_vocab, max_len_support=max_len_support))

    if args.model == "ppm":
        model: SeqModel = PPMMethodC(max_order=args.max_order, alphabet_size=alphabet_size)
        model.fit_on_token_streams(train_token_rows, reset_ctx_per_row=bool(args.reset_ctx_per_row))
    else:
        model = KTMix(max_order=args.max_order, alphabet_size=alphabet_size, weights=args.ktmix_weights)
        model.fit_on_token_streams(train_token_rows, reset_ctx_per_row=bool(args.reset_ctx_per_row))

    if args.save_model:
        with (out_dir / "model.pkl").open("wb") as f:
            pickle.dump({"model": model, "args": vars(args)}, f)

    csv_path = out_dir / "varorder_bbp.csv"
    fieldnames = [
        "split", "facade_id", "filename", "y1", "y2", "H", "W", "dtype", "unique_vals",
        "model", "max_order", "ktmix_weights", "reset_ctx_per_row",
        "type_renorm", "len_remainder_renorm",
        "label_vocab", "max_len_support", "alphabet_size",
        "total_bits", "bits_label", "bits_len",
        "bbp", "bbp_label", "bbp_len", "len_frac_bits",
        "tile", "tiles_h", "tiles_w", "heat_min", "heat_p50", "heat_p90", "heat_max",
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
                mask=mask,
                model=model,
                label_vocab=label_vocab,
                max_order=int(args.max_order),
                max_len_support=int(max_len_support),
                tile=(args.tile if args.save_heatmaps else 0),
                reset_ctx_per_row=bool(args.reset_ctx_per_row),
                type_renorm=bool(args.type_renorm),
                len_remainder_renorm=bool(args.len_remainder_renorm),
                renorm_cache_max=int(args.renorm_cache_max),
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
                hflat = heat.astype(np.float32).ravel()
                tile_sz = args.tile
                tiles_h = heat.shape[0]
                tiles_w = heat.shape[1]
                heat_min = float(np.min(hflat))
                heat_p50 = float(np.quantile(hflat, 0.50))
                heat_p90 = float(np.quantile(hflat, 0.90))
                heat_max = float(np.max(hflat))

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
                model=str(args.model), max_order=int(args.max_order),
                ktmix_weights=str(args.ktmix_weights) if args.model == "ktmix" else "",
                reset_ctx_per_row=bool(args.reset_ctx_per_row),
                type_renorm=bool(args.type_renorm),
                len_remainder_renorm=bool(args.len_remainder_renorm),
                label_vocab=int(label_vocab),
                max_len_support=int(max_len_support),
                alphabet_size=int(alphabet_size),
                total_bits=float(total_bits),
                bits_label=float(bits_label),
                bits_len=float(bits_len),
                bbp=float(bbp),
                bbp_label=float(bbp_label),
                bbp_len=float(bbp_len),
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