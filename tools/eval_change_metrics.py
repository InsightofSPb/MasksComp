#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from maskscomp.change_detection import iter_tiles_2d, read_mask, read_pairs_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tile-wise change detection metrics from heatmaps")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--heatmap-dir", type=Path, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)

    # existing
    p.add_argument("--label-from", choices=["residual_C", "masks"], default="residual_C")

    # NEW
    p.add_argument(
        "--label-mode",
        choices=["area", "class_any", "class_frac"],
        default="area",
        help=(
            "area: label by change area fraction >= tau; "
            "class_any: label if any (changed AND class in pos_classes) exists; "
            "class_frac: label if fraction of (changed AND class in pos_classes) >= tau."
        ),
    )
    p.add_argument(
        "--pos-classes",
        type=str,
        default=None,
        help="Comma-separated integer class ids, e.g. '12,13,27'. Required for label-mode class_*.",
    )

    p.add_argument("--append", action="store_true")

    # existing
    p.add_argument("--ids-txt", type=Path, default=None)
    p.add_argument("--max-items", type=int, default=None)
    return p.parse_args()


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _load_ids(ids_txt: Path) -> set[str]:
    ids = set()
    for ln in ids_txt.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            ids.add(s)
    return ids


def _get_write_mode(out_path: Path, append: bool) -> tuple[str, bool]:
    if not append:
        return "w", True
    if not out_path.exists() or out_path.stat().st_size == 0:
        return "w", True
    with out_path.open("r", encoding="utf-8", newline="") as f:
        has_header = bool(f.readline().strip())
    return "a", not has_header


def _parse_pos_classes(s: Optional[str]) -> Optional[Set[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out: Set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.add(int(tok))
    return out if out else None


def _compute_diff_and_cur(args: argparse.Namespace, r) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      diff: uint8 HxW in {0,1} (changed pixels)
      cur:  uint8/uint16 HxW current labels (only if label_from==masks; else None)
    """
    stem = Path(r.cur_path).stem

    if args.label_from == "residual_C":
        c = read_mask(args.data_root / r.pair_id / "residual_C" / f"{stem}.png")
        diff = (c != 0).astype(np.uint8)
        return diff, None
    else:
        prev = read_mask(args.data_root / r.prev_path)
        cur = read_mask(args.data_root / r.cur_path)
        diff = (cur != prev).astype(np.uint8)
        return diff, cur


def _compute_pos_indicator(args: argparse.Namespace, r, diff: np.ndarray, cur: Optional[np.ndarray], pos: Set[int]) -> np.ndarray:
    """
    pos_indicator[u]=1 iff pixel is changed AND its current class is in pos_classes.
    """
    stem = Path(r.cur_path).stem

    if args.label_from == "residual_C":
        # Use residual_V as "current label at changed pixels" (0 outside change), gate by diff.
        v = read_mask(args.data_root / r.pair_id / "residual_V" / f"{stem}.png")
        in_pos = np.isin(v.astype(np.int32, copy=False), np.fromiter(pos, dtype=np.int32))
        return (diff.astype(bool) & in_pos).astype(np.uint8)
    else:
        assert cur is not None
        in_pos = np.isin(cur.astype(np.int32, copy=False), np.fromiter(pos, dtype=np.int32))
        return (diff.astype(bool) & in_pos).astype(np.uint8)


def main() -> None:
    args = parse_args()

    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split]

    if args.ids_txt is not None:
        keep = _load_ids(args.ids_txt)
        rows = [r for r in rows if str(r.pair_id) in keep]

    if args.max_items is not None:
        rows = rows[: int(args.max_items)]

    pos_classes = _parse_pos_classes(args.pos_classes)
    if args.label_mode != "area" and not pos_classes:
        raise SystemExit("--label-mode class_any/class_frac requires --pos-classes")

    all_scores: List[float] = []
    all_labels: List[int] = []
    hit_count = 0
    recall_num = 0
    recall_den = 0
    evaluated_rows = 0

    for r in rows:
        stem = Path(r.cur_path).stem
        heat_path = args.heatmap_dir / r.pair_id / f"{stem}.npy"
        if not heat_path.exists():
            continue

        heat = np.load(heat_path)
        diff, cur = _compute_diff_and_cur(args, r)

        if args.label_mode == "area":
            gt = diff
        else:
            gt = _compute_pos_indicator(args, r, diff, cur, pos_classes)  # type: ignore[arg-type]

        labels = []
        if args.label_mode == "class_any":
            for _y, _x, tile in iter_tiles_2d(gt, args.tile_size, args.stride):
                labels.append(1 if tile.any() else 0)
        else:
            # area or class_frac: threshold by mean >= tau
            for _y, _x, tile in iter_tiles_2d(gt, args.tile_size, args.stride):
                labels.append(1 if float(tile.mean()) >= float(args.tau) else 0)

        labels = np.asarray(labels, dtype=np.int64)
        scores = heat.reshape(-1).astype(np.float64)

        if scores.size != labels.size:
            n = min(scores.size, labels.size)
            scores = scores[:n]
            labels = labels[:n]

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

        k = min(args.topk, scores.size)
        if k > 0:
            top_idx = np.argsort(scores)[::-1][:k]
            hit_count += int(labels[top_idx].any())
            recall_num += int(labels[top_idx].sum())
            recall_den += int(labels.sum())

        evaluated_rows += 1

    y = np.asarray(all_labels, dtype=np.int64)
    s = np.asarray(all_scores, dtype=np.float64)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "method": args.method,
        "label_from": args.label_from,
        "label_mode": args.label_mode,
        "pos_classes": (args.pos_classes or ""),
        "tau": args.tau,
        "topk": args.topk,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "ROC-AUC": _safe_auc(y, s),
        "PR-AUC": _safe_ap(y, s),
        "Hit@K": float(hit_count) / max(1, evaluated_rows),
        "Recall@K": float(recall_num) / max(1, recall_den),
        "n_pairs": int(evaluated_rows),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode, write_header = _get_write_mode(args.out, args.append)

    with args.out.open(mode, newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            wr.writeheader()
        wr.writerow(metrics)

    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
