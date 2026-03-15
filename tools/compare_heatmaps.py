#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import cv2

from maskscomp.change_detection import read_pairs_csv, read_mask


def parse_args():
    p = argparse.ArgumentParser("compare_heatmaps")
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--ids-txt", type=Path, required=True)

    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--mask-subdir", type=str, default="residual_V")

    p.add_argument("--heat-a", type=Path, required=True)
    p.add_argument("--heat-b", type=Path, required=True)

    p.add_argument("--label-a", type=str, default="classic")
    p.add_argument("--label-b", type=str, default="msdzip")

    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)

    p.add_argument("--margin", type=int, default=16)
    p.add_argument("--boundary-alpha", type=float, default=0.6)

    # NEW: show most disagreeing tiles
    p.add_argument("--diff-topn", type=int, default=3, help="How many most disagreeing tiles to show per pair.")
    p.add_argument("--diff-metric", choices=["rank", "abs"], default="rank",
                  help="rank: |rankA-rankB|; abs: |normA-normB| with per-map minmax.")
    p.add_argument("--draw-boxes", action="store_true", help="Draw rectangles for selected tiles on full images.")

    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-items", type=int, default=None)
    return p.parse_args()


def load_ids(p: Path) -> list[str]:
    out: list[str] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        t = ln.strip()
        if t:
            out.append(t)
    return out


def norm_to_u8(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32, copy=False)
    mn = float(np.min(h))
    mx = float(np.max(h))
    if mx <= mn + 1e-12:
        return np.zeros_like(h, dtype=np.uint8)
    x = (h - mn) / (mx - mn)
    return (255.0 * x).clip(0, 255).astype(np.uint8)


def upsample_heat(heat: np.ndarray, H: int, W: int) -> np.ndarray:
    u8 = norm_to_u8(heat)
    return cv2.resize(u8, (W, H), interpolation=cv2.INTER_NEAREST)


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8, copy=False)
    gx = cv2.Sobel(m, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_16S, 0, 1, ksize=3)
    g = (np.abs(gx) + np.abs(gy)).astype(np.float32)
    g = (g > 0).astype(np.uint8) * 255
    g = cv2.dilate(g, np.ones((3, 3), np.uint8), iterations=1)
    return g


def overlay(mask: np.ndarray, heat: np.ndarray, alpha: float, title: str) -> np.ndarray:
    H, W = mask.shape
    hm = upsample_heat(heat, H, W)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

    base = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(base, 0.35, hm_color, 0.65, 0)

    b = boundary_from_mask(mask)
    b3 = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(out, 1.0, b3, float(alpha), 0)

    cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _norm01_flat(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn + 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)


def top_diff_tiles(heat_a: np.ndarray, heat_b: np.ndarray, n: int, metric: str) -> tuple[list[int], list[float]]:
    fa = heat_a.reshape(-1).astype(np.float64)
    fb = heat_b.reshape(-1).astype(np.float64)
    m = min(fa.size, fb.size)
    fa = fa[:m]
    fb = fb[:m]
    if m == 0 or n <= 0:
        return [], []

    if metric == "abs":
        da = _norm01_flat(fa)
        db = _norm01_flat(fb)
        diff = np.abs(da - db)
    else:
        order_a = np.argsort(-fa)
        order_b = np.argsort(-fb)
        rank_a = np.empty_like(order_a)
        rank_b = np.empty_like(order_b)
        rank_a[order_a] = np.arange(m)
        rank_b[order_b] = np.arange(m)
        diff = np.abs(rank_a - rank_b).astype(np.float64)

    n = min(n, m)
    idx = np.argpartition(diff, -n)[-n:]
    idx = idx[np.argsort(diff[idx])[::-1]]
    return [int(i) for i in idx.tolist()], [float(diff[i]) for i in idx.tolist()]


def idx_to_xy(idx: int, grid_w: int, stride: int) -> tuple[int, int]:
    gy = idx // grid_w
    gx = idx % grid_w
    return gy * stride, gx * stride


def crop_box(H: int, W: int, y0: int, x0: int, tile: int, margin: int) -> tuple[int, int, int, int]:
    y1 = y0 + tile
    x1 = x0 + tile
    yy0 = max(0, y0 - margin)
    xx0 = max(0, x0 - margin)
    yy1 = min(H, y1 + margin)
    xx1 = min(W, x1 + margin)
    return yy0, xx0, yy1, xx1


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = load_ids(args.ids_txt)
    if args.max_items is not None:
        ids = ids[: int(args.max_items)]
    ids_set = set(ids)

    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split and r.pair_id in ids_set]
    rows_map = {r.pair_id: r for r in rows}

    for pid in ids:
        if pid not in rows_map:
            continue
        r = rows_map[pid]
        stem = Path(r.cur_path).stem

        pa = args.heat_a / pid / f"{stem}.npy"
        pb = args.heat_b / pid / f"{stem}.npy"
        if not pa.exists() or not pb.exists():
            continue

        ha = np.load(pa)
        hb = np.load(pb)

        mask_path = args.data_root / pid / args.mask_subdir / f"{stem}.png"
        if not mask_path.exists():
            cand = list((args.data_root / pid / args.mask_subdir).glob("*.png"))
            if not cand:
                continue
            mask_path = cand[0]
        mask = read_mask(mask_path)

        full_a = overlay(mask, ha, args.boundary_alpha, f"{args.label_a} (full)")
        full_b = overlay(mask, hb, args.boundary_alpha, f"{args.label_b} (full)")

        diff_idxs, diff_vals = top_diff_tiles(ha, hb, args.diff_topn, args.diff_metric)

        # draw rectangles for selected tiles on full images
        if args.draw_boxes and len(diff_idxs) > 0:
            H, W = mask.shape
            grid_w = ha.shape[1] if ha.ndim == 2 else int(round(np.sqrt(ha.size)))
            for i, idx in enumerate(diff_idxs):
                y0, x0 = idx_to_xy(idx, grid_w, args.stride)
                yy0, xx0, yy1, xx1 = crop_box(H, W, y0, x0, args.tile_size, 0)  # just tile box
                cv2.rectangle(full_a, (xx0, yy0), (xx1, yy1), (255, 255, 255), 2)
                cv2.rectangle(full_b, (xx0, yy0), (xx1, yy1), (255, 255, 255), 2)
                cv2.putText(full_a, str(i+1), (xx0+5, yy0+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(full_b, str(i+1), (xx0+5, yy0+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # build rows: first row is full A/B, next rows are crops for each diff tile
        rows_img = []
        top = cv2.hconcat([full_a, full_b])
        rows_img.append(top)

        H, W = mask.shape
        grid_w = ha.shape[1] if ha.ndim == 2 else int(round(np.sqrt(ha.size)))

        for i, (idx, dv) in enumerate(zip(diff_idxs, diff_vals), start=1):
            y0, x0 = idx_to_xy(idx, grid_w, args.stride)
            yy0, xx0, yy1, xx1 = crop_box(H, W, y0, x0, args.tile_size, args.margin)

            crop_a = full_a[yy0:yy1, xx0:xx1].copy()
            crop_b = full_b[yy0:yy1, xx0:xx1].copy()

            crop_a = annotate(crop_a, f"{args.label_a}  diff#{i}  idx={idx}  d={dv:.3g}")
            crop_b = annotate(crop_b, f"{args.label_b}  diff#{i}  idx={idx}  d={dv:.3g}")

            row = cv2.hconcat([crop_a, crop_b])

            # make width match top for clean vconcat
            if row.shape[1] != top.shape[1]:
                row = cv2.resize(row, (top.shape[1], row.shape[0]), interpolation=cv2.INTER_AREA)
            rows_img.append(row)

        panel = cv2.vconcat(rows_img)
        out_path = args.out_dir / f"{pid}__{stem}.png"
        cv2.imwrite(str(out_path), panel)

    print("[OK] wrote panels to", args.out_dir)


if __name__ == "__main__":
    main()
