#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import cv2


def normalize_local(h, p_lo=5.0, p_hi=95.0):
    lo = np.percentile(h, p_lo)
    hi = np.percentile(h, p_hi)
    if hi <= lo + 1e-12:
        return np.zeros_like(h, dtype=np.float32), float(lo), float(hi)
    x = (h - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32), float(lo), float(hi)


def normalize_global(h, vmin, vmax):
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin + 1e-12:
        return np.zeros_like(h, dtype=np.float32), vmin, vmax
    x = (h - vmin) / (vmax - vmin)
    return np.clip(x, 0.0, 1.0).astype(np.float32), vmin, vmax


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    # boundaries where label changes with right/down neighbor
    m = mask.astype(np.int32)
    b = np.zeros_like(m, dtype=np.uint8)
    b[:, 1:] |= (m[:, 1:] != m[:, :-1]).astype(np.uint8)
    b[1:, :] |= (m[1:, :] != m[:-1, :]).astype(np.uint8)
    # thicken a bit
    k = np.ones((3, 3), np.uint8)
    b = cv2.dilate(b, k, iterations=1)
    return (b * 255).astype(np.uint8)


def render_heatmap(
    heat: np.ndarray,
    cell: int = 64,
    cmap=cv2.COLORMAP_TURBO,
    annotate: bool = True,
    value_fmt: str = "{:.3f}",
):
    th, tw = heat.shape
    heat_up = cv2.resize(heat, (tw * cell, th * cell), interpolation=cv2.INTER_NEAREST)

    img_u8 = (heat_up * 255.0 + 0.5).astype(np.uint8)
    img_col = cv2.applyColorMap(img_u8, cmap)

    if annotate and cell >= 40:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for y in range(th):
            for x in range(tw):
                v = float(heat[y, x])
                text = value_fmt.format(v)
                org = (x * cell + 3, (y + 1) * cell - 6)
                cv2.putText(img_col, text, org, font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return img_col


def main():
    ap = argparse.ArgumentParser("Render tile heatmap .npy nicely (upscale + colormap + optional overlay)")
    ap.add_argument("--heat-npy", required=True, help="Path to heatmap .npy")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--cell", type=int, default=64, help="Pixels per tile (upscale factor)")
    ap.add_argument("--mode", choices=["local", "global"], default="local")
    ap.add_argument("--p-lo", type=float, default=5.0)
    ap.add_argument("--p-hi", type=float, default=95.0)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--mask-png", default=None, help="Optional mask png to overlay boundaries")
    ap.add_argument("--alpha", type=float, default=0.35, help="Overlay alpha for boundaries")
    args = ap.parse_args()

    heat = np.load(args.heat_npy).astype(np.float32)

    if args.mode == "local":
        h01, lo, hi = normalize_local(heat, args.p_lo, args.p_hi)
        legend = f"local p{args.p_lo:.0f}-p{args.p_hi:.0f}: [{lo:.4f}, {hi:.4f}]"
    else:
        if args.vmin is None or args.vmax is None:
            raise SystemExit("global mode requires --vmin and --vmax")
        h01, lo, hi = normalize_global(heat, args.vmin, args.vmax)
        legend = f"global: [{lo:.4f}, {hi:.4f}]"

    img = render_heatmap(h01, cell=args.cell, annotate=args.annotate)

    if args.mask_png is not None:
        m = cv2.imread(args.mask_png, cv2.IMREAD_UNCHANGED)
        if m is None or m.ndim != 2:
            raise SystemExit("Failed to read mask png (need 1-channel mask)")
        b = mask_boundaries(m)
        # resize boundaries to same size as heatmap rendering
        b_up = cv2.resize(b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        b_col = cv2.cvtColor(b_up, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 1.0, b_col, float(args.alpha), 0.0)

    # legend strip
    strip = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, legend, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out_img = np.vstack([strip, img])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()