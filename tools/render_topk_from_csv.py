#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import math


def normalize_local(h: np.ndarray, p_lo=5.0, p_hi=95.0):
    lo = float(np.percentile(h, p_lo))
    hi = float(np.percentile(h, p_hi))
    if hi <= lo + 1e-12:
        return np.zeros_like(h, dtype=np.float32), lo, hi
    x = (h - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32), lo, hi


def normalize_global(h: np.ndarray, vmin: float, vmax: float):
    vmin = float(vmin)
    vmax = float(vmax)
    if vmax <= vmin + 1e-12:
        return np.zeros_like(h, dtype=np.float32), vmin, vmax
    x = (h - vmin) / (vmax - vmin)
    return np.clip(x, 0.0, 1.0).astype(np.float32), vmin, vmax


def mask_boundaries(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int32)
    b = np.zeros_like(m, dtype=np.uint8)
    b[:, 1:] |= (m[:, 1:] != m[:, :-1]).astype(np.uint8)
    b[1:, :] |= (m[1:, :] != m[:-1, :]).astype(np.uint8)
    b = cv2.dilate(b * 255, np.ones((3, 3), np.uint8), iterations=1)
    return b


def render_heatmap(heat01: np.ndarray, cell: int = 90, annotate: bool = True):
    th, tw = heat01.shape
    up = cv2.resize(heat01, (tw * cell, th * cell), interpolation=cv2.INTER_NEAREST)
    img_u8 = (up * 255.0 + 0.5).astype(np.uint8)
    img = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)

    if annotate and cell >= 40:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for y in range(th):
            for x in range(tw):
                v = float(heat01[y, x])
                txt = f"{v:.3f}"
                org = (x * cell + 4, (y + 1) * cell - 8)
                cv2.putText(img, txt, org, font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def find_bg_image(bg_root: Path, facade_id: str, stem: str, bg_subdir: str):
    """
    Map warped mask stem -> background stem depending on bg_subdir naming:
      warped_masks:  {y1}_{y2}_warped.png
      overlays:      {y1}_{y2}_overlay.png
      diff_maps:     {y1}_{y2}_diff.png
    Falls back to stem.* and then prefix match.
    """
    d = bg_root / facade_id / bg_subdir
    if not d.exists():
        return None

    # derive target stem
    bg_stem = stem
    if stem.endswith("_warped"):
        if bg_subdir == "overlays":
            bg_stem = stem[:-7] + "_overlay"   # drop "_warped"
        elif bg_subdir == "diff_maps":
            bg_stem = stem[:-7] + "_diff"
        # else: keep original stem

    # try exact bg_stem
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = d / f"{bg_stem}{ext}"
        if p.exists():
            return p

    # fallback: exact original stem
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = d / f"{stem}{ext}"
        if p.exists():
            return p

    # fallback: any file starting with bg_stem
    cand = sorted(d.glob(f"{bg_stem}.*"))
    if cand:
        return cand[0]

    # last resort: any file starting with original stem
    cand = sorted(d.glob(f"{stem}.*"))
    return cand[0] if cand else None


def add_strip(img: np.ndarray, text: str):
    strip_h = 30
    strip = np.zeros((strip_h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(strip, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def main():
    ap = argparse.ArgumentParser("Pick top-K by heat_p90 from CSV and render heatmaps to output folder.")
    ap.add_argument("--csv", required=True, help="Path to ngram_bbp.csv (with heat stats)")
    ap.add_argument("--heat-root", required=True, help="Folder with heatmaps_tiles (contains facade_id/stem.npy)")
    ap.add_argument("--out-dir", required=True, help="Where to write rendered PNGs")

    ap.add_argument("--split", default="val", choices=["train", "val"], help="Which split to select")
    ap.add_argument("--topk", type=int, default=10, help="Top-K examples by heat_p90")
    ap.add_argument("--unique-facades", action="store_true", help="Keep at most one sample per facade_id")

    ap.add_argument("--cell", type=int, default=90, help="Pixels per tile for rendering")
    ap.add_argument("--annotate", action="store_true", help="Write numeric tile values on the image")

    ap.add_argument("--render-local", action="store_true", help="Render local p5-p95 normalized version")
    ap.add_argument("--render-global", action="store_true", help="Render global vmin-vmax normalized version")
    ap.add_argument("--global-vmin", type=float, default=0.04)
    ap.add_argument("--global-vmax", type=float, default=0.18)
    ap.add_argument("--auto-global", action="store_true",
                    help="Compute global vmin/vmax from selected heatmaps (p5/p95 over ALL tiles)")

    ap.add_argument("--data-root", default=None,
                    help="Facades root for masks, e.g. /home/.../facades (to overlay boundaries)")
    ap.add_argument("--mask-subdir", default="warped_masks")
    ap.add_argument("--boundary-alpha", type=float, default=0.35)

    ap.add_argument("--bg-root", default=None,
                    help="Facades root for background images/overlays (optional)")
    ap.add_argument("--bg-subdir", default="overlays",
                    help="Subdir inside each facade folder to search backgrounds")
    ap.add_argument("--bg-alpha", type=float, default=0.45,
                    help="Heat overlay strength on background (0..1)")
    ap.add_argument("--panel", action="store_true",
                    help="If bg exists, output side-by-side panel: [bg | bg+heat]")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    heat_root = Path(args.heat_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "heat_p90" not in df.columns:
        raise SystemExit("CSV has no heat_p90 column (did you run with --save-heatmaps?)")

    df["heat_p90"] = pd.to_numeric(df["heat_p90"], errors="coerce")
    df = df[df["split"] == args.split].copy()
    df = df[df["heat_p90"].notna()].copy()
    df = df.sort_values("heat_p90", ascending=False)

    if args.unique_facades:
        df = df.drop_duplicates(subset=["facade_id"], keep="first")

    df = df.head(args.topk)

    if df.empty:
        raise SystemExit("No rows selected (check split and heat_p90 availability).")

    # Decide which renders to produce
    render_local = args.render_local or (not args.render_local and not args.render_global)
    render_global = args.render_global or (not args.render_local and not args.render_global)

    # Auto global range from selected heatmaps (p5..p95 over all tiles)
    gvmin, gvmax = float(args.global_vmin), float(args.global_vmax)
    if render_global and args.auto_global:
        all_vals = []
        for _, row in df.iterrows():
            facade_id = str(row["facade_id"])
            filename = str(row["filename"])
            stem = Path(filename).stem
            npy = heat_root / facade_id / f"{stem}.npy"
            if npy.exists():
                h = np.load(npy).astype(np.float32).ravel()
                all_vals.append(h)
        if all_vals:
            all_vals = np.concatenate(all_vals, axis=0)
            gvmin = float(np.percentile(all_vals, 5.0))
            gvmax = float(np.percentile(all_vals, 95.0))
            if gvmax <= gvmin + 1e-12:
                gvmax = gvmin + 1e-3

    # Render loop
    index_rows = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        facade_id = str(row["facade_id"])
        filename = str(row["filename"])
        stem = Path(filename).stem
        heat_p90 = float(row["heat_p90"])
        bbp = float(row["bbp"]) if "bbp" in row and pd.notna(row["bbp"]) else float("nan")

        npy = heat_root / facade_id / f"{stem}.npy"
        if not npy.exists():
            print(f"[WARN] missing heatmap: {npy}")
            continue
        heat = np.load(npy).astype(np.float32)

        # optional boundaries from mask
        boundary_img = None
        if args.data_root is not None:
            mask_path = Path(args.data_root) / facade_id / args.mask_subdir / filename
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if m is not None and m.ndim == 2:
                    b = mask_boundaries(m)
                    boundary_img = b  # single channel uint8
            else:
                print(f"[WARN] missing mask for boundaries: {mask_path}")

        # optional background image
        bg_img = None
        bg_path = None
        if args.bg_root is not None:
            bg_path = find_bg_image(Path(args.bg_root), facade_id, stem, args.bg_subdir)
            if bg_path is not None:
                bg_img = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)

        def overlay_boundaries(img_bgr: np.ndarray):
            if boundary_img is None:
                return img_bgr
            b_up = cv2.resize(boundary_img, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            b_col = cv2.cvtColor(b_up, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(img_bgr, 1.0, b_col, float(args.boundary_alpha), 0.0)

        def maybe_make_panel(heat_bgr: np.ndarray):
            # If bg exists and panel requested: [bg | bg+heat]
            if bg_img is None or not args.panel:
                return heat_bgr
            bg_res = cv2.resize(bg_img, (heat_bgr.shape[1], heat_bgr.shape[0]), interpolation=cv2.INTER_AREA)
            over = cv2.addWeighted(bg_res, 1.0 - float(args.bg_alpha), heat_bgr, float(args.bg_alpha), 0.0)
            over = overlay_boundaries(over)
            bg_res = overlay_boundaries(bg_res)
            return np.hstack([bg_res, over])

        # local
        if render_local:
            h01, lo, hi = normalize_local(heat, 5.0, 95.0)
            img = render_heatmap(h01, cell=args.cell, annotate=args.annotate)
            img = overlay_boundaries(img)
            img = maybe_make_panel(img)
            title = f"{rank:02d} {facade_id} {stem} | heat_p90={heat_p90:.4f} bbp={bbp:.4f} | local p5-p95 [{lo:.4f},{hi:.4f}]"
            img = add_strip(img, title)

            out = out_dir / f"{rank:02d}_{facade_id}_{stem}_local.png"
            cv2.imwrite(str(out), img)
            index_rows.append({"rank": rank, "mode": "local", "out": str(out), "facade_id": facade_id, "stem": stem,
                               "heat_p90": heat_p90, "bbp": bbp, "bg": str(bg_path) if bg_path else ""})

        # global
        if render_global:
            h01, lo, hi = normalize_global(heat, gvmin, gvmax)
            img = render_heatmap(h01, cell=args.cell, annotate=args.annotate)
            img = overlay_boundaries(img)
            img = maybe_make_panel(img)
            title = f"{rank:02d} {facade_id} {stem} | heat_p90={heat_p90:.4f} bbp={bbp:.4f} | global [{gvmin:.4f},{gvmax:.4f}]"
            img = add_strip(img, title)

            out = out_dir / f"{rank:02d}_{facade_id}_{stem}_global.png"
            cv2.imwrite(str(out), img)
            index_rows.append({"rank": rank, "mode": "global", "out": str(out), "facade_id": facade_id, "stem": stem,
                               "heat_p90": heat_p90, "bbp": bbp, "bg": str(bg_path) if bg_path else ""})

    # write index
    if index_rows:
        idx = pd.DataFrame(index_rows)
        idx_path = out_dir / "index_topk.csv"
        idx.to_csv(idx_path, index=False)
        print(f"[OK] wrote {idx_path}")
    print(f"[OK] rendered to {out_dir}")


if __name__ == "__main__":
    main()