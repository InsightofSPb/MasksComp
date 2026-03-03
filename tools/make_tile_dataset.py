#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from maskscomp.change_detection import iter_tiles_2d, read_pairs_csv, tile_grid_shape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tile dataset for tile-reset MSDZip eval")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--ids-txt", type=Path, default=None)
    p.add_argument("--src-subdir", type=str, default="residual_V")
    p.add_argument("--tile-size", type=int, required=True)
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    return p.parse_args()


def _load_ids(ids_txt: Path) -> set[str]:
    ids = set()
    for ln in ids_txt.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            ids.add(s)
    return ids


def main() -> None:
    args = parse_args()
    if args.stride != args.tile_size:
        raise SystemExit("For tile-reset protocol use stride == tile-size (non-overlapping tiles).")
    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split]
    if args.ids_txt is not None:
        keep = _load_ids(args.ids_txt)
        rows = [r for r in rows if str(r.pair_id) in keep]

    args.out_root.mkdir(parents=True, exist_ok=True)
    split_dir = args.out_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    pair_ids = sorted({str(r.pair_id) for r in rows})
    (split_dir / f"facade_{args.split}.txt").write_text("\n".join(pair_ids) + "\n", encoding="utf-8")

    per_pair_meta: dict[str, list[dict]] = {}

    for r in rows:
        pair_id = str(r.pair_id)
        stem = Path(r.cur_path).stem
        src = args.data_root / pair_id / args.src_subdir / f"{stem}.png"
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read: {src}")
        if img.ndim == 3 and img.shape[2] == 3:
            if (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all():
                img = img[..., 0]
            else:
                raise ValueError(f"Expected single-channel residual image, got RGB content in {src}")
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image for {src}, got shape={img.shape}")

        h, w = img.shape
        gh, gw = tile_grid_shape(h, w, args.tile_size, args.stride)

        out_tiles = args.out_root / pair_id / "tiles"
        out_tiles.mkdir(parents=True, exist_ok=True)

        idx = 0
        for _y, _x, tile in iter_tiles_2d(img, args.tile_size, args.stride):
            gy, gx = divmod(idx, gw)
            out_name = f"{stem}__gy{gy}__gx{gx}.png"
            ok = cv2.imwrite(str(out_tiles / out_name), tile)
            if not ok:
                raise RuntimeError(f"Failed to write tile: {out_tiles / out_name}")
            idx += 1

        per_pair_meta.setdefault(pair_id, []).append(
            {
                "stem": stem,
                "H": int(h),
                "W": int(w),
                "tile": int(args.tile_size),
                "stride": int(args.stride),
                "gh": int(gh),
                "gw": int(gw),
                "src_subdir": args.src_subdir,
            }
        )

    for pair_id, items in per_pair_meta.items():
        out_meta = args.out_root / pair_id / "tiles_meta.json"
        payload = {
            "pair_id": pair_id,
            "split": args.split,
            "tile": int(args.tile_size),
            "stride": int(args.stride),
            "items": items,
        }
        out_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote tile dataset: {args.out_root} (pairs={len(pair_ids)}, images={len(rows)})")


if __name__ == "__main__":
    main()
