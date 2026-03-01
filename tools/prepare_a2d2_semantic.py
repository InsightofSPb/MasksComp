#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
import sys

import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        
from maskscomp.datasets.a2d2 import parse_class_list, rgb_mask_to_id

CAMERA_CHOICES = [
    "front_center",
    "front_left",
    "front_right",
    "side_left",
    "side_right",
    "rear_center",
]

FRAME_ID_RE = re.compile(r"(\d{9})(?=\.png$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert A2D2 semantic RGB labels to class-id PNG masks")
    parser.add_argument("--a2d2-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--camera", type=str, default="front_center", choices=CAMERA_CHOICES)
    parser.add_argument("--out-subdir", type=str, default="warped_masks")
    parser.add_argument("--limit-per-scene", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--write-manifest", type=Path, default=None)
    parser.add_argument("--unknown", type=str, choices=["error", "map0", "map_last"], default="error")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.limit_per_scene is not None and args.limit_per_scene < 1:
        raise ValueError("--limit-per-scene must be >= 1 when provided")
    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError("--val-ratio must be in [0, 1]")
    return args


def frame_id_from_name(path: Path) -> int:
    m = FRAME_ID_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot extract 9-digit frame id from filename: {path.name}")
    return int(m.group(1))


def discover_scenes(a2d2_root: Path, camera: str) -> list[Path]:
    scene_paths = []
    cam_dir_name = f"cam_{camera}"
    for p in sorted(a2d2_root.iterdir()):
        if not p.is_dir():
            continue
        label_dir = p / "label" / cam_dir_name
        if label_dir.is_dir() and any(label_dir.glob("*.png")):
            scene_paths.append(p)
    return scene_paths


def split_scenes(scene_ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(scene_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_ratio))
    if val_ratio > 0 and len(ids) > 1:
        n_val = min(max(1, n_val), len(ids) - 1)
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


def main() -> None:
    args = parse_args()

    a2d2_root = args.a2d2_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    class_list_path = a2d2_root / "class_list.json"
    rgb2id = parse_class_list(class_list_path)
    num_classes = len(rgb2id)

    unknown_id = None
    if args.unknown == "map0":
        unknown_id = 0
    elif args.unknown == "map_last":
        unknown_id = num_classes

    scenes = discover_scenes(a2d2_root, args.camera)
    if not scenes:
        raise RuntimeError(f"No scenes found in {a2d2_root} with label/cam_{args.camera}/*.png")

    manifest_path = args.write_manifest or (out_root / "manifest.csv")
    splits_dir = args.splits_dir or (out_root / "splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_file, fieldnames=["scene_id", "frame_id", "src_rel", "dst_rel", "H", "W"])
    writer.writeheader()

    total_frames = 0
    converted_frames = 0
    skipped_frames = 0
    sizes: list[tuple[int, int]] = []

    try:
        for scene_path in scenes:
            scene_id = scene_path.name
            src_dir = scene_path / "label" / f"cam_{args.camera}"
            files = sorted(src_dir.glob("*.png"), key=frame_id_from_name)
            files = files[:: args.stride]
            if args.limit_per_scene is not None:
                files = files[: args.limit_per_scene]

            dst_dir = out_root / scene_id / args.out_subdir
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src_path in files:
                frame_id = frame_id_from_name(src_path)
                dst_path = dst_dir / src_path.name

                if args.skip_existing and dst_path.exists():
                    skipped_frames += 1
                    try:
                        with Image.open(dst_path) as dst_img:
                            w, h = dst_img.size
                        sizes.append((h, w))
                    except Exception:
                        pass
                else:
                    with Image.open(src_path) as img:
                        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
                    id_mask = rgb_mask_to_id(rgb, rgb2id, unknown_id=unknown_id)
                    Image.fromarray(id_mask).save(dst_path)
                    h, w = id_mask.shape
                    sizes.append((h, w))
                    converted_frames += 1

                total_frames += 1
                writer.writerow(
                    {
                        "scene_id": scene_id,
                        "frame_id": frame_id,
                        "src_rel": str(src_path.relative_to(a2d2_root)),
                        "dst_rel": str(dst_path.relative_to(out_root)),
                        "H": h,
                        "W": w,
                    }
                )
    finally:
        manifest_file.close()

    scene_ids = [p.name for p in scenes]
    train_ids, val_ids = split_scenes(scene_ids, args.val_ratio, args.seed)

    (splits_dir / "facade_train.txt").write_text("\n".join(train_ids) + ("\n" if train_ids else ""), encoding="utf-8")
    (splits_dir / "facade_val.txt").write_text("\n".join(val_ids) + ("\n" if val_ids else ""), encoding="utf-8")

    hs = [h for h, _ in sizes] or [0]
    ws = [w for _, w in sizes] or [0]
    print("A2D2 semantic conversion completed")
    print(f"  scenes: {len(scene_ids)}")
    print(f"  frames selected: {total_frames}")
    print(f"  frames converted: {converted_frames}")
    print(f"  frames skipped(existing): {skipped_frames}")
    print(f"  num_classes: {num_classes}")
    print(f"  output_dtype: {'uint8' if (num_classes + (1 if unknown_id is not None and unknown_id == num_classes else 0)) <= 256 else 'uint16'}")
    print(
        "  image_size_stats(HxW): "
        f"h[min={min(hs)}, max={max(hs)}], w[min={min(ws)}, max={max(ws)}]"
    )
    if args.unknown == "error":
        print("  unknown_color_stats: strict mode (error on first unknown color)")
    else:
        print(f"  unknown_color_stats: mapped unknown colors to id={unknown_id}")
    print(f"  manifest: {manifest_path}")
    print(f"  splits: {splits_dir}")


if __name__ == "__main__":
    main()
