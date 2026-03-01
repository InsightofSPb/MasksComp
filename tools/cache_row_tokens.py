#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maskscomp.lm_entropy import collect_images, read_mask_png

CACHE_VERSION = "rle_row_v1"
ABOVE_SENTINEL = np.uint16(65535)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build row-wise RLE disk cache for mask PNGs")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="warped_masks")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--manifest-out", type=Path, required=True)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--include-above-features", action="store_true")
    p.add_argument("--dtype-label", choices=["uint16"], default="uint16")
    p.add_argument("--dtype-len", choices=["uint16", "int32"], default="uint16")
    p.add_argument("--dtype-start", choices=["uint16", "int32"], default="uint16")
    p.add_argument("--compress", choices=["none", "npz"], default="npz")
    p.add_argument("--verify", type=int, default=0)
    return p.parse_args()


def _dtype_from_name(name: str) -> np.dtype:
    return np.dtype(name)


def encode_mask_rle(mask: np.ndarray, include_above: bool) -> Dict[str, np.ndarray]:
    h, w = int(mask.shape[0]), int(mask.shape[1])
    row_ptr = np.zeros((h + 1,), dtype=np.int32)
    starts: List[int] = []
    lens: List[int] = []
    labels: List[int] = []
    above_label: List[int] = []
    above_same: List[int] = []

    total = 0
    for y in range(h):
        row = mask[y]
        changes = np.flatnonzero(row[1:] != row[:-1]) + 1
        run_starts = np.concatenate(([0], changes))
        run_ends = np.concatenate((changes, [w]))
        for s, e in zip(run_starts.tolist(), run_ends.tolist()):
            lab = int(row[s])
            starts.append(int(s))
            lens.append(int(e - s))
            labels.append(lab)
            if include_above:
                if y == 0:
                    above_label.append(int(ABOVE_SENTINEL))
                    above_same.append(0)
                else:
                    abv = int(mask[y - 1, s])
                    above_label.append(abv)
                    above_same.append(1 if abv == lab else 0)
            total += 1
        row_ptr[y + 1] = total

    out = {
        "row_ptr": row_ptr,
        "starts": np.asarray(starts),
        "lens": np.asarray(lens),
        "labels": np.asarray(labels),
    }
    if include_above:
        out["above_label"] = np.asarray(above_label)
        out["above_same"] = np.asarray(above_same)
    return out


def cast_arrays(arrs: Dict[str, np.ndarray], w: int, dtype_start: np.dtype, dtype_len: np.dtype, dtype_label: np.dtype) -> Dict[str, np.ndarray]:
    out = {
        "row_ptr": arrs["row_ptr"].astype(np.int32, copy=False),
        "starts": arrs["starts"].astype(dtype_start, copy=False),
        "lens": arrs["lens"].astype(dtype_len, copy=False),
        "labels": arrs["labels"].astype(dtype_label, copy=False),
    }
    if np.issubdtype(dtype_start, np.unsignedinteger) and w > np.iinfo(dtype_start).max:
        raise ValueError(f"Image width {w} does not fit dtype-start={dtype_start}")
    if np.issubdtype(dtype_len, np.unsignedinteger) and w > np.iinfo(dtype_len).max:
        raise ValueError(f"Image width {w} does not fit dtype-len={dtype_len}")

    if "above_label" in arrs:
        out["above_label"] = arrs["above_label"].astype(np.uint16, copy=False)
        out["above_same"] = arrs["above_same"].astype(np.uint8, copy=False)
    return out


def save_cache(path: Path, compress: str, payload: Dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress == "npz":
        out_path = path.with_suffix(".npz")
        np.savez_compressed(out_path, **payload)
        return out_path
    out_path = path.with_suffix(".npy")
    np.save(out_path, payload, allow_pickle=True)
    return out_path


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    obj = np.load(path, allow_pickle=True)
    return obj.item()


def verify_cache(mask: np.ndarray, arrs: Dict[str, np.ndarray]) -> Tuple[bool, str]:
    h, w = mask.shape
    row_ptr = arrs["row_ptr"]
    starts = arrs["starts"]
    lens = arrs["lens"]
    labels = arrs["labels"]
    if row_ptr.shape[0] != h + 1:
        return False, "row_ptr size mismatch"
    for y in range(h):
        a = int(row_ptr[y])
        b = int(row_ptr[y + 1])
        s = starts[a:b].astype(np.int64)
        l = lens[a:b].astype(np.int64)
        labs = labels[a:b].astype(np.int64)
        if s.size == 0:
            return False, f"row {y} has no runs"
        if int(s[0]) != 0:
            return False, f"row {y} first start != 0"
        if not np.all((l >= 1) & (l <= w)):
            return False, f"row {y} invalid lengths"
        if int(np.sum(l)) != w:
            return False, f"row {y} run lengths do not cover width"
        rec = np.empty((w,), dtype=np.int64)
        x = 0
        for j in range(s.size):
            if int(s[j]) != x:
                return False, f"row {y} start mismatch at run {j}"
            x2 = x + int(l[j])
            rec[x:x2] = int(labs[j])
            x = x2
        if not np.array_equal(rec, mask[y].astype(np.int64)):
            return False, f"row {y} reconstructed labels mismatch"
    return True, "ok"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    records = collect_images(args.data_root, subdir=args.subdir)
    if args.shuffle:
        rng.shuffle(records)
    if args.max_files is not None:
        records = records[: int(args.max_files)]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    dtype_label = _dtype_from_name(args.dtype_label)
    dtype_len = _dtype_from_name(args.dtype_len)
    dtype_start = _dtype_from_name(args.dtype_start)

    verify_pool: List[Tuple[Path, Path]] = []
    with args.manifest_out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=["facade_id", "rel_path", "cache_path", "H", "W", "dtype", "unique_vals", "total_runs"],
        )
        wr.writeheader()

        for rec in records:
            mask = read_mask_png(rec.path)
            if mask is None or mask.ndim != 2:
                print(f"[WARN] Skipping invalid mask: {rec.path}")
                continue

            h, w = int(mask.shape[0]), int(mask.shape[1])
            arrs = encode_mask_rle(mask, include_above=args.include_above_features)
            arrs = cast_arrays(arrs, w=w, dtype_start=dtype_start, dtype_len=dtype_len, dtype_label=dtype_label)

            rel_cache_stem = Path(rec.rel_path).with_suffix("")
            cache_path = save_cache(args.cache_dir / rel_cache_stem, compress=args.compress, payload=arrs)
            cache_rel = str(cache_path.relative_to(args.cache_dir))

            wr.writerow(
                {
                    "facade_id": rec.facade_id,
                    "rel_path": rec.rel_path,
                    "cache_path": cache_rel,
                    "H": h,
                    "W": w,
                    "dtype": str(mask.dtype),
                    "unique_vals": "|".join(str(int(v)) for v in np.unique(mask)),
                    "total_runs": int(arrs["labels"].shape[0]),
                }
            )
            verify_pool.append((rec.path, cache_path))

    schema = {
        "row_ptr": "int32[H+1]",
        "starts": str(dtype_start),
        "lens": str(dtype_len),
        "labels": str(dtype_label),
        "above_label": "uint16" if args.include_above_features else None,
        "above_same": "uint8" if args.include_above_features else None,
    }
    readme = {
        "version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "schema": schema,
    }
    (args.cache_dir / "README_cache.json").write_text(json.dumps(readme, indent=2), encoding="utf-8")

    if args.verify > 0 and verify_pool:
        k = min(int(args.verify), len(verify_pool))
        sampled = rng.sample(verify_pool, k=k)
        for src_path, cache_path in sampled:
            mask = read_mask_png(src_path)
            assert mask is not None
            ok, msg = verify_cache(mask, load_cache(cache_path))
            if not ok:
                raise RuntimeError(f"Verification failed for {src_path}: {msg}")
        print(f"[INFO] Verification passed for {k} files")

    print(f"[INFO] Done. Processed {len(records)} files -> {args.manifest_out}")


if __name__ == "__main__":
    main()
