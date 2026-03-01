#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maskscomp.lm_entropy import CachedRowTokenDataset, load_cache_manifest, read_mask_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roundtrip-check row cache against source masks")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--num-files", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_cache(cache_path: Path) -> dict:
    if cache_path.suffix == ".npz":
        with np.load(cache_path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    obj = np.load(cache_path, allow_pickle=True)
    return obj.item()


def check_row(mask_row: np.ndarray, starts: np.ndarray, lens: np.ndarray, labels: np.ndarray) -> bool:
    w = int(mask_row.shape[0])
    if starts.size == 0 or int(starts[0]) != 0:
        return False
    if int(np.sum(lens.astype(np.int64))) != w:
        return False
    rec = np.empty((w,), dtype=np.int64)
    x = 0
    for s, ln, lab in zip(starts.tolist(), lens.tolist(), labels.tolist()):
        if int(s) != x or int(ln) <= 0:
            return False
        rec[x : x + int(ln)] = int(lab)
        x += int(ln)
    return np.array_equal(rec, mask_row.astype(np.int64))


def main() -> None:
    args = parse_args()
    entries = load_cache_manifest(args.manifest)
    if not entries:
        raise RuntimeError("Empty manifest")

    rng = random.Random(args.seed)
    chosen = rng.sample(entries, k=min(args.num_files, len(entries)))

    ds = CachedRowTokenDataset(
        cache_root=args.cache_root,
        manifest_csv=args.manifest,
        allowed_facade_ids=None,
        label_bos_idx=0,
        no_above_idx=0,
        label_to_idx=None,
        use_2d_context=False,
        lru_cache_size=2,
    )
    print(f"[INFO] n_files={ds.n_files} total_rows={len(ds)}")
    probe = sorted(set([0, max(0, len(ds) // 2), max(0, len(ds) - 1)]))
    for idx in probe:
        file_i, row_y = ds._locate(idx)
        back_idx = int(ds.file_row_offsets[file_i] + row_y)
        print(f"[MAP] global_idx={idx} -> file_i={file_i}, row_y={row_y}, back_idx={back_idx}")

    checked_rows = 0
    total_runs = 0
    for e in chosen:
        mask = read_mask_png(args.data_root / e["rel_path"])
        if mask is None:
            raise RuntimeError(f"Invalid source mask: {e['rel_path']}")
        cache = _load_cache(args.cache_root / e["cache_path"])
        row_ptr = cache["row_ptr"]
        starts = cache["starts"]
        lens = cache["lens"]
        labels = cache["labels"]

        h, _w = mask.shape
        if row_ptr.shape[0] != h + 1:
            raise RuntimeError(f"row_ptr mismatch: {e['cache_path']}")

        for y in range(h):
            a = int(row_ptr[y])
            b = int(row_ptr[y + 1])
            ok = check_row(mask[y], starts[a:b], lens[a:b], labels[a:b])
            if not ok:
                raise RuntimeError(f"Row mismatch: {e['rel_path']} row={y}")
            checked_rows += 1
            total_runs += (b - a)

    print(f"[OK] checked_files={len(chosen)} checked_rows={checked_rows} total_runs={total_runs}")


if __name__ == "__main__":
    main()
