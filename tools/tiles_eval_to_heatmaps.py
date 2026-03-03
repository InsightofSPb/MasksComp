#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TILE_RE = re.compile(r"^(?P<stem>.+)__gy(?P<gy>\d+)__gx(?P<gx>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble tile-eval CSV into per-image tile heatmaps")
    p.add_argument("--tile-ds-root", type=Path, required=True)
    p.add_argument("--tiles-eval-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _load_meta(tile_ds_root: Path, pair_id: str) -> dict[str, dict]:
    meta_path = tile_ds_root / pair_id / "tiles_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing tiles_meta.json for pair {pair_id}: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    out = {}
    for it in items:
        out[str(it["stem"])] = {
            "H": int(it["H"]),
            "W": int(it["W"]),
            "gh": int(it["gh"]),
            "gw": int(it["gw"]),
            "tile": int(it["tile"]),
            "stride": int(it["stride"]),
        }
    return out


def main() -> None:
    args = parse_args()

    grouped: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    split_by_key: dict[tuple[str, str], str] = {}

    with args.tiles_eval_csv.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            pair_id = str(row["facade_id"])
            rel_path = str(row["rel_path"])
            split = str(row.get("split", "val"))
            name = Path(rel_path).name
            m = TILE_RE.match(name)
            if m is None:
                continue
            stem = m.group("stem")
            gy = int(m.group("gy"))
            gx = int(m.group("gx"))
            bits = float(row["bits_total"])
            key = (pair_id, stem)
            grouped[key][(gy, gx)] = bits
            split_by_key[key] = split

    heat_root = args.out_dir / "heatmaps_tiles"
    heat_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "change_tiles_scores.csv"

    meta_cache: dict[str, dict[str, dict]] = {}

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "facade_id",
                "filename",
                "bbp",
                "heat_p90",
                "heat_max",
                "heat_p50",
                "heat_min",
                "heatmap_path",
            ],
        )
        wr.writeheader()

        for (pair_id, stem), tile_bits in sorted(grouped.items()):
            if pair_id not in meta_cache:
                meta_cache[pair_id] = _load_meta(args.tile_ds_root, pair_id)
            if stem not in meta_cache[pair_id]:
                raise KeyError(f"No meta for pair={pair_id}, stem={stem}")
            meta = meta_cache[pair_id][stem]

            gh, gw = int(meta["gh"]), int(meta["gw"])
            h, w_img = int(meta["H"]), int(meta["W"])
            heat = np.zeros((gh, gw), dtype=np.float32)
            for (gy, gx), bits in tile_bits.items():
                if 0 <= gy < gh and 0 <= gx < gw:
                    heat[gy, gx] = float(bits)

            out_pair = heat_root / pair_id
            out_pair.mkdir(parents=True, exist_ok=True)
            out_npy = out_pair / f"{stem}.npy"
            np.save(out_npy, heat)

            bbp = float(heat.sum()) / max(1, h * w_img)
            wr.writerow(
                {
                    "split": split_by_key[(pair_id, stem)],
                    "facade_id": pair_id,
                    "filename": f"{stem}.png",
                    "bbp": bbp,
                    "heat_p90": float(np.percentile(heat, 90)),
                    "heat_max": float(np.max(heat)),
                    "heat_p50": float(np.percentile(heat, 50)),
                    "heat_min": float(np.min(heat)),
                    "heatmap_path": str(out_npy.relative_to(args.out_dir)),
                }
            )

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
