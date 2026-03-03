#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maskscomp.change_detection import (
    compress_bytes,
    configure_logging,
    iter_tiles_2d,
    pack_change_mask_bits,
    percentile_or_nan,
    read_mask,
    read_pairs_csv,
    set_deterministic_seed,
    tile_grid_shape,
)

LOGGER = logging.getLogger("eval_change_tiles")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute tile-wise surprise maps from residual representations")

    p.add_argument(
        "--mode",
        choices=["classic_residual", "iid_residual", "lm_residual", "changed_frac"],
        required=True,
        help=(
            "classic_residual: bits-per-tile via codec on C_tile and V_tile; "
            "iid_residual (alias lm_residual): IID baseline (Bernoulli for C + histogram for V-on-changed); "
            "changed_frac: heatmap is fraction of changed pixels per tile (unitless)."
        ),
    )

    p.add_argument("--data-root", type=Path, required=True,
                   help="Residual dataset root containing <pair_id>/residual_{C,V}/")
    p.add_argument("--pairs-csv", type=Path, required=True,
                   help="Pairs CSV produced by make_pairs.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)

    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=64)

    p.add_argument("--codec", type=str, default="lzma", help="Used in classic_residual mode")
    p.add_argument("--level", type=int, default=6, help="Used in classic_residual mode")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    # NEW: subset / speed controls
    p.add_argument("--ids-txt", type=Path, default=None,
                   help="Optional txt with pair_id to process (one per line).")
    p.add_argument("--max-items", type=int, default=None,
                   help="Optional limit on number of pairs to process (after filtering).")
    p.add_argument("--skip-existing", action="store_true",
                   help="If heatmap file already exists, do not recompute it (still writes CSV row).")

    # Optional: compute hybrid bbp (classic C bits + MSDZip V bits) and write it into CSV column `bbp`.
    p.add_argument(
        "--use-hybrid-bbp",
        action="store_true",
        help=(
            "If set, CSV column `bbp` will be set to hybrid bbp_new_sum = (bits_C_classic + bits_V_nn)/(H*W). "
            "Requires --classic-csv and --msdzip-v-csv."
        ),
    )
    p.add_argument(
        "--classic-csv",
        type=Path,
        default=None,
        help=(
            "Path to residual_codecs CSV produced by bench_residual_codecs.py "
            "(must contain pair_id, bits_C, codec, level, split). "
            "Used only if --use-hybrid-bbp is set."
        ),
    )
    p.add_argument("--classic-codec", type=str, default="lzma", help="Filter codec for bits_C (hybrid bbp).")
    p.add_argument("--classic-level", type=int, default=6, help="Filter level for bits_C (hybrid bbp).")
    p.add_argument(
        "--msdzip-v-csv",
        type=Path,
        default=None,
        help=(
            "Path to MSDZip eval CSV for residual_V (val_per_image.csv). "
            "Must contain bits_total and an id column (pair_id or facade_id). Used only if --use-hybrid-bbp is set."
        ),
    )
    p.add_argument(
        "--msdzip-id-col",
        type=str,
        default=None,
        help="ID column in msdzip-v-csv. If not set, tries pair_id then facade_id.",
    )
    p.add_argument("--msdzip-bits-col", type=str, default="bits_total",
                   help="Bits column in msdzip-v-csv (default bits_total).")

    return p.parse_args()


def _estimate_probs(records, data_root: Path):
    """IID baseline: C ~ Bernoulli(p1), V_on_changed ~ categorical(hist over changed values)."""
    hist_v = np.ones(256, dtype=np.float64)
    c1 = 1.0
    c0 = 1.0
    for r in records:
        stem = Path(r.cur_path).stem
        c = read_mask(data_root / r.pair_id / "residual_C" / f"{stem}.png")
        v = read_mask(data_root / r.pair_id / "residual_V" / f"{stem}.png")
        c1 += float(c.sum())
        c0 += float(c.size - c.sum())
        vals = v[c > 0]
        if vals.size:
            bins = np.bincount(vals.astype(np.uint8), minlength=256)
            hist_v += bins
    p1 = c1 / (c1 + c0)
    p0 = c0 / (c1 + c0)
    pv = hist_v / hist_v.sum()
    return p0, p1, pv


def _load_hybrid_maps(
    classic_csv: Path,
    msdzip_v_csv: Path,
    split: str,
    classic_codec: str,
    classic_level: int,
    msdzip_id_col: Optional[str],
    msdzip_bits_col: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
      bitsC_map[pair_id] = bits_C_classic for (codec, level, split)
      bitsV_map[pair_id] = bits_V_nn (msdzip bits_total) for split
    """
    dfc = __import__("pandas").read_csv(classic_csv)
    dfc["codec"] = dfc["codec"].astype(str).str.strip().str.lower()
    dfc["level"] = __import__("pandas").to_numeric(dfc["level"], errors="coerce")
    dfc = dfc[
        (dfc["codec"] == classic_codec.strip().lower())
        & (dfc["level"] == float(classic_level))
        & (dfc["split"] == split)
    ].copy()
    if "pair_id" not in dfc.columns or "bits_C" not in dfc.columns:
        raise ValueError(f"classic-csv must contain pair_id and bits_C. Columns: {list(dfc.columns)}")
    dfc["bits_C"] = __import__("pandas").to_numeric(dfc["bits_C"], errors="coerce")
    bitsC_map = dict(zip(dfc["pair_id"].astype(str), dfc["bits_C"].astype(float)))

    dfv = __import__("pandas").read_csv(msdzip_v_csv)
    if msdzip_id_col is None:
        if "pair_id" in dfv.columns:
            msdzip_id_col = "pair_id"
        elif "facade_id" in dfv.columns:
            msdzip_id_col = "facade_id"
        else:
            raise ValueError(f"msdzip-v-csv must contain pair_id or facade_id. Columns: {list(dfv.columns)}")
    if msdzip_bits_col not in dfv.columns:
        raise ValueError(f"msdzip-v-csv missing bits column {msdzip_bits_col}. Columns: {list(dfv.columns)}")

    dfv[msdzip_bits_col] = __import__("pandas").to_numeric(dfv[msdzip_bits_col], errors="coerce")
    bitsV_map = dict(zip(dfv[msdzip_id_col].astype(str), dfv[msdzip_bits_col].astype(float)))
    return bitsC_map, bitsV_map


def _load_ids(ids_txt: Path) -> set:
    ids = set()
    for ln in ids_txt.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            ids.add(s)
    return ids


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_deterministic_seed(args.seed)

    # Backwards-compat alias
    mode = args.mode
    if mode == "lm_residual":
        mode = "iid_residual"

    rows = [r for r in read_pairs_csv(args.pairs_csv) if r.split == args.split]

    # NEW: filter by ids
    if args.ids_txt is not None:
        keep = _load_ids(args.ids_txt)
        before = len(rows)
        rows = [r for r in rows if str(r.pair_id) in keep]
        LOGGER.info("Filtered by ids-txt: %d -> %d rows", before, len(rows))

    # NEW: limit items
    if args.max_items is not None:
        rows = rows[: int(args.max_items)]
        LOGGER.info("Limited to max-items=%d rows", len(rows))

    heat_root = args.out_dir / "heatmaps_tiles"
    heat_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "change_tiles_scores.csv"

    # IID params
    p0 = p1 = 0.5
    pv = np.ones(256, dtype=np.float64) / 256.0
    if mode == "iid_residual":
        p0, p1, pv = _estimate_probs(rows, args.data_root)
        LOGGER.info("IID probs estimated: p1=%.6f p0=%.6f", p1, p0)

    # Optional: hybrid bbp maps (classic bits_C + msdzip bits_V)
    bitsC_map: Dict[str, float] = {}
    bitsV_map: Dict[str, float] = {}
    if args.use_hybrid_bbp:
        if args.classic_csv is None or args.msdzip_v_csv is None:
            raise SystemExit("--use-hybrid-bbp requires --classic-csv and --msdzip-v-csv")
        bitsC_map, bitsV_map = _load_hybrid_maps(
            classic_csv=args.classic_csv,
            msdzip_v_csv=args.msdzip_v_csv,
            split=args.split,
            classic_codec=args.classic_codec,
            classic_level=args.classic_level,
            msdzip_id_col=args.msdzip_id_col,
            msdzip_bits_col=args.msdzip_bits_col,
        )
        LOGGER.info(
            "Hybrid maps loaded: bitsC=%d entries, bitsV=%d entries (codec=%s level=%d)",
            len(bitsC_map), len(bitsV_map), args.classic_codec, args.classic_level
        )

    # Speedups for classic_residual: cache compressed size for all-zero tiles
    zero_c_bits: Optional[float] = None
    zero_v_bits: Optional[float] = None

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "facade_id",
                "filename",
                "bbp",
                "heat_min",
                "heat_p50",
                "heat_p90",
                "heat_max",
                "heatmap_path",
                "bbp_heat",
                "bbp_hybrid",
                "bits_C_classic",
                "bits_V_nn",
            ],
        )
        wr.writeheader()

        for r in rows:
            stem = Path(r.cur_path).stem
            c = read_mask(args.data_root / r.pair_id / "residual_C" / f"{stem}.png")
            v = read_mask(args.data_root / r.pair_id / "residual_V" / f"{stem}.png")

            h, w = c.shape
            gh, gw = tile_grid_shape(h, w, args.tile_size, args.stride)

            npy_dir = heat_root / r.pair_id
            npy_dir.mkdir(parents=True, exist_ok=True)
            npy_path = npy_dir / f"{stem}.npy"

            # NEW: skip recompute if exists
            if args.skip_existing and npy_path.exists():
                heat = np.load(npy_path).astype(np.float32, copy=False)
            else:
                heat = np.zeros((gh, gw), dtype=np.float32)

                idx = 0
                for y, x, c_tile in iter_tiles_2d(c, args.tile_size, args.stride):
                    gy, gx = divmod(idx, gw)
                    v_tile = v[y: y + args.tile_size, x: x + args.tile_size]

                    if mode == "classic_residual":
                        # bits for C_tile
                        if c_tile.sum() == 0:
                            if zero_c_bits is None:
                                zero_c_bits = 8.0 * len(compress_bytes(pack_change_mask_bits(c_tile), args.codec, args.level))
                            bits_c = zero_c_bits
                        else:
                            bits_c = 8.0 * len(compress_bytes(pack_change_mask_bits(c_tile), args.codec, args.level))

                        # bits for V_tile
                        if (v_tile == 0).all():
                            if zero_v_bits is None:
                                zero_v_bits = 8.0 * len(compress_bytes(v_tile.astype(np.uint8, copy=False).tobytes(), args.codec, args.level))
                            bits_v = zero_v_bits
                        else:
                            bits_v = 8.0 * len(compress_bytes(v_tile.astype(np.uint8, copy=False).tobytes(), args.codec, args.level))

                        bits = float(bits_c + bits_v)

                    elif mode == "iid_residual":
                        c_flat = c_tile.reshape(-1)
                        bits_c = float(-(np.log2(p1) * c_flat.sum() + np.log2(p0) * (c_flat.size - c_flat.sum())))
                        changed = v_tile[c_tile > 0].reshape(-1)
                        bits_v = float(-np.log2(pv[np.clip(changed, 0, 255)]).sum()) if changed.size else 0.0
                        bits = float(bits_c + bits_v)

                    else:
                        # changed_frac
                        bits = float((c_tile != 0).mean())

                    heat[gy, gx] = bits
                    idx += 1

                np.save(npy_path, heat)

            bbp_heat = float(heat.sum()) / max(1, h * w) if mode != "changed_frac" else float(c.mean())

            bbp_hybrid = np.nan
            bits_C_classic = np.nan
            bits_V_nn = np.nan
            if args.use_hybrid_bbp:
                pid = str(r.pair_id)
                bits_C_classic = float(bitsC_map.get(pid, np.nan))
                bits_V_nn = float(bitsV_map.get(pid, np.nan))
                if np.isfinite(bits_C_classic) and np.isfinite(bits_V_nn):
                    bbp_hybrid = float((bits_C_classic + bits_V_nn) / max(1, h * w))

            bbp_out = bbp_heat
            if args.use_hybrid_bbp and np.isfinite(bbp_hybrid):
                bbp_out = bbp_hybrid

            wr.writerow(
                {
                    "split": r.split,
                    "facade_id": r.pair_id,
                    "filename": f"{stem}.png",
                    "bbp": bbp_out,
                    "heat_min": float(heat.min()),
                    "heat_p50": percentile_or_nan(heat.ravel(), 50),
                    "heat_p90": percentile_or_nan(heat.ravel(), 90),
                    "heat_max": float(heat.max()),
                    "heatmap_path": str(npy_path),
                    "bbp_heat": bbp_heat,
                    "bbp_hybrid": bbp_hybrid,
                    "bits_C_classic": bits_C_classic,
                    "bits_V_nn": bits_V_nn,
                }
            )

    LOGGER.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()