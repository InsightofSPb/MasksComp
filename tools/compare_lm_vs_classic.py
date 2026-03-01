"""Compare LM validation bbp against classical codec baselines.

Supports generic dataset IDs (not only facade_id), so it also fits sequential
frame datasets such as A2D2.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_CLASSIC_COLUMNS = {
    "stream",
    "codec",
    "level",
    "H",
    "W",
    "bytes_out",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare LM val_bbp from train_log.csv against weighted classical bbp "
            "from classic_codecs.csv, with optional split-ID filtering."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lm-log", type=Path, required=True, help="Path to LM train_log.csv")
    p.add_argument(
        "--classic-csv",
        type=Path,
        required=True,
        help="Path to classic_codecs.csv (from tools/bench_mask_codecs.py)",
    )
    p.add_argument(
        "--val-facades",
        type=Path,
        default=None,
        help="Backward-compatible alias for --split-ids.",
    )
    p.add_argument(
        "--split-ids",
        type=Path,
        default=None,
        help="Optional file with one ID per line for filtering classical rows.",
    )
    p.add_argument(
        "--id-column",
        type=str,
        default="facade_id",
        help="ID column in classic CSV used for filtering/reporting (e.g. facade_id, sequence_id).",
    )
    p.add_argument(
        "--lm-epoch",
        type=str,
        default="best",
        help="Which LM epoch to use: 'best', 'last', or an explicit epoch number",
    )
    p.add_argument("--top-k", type=int, default=10, help="How many best rows to print")
    p.add_argument("--out-csv", type=Path, default=None, help="Optional path for aggregated CSV")
    p.add_argument("--show-ids", action="store_true", help="Print IDs used after filtering")
    p.add_argument("--max-ids-print", type=int, default=50, help="Max IDs printed with --show-ids")
    return p.parse_args()


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _row_id(row: dict[str, str], id_column: str) -> str:
    return str(row.get(id_column, "")).strip()


def _load_ids_file(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _read_lm_log(path: Path) -> list[dict[str, float]]:
    rows = _read_csv_rows(path)
    out: list[dict[str, float]] = []
    for r in rows:
        if "epoch" not in r or "val_bbp" not in r:
            raise KeyError("LM log must contain 'epoch' and 'val_bbp' columns")
        out.append({"epoch": int(r["epoch"]), "val_bbp": float(r["val_bbp"])})
    if not out:
        raise ValueError("LM log has no rows")
    return out


def _select_lm_row(lm_rows: list[dict[str, float]], lm_epoch: str) -> dict[str, float]:
    if lm_epoch == "best":
        return min(lm_rows, key=lambda x: float(x["val_bbp"]))
    if lm_epoch == "last":
        return max(lm_rows, key=lambda x: int(x["epoch"]))
    try:
        epoch_num = int(lm_epoch)
    except ValueError as exc:
        raise ValueError("--lm-epoch must be 'best', 'last', or integer") from exc
    matches = [r for r in lm_rows if int(r["epoch"]) == epoch_num]
    if not matches:
        raise ValueError(f"Requested epoch {epoch_num} not found in LM log")
    return matches[-1]


def _validate_classic_header(rows: list[dict[str, str]], id_column: str) -> None:
    if not rows:
        raise ValueError("classical CSV has no rows")
    missing = sorted(REQUIRED_CLASSIC_COLUMNS - set(rows[0].keys()))
    if missing:
        raise KeyError(f"classic CSV missing required columns: {missing}")
    if id_column not in rows[0]:
        print(f"[warning] id column '{id_column}' not found in classic CSV; id-based filtering/reporting disabled")


def aggregate_classic(rows: list[dict[str, str]], id_column: str) -> list[dict[str, object]]:
    _validate_classic_header(rows, id_column)
    grouped: dict[tuple[str, str, str], dict[str, object]] = defaultdict(
        lambda: {"rows": 0, "ids": set(), "images": set(), "bytes_out_sum": 0.0, "pixels_sum": 0.0}
    )
    has_filename = "filename" in rows[0]
    id_present = id_column in rows[0]

    for row in rows:
        key = (str(row["stream"]), str(row["codec"]), str(row["level"]))
        agg = grouped[key]
        row_id = _row_id(row, id_column) if id_present else ""

        agg["rows"] = int(agg["rows"]) + 1
        cast_ids = agg["ids"]
        assert isinstance(cast_ids, set)
        if row_id:
            cast_ids.add(row_id)

        cast_imgs = agg["images"]
        assert isinstance(cast_imgs, set)
        img_key = str(row.get("filename", agg["rows"])) if has_filename else str(agg["rows"])
        cast_imgs.add((row_id, img_key))

        h = float(row["H"])
        w = float(row["W"])
        agg["bytes_out_sum"] = float(agg["bytes_out_sum"]) + float(row["bytes_out"])
        agg["pixels_sum"] = float(agg["pixels_sum"]) + max(1.0, h * w)

    out: list[dict[str, object]] = []
    for (stream, codec, level), agg in grouped.items():
        pixels_sum = max(1.0, float(agg["pixels_sum"]))
        out.append(
            {
                "stream": stream,
                "codec": codec,
                "level": level,
                "rows": int(agg["rows"]),
                "ids": len(agg["ids"]) if isinstance(agg["ids"], set) else 0,
                "images": len(agg["images"]) if isinstance(agg["images"], set) else int(agg["rows"]),
                "bytes_out_sum": float(agg["bytes_out_sum"]),
                "pixels_sum": pixels_sum,
                "bbp_weighted": float(agg["bytes_out_sum"]) * 8.0 / pixels_sum,
            }
        )
    out.sort(key=lambda r: float(r["bbp_weighted"]))
    return out


def _print_top(rows: list[dict[str, object]], top_k: int) -> None:
    print("stream        codec   level   bbp_weighted      rows   images   ids")
    for r in rows[: max(1, top_k)]:
        print(
            f"{str(r['stream']):12}  {str(r['codec']):6}  {str(r['level']):5}  "
            f"{float(r['bbp_weighted']):12.10f}  {int(r['rows']):8d}  {int(r['images']):8d}  {int(r['ids']):5d}"
        )


def _write_aggregated_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["stream", "codec", "level", "rows", "images", "ids", "bytes_out_sum", "pixels_sum", "bbp_weighted"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.val_facades is not None and args.split_ids is not None:
        raise ValueError("Use only one of --val-facades or --split-ids")

    lm_log_path = _resolve_file(args.lm_log, "--lm-log")
    classic_csv_path = _resolve_file(args.classic_csv, "--classic-csv")

    lm_rows = _read_lm_log(lm_log_path)
    lm_row = _select_lm_row(lm_rows, args.lm_epoch)
    lm_epoch = int(lm_row["epoch"])
    lm_val_bbp = float(lm_row["val_bbp"])

    classic_all = _read_csv_rows(classic_csv_path)
    _validate_classic_header(classic_all, args.id_column)

    split_info = "all_rows"
    classic_used = classic_all
    ids_path = args.split_ids if args.split_ids is not None else args.val_facades
    requested_ids: set[str] | None = None

    if ids_path is not None:
        ids_file = _resolve_file(ids_path, "--split-ids/--val-facades")
        requested_ids = _load_ids_file(ids_file)
        if args.id_column not in classic_all[0]:
            raise KeyError(f"id column '{args.id_column}' not found in classic CSV")
        classic_used = [r for r in classic_all if _row_id(r, args.id_column) in requested_ids]
        split_info = f"split_ids={len(requested_ids)} id_column={args.id_column}"

    if not classic_used:
        raise ValueError("No classical rows remain after filtering")

    used_ids = sorted({_row_id(r, args.id_column) for r in classic_used if _row_id(r, args.id_column)})
    used_images = {( _row_id(r, args.id_column), str(r.get("filename", i)) ) for i, r in enumerate(classic_used)}

    agg = aggregate_classic(classic_used, id_column=args.id_column)
    best = agg[0]
    best_bbp = float(best["bbp_weighted"])
    ratio = lm_val_bbp / best_bbp
    delta_pct = (ratio - 1.0) * 100.0

    print(f"lm_epoch={lm_epoch}")
    print(f"lm_val_bbp={lm_val_bbp:.10f}")
    print(f"classic_rows_total={len(classic_all)}")
    print(f"classic_rows_used={len(classic_used)} ({split_info})")
    print(f"classic_ids_used={len(used_ids)}")
    print(f"classic_images_used={len(used_images)}")

    if len(used_ids) <= 1:
        print("NOTE: only 1 ID is used in comparison; this can happen for small splits.")
    if requested_ids is not None and len(requested_ids) <= 1:
        print("NOTE: provided split-ID file contains <=1 ID.")

    if args.show_ids:
        listed = used_ids[: max(1, args.max_ids_print)]
        print(f"ids_listed={len(listed)}")
        for x in listed:
            print(f"  - {x}")
        if len(used_ids) > len(listed):
            print(f"  ... ({len(used_ids) - len(listed)} more)")

    print("\nTop classical combinations (lower bbp is better):")
    _print_top(agg, top_k=args.top_k)

    print("\nBest-vs-LM:")
    print(f"best_classic={best['stream']}/{best['codec']}/lvl={best['level']} bbp={best_bbp:.10f}")
    print(f"lm_over_best_ratio={ratio:.6f}x")
    print(f"lm_minus_best_pct={delta_pct:+.2f}%")

    if args.out_csv is not None:
        out_path = args.out_csv.expanduser().resolve()
        _write_aggregated_csv(out_path, agg)
        print(f"saved_aggregated_csv={out_path}")


if __name__ == "__main__":
    main()