"""Compare LM validation bbp against classical codec baselines.

This utility aligns splits by facade IDs and reports weighted bits-per-pixel (bbp)
for classical codecs so they can be compared directly against LM val_bbp.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_CLASSIC_COLUMNS = {
    "facade_id",
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
            "from classic_codecs.csv, optionally filtered by LM val split facades."
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
        help=(
            "Optional facade_val.txt from LM training output. If provided, classical "
            "rows are filtered to this split before aggregation."
        ),
    )
    p.add_argument(
        "--lm-epoch",
        type=str,
        default="best",
        help="Which LM epoch to use: 'best', 'last', or an explicit epoch number",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best classical combinations to print",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to save aggregated classical combinations",
    )
    return p.parse_args()


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    return rows


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
        raise ValueError("--lm-epoch must be 'best', 'last', or an integer") from exc

    matches = [r for r in lm_rows if int(r["epoch"]) == epoch_num]
    if not matches:
        raise ValueError(f"Requested epoch {epoch_num} not found in LM log")
    return matches[-1]


def _load_val_facades(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {line.strip() for line in text.splitlines() if line.strip()}


def _validate_classic_header(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("classical CSV has no rows")
    missing = sorted(REQUIRED_CLASSIC_COLUMNS - set(rows[0].keys()))
    if missing:
        raise KeyError(f"classic CSV missing required columns: {missing}")


def aggregate_classic(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    _validate_classic_header(rows)

    grouped: dict[tuple[str, str, str], dict[str, object]] = defaultdict(
        lambda: {
            "rows": 0,
            "facades": set(),
            "bytes_out_sum": 0.0,
            "pixels_sum": 0.0,
        }
    )

    for row in rows:
        key = (str(row["stream"]), str(row["codec"]), str(row["level"]))
        agg = grouped[key]

        h = float(row["H"])
        w = float(row["W"])
        bytes_out = float(row["bytes_out"])
        pixels = max(1.0, h * w)

        agg["rows"] = int(agg["rows"]) + 1
        cast_facades = agg["facades"]
        assert isinstance(cast_facades, set)
        cast_facades.add(str(row["facade_id"]))
        agg["bytes_out_sum"] = float(agg["bytes_out_sum"]) + bytes_out
        agg["pixels_sum"] = float(agg["pixels_sum"]) + pixels

    table: list[dict[str, object]] = []
    for (stream, codec, level), agg in grouped.items():
        pixels_sum = max(1.0, float(agg["pixels_sum"]))
        bbp_weighted = float(agg["bytes_out_sum"]) * 8.0 / pixels_sum
        facades_count = len(agg["facades"]) if isinstance(agg["facades"], set) else 0
        table.append(
            {
                "stream": stream,
                "codec": codec,
                "level": level,
                "rows": int(agg["rows"]),
                "facades": facades_count,
                "bytes_out_sum": float(agg["bytes_out_sum"]),
                "pixels_sum": float(agg["pixels_sum"]),
                "bbp_weighted": bbp_weighted,
            }
        )

    table.sort(key=lambda r: float(r["bbp_weighted"]))
    return table


def _print_top(rows: list[dict[str, object]], top_k: int) -> None:
    top = rows[: max(1, top_k)]
    print("stream        codec   level   bbp_weighted      rows   facades")
    for r in top:
        print(
            f"{str(r['stream']):12}  {str(r['codec']):6}  {str(r['level']):5}  "
            f"{float(r['bbp_weighted']):12.10f}  {int(r['rows']):8d}  {int(r['facades']):7d}"
        )


def _write_aggregated_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stream",
        "codec",
        "level",
        "rows",
        "facades",
        "bytes_out_sum",
        "pixels_sum",
        "bbp_weighted",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()

    lm_log_path = _resolve_file(args.lm_log, "--lm-log")
    classic_csv_path = _resolve_file(args.classic_csv, "--classic-csv")

    lm_rows = _read_lm_log(lm_log_path)
    lm_row = _select_lm_row(lm_rows, args.lm_epoch)
    lm_epoch = int(lm_row["epoch"])
    lm_val_bbp = float(lm_row["val_bbp"])

    classic_rows_all = _read_csv_rows(classic_csv_path)
    total_rows = len(classic_rows_all)

    split_info = "all_rows"
    classic_rows_used = classic_rows_all

    if args.val_facades is not None:
        val_path = _resolve_file(args.val_facades, "--val-facades")
        val_facades = _load_val_facades(val_path)
        classic_rows_used = [r for r in classic_rows_all if str(r.get("facade_id", "")) in val_facades]
        split_info = f"val_facades={len(val_facades)}"

    if not classic_rows_used:
        raise ValueError("No classical rows remain after filtering")

    agg = aggregate_classic(classic_rows_used)
    best = agg[0]
    best_bbp = float(best["bbp_weighted"])
    ratio = lm_val_bbp / best_bbp
    delta_pct = (ratio - 1.0) * 100.0

    print(f"lm_epoch={lm_epoch}")
    print(f"lm_val_bbp={lm_val_bbp:.10f}")
    print(f"classic_rows_total={total_rows}")
    print(f"classic_rows_used={len(classic_rows_used)} ({split_info})")
    print()

    print("Top classical combinations (lower bbp is better):")
    _print_top(agg, top_k=args.top_k)
    print()

    print("Best-vs-LM:")
    print(
        "best_classic="
        f"{best['stream']}/{best['codec']}/lvl={best['level']} "
        f"bbp={best_bbp:.10f}"
    )
    print(f"lm_over_best_ratio={ratio:.6f}x")
    print(f"lm_minus_best_pct={delta_pct:+.2f}%")

    if args.out_csv is not None:
        out_path = args.out_csv.expanduser().resolve()
        _write_aggregated_csv(out_path, agg)
        print(f"saved_aggregated_csv={out_path}")


if __name__ == "__main__":
    main()