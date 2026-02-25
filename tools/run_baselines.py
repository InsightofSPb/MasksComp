#!/usr/bin/env python3
"""Compute baseline compression metrics for int32 token streams."""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaselineRow:
    dataset: str
    method: str
    level: int
    input_bytes: int
    compressed_bytes: int
    bits_per_pixel: float | None
    bits_per_token: float | None


def parse_levels(raw: str, flag: str, parser: argparse.ArgumentParser) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            parser.error(f"{flag} must contain only integers, got '{part}'")
    if not values:
        parser.error(f"{flag} cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute LZMA/ZSTD baseline compression for *.int32.bin token stream files "
            "and export CSV + Markdown summaries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--streams-dir",
        required=True,
        type=Path,
        help="Directory containing *.int32.bin streams and matching *.stats.json files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where baselines.csv and baselines.md will be written.",
    )
    parser.add_argument(
        "--lzma-presets",
        default="6",
        help="Comma-separated LZMA presets.",
    )
    parser.add_argument(
        "--zstd-levels",
        default="3",
        help="Comma-separated ZSTD levels.",
    )
    parser.add_argument(
        "--skip-zstd",
        action="store_true",
        help="Skip ZSTD runs even if zstd binary/python package is available.",
    )
    args = parser.parse_args()

    args.streams_dir = args.streams_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.streams_dir.is_dir():
        parser.error(f"--streams-dir is not a directory: {args.streams_dir}")

    args.lzma_presets = parse_levels(args.lzma_presets, "--lzma-presets", parser)
    args.zstd_levels = parse_levels(args.zstd_levels, "--zstd-levels", parser)
    return args


def warn(msg: str) -> None:
    print(f"[warning] {msg}", file=sys.stderr)


def load_stats(stats_path: Path) -> tuple[int | None, int | None]:
    if not stats_path.exists():
        warn(f"stats file not found: {stats_path}; bits/pixel and bits/token will be empty")
        return None, None

    try:
        with stats_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"failed to load stats {stats_path}: {exc}; bits metrics will be empty")
        return None, None

    total_pixels = payload.get("total_pixels")
    total_tokens = payload.get("total_tokens")

    if not isinstance(total_pixels, int) or total_pixels <= 0:
        warn(f"invalid total_pixels in {stats_path}; bits/pixel will be empty")
        total_pixels = None
    if not isinstance(total_tokens, int) or total_tokens <= 0:
        warn(f"invalid total_tokens in {stats_path}; bits/token will be empty")
        total_tokens = None

    return total_pixels, total_tokens


def compress_lzma(data: bytes, preset: int) -> int:
    compressed = lzma.compress(data, preset=preset)
    return len(compressed)


def detect_zstd_backend(skip_zstd: bool) -> str | None:
    if skip_zstd:
        print("[info] ZSTD skipped due to --skip-zstd")
        return None

    if shutil.which("zstd"):
        return "binary"

    try:
        import zstandard  # noqa: F401

        return "python"
    except ImportError:
        warn("zstd binary and python package 'zstandard' are unavailable; skipping zstd")
        return None


def compress_zstd(in_path: Path, data: bytes, level: int, backend: str) -> int:
    if backend == "binary":
        proc = subprocess.run(
            ["zstd", f"-{level}", "-q", "-c", str(in_path)],
            capture_output=True,
            check=True,
        )
        return len(proc.stdout)

    if backend == "python":
        import zstandard

        compressed = zstandard.ZstdCompressor(level=level).compress(data)
        return len(compressed)

    raise ValueError(f"unsupported zstd backend: {backend}")


def dataset_name(stream_path: Path) -> str:
    return stream_path.name.removesuffix(".int32.bin")


def fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def write_csv(path: Path, rows: list[BaselineRow]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "level",
        "input_bytes",
        "compressed_bytes",
        "bits_per_pixel",
        "bits_per_token",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "method": row.method,
                    "level": row.level,
                    "input_bytes": row.input_bytes,
                    "compressed_bytes": row.compressed_bytes,
                    "bits_per_pixel": fmt_float(row.bits_per_pixel),
                    "bits_per_token": fmt_float(row.bits_per_token),
                }
            )


def write_markdown(path: Path, rows: list[BaselineRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| dataset | method | level | input_bytes | compressed_bytes | bits_per_pixel | bits_per_token |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| "
                f"{row.dataset} | {row.method} | {row.level} | {row.input_bytes} | {row.compressed_bytes} | "
                f"{fmt_float(row.bits_per_pixel)} | {fmt_float(row.bits_per_token)} |\n"
            )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    streams = sorted(args.streams_dir.glob("*.int32.bin"))
    if not streams:
        warn(f"no stream files found in {args.streams_dir}")

    zstd_backend = detect_zstd_backend(args.skip_zstd)
    if zstd_backend == "binary":
        print("[info] Using zstd binary")
    elif zstd_backend == "python":
        print("[info] Using python zstandard backend")

    rows: list[BaselineRow] = []

    for stream_path in streams:
        data = stream_path.read_bytes()
        input_bytes = len(data)
        ds = dataset_name(stream_path)

        stats_path = stream_path.with_suffix("").with_suffix(".stats.json")
        total_pixels, total_tokens = load_stats(stats_path)

        for preset in args.lzma_presets:
            compressed_bytes = compress_lzma(data, preset=preset)
            bits_per_pixel = (
                8.0 * compressed_bytes / total_pixels if total_pixels is not None else None
            )
            bits_per_token = (
                8.0 * compressed_bytes / total_tokens if total_tokens is not None else None
            )
            rows.append(
                BaselineRow(
                    dataset=ds,
                    method="lzma",
                    level=preset,
                    input_bytes=input_bytes,
                    compressed_bytes=compressed_bytes,
                    bits_per_pixel=bits_per_pixel,
                    bits_per_token=bits_per_token,
                )
            )

        if zstd_backend is not None:
            for level in args.zstd_levels:
                compressed_bytes = compress_zstd(
                    in_path=stream_path,
                    data=data,
                    level=level,
                    backend=zstd_backend,
                )
                bits_per_pixel = (
                    8.0 * compressed_bytes / total_pixels if total_pixels is not None else None
                )
                bits_per_token = (
                    8.0 * compressed_bytes / total_tokens if total_tokens is not None else None
                )
                rows.append(
                    BaselineRow(
                        dataset=ds,
                        method="zstd",
                        level=level,
                        input_bytes=input_bytes,
                        compressed_bytes=compressed_bytes,
                        bits_per_pixel=bits_per_pixel,
                        bits_per_token=bits_per_token,
                    )
                )

        print(f"[info] Processed {stream_path.name} ({input_bytes} bytes)")

    rows.sort(key=lambda r: (r.dataset, r.method, r.level))

    csv_path = args.out_dir / "baselines.csv"
    md_path = args.out_dir / "baselines.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"[done] Wrote {csv_path}")
    print(f"[done] Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
