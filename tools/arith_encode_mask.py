#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import json
import torch

from maskscomp.entropy_coding.lm_codec import OnlineConfig, encode_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arithmetic encode mask(s) with LM entropy model")
    p.add_argument("--input", type=Path)
    p.add_argument("--input-list", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--subdir", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-2d-context", action="store_true")
    p.add_argument("--mode", choices=["pretrained", "online"], default="pretrained")
    p.add_argument("--online-optimizer", type=str, default="sgd")
    p.add_argument("--online-lr", type=float, default=1e-3)
    p.add_argument("--online-steps-per-row", type=int, default=1)
    p.add_argument("--online-clip", type=float, default=1.0)
    p.add_argument("--online-after", choices=["row", "image"], default="row")
    p.add_argument("--dump-stats", type=Path, default=None)
    args = p.parse_args()
    if bool(args.input is None) == bool(args.input_list is None):
        raise ValueError("Specify exactly one of --input or --input-list")
    return args


def main() -> None:
    args = parse_args()
    online = OnlineConfig(
        mode=args.mode,
        optimizer=args.online_optimizer,
        lr=args.online_lr,
        steps_per_row=args.online_steps_per_row,
        clip=args.online_clip,
        online_after=args.online_after,
    )
    device = torch.device(args.device)

    if args.input is not None:
        out_bin = args.out
        if out_bin.suffix != ".bin":
            out_bin = out_bin.with_suffix(".bin")
        stats = encode_file(args.input, args.checkpoint, out_bin, args.use_2d_context, online, device, args.dump_stats)
        print(json.dumps({"input": str(args.input), "out": str(out_bin), **stats}, indent=2))
        return

    lines = [ln.strip() for ln in args.input_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for rel in lines:
        in_path = Path(rel)
        if args.subdir:
            in_path = Path(args.subdir) / in_path
        out_bin = out_dir / Path(rel).with_suffix(".bin")
        stats = encode_file(in_path, args.checkpoint, out_bin, args.use_2d_context, online, device, None)
        all_stats.append(stats)
    if args.dump_stats is not None:
        args.dump_stats.parent.mkdir(parents=True, exist_ok=True)
        args.dump_stats.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
