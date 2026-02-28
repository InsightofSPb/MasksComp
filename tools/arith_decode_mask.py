#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from maskscomp.entropy_coding.lm_codec import (
    checkpoint_sha,
    decode_mask,
    load_checkpoint_and_model,
    read_bin,
    set_deterministic,
)
from maskscomp.lm_entropy import read_mask_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arithmetic decode mask bitstream")
    p.add_argument("--bin", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-png", type=Path, required=True)
    p.add_argument("--verify-against", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(0)

    header, payload = read_bin(args.bin)
    use_2d = bool(header["use_2d_context"])
    if header.get("mode") == "online" and args.device != "cpu":
        raise ValueError(
            "Online decode must run on CPU for deterministic synchronization"
        )

    _, model = load_checkpoint_and_model(
        args.checkpoint, use_2d, torch.device(args.device)
    )

    if header.get("checkpoint_sha256") is None:
        raise ValueError("Missing checkpoint hash in header")
    if header["checkpoint_sha256"] != checkpoint_sha(args.checkpoint):
        raise ValueError("Checkpoint hash does not match encoded file")

    mask = decode_mask(payload, header, model, use_2d, torch.device(args.device))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out_png), mask):
        raise RuntimeError(f"Failed to write {args.out_png}")

    if args.verify_against is not None:
        gt = read_mask_png(args.verify_against)
        if gt is None:
            raise ValueError("Invalid verify-against mask")
        if (
            gt.shape != mask.shape
            or gt.dtype != mask.dtype
            or not np.array_equal(gt, mask)
        ):
            raise RuntimeError("Decoded mask mismatch against verify-against")


if __name__ == "__main__":
    main()
