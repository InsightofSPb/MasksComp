from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


def parse_class_list(path: Path) -> dict[tuple[int, int, int], int]:
    """Parse A2D2 class_list.json and return RGB -> contiguous class id mapping."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in class list at {path}, got {type(data)!r}")

    rgb2id: dict[tuple[int, int, int], int] = {}
    for idx, (raw_rgb, raw_name) in enumerate(data.items()):
        if isinstance(raw_name, dict):
            name = raw_name.get("name", str(raw_name))
        else:
            name = raw_name if isinstance(raw_name, str) else str(raw_name)

        if not isinstance(raw_rgb, str) or not raw_rgb.startswith("#") or len(raw_rgb) != 7:
            raise ValueError(
                f"Invalid color key {raw_rgb!r} for class {name!r} in {path}; expected '#RRGGBB'"
            )

        try:
            rgb = tuple(int(raw_rgb[i : i + 2], 16) for i in (1, 3, 5))
        except ValueError as exc:
            raise ValueError(f"Invalid hex color {raw_rgb!r} for class {name!r} in {path}") from exc

        rgb_tuple = (rgb[0], rgb[1], rgb[2])
        if rgb_tuple in rgb2id:
            raise ValueError(f"Duplicate RGB color {raw_rgb} in {path}")
        rgb2id[rgb_tuple] = idx

    return rgb2id


def _normalize_rgb_array(mask_rgb: np.ndarray) -> np.ndarray:
    mask_rgb = np.asarray(mask_rgb)
    if mask_rgb.ndim == 2:
        raise ValueError(
            "Expected RGB(A) mask array with shape (H, W, 3|4). "
            "Got 2D array; paletted images should be converted to RGB with PIL first."
        )
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] not in (3, 4):
        raise ValueError(f"Expected mask shape (H, W, 3|4), got {mask_rgb.shape}")

    if mask_rgb.shape[2] == 4:
        mask_rgb = mask_rgb[:, :, :3]
    if mask_rgb.dtype != np.uint8:
        mask_rgb = mask_rgb.astype(np.uint8, copy=False)
    return mask_rgb


def rgb_mask_to_id(
    mask_rgb: np.ndarray,
    rgb2id: Mapping[tuple[int, int, int], int],
    unknown_id: int | None = None,
    max_unknown_report: int = 10,
) -> np.ndarray:
    """Convert RGB semantic mask to 2D class-id mask.

    When ``unknown_id`` is None, unknown colors raise ValueError with a compact report.
    Otherwise unknown colors are mapped to ``unknown_id``.
    """
    rgb = _normalize_rgb_array(mask_rgb)
    h, w = rgb.shape[:2]

    n_classes = len(rgb2id) + (1 if unknown_id is not None else 0)
    dtype = np.uint8 if n_classes <= 256 else np.uint16
    out = np.full((h, w), unknown_id if unknown_id is not None else 0, dtype=dtype)

    unknown_counts: dict[tuple[int, int, int], int] = {}

    flat_rgb = rgb.reshape(-1, 3)
    flat_out = out.reshape(-1)
    for i, pix in enumerate(flat_rgb):
        key = (int(pix[0]), int(pix[1]), int(pix[2]))
        class_id = rgb2id.get(key)
        if class_id is None:
            if unknown_id is None:
                unknown_counts[key] = unknown_counts.get(key, 0) + 1
            else:
                flat_out[i] = int(unknown_id)
        else:
            flat_out[i] = int(class_id)

    if unknown_counts and unknown_id is None:
        total_unknown = sum(unknown_counts.values())
        top = sorted(unknown_counts.items(), key=lambda x: x[1], reverse=True)[:max_unknown_report]
        top_msg = ", ".join(f"{rgb}:{count}" for rgb, count in top)
        raise ValueError(
            f"Found {len(unknown_counts)} unknown RGB colors ({total_unknown} pixels). "
            f"Examples: {top_msg}."
        )

    return out
