from __future__ import annotations

import csv
from pathlib import Path


COMMON_PREV_COLS = ["prev_sample_id", "prev_id", "sample_id_prev", "prev_sample", "prev_path", "image_prev"]
COMMON_CURR_COLS = ["curr_sample_id", "cur_sample_id", "curr_id", "sample_id_curr", "cur_path", "image_curr"]
COMMON_PAIR_COLS = ["pair_id", "id", "pair"]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _pick_col(fieldnames: list[str], candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in fieldnames:
            return c
    raise ValueError(f"Could not infer {label} column. Provide one of: {candidates}; found={fieldnames}")


def infer_pair_columns(fieldnames: list[str]) -> tuple[str, str, str]:
    pair_col = _pick_col(fieldnames, COMMON_PAIR_COLS, "pair_id")
    prev_col = _pick_col(fieldnames, COMMON_PREV_COLS, "prev sample")
    curr_col = _pick_col(fieldnames, COMMON_CURR_COLS, "curr sample")
    return pair_col, prev_col, curr_col


def load_semantic_index(path: Path) -> dict[str, dict[str, str]]:
    rows = load_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            out[sid] = row
    return out


def resolve_semantic_row(index: dict[str, dict[str, str]], key: str) -> dict[str, str]:
    key_norm = str(key).strip()
    if key_norm in index:
        return index[key_norm]

    # Path fallback resolution: compare to image_path basename/stem.
    key_path = Path(key_norm)
    for row in index.values():
        ip = Path(str(row.get("image_path", "")))
        if ip == key_path or ip.name == key_path.name or ip.stem == key_path.stem:
            return row
    raise KeyError(f"Could not resolve semantic artifacts for key={key}")
