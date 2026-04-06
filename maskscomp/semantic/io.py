from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

from .types import SemanticSample

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def make_sample_id(path: Path, rel_hint: str | None = None) -> str:
    """Build a stable sample id from relative path + hash suffix."""
    key = rel_hint or path.as_posix()
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    stem = path.stem.replace(" ", "_")
    return f"{stem}__{digest}"


def discover_images(input_root: Path) -> list[Path]:
    """Recursively discover image files under input_root."""
    images: list[Path] = []
    for p in sorted(input_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def _read_manifest_rows(manifest_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames or []
        return fieldnames, [dict(r) for r in rd]


def load_samples(
    input_root: Path,
    manifest_csv: Path | None = None,
    image_col: str = "image_path",
    limit: int | None = None,
) -> list[SemanticSample]:
    """Load samples either from scan mode or manifest mode."""
    out: list[SemanticSample] = []
    if manifest_csv is None:
        for p in discover_images(input_root):
            rel = p.relative_to(input_root).as_posix()
            out.append(SemanticSample(sample_id=make_sample_id(p, rel), image_path=p, meta={"image_path": rel}))
            if limit is not None and len(out) >= limit:
                break
        return out

    fieldnames, rows = _read_manifest_rows(manifest_csv)
    if image_col not in fieldnames:
        raise ValueError(f"Column '{image_col}' is missing in {manifest_csv}; columns={fieldnames}")

    for row in rows:
        raw_path = row.get(image_col, "")
        if not raw_path:
            continue
        p = Path(raw_path)
        if not p.is_absolute():
            p = (input_root / p).resolve()
        sample_id = row.get("sample_id") or make_sample_id(p, row.get(image_col))
        row_meta = {k: v for k, v in row.items() if k != image_col}
        row_meta["image_path"] = raw_path
        out.append(SemanticSample(sample_id=str(sample_id), image_path=p, meta=row_meta))
        if limit is not None and len(out) >= limit:
            break
    return out


def write_index_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    """Write semantic artifact index table."""
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "sample_id",
        "image_path",
        "mask_path",
        "probs_path",
        "features_path",
        "overlay_path",
        "status",
        "height",
        "width",
        "feature_h",
        "feature_w",
        "feature_channels",
    ]
    extra_fields: list[str] = []
    for row in rows_list:
        for k in row.keys():
            if k not in base_fields and k not in extra_fields:
                extra_fields.append(str(k))
    fields = base_fields + extra_fields

    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows_list:
            wr.writerow(row)
