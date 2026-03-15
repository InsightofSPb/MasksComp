from __future__ import annotations

import csv
import fnmatch
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from maskscomp.models.msdzip import MixedModel


@dataclass(frozen=True)
class TemporalPairRecord:
    pair_id: str
    sample_id: str
    facade_id: str
    prev_path: str
    cur_path: str
    year_a: str = ""
    year_b: str = ""
    split: str = "train"
    delta_years: str = ""
    homography_path: str = ""
    valid_mask_path: str = ""
    lposs_prev_path: str = ""
    lposs_cur_path: str = ""
    superpixel_labels_path: str = ""


def _guess_split(csv_name: str) -> str:
    n = csv_name.lower()
    if "val" in n:
        return "val"
    if "test" in n:
        return "test"
    return "train"


def _basename_no_ext(path_like: str) -> str:
    return Path(str(path_like)).stem


def read_pairs_csv(path: Path) -> List[TemporalPairRecord]:
    rows: List[TemporalPairRecord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = set(rd.fieldnames or [])

        has_new = {"pair_id", "sample_id", "prev_path", "cur_path"}.issubset(cols)
        has_existing = {"pair_id", "facade_id", "year_a", "year_b", "mask_a", "mask_b"}.issubset(cols)

        if not (has_new or has_existing):
            raise ValueError(
                f"Unsupported pair CSV format in {path}. Expected either "
                "(pair_id,sample_id,prev_path,cur_path[,split...]) or "
                "(pair_id,facade_id,year_a,year_b,mask_a,mask_b[,delta_years])."
            )

        default_split = _guess_split(path.name)
        for r in rd:
            if has_existing:
                facade_id = str(r["facade_id"])
                year_a = str(r.get("year_a", "") or "")
                year_b = str(r.get("year_b", "") or "")
                rows.append(
                    TemporalPairRecord(
                        pair_id=str(r["pair_id"]),
                        sample_id=facade_id,
                        facade_id=facade_id,
                        prev_path=str(r["mask_a"]),
                        cur_path=str(r["mask_b"]),
                        year_a=year_a,
                        year_b=year_b,
                        split=str(r.get("split", default_split) or default_split),
                        delta_years=str(r.get("delta_years", "") or ""),
                    )
                )
            else:
                sample_id = str(r["sample_id"])
                rows.append(
                    TemporalPairRecord(
                        pair_id=str(r["pair_id"]),
                        sample_id=sample_id,
                        facade_id=str(r.get("facade_id", sample_id) or sample_id),
                        prev_path=str(r["prev_path"]),
                        cur_path=str(r["cur_path"]),
                        year_a=str(r.get("year_a", "") or ""),
                        year_b=str(r.get("year_b", "") or ""),
                        split=str(r.get("split", default_split) or default_split),
                        delta_years=str(r.get("delta_years", "") or ""),
                        homography_path=str(r.get("homography_path", "") or ""),
                        valid_mask_path=str(r.get("valid_mask_path", "") or ""),
                        lposs_prev_path=str(r.get("lposs_prev_path", "") or ""),
                        lposs_cur_path=str(r.get("lposs_cur_path", "") or ""),
                        superpixel_labels_path=str(r.get("superpixel_labels_path", "") or ""),
                    )
                )
    return rows


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def read_homography(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        h = np.load(path)
    elif path.suffix.lower() == ".npz":
        z = np.load(path)
        key = "H" if "H" in z else list(z.keys())[0]
        h = z[key]
    else:
        h = np.loadtxt(path, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {h.shape} from {path}")
    return h


def align_prev_to_cur(prev_rgb: np.ndarray, cur_shape: Tuple[int, int], homography: Optional[np.ndarray]) -> np.ndarray:
    h, w = cur_shape
    if homography is None:
        if prev_rgb.shape[:2] != (h, w):
            raise ValueError("prev and cur must match in size if no homography is provided")
        return prev_rgb
    warped = cv2.warpPerspective(
        prev_rgb,
        homography,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def _path_candidates(root: Path, patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).lower()
        for pat in patterns:
            if fnmatch.fnmatch(rel, pat.lower()):
                out.append(p)
                break
    return out


def _score_asset(path: Path, year_a: str, year_b: str, hint_prev: str, hint_cur: str, target: str) -> int:
    n = str(path).lower()
    s = 0
    if year_a and year_a in n:
        s += 2
    if year_b and year_b in n:
        s += 2
    if target == "prev_aligned":
        for k in ["align", "aligned", "warp", "warped", "to_", "to"]:
            if k in n:
                s += 3
        if hint_prev and hint_prev in n:
            s += 3
        if hint_cur and hint_cur in n:
            s += 2
    elif target == "cur":
        for k in ["cur", "current", "ref", "target", "year_b", "b"]:
            if k in n:
                s += 2
        if hint_cur and hint_cur in n:
            s += 4
    return s


def resolve_prealigned_assets(prealigned_root: Path, rec: TemporalPairRecord) -> Dict[str, str]:
    facade_dir = prealigned_root / rec.facade_id
    if not facade_dir.exists():
        raise FileNotFoundError(
            f"Prealigned facade directory not found for pair_id={rec.pair_id} facade_id={rec.facade_id}: {facade_dir}"
        )

    img_ext = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp"]
    files = [p for p in facade_dir.rglob("*") if p.is_file()]
    imgs = [p for p in files if any(fnmatch.fnmatch(p.name.lower(), e) for e in img_ext)]

    ya = str(rec.year_a)
    yb = str(rec.year_b)
    hint_prev = _basename_no_ext(rec.prev_path).lower()
    hint_cur = _basename_no_ext(rec.cur_path).lower()

    prev_pool = [
        p for p in imgs if (ya in str(p) if ya else True) and (yb in str(p) if yb else True) and any(k in str(p).lower() for k in ["align", "warp", "to", "aligned"])
    ]
    cur_pool = [p for p in imgs if (yb in str(p) if yb else True)]
    if not prev_pool:
        prev_pool = imgs
    if not cur_pool:
        cur_pool = imgs

    prev_ranked = sorted(prev_pool, key=lambda p: _score_asset(p, ya, yb, hint_prev, hint_cur, "prev_aligned"), reverse=True)
    cur_ranked = sorted(cur_pool, key=lambda p: _score_asset(p, ya, yb, hint_prev, hint_cur, "cur"), reverse=True)

    if not prev_ranked or not cur_ranked:
        searched = [str(p.relative_to(facade_dir)) for p in files[:200]]
        raise RuntimeError(
            "Could not resolve prealigned pair assets. "
            f"pair_id={rec.pair_id} facade_id={rec.facade_id} year_a={rec.year_a} year_b={rec.year_b} "
            f"searched_under={facade_dir} files_sample={searched}"
        )

    prev_aligned = prev_ranked[0]
    cur_img = cur_ranked[0]

    valid_mask = None
    for p in imgs:
        n = str(p).lower()
        if any(k in n for k in ["valid", "roi", "facade_mask", "fg_mask", "mask_valid"]):
            valid_mask = p
            break

    superpixel = None
    for p in files:
        n = str(p).lower()
        if any(k in n for k in ["superpixel", "spx", ".labels.npz", "labels.npz"]):
            superpixel = p
            break

    overlay = None
    for p in imgs:
        n = str(p).lower()
        if any(k in n for k in ["overlay", "viz", "visual", "blend"]):
            overlay = p
            break

    return {
        "facade_dir": str(facade_dir),
        "prev_aligned_rgb": str(prev_aligned),
        "cur_rgb": str(cur_img),
        "valid_mask": str(valid_mask) if valid_mask else "",
        "superpixel_labels": str(superpixel) if superpixel else "",
        "overlay_viz": str(overlay) if overlay else "",
    }


def read_geom_metadata(geom_root: Optional[Path], rec: TemporalPairRecord) -> Dict[str, object]:
    fields = {
        "num_matches": "",
        "num_inliers": "",
        "inlier_ratio": "",
        "iou_fg": "",
        "iou_edge": "",
        "status_quality": "",
        "geom_path": "",
    }
    if geom_root is None:
        return fields
    gp = geom_root / rec.facade_id / f"{rec.year_a}_{rec.year_b}.json"
    if not gp.exists():
        return fields
    data = json.loads(gp.read_text(encoding="utf-8"))
    fields["num_matches"] = data.get("num_matches", "")
    fields["num_inliers"] = data.get("num_inliers", "")
    fields["inlier_ratio"] = data.get("inlier_ratio", "")
    fields["iou_fg"] = data.get("iou_fg", "")
    fields["iou_edge"] = data.get("iou_edge", "")
    fields["status_quality"] = data.get("status_quality", "")
    fields["geom_path"] = str(gp)
    return fields


def modular_residual(cur_rgb: np.ndarray, aligned_prev_rgb: np.ndarray) -> np.ndarray:
    return ((cur_rgb.astype(np.int16) - aligned_prev_rgb.astype(np.int16)) % 256).astype(np.uint8)


def signed_residual_zigzag(cur_rgb: np.ndarray, aligned_prev_rgb: np.ndarray) -> np.ndarray:
    d = cur_rgb.astype(np.int16) - aligned_prev_rgb.astype(np.int16)
    zz = np.where(d >= 0, 2 * d, -2 * d - 1)
    return zz.astype(np.uint16)


def serialize_tile(tile: np.ndarray, mode: str = "interleaved") -> np.ndarray:
    if tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 tile, got {tile.shape}")
    if mode == "interleaved":
        return tile.reshape(-1)
    if mode == "planar":
        return np.concatenate([tile[..., 0].reshape(-1), tile[..., 1].reshape(-1), tile[..., 2].reshape(-1)], axis=0)
    raise ValueError(f"Unsupported serialize mode: {mode}")


def iter_tiles(
    arr: np.ndarray,
    tile_size: int,
    stride: int,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_fraction: float = 0.0,
) -> Iterator[Tuple[int, int, np.ndarray, float]]:
    h, w = arr.shape[:2]
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = arr[y : y + tile_size, x : x + tile_size]
            frac = 1.0
            if valid_mask is not None:
                m = valid_mask[y : y + tile_size, x : x + tile_size]
                frac = float(np.mean(m > 0))
            if frac >= float(min_valid_fraction):
                yield y, x, tile, frac


def png_size_bytes(tile_rgb: np.ndarray, compress_level: int = 6) -> int:
    bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(compress_level)])
    if not ok:
        raise RuntimeError("Failed to PNG-encode tile")
    return int(enc.size)


class ByteMSDZipLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        timesteps: int = 64,
        vocab_dim: int = 32,
        hidden_dim: int = 128,
        ffn_dim: int = 256,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.timesteps = int(timesteps)
        self.token_embed = nn.Embedding(self.vocab_size, vocab_dim)
        self.in_proj = nn.Linear(vocab_dim, hidden_dim)
        self.backbone = MixedModel(
            timesteps=int(timesteps),
            hidden_dim=int(hidden_dim),
            ffn_dim=int(ffn_dim),
            layers=int(layers),
            dropout=float(dropout),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(self.token_embed(x))
        z = self.backbone(self.dropout(h))
        return self.head(z)


@torch.no_grad()
def compute_sequence_bits(model: nn.Module, seq: np.ndarray, device: torch.device, timesteps: int, batch_size: int = 4096) -> float:
    if seq.size <= 1:
        return 0.0
    seq_i = np.asarray(seq, dtype=np.int64)
    contexts: List[np.ndarray] = []
    targets: List[int] = []
    for t in range(1, seq_i.size):
        start = max(0, t - timesteps)
        ctx = np.zeros((timesteps,), dtype=np.int64)
        window = seq_i[start:t]
        ctx[-window.size :] = window
        contexts.append(ctx)
        targets.append(int(seq_i[t]))
    total_bits = 0.0
    for i in range(0, len(contexts), int(batch_size)):
        xb = torch.as_tensor(np.stack(contexts[i : i + batch_size]), dtype=torch.long, device=device)
        yb = torch.as_tensor(np.asarray(targets[i : i + batch_size], dtype=np.int64), dtype=torch.long, device=device)
        logits = model(xb)[:, -1, :]
        logp = F.log_softmax(logits, dim=-1)
        nll_nat = -logp.gather(-1, yb.unsqueeze(-1)).squeeze(-1)
        total_bits += float((nll_nat / math.log(2.0)).sum().item())
    return total_bits


def score_to_heatmap(tile_rows: Sequence[Dict[str, float]], h: int, w: int, tile_size: int) -> np.ndarray:
    score = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for r in tile_rows:
        y = int(r["tile_y"])
        x = int(r["tile_x"])
        v = float(r["lm_bits"])
        score[y : y + tile_size, x : x + tile_size] += v
        count[y : y + tile_size, x : x + tile_size] += 1.0
    out = np.zeros_like(score)
    np.divide(score, np.maximum(1.0, count), out=out)
    return out


def overlay_heatmap_on_rgb(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h01 = heatmap.copy().astype(np.float32)
    if np.isfinite(h01).any():
        lo, hi = np.percentile(h01[np.isfinite(h01)], [5.0, 95.0])
        if hi > lo:
            h01 = np.clip((h01 - lo) / (hi - lo), 0.0, 1.0)
        else:
            h01 = np.zeros_like(h01)
    else:
        h01 = np.zeros_like(h01)
    hm_u8 = (h01 * 255).astype(np.uint8)
    cm = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    out = (rgb.astype(np.float32) * (1.0 - alpha) + cm.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out
