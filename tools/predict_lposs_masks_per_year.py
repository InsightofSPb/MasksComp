#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict per-year semantic masks with LPOSS official inference helpers.

- Enumerates GT masks under --gt-masks-dir
- Resolves RGB images under --images-dir (same relpath+stem with ext fallback)
- Runs LPOSS inference (lposs_predict_map expects BGR)
- Saves predicted class-index PNG masks mirroring GT relative paths into --out-dir
- Writes manifest CSV: out_dir/pred_masks_manifest.csv

Fixes:
- Guard against seg_model being a dict (common when someone returns cfg/ckpt by mistake).
- Recovery path: build_model(cfg, class_names) + load_checkpoint(model, checkpoint_path).
- Fail-fast/debug options for quick 1-2 image runs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

MASK_EXTS = {".png", ".tif", ".tiff"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
ARTIFACT_DIR_TOKENS = ("/overlays/", "/diff_maps/", "/warped_masks/", "/spx/")
AUG_PREFIX = "aug_"


@dataclass
class ManifestRow:
    key_relpath: str
    gt_mask_path: str
    image_path: str
    pred_mask_path: str
    status: str
    error: str


def iter_gt_masks(gt_masks_dir: Path) -> Iterator[Path]:
    for p in sorted(gt_masks_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            yield p


def is_candidate_rgb(path: Path) -> bool:
    p = path.as_posix().lower()
    if any(tok in p for tok in ARTIFACT_DIR_TOKENS):
        return False
    if path.name.lower().startswith(AUG_PREFIX):
        return False
    return path.suffix.lower() in IMAGE_EXTS


def resolve_rgb_image(gt_mask_path: Path, gt_masks_dir: Path, images_dir: Path) -> Optional[Path]:
    rel = gt_mask_path.relative_to(gt_masks_dir)
    rel_dir = rel.parent
    stem = gt_mask_path.stem

    # Primary: same rel_dir + same stem, extension fallback
    for ext in IMAGE_EXTS:
        cand = images_dir / rel_dir / f"{stem}{ext}"
        if cand.exists() and cand.is_file() and is_candidate_rgb(cand):
            return cand

    # Fallback: search inside facade folder (first component of relpath)
    parts = rel.parts
    if parts:
        facade_id = parts[0]
        facade_root = images_dir / facade_id
        if facade_root.exists():
            for cand in sorted(facade_root.rglob(f"{stem}.*")):
                if cand.is_file() and is_candidate_rgb(cand):
                    return cand

    return None


def pred_out_path(gt_mask_path: Path, gt_masks_dir: Path, out_dir: Path) -> Path:
    rel = gt_mask_path.relative_to(gt_masks_dir)
    return (out_dir / rel).with_suffix(".png")


def save_pred_mask(pred: np.ndarray, out_path: Path, num_classes_hint: int) -> None:
    max_class = int(np.max(pred)) if pred.size else 0
    num_classes = max(int(num_classes_hint), max_class + 1)
    arr = pred.astype(np.uint8 if num_classes <= 255 else np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), arr)
    if not ok:
        raise RuntimeError(f"Failed to write png: {out_path}")


def write_manifest(rows: List[ManifestRow], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "pred_masks_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key_relpath", "gt_mask_path", "image_path", "pred_mask_path", "status", "error"])
        for r in rows:
            w.writerow([r.key_relpath, r.gt_mask_path, r.image_path, r.pred_mask_path, r.status, r.error])
    return manifest_path


class LPOSSRunner:
    """
    Loads /home/sasha/LPOSS/tools/lposs_inference.py dynamically and builds a real seg_model.
    Uses:
      - build_lposs_inferencer(config_path, checkpoint_path, dataset_config, device)
      - load_checkpoint(model, checkpoint_path)
      - build_model(config, class_names)
      - lposs_predict_map(image_bgr, seg_model, patch_size)
    """

    def __init__(
        self,
        repo_root: Path,
        inference_path: Path,
        config_path: Path,
        checkpoint_path: Path,
        dataset_config: Path,
        device: str,
        debug: bool = False,
    ) -> None:
        self.repo_root = repo_root
        self.inference_path = inference_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dataset_config = dataset_config
        self.device = device
        self.debug = debug

        self.seg_model: Optional[torch.nn.Module] = None
        self.class_names: List[str] = []
        self.palette = None
        self.patch_size: int = 0

        self._load_module_and_build()

    def _load_module_and_build(self) -> None:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"LPOSS repo root not found: {self.repo_root}")
        if not self.inference_path.exists():
            raise FileNotFoundError(f"LPOSS inference helper not found: {self.inference_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"LPOSS config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"LPOSS checkpoint not found: {self.checkpoint_path}")
        if not self.dataset_config.exists():
            raise FileNotFoundError(f"LPOSS dataset config not found: {self.dataset_config}")

        sys.path.insert(0, str(self.repo_root))

        spec = importlib.util.spec_from_file_location("lposs_inference_dynamic", self.inference_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import {self.inference_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        build_lposs_inferencer = getattr(module, "build_lposs_inferencer", None)
        lposs_predict_map = getattr(module, "lposs_predict_map", None)
        load_checkpoint = getattr(module, "load_checkpoint", None)
        build_model = getattr(module, "build_model", None)

        if not callable(build_lposs_inferencer) or not callable(lposs_predict_map):
            raise RuntimeError("lposs_inference.py must expose build_lposs_inferencer and lposs_predict_map")

        if not callable(load_checkpoint):
            raise RuntimeError("lposs_inference.py must expose load_checkpoint(model, checkpoint_path)")

        # Keep references
        self._module = module
        self._lposs_predict_map = lposs_predict_map
        self._load_checkpoint = load_checkpoint
        self._build_model = build_model if callable(build_model) else None

        # 1) Try official builder (correct signature)
        try:
            seg_model, class_names, palette, patch_size = build_lposs_inferencer(
                config_path=Path(self.config_path),
                checkpoint_path=Path(self.checkpoint_path),
                dataset_config=Path(self.dataset_config),
                device=str(self.device),
            )
        except TypeError:
            seg_model, class_names, palette, patch_size = build_lposs_inferencer(
                Path(self.config_path),
                Path(self.checkpoint_path),
                Path(self.dataset_config),
                str(self.device),
            )

        self.class_names = list(class_names) if isinstance(class_names, list) else []
        self.palette = palette
        self.patch_size = int(patch_size) if patch_size is not None else 0

        # 2) Validate seg_model; if it's not a Module, try to recover using build_model+load_checkpoint
        if isinstance(seg_model, torch.nn.Module):
            self.seg_model = seg_model
        else:
            # Recovery: treat seg_model as "config-like" (dict/DictConfig/etc.)
            if self._build_model is None:
                raise RuntimeError(
                    f"build_lposs_inferencer returned non-Module type={type(seg_model)} "
                    "and build_model is unavailable for recovery."
                )

            cfg_like = seg_model
            if self.debug:
                print(f"[debug] build_lposs_inferencer returned {type(seg_model)}; trying recovery via build_model+load_checkpoint")

            model = self._build_model(cfg_like, self.class_names)
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError(f"build_model returned non-Module type={type(model)}")

            # load checkpoint weights (knows about 'model_state')
            self._load_checkpoint(model, Path(self.checkpoint_path))
            self.seg_model = model

        # 3) Finalize model
        assert self.seg_model is not None
        self.seg_model.to(self.device)
        self.seg_model.eval()

        if self.patch_size <= 0:
            # patch_size is required by lposs_predict_map; give a sane fallback
            self.patch_size = 512

        if self.debug:
            print(f"[debug] seg_model type={type(self.seg_model)} callable={callable(self.seg_model)} patch_size={self.patch_size} n_classes={len(self.class_names)}")

    @property
    def num_classes(self) -> int:
        return len(self.class_names) if self.class_names else 1

    def predict(self, bgr_image: np.ndarray) -> np.ndarray:
        pred = self._lposs_predict_map(
            image_bgr=bgr_image,
            seg_model=self.seg_model,
            patch_size=self.patch_size,
        )

        pred_arr = np.asarray(pred)

        # If lposs_predict_map returns logits/probs (H,W,C or C,H,W), convert to class map
        if pred_arr.ndim == 3:
            # HWC
            if pred_arr.shape[2] <= 4096 and pred_arr.shape[0] == bgr_image.shape[0] and pred_arr.shape[1] == bgr_image.shape[1]:
                pred_arr = np.argmax(pred_arr, axis=2)
            # CHW
            elif pred_arr.shape[0] <= 4096:
                pred_arr = np.argmax(pred_arr, axis=0)
            else:
                raise RuntimeError(f"Unexpected 3D output shape: {pred_arr.shape}")

        if pred_arr.ndim != 2:
            raise RuntimeError(f"Expected [H,W] prediction map, got shape {pred_arr.shape}")

        return pred_arr.astype(np.int32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-masks-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--lposs-repo-root", type=Path, required=True)
    p.add_argument("--lposs-inference-path", type=Path, required=True)
    p.add_argument("--lposs-config", type=Path, required=True)
    p.add_argument("--lposs-checkpoint", type=Path, required=True)
    p.add_argument("--lposs-dataset-config", type=Path, required=True)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-items", type=int, default=None)

    p.add_argument("--debug", action="store_true")
    p.add_argument("--fail-fast", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    gt_masks = list(iter_gt_masks(args.gt_masks_dir))
    if args.max_items is not None:
        gt_masks = gt_masks[: args.max_items]

    if not gt_masks:
        manifest = write_manifest([], args.out_dir)
        print(f"No GT masks found. manifest={manifest}")
        return 0

    runner: Optional[LPOSSRunner] = None
    if not args.dry_run:
        runner = LPOSSRunner(
            repo_root=args.lposs_repo_root,
            inference_path=args.lposs_inference_path,
            config_path=args.lposs_config,
            checkpoint_path=args.lposs_checkpoint,
            dataset_config=args.lposs_dataset_config,
            device=args.device,
            debug=args.debug,
        )

    rows: List[ManifestRow] = []
    for gt_path in tqdm(gt_masks, desc="Processing", unit="mask"):
        key_relpath = gt_path.relative_to(args.gt_masks_dir).as_posix()
        out_path = pred_out_path(gt_path, args.gt_masks_dir, args.out_dir)
        img_path = resolve_rgb_image(gt_path, args.gt_masks_dir, args.images_dir)

        if img_path is None:
            rows.append(ManifestRow(key_relpath, str(gt_path), "", str(out_path), "missing_image", "No matching RGB image found"))
            continue

        # Guard: never use GT mask as image
        if img_path.resolve() == gt_path.resolve():
            rows.append(ManifestRow(key_relpath, str(gt_path), str(img_path), str(out_path), "missing_image", "Resolved image equals GT mask (guarded)"))
            continue

        if args.dry_run:
            print(f"[DRY] {gt_path} -> {img_path} -> {out_path}")
            rows.append(ManifestRow(key_relpath, str(gt_path), str(img_path), str(out_path), "ok", ""))
            continue

        assert runner is not None
        try:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cv2.imread failed: {img_path}")

            pred = runner.predict(bgr)
            save_pred_mask(pred, out_path, runner.num_classes)

            rows.append(ManifestRow(key_relpath, str(gt_path), str(img_path), str(out_path), "ok", ""))
        except Exception as exc:
            err = str(exc)
            if args.debug:
                err = traceback.format_exc()
            rows.append(ManifestRow(key_relpath, str(gt_path), str(img_path), str(out_path), "failed", err))
            if args.fail_fast:
                manifest = write_manifest(rows, args.out_dir)
                print(f"FAIL-FAST. manifest={manifest}")
                raise

    manifest = write_manifest(rows, args.out_dir)
    counts = {"ok": 0, "missing_image": 0, "failed": 0}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1

    print(
        "Done. "
        f"total={len(rows)} ok={counts['ok']} missing_image={counts['missing_image']} "
        f"failed={counts['failed']} manifest={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())