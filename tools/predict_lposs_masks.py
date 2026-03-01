"""Predict LPOSS segmentation masks for facade samples.

This script mirrors `<facade>/<template-subdir>/*.png` filenames into
`<facade>/<out-subdir>/` while running LPOSS/MaskCLIP inference on the
corresponding RGB image.

Notes:
- The script is designed to work with a separate LPOSS repo (default: /home/sasha/LPOSS)
  without installation by inserting that repo into ``sys.path`` at runtime.
- It prefers using helper functions from ``tools/lposs_inference.py`` in the LPOSS repo
  (``build_lposs_inferencer``, ``lposs_predict_map``, ``load_checkpoint``).
- If helpers are unavailable, the script raises a clear error describing what to provide.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    np = None  # type: ignore[assignment]
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    def tqdm(iterable, **_kwargs):
        return iterable


DEFAULT_LPOSS_REPO = Path("/home/sasha/LPOSS")
DEFAULT_CKPT = (
    "/home/sasha/LPOSS/outputs/finetune_20260223-193147_lr5e-05_depth-1_bs16/"
    "epoch_0035_val_loss_0.9876.pth"
)


@dataclass(frozen=True)
class Sample:
    facade_id: str
    template_mask_path: Path


@dataclass(frozen=True)
class ResolvedSample:
    sample: Sample
    image_path: Path
    pred_mask_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict LPOSS masks and export PNG class maps")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--template-subdir", type=str, default="warped_masks")
    p.add_argument("--images-subdir", type=str, default="AUTO")
    p.add_argument("--out-subdir", type=str, default="pred_masks")
    p.add_argument("--out-manifest", type=Path, default=None)

    p.add_argument("--lposs-repo-root", type=Path, default=DEFAULT_LPOSS_REPO)
    p.add_argument(
        "--lposs-inference-path",
        type=Path,
        default=None,
        help="Optional explicit path to LPOSS tools/lposs_inference.py",
    )
    p.add_argument("--lposs-config", type=Path, default=Path("/home/sasha/LPOSS/configs/lposs.yaml"))
    p.add_argument("--lposs-checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    p.add_argument("--lposs-dataset-config", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0")

    p.add_argument("--use-tiling", action="store_true")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--tile-overlap", type=int, default=0)

    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--save-probs-npy", action="store_true")

    p.add_argument("--mask-to-image-regex", type=str, default=None)
    p.add_argument("--mask-to-image-repl", type=str, default="")
    p.add_argument(
        "--image-ext-priority",
        type=str,
        default=".jpg,.jpeg,.png,.tif,.tiff",
        help="Comma-separated list of extension priority",
    )

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-items", type=int, default=None)

    return p.parse_args()


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    """Call function with kwargs filtered to explicit params unless **kwargs is present."""
    sig = inspect.signature(fn)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**accepted)


def _build_kwargs_from_signature(fn: Any, alias_map: dict[str, Any]) -> dict[str, Any]:
    """Construct kwargs for callable based on its named parameters and aliases."""
    sig = inspect.signature(fn)
    out: dict[str, Any] = {}
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if p.name in alias_map:
            out[p.name] = alias_map[p.name]
        elif p.default is inspect._empty:
            raise RuntimeError(f"Missing required parameter '{p.name}' for call")
    return out


def _invoke_build_inferencer(
    build_fn: Any,
    config_path: Path,
    checkpoint_path: Path,
    dataset_config: Path,
    device: str,
) -> Any:
    """Call build_lposs_inferencer across known signature variants."""
    attempts: list[dict[str, Any]] = [
        {
            "lposs_config": config_path,
            "lposs_checkpoint": checkpoint_path,
            "lposs_dataset_config": dataset_config,
            "dataset_config": dataset_config,
            "device": device,
            "torch_device": device,
        },
        {
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "dataset_config_path": dataset_config,
            "dataset_config": dataset_config,
            "device": device,
            "torch_device": device,
        },
        {
            "config": config_path,
            "checkpoint": checkpoint_path,
            "dataset_config": dataset_config,
            "device": device,
            "torch_device": device,
        },
    ]

    errors: list[str] = []
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return _call_with_supported_kwargs(build_fn, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            errors.append(f"kwargs_attempt failed: {exc}")

    # Signature-guided fallback: build kwargs by aliases and call directly.
    alias_map = {
        "config": config_path,
        "config_path": config_path,
        "lposs_config": config_path,
        "checkpoint": checkpoint_path,
        "checkpoint_path": checkpoint_path,
        "lposs_checkpoint": checkpoint_path,
        "dataset_config": dataset_config,
        "dataset_config_path": dataset_config,
        "lposs_dataset_config": dataset_config,
        "device": device,
        "torch_device": device,
    }
    try:
        guessed_kwargs = _build_kwargs_from_signature(build_fn, alias_map)
        return build_fn(**guessed_kwargs)
    except Exception as exc:  # pragma: no cover
        last_error = exc
        errors.append(f"signature_guided failed: {exc}")

    mixed_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((config_path, checkpoint_path, dataset_config), {"device": device}),
        ((config_path, checkpoint_path, dataset_config), {"torch_device": device}),
        ((config_path, checkpoint_path), {"dataset_config": dataset_config, "device": device}),
    ]
    for pargs, pkwargs in mixed_attempts:
        try:
            return build_fn(*pargs, **pkwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            errors.append(f"mixed_attempt failed: args={len(pargs)} kwargs={list(pkwargs.keys())}: {exc}")

    positional_attempts = [
        (config_path, checkpoint_path, dataset_config, device),
    ]
    for pargs in positional_attempts:
        try:
            return build_fn(*pargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            errors.append(f"positional_attempt failed: args={len(pargs)}: {exc}")

    err_tail = " | ".join(errors[-4:]) if errors else str(last_error)
    raise RuntimeError(f"Failed to build LPOSS inferencer: {last_error}; attempts={err_tail}")


def _invoke_predict_map(predict_fn: Any, inferencer: Any, image_bgr: np.ndarray) -> np.ndarray:
    """Call lposs_predict_map across known signature variants."""
    candidates: list[dict[str, Any]] = [
        {"inferencer": inferencer, "image_bgr": image_bgr, "image": image_bgr},
        {"seg_model": inferencer[0], "patch_size": inferencer[3], "image_bgr": image_bgr},
        {"model": inferencer, "image": image_bgr},
    ]
    last_error: Exception | None = None
    for kwargs in candidates:
        try:
            return np.asarray(_call_with_supported_kwargs(predict_fn, **kwargs))
        except Exception as exc:  # pragma: no cover
            last_error = exc

    # positional variants
    pargs = [
        (image_bgr, inferencer),
        (image_bgr, inferencer[0], inferencer[3]) if isinstance(inferencer, tuple) and len(inferencer) >= 4 else None,
    ]
    for args in pargs:
        if args is None:
            continue
        try:
            return np.asarray(predict_fn(*args))
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise RuntimeError(f"Failed to run LPOSS prediction: {last_error}")


def _normalize_probs_chw(probs: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Normalize probability tensor layout to C,H,W."""
    np_local = _get_np()
    if probs.ndim != 3:
        raise ValueError(f"Expected 3D probs tensor, got {probs.shape}")
    h, w = image_hw
    if probs.shape[0] == h and probs.shape[1] == w:
        # H,W,C
        return np_local.moveaxis(probs, -1, 0)
    if probs.shape[1] == h and probs.shape[2] == w:
        # C,H,W
        return probs
    raise ValueError(f"Unexpected probs shape {probs.shape} for image size {(h,w)}")


def _get_np() -> Any:
    """Import/validate numpy lazily so --help can run without runtime deps."""
    if np is None:  # pragma: no cover - environment dependent
        raise RuntimeError("NumPy is required for inference. Install numpy in the active environment.")
    return np


def _get_cv2() -> Any:
    """Import cv2 lazily so argument parsing/help works without OpenCV installed."""
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV (cv2) is required for inference and writing outputs. "
            "Install opencv-python in the active environment."
        ) from exc
    return cv2


def load_lposs_helpers(repo_root: Path, inference_path: Path | None) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(repo_root))
    target = inference_path or (repo_root / "tools" / "lposs_inference.py")
    if not target.is_file():
        raise FileNotFoundError(
            f"LPOSS inference helper not found at: {target}. "
            "Provide --lposs-inference-path or ensure --lposs-repo-root is correct."
        )

    spec = importlib.util.spec_from_file_location("_lposs_inference", str(target))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = [
        name
        for name in ("build_lposs_inferencer", "lposs_predict_map", "load_checkpoint")
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            "LPOSS helper module is missing required symbols: "
            f"{', '.join(missing)} at {target}"
        )
    return module.build_lposs_inferencer, module.lposs_predict_map, module.load_checkpoint


def discover_template_masks(data_root: Path, template_subdir: str) -> list[Sample]:
    samples: list[Sample] = []
    for p in sorted(data_root.glob(f"*/{template_subdir}/*.png")):
        facade_id = p.parent.parent.name
        samples.append(Sample(facade_id=facade_id, template_mask_path=p))
    return samples


def _parse_ext_priority(exts_csv: str) -> list[str]:
    out: list[str] = []
    for raw in exts_csv.split(","):
        ext = raw.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        out.append(ext)
    return out


def _maybe_transform_stem(stem: str, regex: str | None, repl: str) -> str:
    if not regex:
        return stem
    return re.sub(regex, repl, stem)


def _find_candidate_in_dir(base_dir: Path, stem: str, exts: list[str]) -> Path | None:
    for ext in exts:
        c = base_dir / f"{stem}{ext}"
        if c.is_file():
            return c
    return None


def resolve_image_path(
    facade_dir: Path,
    template_mask_path: Path,
    images_subdir: str,
    exts: list[str],
    mask_to_image_regex: str | None,
    mask_to_image_repl: str,
) -> Path | None:
    stem = template_mask_path.stem
    stem2 = _maybe_transform_stem(stem, mask_to_image_regex, mask_to_image_repl)

    if images_subdir != "AUTO":
        d = facade_dir / images_subdir
        if d.is_dir():
            p = _find_candidate_in_dir(d, stem2, exts) or _find_candidate_in_dir(d, stem, exts)
            if p is not None:
                return p

    for fallback in ("images", "warped_images"):
        d = facade_dir / fallback
        if d.is_dir():
            p = _find_candidate_in_dir(d, stem2, exts) or _find_candidate_in_dir(d, stem, exts)
            if p is not None:
                return p

    # Fallback: search under whole facade folder by stem.
    for ext in exts:
        matches = sorted(facade_dir.glob(f"**/{stem2}{ext}"))
        if matches:
            return matches[0]
    for ext in exts:
        matches = sorted(facade_dir.glob(f"**/{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def auto_find_dataset_config(repo_root: Path) -> Path | None:
    candidates = sorted(repo_root.glob("**/*facade*.py")) + sorted(repo_root.glob("**/*facades*.py"))
    for c in candidates:
        try:
            text = c.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if "classes" in text or "palette" in text or "metainfo" in text:
            return c
    return None


def build_inferencer(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    build_fn, pred_fn, _load_ckpt = load_lposs_helpers(args.lposs_repo_root, args.lposs_inference_path)

    dataset_cfg = args.lposs_dataset_config
    if dataset_cfg is None:
        dataset_cfg = auto_find_dataset_config(args.lposs_repo_root)
        if dataset_cfg is None:
            raise ValueError(
                "Could not auto-find --lposs-dataset-config under LPOSS repo. "
                "Please pass --lposs-dataset-config explicitly."
            )
        print(f"[info] Auto-selected dataset config: {dataset_cfg}")

    inferencer = _invoke_build_inferencer(
        build_fn=build_fn,
        config_path=args.lposs_config,
        checkpoint_path=args.lposs_checkpoint,
        dataset_config=dataset_cfg,
        device=args.device,
    )

    meta = {"dataset_config": str(dataset_cfg)}
    return inferencer, pred_fn, meta


def predict_probs(
    image_bgr: np.ndarray,
    inferencer: Any,
    predict_fn: Any,
    use_tiling: bool,
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    if not use_tiling:
        probs = _invoke_predict_map(predict_fn=predict_fn, inferencer=inferencer, image_bgr=image_bgr)
        return _normalize_probs_chw(probs, image_bgr.shape[:2])

    h, w = image_bgr.shape[:2]
    step = max(1, tile_size - tile_overlap)
    probs_acc: np.ndarray | None = None
    weight = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(1, h - tile_size + 1), step))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    for y0 in ys:
        for x0 in xs:
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            tile = image_bgr[y0:y1, x0:x1]
            tile_probs = _invoke_predict_map(predict_fn=predict_fn, inferencer=inferencer, image_bgr=tile)
            tile_probs = _normalize_probs_chw(tile_probs, tile.shape[:2])
            c, th, tw = tile_probs.shape

            if (th, tw) != (y1 - y0, x1 - x0):
                raise ValueError(
                    "Tile prediction shape mismatch: "
                    f"tile {(y1 - y0, x1 - x0)} vs probs {(th, tw)}"
                )

            if probs_acc is None:
                probs_acc = np.zeros((c, h, w), dtype=np.float32)

            probs_acc[:, y0:y1, x0:x1] += tile_probs.astype(np.float32, copy=False)
            weight[y0:y1, x0:x1] += 1.0

    if probs_acc is None:
        raise RuntimeError("No tiles processed")

    weight = np.maximum(weight, 1e-6)
    probs_acc /= weight[None, :, :]
    return probs_acc


def probs_to_mask(probs_chw: np.ndarray) -> np.ndarray:
    np_local = _get_np()
    if probs_chw.ndim != 3:
        raise ValueError(f"Expected C,H,W probs, got {probs_chw.shape}")
    return np_local.argmax(probs_chw, axis=0).astype(np.int64)


def overlay_mask_on_image(image_rgb: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    # deterministic pseudo-palette from label id
    cv2 = _get_cv2()
    np = _get_np()
    lbl = pred_mask.astype(np.int32)
    color = np.zeros((*lbl.shape, 3), dtype=np.uint8)
    color[..., 0] = (lbl * 37) % 255
    color[..., 1] = (lbl * 67) % 255
    color[..., 2] = (lbl * 97) % 255
    return cv2.addWeighted(image_rgb, 0.55, color, 0.45, 0.0)


def save_mask_png(mask: np.ndarray, path: Path, num_classes: int) -> None:
    cv2 = _get_cv2()
    np = _get_np()
    path.parent.mkdir(parents=True, exist_ok=True)
    if num_classes <= 255:
        out = mask.astype(np.uint8)
    else:
        out = mask.astype(np.uint16)
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise RuntimeError(f"Failed to write PNG: {path}")


def main() -> None:
    args = parse_args()
    np = _get_np() if not args.dry_run else None
    cv2 = _get_cv2() if not args.dry_run else None
    exts = _parse_ext_priority(args.image_ext_priority)
    samples = discover_template_masks(args.data_root, args.template_subdir)
    if args.max_items is not None:
        samples = samples[: max(0, args.max_items)]

    if not samples:
        raise SystemExit(
            f"No template masks found under {args.data_root}/*/{args.template_subdir}/*.png"
        )

    out_manifest = args.out_manifest
    if out_manifest is None:
        out_manifest = args.data_root / args.out_subdir / "pred_masks_manifest.csv"

    resolved: list[ResolvedSample] = []
    manifest_rows: list[dict[str, str]] = []

    for s in samples:
        facade_dir = s.template_mask_path.parent.parent
        rel_template = s.template_mask_path.relative_to(args.data_root)
        out_path = facade_dir / args.out_subdir / s.template_mask_path.name
        img_path = resolve_image_path(
            facade_dir,
            s.template_mask_path,
            images_subdir=args.images_subdir,
            exts=exts,
            mask_to_image_regex=args.mask_to_image_regex,
            mask_to_image_repl=args.mask_to_image_repl,
        )
        if img_path is None:
            manifest_rows.append(
                {
                    "facade_id": s.facade_id,
                    "template_mask_relpath": str(rel_template),
                    "image_relpath": "",
                    "pred_mask_relpath": str(out_path.relative_to(args.data_root)),
                    "status": "skipped_missing_image",
                    "error": "image_not_found",
                }
            )
            continue
        resolved.append(ResolvedSample(sample=s, image_path=img_path, pred_mask_path=out_path))
        manifest_rows.append(
            {
                "facade_id": s.facade_id,
                "template_mask_relpath": str(rel_template),
                "image_relpath": str(img_path.relative_to(args.data_root)),
                "pred_mask_relpath": str(out_path.relative_to(args.data_root)),
                "status": "pending",
                "error": "",
            }
        )

    if args.dry_run:
        for r in resolved:
            print(
                f"[dry-run] {r.sample.template_mask_path.relative_to(args.data_root)} "
                f"-> {r.image_path.relative_to(args.data_root)} "
                f"-> {r.pred_mask_path.relative_to(args.data_root)}"
            )
    else:
        inferencer, predict_fn, _meta = build_inferencer(args)
        label_hist: np.ndarray | None = None

        pending_idx = 0
        for i, row in enumerate(tqdm(manifest_rows, desc="predict", unit="img")):
            if row["status"] != "pending":
                continue
            rs = resolved[pending_idx]
            pending_idx += 1
            try:
                assert cv2 is not None
                bgr = cv2.imread(str(rs.image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Failed to read image: {rs.image_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                probs = predict_probs(
                    image_bgr=bgr,
                    inferencer=inferencer,
                    predict_fn=predict_fn,
                    use_tiling=args.use_tiling,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                )
                pred_mask = probs_to_mask(probs)
                num_classes = int(probs.shape[0])
                save_mask_png(pred_mask, rs.pred_mask_path, num_classes=num_classes)

                if args.save_probs_npy:
                    npy_path = rs.pred_mask_path.with_suffix(".probs.npy")
                    np.save(str(npy_path), probs.astype(np.float16))

                if args.save_overlays:
                    ov = overlay_mask_on_image(rgb, pred_mask)
                    ov_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
                    ov_path = rs.pred_mask_path.with_name(rs.pred_mask_path.stem + "_overlay.png")
                    ov_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(ov_path), ov_bgr):
                        raise RuntimeError(f"Failed to write overlay: {ov_path}")

                flat = pred_mask.reshape(-1)
                max_id = int(flat.max()) if flat.size else 0
                if label_hist is None:
                    label_hist = np.zeros(max(1, max_id + 1), dtype=np.int64)
                if max_id >= label_hist.shape[0]:
                    tmp = np.zeros(max_id + 1, dtype=np.int64)
                    tmp[: label_hist.shape[0]] = label_hist
                    label_hist = tmp
                binc = np.bincount(flat, minlength=label_hist.shape[0])
                label_hist += binc.astype(np.int64)

                manifest_rows[i]["status"] = "processed"
                manifest_rows[i]["error"] = ""
            except Exception as exc:  # pragma: no cover - runtime integration issues
                manifest_rows[i]["status"] = "failed"
                manifest_rows[i]["error"] = str(exc)

        if label_hist is not None:
            nz = np.flatnonzero(label_hist)
            hist_str = ", ".join(f"{int(k)}:{int(label_hist[k])}" for k in nz[:50])
            more = " ..." if len(nz) > 50 else ""
            print(f"[summary] label_hist_nonzero={len(nz)} [{hist_str}{more}]")

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "facade_id",
        "template_mask_relpath",
        "image_relpath",
        "pred_mask_relpath",
        "status",
        "error",
    ]
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(manifest_rows)

    total = len(manifest_rows)
    processed = sum(r["status"] == "processed" for r in manifest_rows)
    skipped = sum(r["status"] == "skipped_missing_image" for r in manifest_rows)
    failed = sum(r["status"] == "failed" for r in manifest_rows)
    pending = sum(r["status"] == "pending" for r in manifest_rows)
    if args.dry_run:
        print(
            f"[dry-run summary] total={total} resolvable={len(resolved)} "
            f"skipped_missing_image={skipped}"
        )
    else:
        print(
            f"[summary] processed={processed} skipped_missing_image={skipped} "
            f"failed={failed} pending={pending} total={total}"
        )
    print(f"[summary] manifest={out_manifest}")


if __name__ == "__main__":
    main()
