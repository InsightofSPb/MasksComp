# Facade Semantic Pipeline (S2 stage)

This document describes the semantic-only stage integrated into `MasksComp`.

## 1) Semantic export

CLI: `tools/export_facade_semantics.py`

### Inputs
- `--input-root`: image root (scan mode)
- `--manifest-csv` + `--image-col`: optional manifest mode
- `--config`: predictor/backend config
- `--checkpoint`: semantic checkpoint path (backend-dependent)
- `--device`: inference device

### Exported artifacts
Under `output-root`:
- `masks/<sample_id>.png`: single-channel class-id map
- `probs/<sample_id>.npz`: `probs_or_logits` (if backend provides it)
- `features/<sample_id>.npz`: compressed semantic features with metadata
  - `feat`
  - `feat_h`, `feat_w`
  - `channels`
  - `source_model`
  - `sample_id`
- `overlays/<sample_id>.jpg`
- `index.csv`

`index.csv` contains at least:
- `sample_id`, `image_path`
- `mask_path`, `probs_path`, `features_path`, `overlay_path`
- `status`, `height`, `width`
- `feature_h`, `feature_w`, `feature_channels`

## 2) Temporal semantic features (aligned pairs)

CLI: `tools/build_temporal_semantic_features.py`

### Pair CSV schema
The script infers common columns. Required logical fields:
- pair id: one of `pair_id|id|pair`
- previous sample key/path: one of `prev_sample_id|prev_id|sample_id_prev|prev_sample|prev_path|image_prev`
- current sample key/path: one of `curr_sample_id|cur_sample_id|curr_id|sample_id_curr|cur_path|image_curr`

### Features per tile
For each tile (top-left origin, row-major, deterministic border handling):
1. `mask_change_density`
2. `class_histogram_drift` (L1 distance between normalized class histograms)
3. `feature_cosine_distance`

Aggregate:

`semantic_score = alpha*mask_change_density + beta*class_histogram_drift + gamma*feature_cosine_distance`

### Outputs
- `temporal_semantic_features.csv`
- `heatmaps_semantic/<pair_id>.npz`
- `previews_semantic/<pair_id>.jpg`

## Notes
- This stage is S2-only; no compressor-conditioning logic (C1) is included yet.
- `Compress_to_prevent` is **reference-only** and not required at runtime.
