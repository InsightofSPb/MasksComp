# MasksComp

## LM entropy model baseline

Train (row-wise RLE token LM with optional 2D context):

```bash
python tools/train_lm_entropy.py \
  --data-root /path/to/data \
  --out-dir output/lm_rle_baseline \
  --epochs 5 \
  --batch-size 64 \
  --use-2d-context
```


MSDZip-style backbone option (`--arch msdzip`) for fixed-context prediction:

```bash
python tools/train_lm_entropy.py \
  --data-root /path/to/data \
  --out-dir output/lm_msdzip \
  --arch msdzip \
  --timesteps 16 \
  --vocab-dim 16 \
  --hidden-dim 128 \
  --ffn-dim 256 \
  --layers 4
```

Run codec matrix with an MSDZip checkpoint (same CLI):

```bash
PYTHONPATH=$(pwd) python tools/run_lm_codec_matrix.py \
  --data-root /path/to/data \
  --subdir warped_masks \
  --checkpoint output/lm_msdzip/checkpoints/best.pt \
  --splits-dir output/lm_msdzip/splits \
  --out-csv output/lm_msdzip/lm_matrix.csv
```

Note: codec scripts enable deterministic PyTorch ops. On CUDA this can require
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) in your environment.

Evaluate ideal bits (cross-entropy) on split:

```bash
python tools/eval_lm_entropy.py \
  --data-root /path/to/data \
  --checkpoint output/lm_rle_baseline/checkpoints/best.pt \
  --split val \
  --splits-dir output/lm_rle_baseline/splits \
  --out-csv output/lm_val_metrics.csv
```

## Row cache for large datasets

For large mask corpora (e.g. A2D2), precompute a row-wise RLE cache to avoid keeping all
`RowItem` objects in RAM:

```bash
python tools/cache_row_tokens.py \
  --data-root /path/to/data \
  --subdir warped_masks \
  --cache-dir /path/to/data/cache_rle_row_v1 \
  --manifest-out /path/to/data/cache_rle_row_v1/manifest.csv \
  --include-above-features \
  --compress npz \
  --verify 2
```

Train using the cache (same model/loss code path, lazy row loading):

```bash
python tools/train_lm_entropy.py \
  --data-root /path/to/data \
  --subdir warped_masks \
  --out-dir output/lm_rle_cached \
  --row-cache-dir /path/to/data/cache_rle_row_v1 \
  --epochs 5
```

Evaluate with cache-backed rows:

```bash
python tools/eval_lm_entropy.py \
  --data-root /path/to/data \
  --checkpoint output/lm_rle_cached/checkpoints/best.pt \
  --split val \
  --splits-dir output/lm_rle_cached/splits \
  --row-cache-dir /path/to/data/cache_rle_row_v1 \
  --out-csv output/lm_val_metrics_cached.csv
```

## A2D2 quickstart

1) Convert A2D2 RGB semantic labels to class-id masks in MasksComp layout:

```bash
PYTHONPATH=$(pwd) python tools/prepare_a2d2_semantic.py \
  --a2d2-root /path/to/a2d2 \
  --out-root output/a2d2_frontcenter_ids \
  --camera front_center \
  --out-subdir warped_masks
```

2) Train LM entropy model on the converted masks:

```bash
PYTHONPATH=$(pwd) python tools/train_lm_entropy.py \
  --data-root output/a2d2_frontcenter_ids \
  --subdir warped_masks \
  --out-dir output/a2d2_lm \
  --epochs 1
```

3) Run LM codec matrix evaluation with the same dataset root/subdir:

```bash
PYTHONPATH=$(pwd) python tools/run_lm_codec_matrix.py \
  --data-root output/a2d2_frontcenter_ids \
  --subdir warped_masks \
  --checkpoint output/a2d2_lm/checkpoints/best.pt \
  --splits-dir output/a2d2_lm/splits \
  --out-csv output/a2d2_lm/lm_matrix.csv
```

## Change detection benchmark (conditional codelength)

1. Build pair lists for train/val:

```bash
python tools/make_pairs.py --dataset a2d2 --data-root DS --subdir warped_masks --splits-dir DS/splits --delta 1 --out-dir OUT/pairs
python tools/make_pairs.py --dataset facades --data-root DS_F --subdir warped_masks --splits-dir DS_F/splits --out-dir OUT/pairs
```

2. Build residual datasets (`C` and `V`):

```bash
python tools/build_residual_dataset.py --data-root DS --subdir warped_masks --pairs-csv OUT/pairs/a2d2_pairs_train.csv --out-root RES_A2D2
```

3. Benchmark conditional classic codecs:

```bash
python tools/bench_residual_codecs.py --data-root RES_A2D2 --splits-dir RES_A2D2/splits --split val --codecs lzma,zstd --levels 1,3,6,9 --out-csv OUT/conditional_classic/a2d2_residual_codecs.csv
```

4. Train/evaluate LM on `residual_C` and `residual_V` with existing `tools/train_lm_entropy.py` and `tools/eval_lm_entropy.py`, then merge:

```bash
python tools/merge_residual_lm.py --csv-c OUT/lm_resC/val_per_image.csv --csv-v OUT/lm_resV/val_per_image.csv --split val --out-csv OUT/conditional_nn/residual_lm_sum.csv
```

5. Tile surprise maps and metrics:

```bash
python tools/eval_change_tiles.py --mode classic_residual --data-root RES_A2D2 --pairs-csv OUT/pairs/a2d2_pairs_val.csv --split val --out-dir OUT/change_tiles/classic
python tools/eval_change_metrics.py --data-root DS --pairs-csv OUT/pairs/a2d2_pairs_val.csv --heatmap-dir OUT/change_tiles/classic/heatmaps_tiles --dataset a2d2 --split val --method classic_residual --out OUT/change_tiles/classic/metrics_change_tiles.csv
```

6. Paper artifacts (tables + optional top-K render):

```bash
python tools/make_paper_artifacts.py --cond-csv OUT/conditional_nn/residual_lm_sum.csv --metrics-csv OUT/change_tiles/classic/metrics_change_tiles.csv --tiles-csv OUT/change_tiles/classic/change_tiles_scores.csv --heat-root OUT/change_tiles/classic/heatmaps_tiles --out-dir output --render-topk 10
```

## Recommended MSDZip tile protocols (A2D2 residual)

### B-mode (global eval → run dumps → tile heatmaps)

```bash
cd /home/sasha/MasksComp
export PYTHONPATH=$(pwd)

RES_ALL=results_a2d2/all
PAIRS=output/a2d2_msdzip_cached/pairs/a2d2_pairs_val.csv
OUTV=/home/sasha/MasksComp/results_a2d2_conditional/conditional_nn/lm_resV
CKPTV=$OUTV/checkpoints/best.pt
DUMP=$OUTV/run_bits_val

python tools/eval_lm_entropy.py \
  --data-root "$RES_ALL" --subdir residual_V \
  --checkpoint "$CKPTV" \
  --splits-dir "$RES_ALL/splits" --split val \
  --arch msdzip --timesteps 16 \
  --out-csv "$OUTV/val_per_image.csv" \
  --dump-run-bits-dir "$DUMP" \
  --batch-size 256 --device cuda

OUT_B=/home/sasha/MasksComp/results_a2d2_conditional/change_tiles/a2d2_B_msdzipV_decomp_t64_s32
python tools/run_bits_to_tile_heatmaps.py \
  --run-bits-dir "$DUMP" \
  --pairs-csv "$PAIRS" \
  --split val \
  --tile-size 64 --stride 32 \
  --out-dir "$OUT_B"

python tools/eval_change_metrics.py \
  --data-root "$RES_ALL" \
  --pairs-csv "$PAIRS" \
  --heatmap-dir "$OUT_B/heatmaps_tiles" \
  --dataset a2d2 --split val --method msdzip_B_t64_s32 \
  --tile-size 64 --stride 32 --tau 0.01 --topk 5 \
  --label-from residual_C \
  --out /home/sasha/MasksComp/results_a2d2_conditional/metrics/a2d2_val_msdzipB_t64_s32.csv
```

### A-mode (tile-reset per tile)

```bash
TILE_DS=/home/sasha/MasksComp/results_a2d2_conditional/tile_ds/a2d2_val_resV_t64_s64_top50
python tools/make_tile_dataset.py \
  --data-root results_a2d2/all \
  --pairs-csv output/a2d2_msdzip_cached/pairs/a2d2_pairs_val.csv \
  --split val \
  --ids-txt /home/sasha/MasksComp/results_a2d2_conditional/subsets/a2d2_val_top50_by_hybrid.txt \
  --src-subdir residual_V \
  --tile-size 64 --stride 64 \
  --out-root "$TILE_DS"

OUT_A_NN=/home/sasha/MasksComp/results_a2d2_conditional/change_tiles/a2d2_A_msdzip_tileReset_t64_s64_top50
python tools/eval_lm_entropy.py \
  --data-root "$TILE_DS" --subdir tiles \
  --checkpoint /home/sasha/MasksComp/results_a2d2_conditional/conditional_nn/lm_resV/checkpoints/best.pt \
  --splits-dir "$TILE_DS/splits" --split val \
  --arch msdzip --timesteps 16 \
  --out-csv "$OUT_A_NN/val_tiles_per_image.csv" \
  --batch-size 256 --device cuda

python tools/tiles_eval_to_heatmaps.py \
  --tile-ds-root "$TILE_DS" \
  --tiles-eval-csv "$OUT_A_NN/val_tiles_per_image.csv" \
  --out-dir "$OUT_A_NN"
```
