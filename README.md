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
