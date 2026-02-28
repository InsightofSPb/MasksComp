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

Evaluate ideal bits (cross-entropy) on split:

```bash
python tools/eval_lm_entropy.py \
  --data-root /path/to/data \
  --checkpoint output/lm_rle_baseline/checkpoints/best.pt \
  --split val \
  --splits-dir output/lm_rle_baseline/splits \
  --out-csv output/lm_val_metrics.csv
```
