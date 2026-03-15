Raw Temporal Compression Signal (MSDZip-style LM + LPOSS side branch)
====================================================================

Reused MasksComp components
---------------------------
- MSDZip-style backbone: reused ``maskscomp.models.msdzip.MixedModel`` as temporal LM core for byte prediction.
- Tile heatmap conventions: reused per-tile scoring + image-aligned heatmap output style.
- Split-file conventions: preserved ``splits/facade_<split>.txt`` and CSV-centric reporting.

New files
---------
- ``maskscomp/raw_temporal.py``
- ``tools/build_raw_temporal_residual_dataset.py``
- ``tools/train_raw_temporal_lm.py``
- ``tools/eval_raw_temporal_lm.py``
- ``tools/compare_raw_temporal_lm_vs_png.py``

Data assumptions
----------------
- Pair CSV required columns: ``pair_id,sample_id,prev_path,cur_path,split``.
- Optional columns: ``homography_path``, ``valid_mask_path``, ``lposs_prev_path,lposs_cur_path``, ``superpixel_labels_path``.
- Milestone-1 residual uses modular residual: ``r = (I_t - I_{t-1->t}) mod 256``.

LPOSS and superpixel usage
--------------------------
- LPOSS remains separate side branch for semantic interpretation only.
- Superpixels are optional postprocessing aggregation over fixed-tile scores.

Minimal runnable commands
-------------------------
::

  python tools/build_raw_temporal_residual_dataset.py --data-root <DATA_ROOT> --pairs-csv <PAIRS_CSV> --out-root <OUT_DATASET> --tile-size 64 --stride 32 --serialize interleaved
  python tools/train_raw_temporal_lm.py --dataset-root <OUT_DATASET> --splits-dir <OUT_DATASET>/splits --out-dir runs/raw_temporal_msdzip --timesteps 64 --epochs 5 --batch-size 512 --device cuda
  python tools/eval_raw_temporal_lm.py --dataset-root <OUT_DATASET> --pairs-csv <PAIRS_CSV> --split val --checkpoint runs/raw_temporal_msdzip/best.pt --out-dir runs/raw_temporal_eval/val --write-heatmaps --overlay-lposs
  python tools/compare_raw_temporal_lm_vs_png.py --tile-csv runs/raw_temporal_eval/val/tile_scores_val.csv --out-csv runs/raw_temporal_eval/val/lm_vs_png_summary.csv
