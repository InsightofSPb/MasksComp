Raw Temporal Compression Signal (MSDZip-style LM + LPOSS side branch)
====================================================================

Reused MasksComp components
---------------------------
- MSDZip-style backbone: reused ``maskscomp.models.msdzip.MixedModel`` as temporal LM core for byte prediction.
- Tile heatmap conventions: reused per-tile scoring + image-aligned heatmap output style.
- Split-file conventions: preserved ``splits/facade_<split>.txt`` and CSV-centric reporting.

What changed in this revision
-----------------------------
- Added a **precomputed alignment mode** to dataset building.
- The builder now accepts existing pair CSVs in either format:

  - ``pair_id,facade_id,year_a,year_b,mask_a,mask_b,delta_years`` (existing LPOSS CSVs; ``mask_a/mask_b`` are treated as raw RGB paths),
  - or the previous custom format.

- In precomputed mode, aligned pair assets are resolved from
  ``<prealigned-root>/<facade_id>/...`` and **no homography warp is recomputed**.
- Geometry JSONs under ``geom/<facade_id>/<year_a>_<year_b>.json`` are parsed for metadata only.

Data assumptions
----------------
- Prealigned assets directory contains pair-consistent aligned RGB rasters per facade.
- MVP signal is modular residual: ``r = (I_t - I_{t-1->t}) mod 256``.
- LPOSS remains side-information only; not part of compressed signal.

Primary builder command (MVP)
-----------------------------
::

  python tools/build_raw_temporal_residual_dataset.py \
    --pairs-csv /home/sasha/LPOSS/datasets/feb_mar_with_years/pairs_consecutive.csv \
    --alignment-mode precomputed \
    --prealigned-root /home/sasha/LPOSS/datasets/feb_mar_with_years/spx_aligning_results/facades \
    --geom-root /home/sasha/LPOSS/datasets/feb_mar_with_years/geom \
    --out-root /home/sasha/MasksComp/outputs/raw_temporal_dataset_v1 \
    --tile-size 64 \
    --stride 32 \
    --serialize interleaved \
    --png-level 6 \
    --write-viz

Training/eval commands
----------------------
::

  python tools/train_raw_temporal_lm.py --dataset-root <OUT_DATASET> --splits-dir <OUT_DATASET>/splits --out-dir runs/raw_temporal_msdzip --timesteps 64 --epochs 5 --batch-size 512 --device cuda
  python tools/eval_raw_temporal_lm.py --dataset-root <OUT_DATASET> --pairs-csv <PAIRS_CSV> --split val --checkpoint runs/raw_temporal_msdzip/best.pt --out-dir runs/raw_temporal_eval/val --write-heatmaps --overlay-lposs
  python tools/compare_raw_temporal_lm_vs_png.py --tile-csv runs/raw_temporal_eval/val/tile_scores_val.csv --out-csv runs/raw_temporal_eval/val/lm_vs_png_summary.csv
