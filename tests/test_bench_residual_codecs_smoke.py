import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np


def test_bench_residual_codecs_smoke(tmp_path: Path) -> None:
    root = tmp_path / "res"
    pair = "pair_1"
    (root / pair / "residual_C").mkdir(parents=True)
    (root / pair / "residual_V").mkdir(parents=True)
    c = np.zeros((8, 8), dtype=np.uint8)
    c[0, 0] = 1
    v = np.zeros((8, 8), dtype=np.uint8)
    v[0, 0] = 7
    cv2.imwrite(str(root / pair / "residual_C" / "x.png"), c)
    cv2.imwrite(str(root / pair / "residual_V" / "x.png"), v)
    splits = root / "splits"
    splits.mkdir()
    (splits / "facade_val.txt").write_text(f"{pair}\n", encoding="utf-8")

    out_csv = tmp_path / "out.csv"
    subprocess.run(
        [
            "python",
            "tools/bench_residual_codecs.py",
            "--data-root",
            str(root),
            "--splits-dir",
            str(splits),
            "--split",
            "val",
            "--codecs",
            "lzma",
            "--levels",
            "1",
            "--out-csv",
            str(out_csv),
        ],
        check=True,
    )

    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert float(rows[0]["bits_sum"]) > 0
