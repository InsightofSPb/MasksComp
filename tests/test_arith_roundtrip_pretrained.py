from pathlib import Path

import cv2
import numpy as np
import torch

from maskscomp.entropy_coding.lm_codec import OnlineConfig, encode_file, read_bin, decode_mask, load_checkpoint_and_model


def _make_ckpt(path: Path) -> Path:
    cfg = {
        "labels": [0, 1, 2],
        "label_to_idx": {0: 0, 1: 1, 2: 2},
        "wmax": 17,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 1,
        "dropout": 0.0,
        "max_seq_len": 64,
        "use_2d_context": False,
    }
    from maskscomp.lm_entropy import LMEntropyModel

    model = LMEntropyModel(3, 17, d_model=16, n_layers=1, n_heads=1, dropout=0.0, max_seq_len=64, use_2d_context=False)
    torch.save({"model_state": model.state_dict(), "config": cfg}, path)
    return path


def test_arith_roundtrip_pretrained(tmp_path: Path):
    mask = np.zeros((16, 17), dtype=np.uint8)
    mask[:, :5] = 1
    mask[:, 5:9] = 2
    inp = tmp_path / "in.png"
    cv2.imwrite(str(inp), mask)

    ckpt = _make_ckpt(tmp_path / "best.pt")
    out_bin = tmp_path / "out.bin"
    encode_file(inp, ckpt, out_bin, False, OnlineConfig(mode="pretrained"), torch.device("cpu"), None)

    header, payload = read_bin(out_bin)
    _, model = load_checkpoint_and_model(ckpt, False, torch.device("cpu"))
    dec = decode_mask(payload, header, model, False, torch.device("cpu"))
    assert dec.dtype == mask.dtype
    assert np.array_equal(dec, mask)
