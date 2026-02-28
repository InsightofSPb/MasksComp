from pathlib import Path

import cv2
import numpy as np
import torch

from maskscomp.entropy_coding.lm_codec import OnlineConfig, decode_mask, encode_file, load_checkpoint_and_model, read_bin


def _make_ckpt(path: Path) -> Path:
    cfg = {
        "labels": [0, 4],
        "label_to_idx": {0: 0, 4: 1},
        "wmax": 17,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 1,
        "dropout": 0.0,
        "max_seq_len": 64,
        "use_2d_context": False,
    }
    from maskscomp.lm_entropy import LMEntropyModel

    model = LMEntropyModel(2, 17, d_model=16, n_layers=1, n_heads=1, dropout=0.0, max_seq_len=64, use_2d_context=False)
    torch.save({"model_state": model.state_dict(), "config": cfg}, path)
    return path


def test_arith_roundtrip_online(tmp_path: Path):
    mask = np.zeros((16, 17), dtype=np.uint8)
    mask[::2, :] = 4
    inp = tmp_path / "in.png"
    cv2.imwrite(str(inp), mask)

    ckpt = _make_ckpt(tmp_path / "best.pt")
    out_bin = tmp_path / "out.bin"
    online = OnlineConfig(mode="online", lr=1e-3, steps_per_row=1, clip=1.0, online_after="row")
    encode_file(inp, ckpt, out_bin, False, online, torch.device("cpu"), None)

    header, payload = read_bin(out_bin)
    _, model = load_checkpoint_and_model(ckpt, False, torch.device("cpu"))
    dec = decode_mask(payload, header, model, False, torch.device("cpu"))
    assert np.array_equal(dec, mask)
