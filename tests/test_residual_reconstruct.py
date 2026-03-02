import numpy as np

from maskscomp.change_detection import build_residual, reconstruct_from_residual


def test_residual_reconstruct_random_uint8() -> None:
    rng = np.random.default_rng(0)
    prev = rng.integers(0, 10, size=(16, 16), dtype=np.uint8)
    cur = rng.integers(0, 10, size=(16, 16), dtype=np.uint8)
    c, v = build_residual(prev, cur)
    recon = reconstruct_from_residual(prev, c, v)
    assert np.array_equal(recon, cur)
