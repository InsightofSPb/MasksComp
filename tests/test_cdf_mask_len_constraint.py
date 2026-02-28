import io

import numpy as np

from maskscomp.entropy_coding.arith import ArithmeticDecoder, ArithmeticEncoder, BitInputStream, BitOutputStream
from maskscomp.entropy_coding.lm_codec import probs_to_cdf


def test_cdf_respects_length_mask_and_decodes_valid_symbol():
    probs = np.ones((8,), dtype=np.float64)
    remaining = 3
    allowed = np.zeros((8,), dtype=bool)
    allowed[1 : remaining + 1] = True
    cdf = probs_to_cdf(probs, allowed_mask=allowed, total_freq=1 << 12)

    freq = np.diff(cdf)
    assert np.all(freq[remaining + 1 :] == 0)
    assert np.all(freq[1 : remaining + 1] > 0)

    sink = io.BytesIO()
    enc = ArithmeticEncoder(BitOutputStream(sink))
    enc.write(cdf, 2)
    enc.finish()

    dec = ArithmeticDecoder(BitInputStream(io.BytesIO(sink.getvalue())))
    sym = dec.read(cdf)
    assert 1 <= sym <= remaining
