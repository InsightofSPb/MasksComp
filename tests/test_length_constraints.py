import torch

from maskscomp.lm_entropy import masked_length_nll_bits
from maskscomp.rle_tokenizer import encode_mask_to_row_tokens


def test_remaining_width_and_masking_constraints():
    row = torch.tensor([[1, 1, 1, 2, 2, 3, 3, 3]], dtype=torch.int64).numpy()
    encoded = encode_mask_to_row_tokens(row)
    r = encoded[0]

    assert r.rem_width[1] == 8
    assert r.rem_width[3] == 5
    assert r.rem_width[5] == 3

    logits = torch.zeros((1, 7, 9), dtype=torch.float32)
    targets = torch.tensor([[0, 3, 0, 2, 0, 3, 0]], dtype=torch.long)
    remaining = torch.tensor([[0, 8, 0, 5, 0, 3, 0]], dtype=torch.long)

    bits = masked_length_nll_bits(logits, targets, remaining)
    # Uniform over allowed lengths => bits at length steps should be log2(remaining)
    assert torch.isclose(bits[0, 1], torch.tensor(3.0), atol=1e-5)
    assert torch.isclose(bits[0, 3], torch.tensor(torch.log2(torch.tensor(5.0))), atol=1e-5)
    assert torch.isclose(bits[0, 5], torch.tensor(torch.log2(torch.tensor(3.0))), atol=1e-5)
