import torch

from src.common.pooling import masked_mean


def test_masked_mean_matches_unmasked_when_no_padding():
    hidden = torch.randn(3, 7, 5)
    mask = torch.ones(3, 7, dtype=torch.long)
    assert torch.allclose(masked_mean(hidden, mask), hidden.mean(dim=1), atol=1e-6)


def test_masked_mean_excludes_padded_positions():
    # Batch 2, length 4, dim 2. First sample has real tokens 0..2, padding at 3.
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 100.0]],  # pad at pos 3
            [[0.0, 1.0], [0.0, 2.0], [100.0, 100.0], [100.0, 100.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    out = masked_mean(hidden, mask)
    assert torch.allclose(out[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 1.5]))


def test_masked_mean_all_padding_is_safe():
    hidden = torch.randn(1, 4, 3)
    mask = torch.zeros(1, 4, dtype=torch.long)
    out = masked_mean(hidden, mask)
    # clamp(min=1) means we divide by 1; sum of (hidden * 0) = 0 → zero vector.
    assert torch.allclose(out, torch.zeros(1, 3))


def test_masked_mean_preserves_dtype():
    hidden = torch.randn(2, 3, 4, dtype=torch.float32)
    mask = torch.ones(2, 3, dtype=torch.long)
    assert masked_mean(hidden, mask).dtype == torch.float32
