import torch

from src.decoder.models import _top_k_top_p_filter


def test_top_k_filter_keeps_exactly_k():
    logits = torch.tensor([[0.1, 5.0, 1.0, 3.0, 2.0]])
    filtered = _top_k_top_p_filter(logits.clone(), top_k=2, top_p=None)
    # Only the top-2 logits (5.0, 3.0) should remain finite.
    assert (filtered > float("-inf")).sum().item() == 2
    assert torch.isfinite(filtered[0, 1])  # 5.0
    assert torch.isfinite(filtered[0, 3])  # 3.0


def test_top_p_filter_keeps_nucleus():
    # Softmax-balanced logits: [0, 0, 10] → probs ≈ [0, 0, ~1].
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    filtered = _top_k_top_p_filter(logits.clone(), top_k=None, top_p=0.9)
    # Top token (index 2) alone already covers >0.9 of mass.
    assert torch.isfinite(filtered[0, 2])
    # The other tokens should be masked to -inf.
    assert torch.isinf(filtered[0, 0])
    assert torch.isinf(filtered[0, 1])


def test_top_p_always_keeps_top_token():
    # Even with top_p=0 we never drop the single highest-probability token.
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    filtered = _top_k_top_p_filter(logits.clone(), top_k=None, top_p=0.0)
    assert torch.isfinite(filtered[0, 2])


def test_no_filter_when_both_none_or_unit():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    out = _top_k_top_p_filter(logits.clone(), top_k=None, top_p=None)
    assert torch.equal(out, logits)
    # top_p=1.0 includes everything
    out = _top_k_top_p_filter(logits.clone(), top_k=None, top_p=1.0)
    assert torch.equal(out, logits)
