import copy

import torch
import torch.nn as nn

from src.encoder.ema import EMAEncoder, momentum_at_step


def _tiny_module() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))


def test_ema_initial_state_equals_online():
    online = _tiny_module()
    ema = EMAEncoder(online)
    for o, t in zip(online.parameters(), ema.target.parameters(), strict=True):
        assert torch.equal(o, t)


def test_ema_target_has_no_grad():
    ema = EMAEncoder(_tiny_module())
    for p in ema.target.parameters():
        assert not p.requires_grad


def test_ema_target_stays_in_eval_mode():
    ema = EMAEncoder(_tiny_module())
    ema.train(True)  # should be ignored
    assert not ema.target.training


def test_ema_m_zero_copies_online():
    online = _tiny_module()
    ema = EMAEncoder(_tiny_module())
    # Modify online; EMA with m=0 should fully adopt online.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    ema.update(online, momentum=0.0)
    for o, t in zip(online.parameters(), ema.target.parameters(), strict=True):
        assert torch.allclose(o, t)


def test_ema_m_one_freezes_target():
    online = _tiny_module()
    ema = EMAEncoder(_tiny_module())
    before = [p.clone() for p in ema.target.parameters()]
    with torch.no_grad():
        for p in online.parameters():
            p.add_(100.0)
    ema.update(online, momentum=1.0)
    for b, a in zip(before, ema.target.parameters(), strict=True):
        assert torch.allclose(b, a)


def test_ema_interpolation_correct():
    online = _tiny_module()
    ema = EMAEncoder(copy.deepcopy(online))
    online_before = [p.clone() for p in online.parameters()]
    # Shift online
    with torch.no_grad():
        for p in online.parameters():
            p.add_(2.0)
    ema.update(online, momentum=0.5)
    for o_before, o_after, t in zip(
        online_before, online.parameters(), ema.target.parameters(), strict=True
    ):
        expected = 0.5 * o_before + 0.5 * o_after
        assert torch.allclose(t, expected, atol=1e-6)


def test_momentum_schedule_monotonic():
    # Cosine schedule from 0.996 → 1.0 should be monotonically non-decreasing.
    values = [momentum_at_step(i, 100, 0.996, 1.0) for i in range(0, 101, 10)]
    for a, b in zip(values[:-1], values[1:], strict=True):
        assert b >= a - 1e-9
    assert abs(values[0] - 0.996) < 1e-6
    assert abs(values[-1] - 1.0) < 1e-6


def test_momentum_handles_degenerate_total_steps():
    # Edge cases: total_steps=0 or step >= total_steps should not crash.
    assert momentum_at_step(0, 0, 0.996, 1.0) == 1.0
    assert momentum_at_step(999, 10, 0.5, 0.9) == 0.9
