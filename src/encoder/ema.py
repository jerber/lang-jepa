import copy
import math

import torch
import torch.nn as nn


class EMAEncoder(nn.Module):
    """Momentum-updated copy of the online encoder used to compute JEPA targets.

    Holds a frozen deep copy of the online encoder. Call .update(online, momentum)
    after each optimizer step to EMA-update parameters and buffers in place.
    """

    def __init__(self, online: nn.Module):
        super().__init__()
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, online: nn.Module, momentum: float) -> None:
        """target = momentum * target + (1 - momentum) * online, param-wise."""
        online_mod = online.module if hasattr(online, "module") else online
        for tp, op in zip(
            self.target.parameters(), online_mod.parameters(), strict=True
        ):
            tp.mul_(momentum).add_(op.detach(), alpha=1.0 - momentum)
        for tb, ob in zip(
            self.target.buffers(), online_mod.buffers(), strict=True
        ):
            if tb.dtype.is_floating_point:
                tb.mul_(momentum).add_(ob.detach(), alpha=1.0 - momentum)
            else:
                tb.copy_(ob)

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def train(self, mode: bool = True):
        # Target encoder stays in eval mode regardless of parent training state,
        # so dropout never perturbs target features.
        return super().train(False)


def momentum_at_step(
    step: int, total_steps: int, start: float, end: float
) -> float:
    """Cosine schedule from start to end over total_steps (matches I-JEPA)."""
    if total_steps <= 0:
        return end
    progress = min(max(step / total_steps, 0.0), 1.0)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))
