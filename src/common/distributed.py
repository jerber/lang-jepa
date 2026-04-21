"""Small DDP helpers shared by encoder and decoder training.

Detects `LOCAL_RANK` in env (set by `torchrun`) to decide whether to
initialize a process group. Single-process runs are a no-op — the same
training scripts work on a laptop and a multi-GPU cluster.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistInfo:
    """Initialize NCCL process group if launched via torchrun, else be a no-op."""
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistInfo(rank=0, world_size=1, local_rank=0, device=device)

    local_rank = int(local_rank_env)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistInfo(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, stripping DDP wrapper if present."""
    return model.module if hasattr(model, "module") else model
