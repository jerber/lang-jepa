import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from src.common.schedulers import CosineWDSchedule, WarmupCosineSchedule


def init_optimizer(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    lr: float,
    weight_decay: float,
    warmup: int,
    total_epochs: int,
    steps_per_epoch: int,
    final_wd: float = 0.0,
    final_lr: float = 0.0,
    use_bfloat16: bool = False,
):
    """Build AdamW with separate weight-decay / no-decay groups.

    Biases and 1-D parameters (LayerNorm / BatchNorm gains) do not get weight
    decay. Returns (optimizer, scaler, lr_scheduler, wd_scheduler).

    `scaler` is only constructed for fp16 paths — bfloat16 autocast needs no
    scaler. Currently callers always pass use_bfloat16=False since the encoder
    train loop uses torch.autocast directly for bfloat16.
    """
    param_groups = [
        {
            "params": [
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            ]
        },
        {
            "params": [
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            ]
        },
        {
            "params": [
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ],
            "WD_exclude": True,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ],
            "WD_exclude": True,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    total_steps = steps_per_epoch * total_epochs

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=int(warmup * steps_per_epoch),
        start_lr=lr * 0.1,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer, ref_wd=weight_decay, final_wd=final_wd, T_max=total_steps
    )
    scaler = GradScaler() if use_bfloat16 and torch.cuda.is_available() else None
    return optimizer, scaler, scheduler, wd_scheduler
