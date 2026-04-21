import logging
import os

import torch
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_path: str,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    epoch: int,
    global_step: int,
    loss: float,
) -> None:
    """Save a consolidated checkpoint including the EMA target encoder."""
    state = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
    }
    try:
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at {checkpoint_path}: {e}")


def load_checkpoint(
    checkpoint_path: str | None,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, int]:
    """Restore state from a checkpoint. Returns (start_epoch, global_step).

    Tolerates older checkpoints that lack target_encoder or global_step.
    """
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        logger.info("No checkpoint found, starting from scratch.")
        return 0, 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        predictor.load_state_dict(checkpoint["predictor"])

        if target_encoder is not None and "target_encoder" in checkpoint:
            target_encoder.load_state_dict(checkpoint["target_encoder"])
        elif target_encoder is not None:
            logger.info(
                "Checkpoint predates target_encoder bundling — seeding the "
                "target encoder from online encoder state."
            )
            target_encoder.load_state_dict(encoder.state_dict())

        optimizer.load_state_dict(checkpoint["opt"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {start_epoch}, step {global_step})."
        )
        return start_epoch, global_step
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        return 0, 0
