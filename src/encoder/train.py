import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from src.common.config import LANGJEPAConfig
from src.common.datasets.fineweb_edu import TextDataset, worker_init_fn
from src.common.datasets.streaming import StreamingTextDataset
from src.common.distributed import DistInfo, setup_distributed, unwrap
from src.common.logging import AverageMeter, CSVLogger
from src.common.pooling import masked_mean
from src.encoder.collator import Batch, Collator
from src.encoder.ema import EMAEncoder, momentum_at_step
from src.encoder.models import TextPredictor, TextTransformer
from src.encoder.utils.checkpoints import load_checkpoint, save_checkpoint
from src.encoder.utils.monitor import TrainingMonitor, ValidationMetrics
from src.encoder.utils.optim import init_optimizer


class _NoopWandb:
    def log(self, *_args, **_kwargs) -> None:
        pass

    def finish(self) -> None:
        pass


def _maybe_init_wandb(
    config: LANGJEPAConfig, run_dir: Path, dist_info: DistInfo
) -> tuple[object, bool]:
    """Initialize wandb only on rank 0, only when enabled + key present."""
    if not dist_info.is_main or not config.logging.log_to_wandb:
        return _NoopWandb(), False
    load_dotenv()
    if "WANDB_API_KEY" not in os.environ:
        print("[wandb] log_to_wandb=true but WANDB_API_KEY not set — skipping.")
        return _NoopWandb(), False
    import wandb

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project="lang-jepa",
        config=config.model_dump(exclude={"data": {"tokenizer"}}),
        name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
        dir=str(run_dir),
    )
    return wandb, True


def _build_train_loader(
    config: LANGJEPAConfig, dist_info: DistInfo
) -> tuple[DataLoader, DataLoader | None]:
    """Return (train_loader, val_loader). val_loader may be None."""
    collator = Collator(
        tokenizer=config.data.tokenizer, max_length=config.model.max_length
    )

    if config.data.streaming:
        train_ds: IterableDataset = StreamingTextDataset(
            train_file=config.data.train_file,
            min_length=config.data.min_length,
            window_size=config.data.window_size,
            min_sentences=config.data.min_sentences,
            val_fraction=config.data.val_fraction,
            split="train",
            shuffle_buffer=config.data.shuffle_buffer,
            rank=dist_info.rank,
            world_size=dist_info.world_size,
            splitter_device=config.data.splitter_device,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        val_loader: DataLoader | None = None
        if config.data.val_fraction > 0.0 and dist_info.is_main:
            val_ds = StreamingTextDataset(
                train_file=config.data.train_file,
                min_length=config.data.min_length,
                window_size=config.data.window_size,
                min_sentences=config.data.min_sentences,
                val_fraction=config.data.val_fraction,
                split="val",
                shuffle_buffer=0,
                rank=0,
                world_size=1,
                splitter_device=config.data.splitter_device,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=config.data.batch_size,
                num_workers=max(1, config.data.num_workers // 2),
                pin_memory=True,
                collate_fn=collator,
            )
        return train_loader, val_loader

    # Non-streaming: build once on each rank (each rank will shard implicitly by
    # taking a different slice via DistributedSampler-style index filtering).
    dataset = TextDataset(
        train_file=config.data.train_file,
        limit=config.data.limit,
        min_length=config.data.min_length,
        min_sentences=config.data.min_sentences,
        window_size=config.data.window_size,
        val_fraction=config.data.val_fraction,
    )
    sampler = None
    if dist_info.is_distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=dist_info.world_size,
            rank=dist_info.rank,
            shuffle=True,
        )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=collator,
        worker_init_fn=worker_init_fn,
    )

    val_loader = None
    if dist_info.is_main and len(dataset.val_view()) > 0:
        val_loader = DataLoader(
            dataset=dataset.val_view(),
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
    return train_loader, val_loader


def _resolve_total_steps(
    config: LANGJEPAConfig, train_loader: DataLoader
) -> tuple[int, int]:
    """Return (total_optimizer_steps, steps_per_epoch).

    When streaming, the user MUST set optimization.max_steps since the dataset
    has no known length. When not streaming, we compute it from the loader.
    """
    accum = config.optimization.grad_accum_steps

    if config.data.streaming:
        if config.optimization.max_steps is None:
            raise ValueError(
                "data.streaming=true requires optimization.max_steps to be set."
            )
        total = config.optimization.max_steps
        steps_per_epoch = max(total // max(config.optimization.epochs, 1), 1)
        return total, steps_per_epoch

    # len(train_loader) is in *forward passes*, not optimizer steps.
    forward_per_epoch = len(train_loader)
    steps_per_epoch = max(forward_per_epoch // accum, 1)
    total = (
        config.optimization.max_steps
        if config.optimization.max_steps is not None
        else steps_per_epoch * config.optimization.epochs
    )
    return total, steps_per_epoch


@torch.no_grad()
def _run_validation(
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    target_encoder: EMAEncoder,
    val_loader: DataLoader,
    config: LANGJEPAConfig,
    device: torch.device,
    autocast_ctx,
) -> dict[str, float]:
    metrics = ValidationMetrics()
    encoder.eval()
    predictor.eval()
    for batch in val_loader:
        context_ids = batch.context_ids.to(device)
        context_mask = batch.padding_masks.to(device)
        target_tokens = config.data.tokenizer(
            batch.target_texts,
            padding=True,
            truncation=True,
            max_length=config.model.max_length,
            return_tensors="pt",
        ).to(device)
        target_mask = target_tokens["attention_mask"]

        with autocast_ctx:
            tgt_hidden = target_encoder(target_tokens["input_ids"], target_mask)
            target_features = F.normalize(masked_mean(tgt_hidden, target_mask), dim=-1)

            context_hidden = encoder(context_ids, context_mask)
            predicted_features = F.normalize(
                predictor(context_hidden, context_mask), dim=-1
            )
        metrics.update(predicted_features, target_features)

    encoder.train()
    predictor.train()
    return metrics.get_metrics()


def train(config: LANGJEPAConfig) -> None:
    """Main training function for LANG-JEPA next-sentence prediction."""

    dist_info = setup_distributed()
    device = dist_info.device

    run_dir = Path(config.logging.log_dir)
    if dist_info.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(run_dir / "config.yaml"))

    wb, wandb_live = _maybe_init_wandb(config, run_dir, dist_info)

    csv_logger = None
    if dist_info.is_main:
        csv_logger = CSVLogger(
            str(run_dir / "training.csv"),
            ("%d", "epoch"),
            ("%d", "step"),
            ("%.5f", "loss"),
            ("%.6f", "lr"),
            ("%.4f", "momentum"),
            ("%.2f", "time(ms)"),
        )

    train_loader, val_loader = _build_train_loader(config, dist_info)
    total_steps, steps_per_epoch = _resolve_total_steps(config, train_loader)

    # Models
    encoder_mod = TextTransformer(config=config).to(device)
    predictor_mod = TextPredictor(
        input_dim=encoder_mod.embed_dim, pred_dim=config.model.pred_dim
    ).to(device)
    target_encoder = EMAEncoder(encoder_mod).to(device)

    encoder: torch.nn.Module = encoder_mod
    predictor: torch.nn.Module = predictor_mod
    if dist_info.is_distributed and torch.cuda.is_available():
        encoder = DDP(encoder_mod, device_ids=[dist_info.local_rank])
        predictor = DDP(predictor_mod, device_ids=[dist_info.local_rank])

    optimizer, scaler, lr_scheduler, wd_scheduler = init_optimizer(
        encoder=unwrap(encoder),
        predictor=unwrap(predictor),
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
        warmup=config.optimization.warmup,
        total_epochs=max(config.optimization.epochs, 1),
        steps_per_epoch=steps_per_epoch,
        final_wd=config.optimization.final_weight_decay,
        final_lr=config.optimization.final_lr,
        use_bfloat16=False,  # bf16 handled via autocast; GradScaler is fp16-only.
    )

    monitor = None
    if dist_info.is_main:
        monitor = TrainingMonitor(
            tokenizer=config.data.tokenizer,
            log_dir=run_dir,
            num_examples=config.logging.num_examples,
            log_to_wandb=wandb_live,
        )

    start_epoch = 0
    global_step = 0
    if config.meta.load_checkpoint:
        start_epoch, global_step = load_checkpoint(
            checkpoint_path=config.meta.checkpoint_path,
            encoder=unwrap(encoder),
            predictor=unwrap(predictor),
            target_encoder=target_encoder.target,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
    if global_step == 0:
        global_step = start_epoch * steps_per_epoch
    forward_count = 0
    loss_meter = AverageMeter()
    accum = config.optimization.grad_accum_steps

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if config.meta.use_bfloat16
        else nullcontext()
    )

    encoder.train()
    predictor.train()

    for epoch in range(start_epoch, config.optimization.epochs):
        epoch_start = time.time()
        loss_meter.reset()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            batch: Batch
            context_ids = batch.context_ids.to(device)
            context_mask = batch.padding_masks.to(device)
            target_tokens = config.data.tokenizer(
                batch.target_texts,
                padding=True,
                truncation=True,
                max_length=config.model.max_length,
                return_tensors="pt",
            ).to(device)
            target_mask = target_tokens["attention_mask"]

            is_accum_boundary = ((forward_count + 1) % accum) == 0
            # Skip DDP all-reduce on non-boundary steps (grads accumulate locally).
            enc_sync = (
                encoder.no_sync()
                if isinstance(encoder, DDP) and not is_accum_boundary
                else nullcontext()
            )
            pred_sync = (
                predictor.no_sync()
                if isinstance(predictor, DDP) and not is_accum_boundary
                else nullcontext()
            )

            with enc_sync, pred_sync, autocast_ctx:
                with torch.no_grad():
                    tgt_hidden = target_encoder(
                        target_tokens["input_ids"], target_mask
                    )
                    target_features = F.normalize(
                        masked_mean(tgt_hidden, target_mask), dim=-1
                    )

                context_hidden = encoder(context_ids, context_mask)
                predicted_features = F.normalize(
                    predictor(context_hidden, context_mask), dim=-1
                )

                if config.optimization.loss_fn == "smooth_l1":
                    loss = F.smooth_l1_loss(predicted_features, target_features)
                else:
                    loss = 1.0 - F.cosine_similarity(
                        predicted_features, target_features
                    ).mean()

                scaled_loss = loss / accum

            scaled_loss.backward()
            forward_count += 1
            loss_meter.update(loss.item())

            if not is_accum_boundary:
                continue

            # Accumulation boundary: optimizer + EMA step.
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr = lr_scheduler.step()
            wd_scheduler.step()

            m = momentum_at_step(
                step=global_step,
                total_steps=total_steps,
                start=config.optimization.momentum_start,
                end=config.optimization.momentum_end,
            )
            target_encoder.update(unwrap(encoder), momentum=m)
            global_step += 1

            if dist_info.is_main and (global_step % config.logging.log_freq == 0):
                elapsed = (time.time() - epoch_start) * 1000.0
                csv_logger.log(
                    epoch + 1, global_step, loss.item(), lr, m, elapsed
                )
                print(
                    f"[Epoch {epoch + 1}, Step {global_step}] "
                    f"loss: {loss_meter.avg:.4f}, lr: {lr:.2e}, momentum: {m:.4f}"
                )
                wb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": lr,
                        "train/momentum": m,
                        "train/step": global_step,
                        "stats/target_features_norm": target_features.norm(dim=1)
                        .mean()
                        .item(),
                        "stats/predicted_features_norm": predicted_features.norm(dim=1)
                        .mean()
                        .item(),
                        "stats/cosine_similarity": F.cosine_similarity(
                            predicted_features, target_features
                        )
                        .mean()
                        .item(),
                    }
                )
                if monitor is not None:
                    monitor.log_training_examples(
                        epoch=epoch,
                        batch_texts=batch.context_texts,
                        target_texts=batch.target_texts,
                        predicted_features=predicted_features.detach(),
                        target_features=target_features.detach(),
                        encoder=unwrap(encoder),
                        predictor=unwrap(predictor),
                    )
                    monitor.log_validation_metrics(
                        epoch=epoch,
                        pred_embeddings=predicted_features.detach(),
                        target_embeddings=target_features.detach(),
                    )

            if global_step >= total_steps:
                break

        # End of epoch
        if val_loader is not None and dist_info.is_main:
            val_metrics = _run_validation(
                encoder=unwrap(encoder),
                predictor=unwrap(predictor),
                target_encoder=target_encoder,
                val_loader=val_loader,
                config=config,
                device=device,
                autocast_ctx=autocast_ctx,
            )
            print(f"[Epoch {epoch + 1}] Validation:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            wb.log(
                {f"val/{k}": v for k, v in val_metrics.items()}
                | {"epoch": epoch + 1}
            )

        if dist_info.is_main and ((epoch + 1) % config.logging.checkpoint_freq == 0):
            ckpt_path = run_dir / f"checkpoint-epoch{epoch + 1}.pth"
            save_checkpoint(
                str(ckpt_path),
                unwrap(encoder),
                unwrap(predictor),
                target_encoder.target,
                optimizer,
                scaler,
                epoch + 1,
                global_step,
                loss_meter.avg,
            )

        if dist_info.is_main:
            wb.log(
                {
                    "epoch/loss": loss_meter.avg,
                    "epoch/time": time.time() - epoch_start,
                    "epoch/number": epoch + 1,
                }
            )

        if global_step >= total_steps:
            break

    if dist_info.is_main:
        print("Training completed successfully.")
        wb.finish()
