from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.distributed import DistInfo, setup_distributed, unwrap
from src.common.logging import AverageMeter
from src.decoder.concept_extractor import ConceptExtractor
from src.decoder.config import DecoderFullConfig
from src.decoder.decoder_dataset import DecoderBatch
from src.decoder.models import ConceptDecoder
from src.decoder.utils.evaluation import ConceptMetrics, SampleGenerator, format_metrics


class DecoderTrainer:
    """Trains a ConceptDecoder given a frozen ConceptExtractor.

    Supports single-GPU, CPU, and DDP (detected via LOCAL_RANK). The extractor
    is always per-rank (frozen, no gradients) and the decoder gets wrapped in
    DDP when world_size > 1.
    """

    def __init__(
        self,
        config: DecoderFullConfig,
        extractor: ConceptExtractor,
        decoder: ConceptDecoder,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        dist_info: DistInfo | None = None,
    ):
        self.config = config
        self.training = config.training
        self.evaluation = config.evaluation
        self.dist_info = dist_info or setup_distributed()
        self.device = self.dist_info.device

        self.extractor = extractor.to(self.device)
        self._decoder_mod = decoder.to(self.device)
        if self.dist_info.is_distributed and torch.cuda.is_available():
            self.decoder: torch.nn.Module = DDP(
                self._decoder_mod, device_ids=[self.dist_info.local_rank]
            )
        else:
            self.decoder = self._decoder_mod

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.optimizer = AdamW(
            self.decoder.parameters(),
            lr=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
        )

        self.metrics = ConceptMetrics(
            self._decoder_mod.tokenizer,
            self.device,
            max_length=self._decoder_mod.config.max_length,
        )
        self.sample_generator = SampleGenerator(
            self.extractor,
            self._decoder_mod,
            self._decoder_mod.tokenizer,
            self.device,
            max_length=self._decoder_mod.config.max_length,
        )

        self.output_dir = Path(self.training.output_dir)
        if self.dist_info.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _process_batch(self, batch: DecoderBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return batch.input_ids.to(self.device), batch.attention_mask.to(self.device)

    def _forward_loss(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        concepts = self.extractor(input_ids, attention_mask)
        logits = self.decoder(concepts, target_ids=input_ids)  # [B, L-1, V]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=self._decoder_mod.config.pad_token_id,
        )

    def train(self) -> None:
        best_loss = float("inf")
        global_step = 0
        forward_count = 0
        loss_meter = AverageMeter()
        accum = self.training.grad_accum_steps

        for epoch in range(self.training.num_epochs):
            if self.dist_info.is_main:
                print(f"\nEpoch {epoch + 1}/{self.training.num_epochs}")
            self.decoder.train()
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            iterator = (
                tqdm(self.train_loader, desc="Training")
                if self.dist_info.is_main
                else self.train_loader
            )
            for batch in iterator:
                input_ids, attention_mask = self._process_batch(batch)
                is_boundary = ((forward_count + 1) % accum) == 0
                sync_ctx = (
                    self.decoder.no_sync()
                    if isinstance(self.decoder, DDP) and not is_boundary
                    else nullcontext()
                )

                with sync_ctx:
                    loss = self._forward_loss(input_ids, attention_mask)
                    (loss / accum).backward()
                forward_count += 1
                loss_meter.update(loss.item())

                if not is_boundary:
                    continue

                if self.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.decoder.parameters(), self.training.grad_clip
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if (
                    self.dist_info.is_main
                    and global_step % self.evaluation.eval_steps == 0
                ):
                    eval_loss = self.evaluate()
                    print(f"\nStep {global_step} - Eval loss: {eval_loss:.4f}")
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        self.save_checkpoint(
                            self.output_dir / "best_decoder.pt",
                            global_step,
                            best_loss,
                        )
                    if self.eval_loader is not None:
                        eval_batch = next(iter(self.eval_loader))
                        samples = self.sample_generator.generate_samples(
                            eval_batch.input_texts,
                            num_samples=self.evaluation.num_samples,
                        )
                        self._print_samples(samples)
                        extended = self.metrics.compute_metrics(
                            extractor=self.extractor,
                            decoder=self._decoder_mod,
                            original_texts=[s["original"] for s in samples],
                            generated_texts=[s["generated"] for s in samples],
                        )
                        print("\nExtended metrics (on a sample):")
                        print(format_metrics(extended))

                if (
                    self.dist_info.is_main
                    and global_step % self.evaluation.save_steps == 0
                ):
                    self.save_checkpoint(
                        self.output_dir / f"decoder_step_{global_step}.pt",
                        global_step,
                        loss_meter.avg,
                    )

                if self.dist_info.is_main and global_step % 10 == 0:
                    print(f"Step {global_step} - Loss: {loss_meter.avg:.4f}")

            if self.dist_info.is_main:
                print(f"Epoch {epoch + 1} finished. Avg loss: {loss_meter.avg:.4f}")
            loss_meter.reset()

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.eval_loader is None:
            return float("inf")

        self.decoder.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.eval_loader:
            input_ids, attention_mask = self._process_batch(batch)
            total_loss += self._forward_loss(input_ids, attention_mask).item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.decoder.train()
        return avg_loss

    def save_checkpoint(self, path: Path, global_step: int, loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": global_step,
                "model_state_dict": unwrap(self.decoder).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
                "config": self._decoder_mod.config,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        unwrap(self.decoder).load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path}")

    def _print_samples(self, samples: list) -> None:
        print("\nGenerated Samples:")
        print("-" * 50)
        for i, sample in enumerate(samples, 1):
            print(f"Sample {i}:")
            print(f"Original : {sample['original']}")
            print(f"Generated: {sample['generated']}")
            print(f"BLEU     : {sample['bleu']:.4f}")
            print()
