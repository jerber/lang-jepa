import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from torch import Tensor
from transformers import PreTrainedTokenizer

import wandb

"""
These metrics will help you understand:

Semantic Similarity (avg_similarity): How close your predictions are to the actual next sentence embeddings
Hit Rate: Whether your model can distinguish the true next sentence from other sentences in the batch
Embedding Norms: If your embeddings are maintaining reasonable magnitudes
Embedding Diversity: If your embeddings are maintaining good separation or collapsing

Low hit rate but high similarity might indicate your model is making "safe" but overly general predictions. Low diversity might indicate embedding collapse. You can use these insights to tune your architecture or training process.
"""


@dataclass
class MonitoringExample:
    """Holds information for a single monitoring example."""

    context_text: str
    target_text: str
    predicted_embedding: Tensor
    target_embedding: Tensor
    similarity_score: float


def _effective_rank(centered: Tensor) -> float:
    """Effective rank = exp(entropy of normalized singular values).

    Measures dimensional collapse: full rank → eff_rank = min(N, D); total
    collapse → eff_rank = 1. Works on centered [N, D] batches.
    """
    if centered.numel() == 0 or centered.shape[0] < 2:
        return 0.0
    # Float32 for stable SVD, even if the batch is bf16.
    svals = torch.linalg.svdvals(centered.float())
    svals = svals[svals > 1e-10]
    if svals.numel() == 0:
        return 0.0
    p = svals / svals.sum()
    entropy = -(p * p.log()).sum()
    return float(entropy.exp().item())


def _off_diag_cov_l1(centered: Tensor) -> float:
    """Sum of abs off-diagonal entries of (centered^T @ centered) / N."""
    n = centered.shape[0]
    if n < 2:
        return 0.0
    cov = (centered.float().T @ centered.float()) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return float(off.abs().sum().item())


class ValidationMetrics:
    """Tracks validation metrics for JEPA model evaluation.

    Includes anti-collapse diagnostics on the target batch: per-dim stddev,
    effective rank, and off-diagonal covariance L1. Healthy training keeps
    stddev above ~0.1, eff_rank comfortably > 1, and covariance bounded.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metric counters."""
        self.total_samples = 0
        self.metrics = {
            "semantic/avg_similarity": 0.0,
            "semantic/hit_rate": 0.0,
            "embeddings/norm": 0.0,
            "embeddings/diversity": 0.0,
            "embeddings/target_std_mean": 0.0,
            "embeddings/target_eff_rank": 0.0,
            "embeddings/target_off_diag_cov_l1": 0.0,
        }

    def update(self, pred_embeddings: Tensor, target_embeddings: Tensor):
        """Update metrics with new batch of embeddings."""
        batch_size = pred_embeddings.shape[0]

        # Compute cosine similarity
        sim_scores = F.cosine_similarity(pred_embeddings, target_embeddings)
        avg_sim = sim_scores.mean().item()

        # Compute contrastive accuracy (hit rate)
        sim_matrix = torch.matmul(pred_embeddings, target_embeddings.T)
        correct_matches = sim_matrix.argmax(dim=-1) == torch.arange(
            len(pred_embeddings), device=pred_embeddings.device
        )
        hit_rate = correct_matches.float().mean().item()

        # Compute embedding norms
        norms = torch.norm(pred_embeddings, dim=-1).mean().item()

        # Compute embedding diversity
        cosine_sim_matrix = torch.matmul(pred_embeddings, pred_embeddings.T)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=pred_embeddings.device)
        diversity = 1 - cosine_sim_matrix[mask].mean().item()

        # Anti-collapse diagnostics computed on the target batch
        with torch.no_grad():
            centered = target_embeddings - target_embeddings.mean(dim=0, keepdim=True)
            target_std_mean = float(target_embeddings.std(dim=0).mean().item())
            target_eff_rank = _effective_rank(centered)
            off_diag_l1 = _off_diag_cov_l1(centered)

        # Update running averages
        weight = batch_size / (self.total_samples + batch_size)
        old_weight = 1 - weight

        self.metrics["semantic/avg_similarity"] = (
            old_weight * self.metrics["semantic/avg_similarity"] + weight * avg_sim
        )
        self.metrics["semantic/hit_rate"] = (
            old_weight * self.metrics["semantic/hit_rate"] + weight * hit_rate
        )
        self.metrics["embeddings/norm"] = (
            old_weight * self.metrics["embeddings/norm"] + weight * norms
        )
        self.metrics["embeddings/diversity"] = (
            old_weight * self.metrics["embeddings/diversity"] + weight * diversity
        )
        self.metrics["embeddings/target_std_mean"] = (
            old_weight * self.metrics["embeddings/target_std_mean"]
            + weight * target_std_mean
        )
        self.metrics["embeddings/target_eff_rank"] = (
            old_weight * self.metrics["embeddings/target_eff_rank"]
            + weight * target_eff_rank
        )
        self.metrics["embeddings/target_off_diag_cov_l1"] = (
            old_weight * self.metrics["embeddings/target_off_diag_cov_l1"]
            + weight * off_diag_l1
        )

        self.total_samples += batch_size

    def get_metrics(self) -> dict[str, float]:
        """Return current metrics."""
        return self.metrics.copy()


class TrainingMonitor:
    """Monitors and logs training progress for JEPA model."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        log_dir: Path = Path("logs/monitor_logs"),
        num_examples: int = 3,
        log_every_n_epochs: int = 1,
        log_to_wandb: bool = True,
    ):
        self.tokenizer = tokenizer
        self.log_dir = Path(log_dir)
        self.num_examples = num_examples
        self.log_every_n_epochs = log_every_n_epochs
        self.log_to_wandb = log_to_wandb
        self.console = Console()
        self.validation_metrics = ValidationMetrics()

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up loggers
        self._setup_loggers()

    def _setup_loggers(self):
        """Initialize different loggers for console and file output."""
        # Console logger
        self.console_logger = logging.getLogger("console")
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)
        self.console_logger.setLevel(logging.INFO)

        # File logger for training examples
        self.file_logger = logging.getLogger("training_examples")
        file_handler = logging.FileHandler(self.log_dir / "training_examples.log")
        file_formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        self.file_logger.setLevel(logging.INFO)

    def log_training_examples(
        self,
        epoch: int,
        batch_texts: list[str],
        target_texts: list[str],
        predicted_features: Tensor,
        target_features: Tensor,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
    ) -> None:
        """Log training examples showing next-sentence prediction performance."""
        if epoch % self.log_every_n_epochs != 0:
            return

        examples = []
        for idx in range(min(self.num_examples, len(batch_texts))):
            # Calculate cosine similarity for this example
            similarity = F.cosine_similarity(
                predicted_features[idx : idx + 1], target_features[idx : idx + 1], dim=1
            ).item()

            example = MonitoringExample(
                context_text=batch_texts[idx],
                target_text=target_texts[idx],
                predicted_embedding=predicted_features[idx].cpu(),
                target_embedding=target_features[idx].cpu(),
                similarity_score=similarity,
            )
            examples.append(example)

        self._display_examples(epoch, examples)
        if self.log_to_wandb:
            self._log_to_wandb(epoch, examples)

    def _display_examples(self, epoch: int, examples: list[MonitoringExample]) -> None:
        """Display training examples in a formatted table."""
        self.file_logger.info(f"\n=== Training Examples (Epoch {epoch}) ===")

        for i, example in enumerate(examples, 1):
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )
            table.add_column("Type", style="cyan", width=20)
            table.add_column("Content", style="green")

            # Format context text for display
            escaped_context = escape(example.context_text)
            context_chunks = [
                escaped_context[i : i + 100]
                for i in range(0, len(escaped_context), 100)
            ]
            table.add_row("Context", "\n".join(context_chunks))

            # Format target text
            escaped_target = escape(example.target_text)
            target_chunks = [
                escaped_target[i : i + 100] for i in range(0, len(escaped_target), 100)
            ]
            table.add_row("Target (Next Sentence)", "\n".join(target_chunks))

            # Add similarity score
            table.add_row("Cosine Similarity", f"{example.similarity_score:.4f}")

            # Add embedding statistics
            pred_norm = example.predicted_embedding.norm().item()
            target_norm = example.target_embedding.norm().item()
            table.add_row(
                "Embedding Stats",
                f"Predicted norm: {pred_norm:.4f}\nTarget norm: {target_norm:.4f}",
            )

            # Print the table to console and log to file
            panel = Panel(table, title=f"Example {i}", border_style="blue")
            self.file_logger.info(panel)
            self.file_logger.info(f"\nExample {i}:")
            self.file_logger.info(f"Context: {example.context_text}")
            self.file_logger.info(f"Target: {example.target_text}")
            self.file_logger.info(f"Similarity: {example.similarity_score:.4f}")
            self.file_logger.info("-" * 80)

    def _log_to_wandb(self, epoch: int, examples: list[MonitoringExample]) -> None:
        """Log examples and metrics to Weights & Biases."""
        # Log examples
        for i, example in enumerate(examples):
            wandb.log(
                {
                    f"examples/context_{i}": example.context_text,
                    f"examples/target_{i}": example.target_text,
                    f"examples/similarity_{i}": example.similarity_score,
                    f"examples/pred_norm_{i}": example.predicted_embedding.norm().item(),
                    f"examples/target_norm_{i}": example.target_embedding.norm().item(),
                    "epoch": epoch,
                }
            )

            # Create similarity histogram for this batch
            if i == 0:  # Only do this once per batch
                wandb.log(
                    {
                        "similarity_distribution": wandb.Histogram(
                            [e.similarity_score for e in examples]
                        ),
                        "epoch": epoch,
                    }
                )

    def log_validation_metrics(
        self,
        epoch: int,
        pred_embeddings: Tensor,
        target_embeddings: Tensor,
    ) -> None:
        """Log validation metrics for the current batch."""
        # Update metrics
        self.validation_metrics.update(pred_embeddings, target_embeddings)

        # Get current metrics
        metrics = self.validation_metrics.get_metrics()

        # Log to console with rich table
        self.console.print("\nValidation Metrics:")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for name, value in metrics.items():
            table.add_row(name, f"{value:.4f}")

        self.console.print(table)

        # Log to wandb if enabled
        if self.log_to_wandb:
            wandb.log({f"val/{k}": v for k, v in metrics.items()})
            wandb.log({"epoch": epoch})

        # Log to file
        self.file_logger.info(f"\nValidation Metrics (Epoch {epoch}):")
        for name, value in metrics.items():
            self.file_logger.info(f"{name}: {value:.4f}")

    def get_current_metrics(self) -> dict[str, float]:
        """Return current validation metrics."""
        return self.validation_metrics.get_metrics()

    def reset_validation_metrics(self) -> None:
        """Reset validation metrics for new epoch."""
        self.validation_metrics.reset()
