from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedTokenizer


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""

    train_file: str = Field(description="Dataset file to use for training")
    batch_size: int = Field(gt=0, description="Training batch size")
    num_workers: int = Field(ge=0, description="Number of data loader workers")
    tokenizer_path: str = Field(description="Path or name of the pretrained tokenizer")
    limit: int = Field(gt=0, description="Limit on number of training samples")
    min_length: int = Field(gt=0, description="Minimum text length to consider")
    min_sentences: int = Field(
        gt=1, default=2, description="Minimum number of sentences required"
    )
    window_size: int = Field(
        gt=0, default=8, description="Sentences of context before the target sentence"
    )
    val_fraction: float = Field(
        ge=0.0, lt=1.0, default=0.0,
        description="Fraction of docs held out for validation",
    )
    streaming: bool = Field(
        default=False,
        description="Use StreamingTextDataset (IterableDataset). Required at "
        "cluster scale; max_steps must then be set since len(dataset) is unknown.",
    )
    shuffle_buffer: int = Field(
        ge=0, default=10_000,
        description="Streaming shuffle buffer size (0 disables shuffle).",
    )
    splitter_device: str = Field(
        default="cpu",
        description="Device for the sentence splitter in workers. 'cpu' avoids "
        "competing with training for GPU memory.",
    )
    tokenizer: PreTrainedTokenizer | None = Field(
        default=None, description="Loaded tokenizer instance"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    max_length: int = Field(gt=0, description="Maximum sequence length")
    pred_dim: int = Field(gt=0, description="Prediction dimension")
    embed_dim: int = Field(gt=0, description="Embedding dimension")
    num_layers: int = Field(gt=0, description="Number of transformer layers")
    num_heads: int = Field(gt=0, description="Number of attention heads")
    mlp_ratio: float = Field(gt=0.0, description="MLP hidden dimension ratio")
    dropout: float = Field(ge=0.0, lt=1.0, description="Dropout rate")
    pretrained: bool = Field(
        default=True,
        description="If True, initialize encoder from AutoModel.from_pretrained; "
        "if False, build a fresh model from config (requires much more data).",
    )


class OptimizationConfig(BaseModel):
    """Configuration for training optimization."""

    epochs: int = Field(gt=0, description="Number of training epochs")
    lr: float = Field(gt=0.0, description="Learning rate")
    warmup: int = Field(ge=0, description="Number of warmup epochs")
    weight_decay: float = Field(ge=0.0, description="Weight decay")
    final_weight_decay: float = Field(ge=0.0, description="Final weight decay")
    final_lr: float = Field(ge=0.0, description="Final learning rate")
    loss_fn: Literal["smooth_l1", "cosine"] = Field(
        default="smooth_l1",
        description="Training loss. smooth_l1 on normalized features matches I-JEPA.",
    )
    momentum_start: float = Field(
        ge=0.0, le=1.0, default=0.996,
        description="Initial EMA momentum for target encoder",
    )
    momentum_end: float = Field(
        ge=0.0, le=1.0, default=1.0,
        description="Final EMA momentum for target encoder",
    )
    grad_accum_steps: int = Field(
        gt=0, default=1,
        description="Forward passes per optimizer step. Effective batch size = "
        "batch_size * grad_accum_steps * world_size.",
    )
    max_steps: int | None = Field(
        default=None, ge=1,
        description="Total optimizer steps. Required when data.streaming=True; "
        "ignored otherwise (computed from epochs * len(dataloader)).",
    )


class LoggingConfig(BaseModel):
    """Configuration for logging and checkpoints."""

    log_dir: str = Field(description="Directory for logs")
    log_freq: int = Field(
        default=50, gt=0, description="Logging frequency in iterations"
    )
    checkpoint_freq: int = Field(
        default=1, gt=0, description="Checkpoint saving frequency in epochs"
    )
    num_examples: int = Field(
        default=3, gt=0, description="Number of examples to log each interval"
    )
    log_to_wandb: bool = Field(
        default=False, description="If True and WANDB_API_KEY set, log to wandb"
    )


class MetaConfig(BaseModel):
    """Meta configuration for training."""

    use_bfloat16: bool = Field(
        default=False, description="Whether to use bfloat16 precision"
    )
    load_checkpoint: bool = Field(
        default=False, description="Whether to load from checkpoint"
    )
    checkpoint_path: str | None = Field(
        default=None, description="Path to checkpoint file"
    )
    use_gradient_checkpointing: bool = Field(
        default=False, description="Whether to use gradient checkpointing"
    )


class LANGJEPAConfig(BaseModel):
    """Main configuration class combining all sub-configs."""

    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    meta: MetaConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LANGJEPAConfig":
        """Load config from YAML file."""
        import yaml

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        import yaml

        config_dict = self.model_dump()
        # Remove tokenizer since it can't be serialized
        if "tokenizer" in config_dict.get("data", {}):
            del config_dict["data"]["tokenizer"]
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)
