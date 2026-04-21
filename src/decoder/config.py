from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DecoderArchConfig(BaseModel):
    """Architecture of the concept decoder."""

    hidden_dim: int = Field(gt=0, description="Internal decoder hidden size")
    num_layers: int = Field(default=4, gt=0)
    num_heads: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_length: int = Field(default=128, gt=0)


class DecoderTrainingHParams(BaseModel):
    """Training hyperparameters for the decoder."""

    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    num_epochs: int = Field(default=10, gt=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    grad_clip: float = Field(default=1.0, ge=0.0)
    grad_accum_steps: int = Field(default=1, gt=0)
    eval_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    output_dir: str = Field(default="outputs/decoder")


class DecoderEvalParams(BaseModel):
    """Evaluation cadence / sample counts for the decoder."""

    eval_steps: int = Field(default=100, gt=0)
    save_steps: int = Field(default=1000, gt=0)
    num_samples: int = Field(default=5, gt=0)


class DecoderFullConfig(BaseModel):
    """Everything needed to train the decoder, loaded from a single yaml."""

    decoder: DecoderArchConfig
    training: DecoderTrainingHParams = Field(default_factory=DecoderTrainingHParams)
    evaluation: DecoderEvalParams = Field(default_factory=DecoderEvalParams)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "DecoderFullConfig":
        with open(yaml_path) as f:
            return cls(**yaml.safe_load(f))
