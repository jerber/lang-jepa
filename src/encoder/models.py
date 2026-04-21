import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel

from src.common.config import LANGJEPAConfig


class TextTransformer(nn.Module):
    """Text encoder wrapping a HuggingFace AutoModel.

    Two modes:
      * pretrained=True: AutoModel.from_pretrained(tokenizer_path). Architecture
        and dims come from the pretrained checkpoint; config.model.* is ignored
        for encoder shape (we still surface embed_dim via .embed_dim). Dropout
        override is applied.
      * pretrained=False: AutoModel.from_config with overridden dims. Fresh
        random init; needs large-scale training to be useful.
    """

    def __init__(self, config: LANGJEPAConfig):
        super().__init__()

        if config.model.pretrained:
            self.encoder = AutoModel.from_pretrained(config.data.tokenizer_path)
            # Override dropout if requested; keep all other dims as pretrained.
            if hasattr(self.encoder.config, "hidden_dropout_prob"):
                self.encoder.config.hidden_dropout_prob = config.model.dropout
            if hasattr(self.encoder.config, "attention_probs_dropout_prob"):
                self.encoder.config.attention_probs_dropout_prob = config.model.dropout
            self.embed_dim = self.encoder.config.hidden_size
        else:
            model_config = AutoConfig.from_pretrained(config.data.tokenizer_path)
            model_config.update(
                {
                    "hidden_size": config.model.embed_dim,
                    "num_hidden_layers": config.model.num_layers,
                    "num_attention_heads": config.model.num_heads,
                    "intermediate_size": int(
                        config.model.embed_dim * config.model.mlp_ratio
                    ),
                    "hidden_dropout_prob": config.model.dropout,
                    "attention_probs_dropout_prob": config.model.dropout,
                    "vocab_size": len(config.data.tokenizer),
                }
            )
            self.encoder = AutoModel.from_config(model_config)
            self.embed_dim = config.model.embed_dim

        if config.meta.use_gradient_checkpointing and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Return token-level last_hidden_state: [batch, seq_len, embed_dim]."""
        outputs = self.encoder(input_ids, attention_mask, return_dict=True)
        return outputs.last_hidden_state


class TextPredictor(nn.Module):
    """Predicts the next-sentence embedding from a context sequence.

    Aggregates context tokens via multi-head attention with a learnable query,
    then projects to pred_dim. Output is NOT normalized here — the train loop
    owns normalization to keep the geometry in one place.

    If pred_dim == input_dim, the projection is a LayerNorm only (no linear mix),
    so predictions live in the same space as target pooled features (I-JEPA-style).
    """

    def __init__(self, input_dim: int, pred_dim: int, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.pred_dim = pred_dim

        self.context_attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        if pred_dim == input_dim:
            self.projection: nn.Module = nn.LayerNorm(pred_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, pred_dim), nn.LayerNorm(pred_dim)
            )

    def forward(
        self, context_feats: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        query = self.query.expand(context_feats.size(0), -1, -1)
        key_padding_mask = (
            ~attention_mask.bool() if attention_mask is not None else None
        )
        context, _ = self.context_attention(
            query=query,
            key=context_feats,
            value=context_feats,
            key_padding_mask=key_padding_mask,
        )
        return self.projection(context.squeeze(1))
