import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.common.pooling import masked_mean
from src.encoder.models import TextTransformer


class ConceptExtractor(nn.Module):
    """Canonical text → concept mapping used by decoder training and evaluation.

    This is the same representation the encoder's target path produces during
    Phase 1 training: masked-mean over tokens, then L2 normalization. Keeping
    this in one place guarantees the decoder inverts *exactly* the map the
    encoder was optimized toward.

    Frozen by default (requires_grad=False, eval mode, no_grad forward).
    """

    def __init__(self, encoder: TextTransformer, normalize: bool = True):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    @torch.no_grad()
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return pooled, optionally normalized concept vectors [batch, embed_dim]."""
        hidden = self.encoder(input_ids, attention_mask)
        pooled = masked_mean(hidden, attention_mask)
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def train(self, mode: bool = True):
        # Always eval; dropout would inject noise into concept targets.
        return super().train(False)
