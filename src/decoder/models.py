from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


@dataclass
class DecoderConfig:
    """Configuration for the concept decoder."""

    embed_dim: int  # Dimension of concept space (canonical concept dim from encoder)
    hidden_dim: int  # Internal dimension of decoder
    vocab_size: int  # Set from tokenizer
    pad_token_id: int  # Set from tokenizer
    bos_token_id: int  # Set from tokenizer
    eos_token_id: int  # Set from tokenizer
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_length: int = 128

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizer,
        embed_dim: int,
        hidden_dim: int | None = None,
        **kwargs,
    ) -> "DecoderConfig":
        """Create config from tokenizer and embedding dimension."""
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.cls_token_id
        if bos is None:
            bos = tokenizer.pad_token_id
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = tokenizer.sep_token_id
        if eos is None:
            eos = tokenizer.pad_token_id
        return cls(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim or embed_dim * 2,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=bos,
            eos_token_id=eos,
            **kwargs,
        )


def _top_k_top_p_filter(
    logits: torch.Tensor, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    """Return logits with tokens outside the top-k / top-p set masked to -inf."""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = torch.where(
            logits < kth, torch.full_like(logits, float("-inf")), logits
        )
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        # Shift right so the token that CROSSES the threshold is still kept,
        # then unconditionally keep the top token.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )
    return logits


class ConceptDecoder(nn.Module):
    """Decoder for converting concept embeddings back to text.

    Input: concept vectors [B, embed_dim] from the canonical ConceptExtractor
    (masked-mean pooled + L2-normalized from the encoder). Training uses teacher
    forcing; generation is autoregressive with per-sample EOS handling.
    """

    def __init__(self, config: DecoderConfig, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.concept_proj = nn.Linear(config.embed_dim, config.hidden_dim)

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.max_length, config.hidden_dim)
        )

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.out_proj = nn.Linear(config.hidden_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _memory(self, concepts: torch.Tensor) -> torch.Tensor:
        """Project [B, embed_dim] concepts → [B, 1, hidden_dim] memory."""
        if concepts.dim() > 2:
            concepts = concepts.reshape(concepts.shape[0], -1)
        return self.concept_proj(concepts).unsqueeze(1)

    def forward(
        self,
        concepts: torch.Tensor,  # [B, embed_dim]
        target_ids: torch.Tensor,  # [B, L] — required; teacher-forcing only
    ) -> torch.Tensor:
        """Teacher-forcing forward. Returns logits for positions 1..L-1 as [B, L-1, V].

        For generation at inference time, call .generate() instead.
        """
        memory = self._memory(concepts)  # [B, 1, H]

        seq_length = target_ids.size(1) - 1
        tgt_emb = self.token_embedding(target_ids[:, :-1])
        tgt_emb = tgt_emb + self.pos_embedding[:, :seq_length]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_length, device=target_ids.device
        )
        hidden = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.out_proj(hidden)

    @torch.no_grad()
    def generate(
        self,
        concepts: torch.Tensor,
        tokenizer: PreTrainedTokenizer | None = None,
        max_length: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        """Autoregressive generation with per-sample EOS handling.

        When any sample emits EOS, it is marked finished; subsequent tokens for
        that sample are replaced with pad so other samples can keep generating.
        Returns decoded strings (special tokens stripped).
        """
        self.eval()
        tokenizer = tokenizer or self.tokenizer
        max_length = max_length or self.config.max_length

        memory = self._memory(concepts)  # [B, 1, H]
        batch_size = concepts.shape[0]
        device = concepts.device

        curr_ids = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            device=device,
            dtype=torch.long,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            seq_len = curr_ids.size(1)
            if seq_len > self.config.max_length:
                break
            tgt_emb = self.token_embedding(curr_ids) + self.pos_embedding[:, :seq_len]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            hidden = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            step_logits = self.out_proj(hidden[:, -1])  # [B, V]

            if do_sample:
                step_logits = step_logits / max(temperature, 1e-5)
                step_logits = _top_k_top_p_filter(step_logits, top_k, top_p)
                probs = F.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = step_logits.argmax(dim=-1)

            # Freeze finished samples: emit pad instead of their predicted token.
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )
            curr_ids = torch.cat([curr_ids, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == self.config.eos_token_id)

            if bool(finished.all()):
                break

        return tokenizer.batch_decode(curr_ids, skip_special_tokens=True)
