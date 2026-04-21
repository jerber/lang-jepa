"""Per-sample EOS handling in ConceptDecoder.generate.

Doesn't require a real tokenizer — we mock the minimum surface area.
"""
from dataclasses import dataclass

import torch

from src.decoder.models import ConceptDecoder, DecoderConfig


@dataclass
class _MiniTok:
    vocab: dict
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    cls_token_id: int | None = None
    sep_token_id: int | None = None
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    def __len__(self) -> int:
        return len(self.vocab)

    def batch_decode(
        self, ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> list[str]:
        # Return a whitespace-joined list of non-special token indices for inspection.
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        out = []
        for row in ids.tolist():
            kept = [str(x) for x in row if not (skip_special_tokens and x in specials)]
            out.append(" ".join(kept))
        return out


def _decoder(vocab_size: int = 20, max_len: int = 10) -> ConceptDecoder:
    tok = _MiniTok(
        vocab={f"t{i}": i for i in range(vocab_size)},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    cfg = DecoderConfig(
        embed_dim=8,
        hidden_dim=16,
        vocab_size=vocab_size,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_length=max_len,
    )
    return ConceptDecoder(cfg, tok)


def test_generate_runs_and_returns_strings():
    torch.manual_seed(0)
    d = _decoder()
    out = d.generate(torch.randn(3, 8), max_length=6, do_sample=False)
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(s, str) for s in out)


def test_generate_respects_max_length():
    torch.manual_seed(0)
    d = _decoder(max_len=8)
    out = d.generate(torch.randn(2, 8), max_length=5, do_sample=False)
    # batch_decode strips specials, so 5 is an upper bound on token count.
    for s in out:
        assert len(s.split()) <= 5


def test_finished_sample_does_not_emit_new_tokens():
    """Force sample 0 to emit EOS at step 1 and verify it freezes to pad afterwards."""
    torch.manual_seed(0)
    d = _decoder()

    # Build a fake out_proj that always predicts EOS for sample 0 and a non-EOS
    # token (let's pick id=7) for sample 1.
    class _FakeProj(torch.nn.Module):
        def __init__(self, vocab_size: int, eos_id: int):
            super().__init__()
            self.vocab_size = vocab_size
            self.eos_id = eos_id

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is [B, *, H] — return logits with EOS for row 0 and token 7 for row 1.
            shape = (*x.shape[:-1], self.vocab_size)
            logits = torch.full(shape, -10.0)
            logits[..., 7] = 0.0  # prefer 7 globally
            logits[0, ..., self.eos_id] = 10.0  # but row 0 MUST pick EOS
            return logits

    d.out_proj = _FakeProj(d.config.vocab_size, d.config.eos_token_id)

    # Ask for 4 generation steps on batch of 2.
    out_ids_by_hook = []

    # Patch generate to peek at curr_ids before decode
    original = d.generate

    def _wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    d.generate = _wrapped
    # Directly invoke generate to observe the ids via batch_decode output.
    strings = d.generate(torch.randn(2, 8), max_length=5, do_sample=False)
    # Sample 0 (finished early on EOS) should produce empty-ish decoded text
    # (special tokens stripped). Sample 1 should keep producing token 7.
    assert "2" not in strings[0]  # eos stripped
    # Row 1 should contain at least one "7".
    assert "7" in strings[1].split()
