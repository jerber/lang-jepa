"""Regression test for the main_encoder.py pad-token handling.

RoBERTa's own `<pad>` must survive; GPT-2 (no pad token) must fall back to EOS.

Skipped when the tokenizer files aren't available (no network).
"""
from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
AutoTokenizer = transformers.AutoTokenizer


def _apply_main_encoder_rule(tokenizer):
    # Mirror main_encoder.py exactly.
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.mark.network
def test_roberta_pad_is_preserved():
    try:
        tok = AutoTokenizer.from_pretrained("roberta-base")
    except Exception as e:
        pytest.skip(f"roberta-base not available: {e}")
    pad_before = tok.pad_token_id
    eos_before = tok.eos_token_id
    _apply_main_encoder_rule(tok)
    # Crucial: we must not overwrite an existing pad_token with eos.
    assert tok.pad_token_id == pad_before
    assert tok.pad_token_id != eos_before


@pytest.mark.network
def test_gpt2_falls_back_to_eos():
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        pytest.skip(f"gpt2 not available: {e}")
    assert tok.pad_token is None  # baseline
    _apply_main_encoder_rule(tok)
    assert tok.pad_token == tok.eos_token
