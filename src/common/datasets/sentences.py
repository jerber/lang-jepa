"""Shared utilities for sentence-boundary handling and dataset splitting.

Both the materialized TextDataset and the StreamingTextDataset reuse these —
keeping them here avoids duplicated logic and cross-file private imports.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class Sentence:
    """One sentence within a parent document, with original-text char span."""

    text: str
    start_idx: int
    end_idx: int


def locate_sentences(text: str, sentences: list[str]) -> list[Sentence]:
    """Find each sentence's exact character span in the original text.

    Sentences that can't be located exactly (e.g. if the splitter normalized
    whitespace) are silently skipped rather than crashing the whole document.
    """
    out: list[Sentence] = []
    search_start = 0
    for sent in sentences:
        try:
            start = text.index(sent, search_start)
        except ValueError:
            continue
        end = start + len(sent)
        out.append(Sentence(text=sent, start_idx=start, end_idx=end))
        search_start = end
    return out


def is_val_doc(text: str, val_fraction: float) -> bool:
    """Stable per-document split: hash a prefix of the text into [0, 1000).

    Using hashlib (rather than Python's PYTHONHASHSEED-salted hash) keeps the
    split reproducible across runs, processes, and distributed ranks — which
    is load-bearing when ranks share data.
    """
    if val_fraction <= 0.0:
        return False
    digest = hashlib.sha256(text[:200].encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 1000
    return bucket < int(val_fraction * 1000)
