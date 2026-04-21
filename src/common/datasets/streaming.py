"""Streaming variant of TextDataset for large-scale / multi-GPU training.

The non-streaming `TextDataset` materializes all pairs into memory — fine for a
few thousand docs, untenable for the sizes DDP benefits from. This module
provides `StreamingTextDataset`, an IterableDataset that streams FineWeb-Edu,
splits into sentences inside each DataLoader worker, and yields (context,
target) pairs on demand.

Sharding semantics:
  * Across DDP ranks (via `rank` / `world_size`): each rank sees a disjoint
    slice so that the union across ranks is the original stream.
  * Within a rank, across DataLoader workers (via worker_init_fn): each
    worker further subdivides the rank's slice so workers don't duplicate.

Sentence splitting runs inside each worker with device='cpu' by default to
avoid competing with the main training process for GPU memory.
"""
from __future__ import annotations

from collections.abc import Iterator

from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info

from src.common.datasets.fineweb_edu import DatasetOutput
from src.common.datasets.sentences import Sentence, is_val_doc, locate_sentences
from src.common.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        *,
        train_file: str,
        min_length: int,
        window_size: int = 8,
        min_sentences: int = 2,
        val_fraction: float = 0.0,
        split: str = "train",
        shuffle_buffer: int = 10_000,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        splitter_device: str = "cpu",
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.train_file = train_file
        self.min_length = min_length
        self.window_size = window_size
        self.min_sentences = min_sentences
        self.val_fraction = val_fraction
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.splitter_device = splitter_device
        self.cache_dir = cache_dir

    def _build_ds(self):
        ds = load_dataset(
            path="HuggingFaceFW/fineweb-edu",
            name=self.train_file,
            split="train",
            streaming=True,
            cache_dir=self.cache_dir,
        )
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + self.rank)
        return ds

    def __iter__(self) -> Iterator[DatasetOutput]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        total_shards = max(self.world_size, 1) * max(num_workers, 1)
        shard_idx = self.rank * max(num_workers, 1) + worker_id

        ds = self._build_ds()
        splitter = SentenceSplitter(SentenceSplitterConfig(device=self.splitter_device))

        for global_idx, doc in enumerate(ds):
            if total_shards > 1 and (global_idx % total_shards) != shard_idx:
                continue

            text = (doc.get("text") or "").strip()
            if len(text) < self.min_length:
                continue

            if (self.split == "val") != is_val_doc(text, self.val_fraction):
                continue

            try:
                sentences = splitter([text])[0]
            except Exception:
                continue
            if len(sentences) < self.min_sentences:
                continue

            sentence_objs = locate_sentences(text, sentences)
            yield from self._build_pairs(text, sentence_objs)

    def _build_pairs(
        self, text: str, sentence_objs: list[Sentence]
    ) -> Iterator[DatasetOutput]:
        for i in range(1, len(sentence_objs)):
            start_sent_idx = max(0, i - self.window_size)
            context = text[
                sentence_objs[start_sent_idx].start_idx : sentence_objs[i - 1].end_idx
            ]
            target = text[sentence_objs[i].start_idx : sentence_objs[i].end_idx]
            yield DatasetOutput(context=context, target=target)
