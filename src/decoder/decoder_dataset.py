import random
from dataclasses import dataclass

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer


@dataclass
class DecoderBatch:
    """Batched data for decoder training."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_texts: list[str]  # Original texts, kept for eval/sampling


class DecoderDataset(Dataset):
    """A thin Dataset over a list of text strings, tokenized on collate.

    Keep this dataset only for texts (not context/target pairs). The decoder
    task is: given concept(text), reconstruct text. Pass in the sentences or
    documents you want the decoder to be able to reconstruct.
    """

    def __init__(
        self, texts: list[str], tokenizer: PreTrainedTokenizer, max_length: int = 128
    ):
        if not all(isinstance(t, str) for t in texts):
            raise TypeError(
                "DecoderDataset expects list[str]; got a non-string item. "
                "If starting from TextDataset.samples, extract .target (or "
                ".context) strings before passing."
            )
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def collate_fn(self, batch: list[str]) -> DecoderBatch:
        encodings = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return DecoderBatch(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            input_texts=batch,
        )


def make_loader(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = DecoderDataset(texts, tokenizer, max_length=max_length)
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )


def split_train_eval(
    texts: list[str],
    eval_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Deterministic train/eval split by a seeded shuffle. Returns (train, eval)."""
    texts = list(texts)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(texts)
    split_idx = int(len(texts) * (1 - eval_ratio))
    return texts[:split_idx], texts[split_idx:]
