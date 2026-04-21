from dataclasses import dataclass

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from src.common.datasets.sentences import Sentence, is_val_doc, locate_sentences
from src.common.datasets.utils.sentence_splitting import (
    SentenceSplitter,
    SentenceSplitterConfig,
)


@dataclass
class DatasetOutput:
    """A single (context, target) training pair: N context sentences → next sentence."""

    context: str
    target: str


@dataclass
class DatasetStats:
    total_docs: int = 0
    docs_processed: int = 0
    docs_rejected_length: int = 0
    docs_rejected_sentences: int = 0
    context_target_pairs: int = 0
    pairs_rejected_length: int = 0
    docs_val: int = 0
    pairs_val: int = 0


class _SampleList(Dataset):
    """Lightweight Dataset over a pre-built list of DatasetOutput samples."""

    def __init__(self, samples: list[DatasetOutput]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetOutput:
        return self.samples[idx]


class TextDataset(Dataset):
    """FineWeb-Edu → (context, next-sentence) pair dataset, materialized in memory.

    Streams documents, splits into sentences via wtpsplit, builds sliding-window
    (context, target) pairs. Supports a deterministic, hash-based document-level
    train/val split: pairs from the same document never cross splits, so there
    is no leakage from validation into training.

    Call `.val_view()` to get the held-out portion as a Dataset. For large-scale
    training that doesn't fit in memory, use `StreamingTextDataset` instead.
    """

    def __init__(
        self,
        *,
        train_file: str,
        limit: int | None,
        min_length: int,
        window_size: int = 8,
        min_sentences: int = 2,
        val_fraction: float = 0.0,
        tokenizer: PreTrainedTokenizer | None = None,
        max_tokens: int | None = None,
        cache_dir: str = "~/.cache/huggingface/datasets",
    ):
        self.samples: list[DatasetOutput] = []
        self.val_samples: list[DatasetOutput] = []
        self.stats = DatasetStats()

        print(
            f"Loading FineWeb-Edu ({train_file}) with {window_size}-sentence "
            f"context window, val_fraction={val_fraction}"
        )
        ds = load_dataset(
            path="HuggingFaceFW/fineweb-edu",
            name=train_file,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )

        splitter = SentenceSplitter(SentenceSplitterConfig())
        pbar = tqdm(total=limit, desc="Processing documents", unit="docs")

        for doc in ds:
            self.stats.total_docs += 1
            text = doc.get("text", "").strip()

            if len(text) < min_length:
                self.stats.docs_rejected_length += 1
                continue

            try:
                sentences = splitter([text])[0]
                if len(sentences) < min_sentences:
                    self.stats.docs_rejected_sentences += 1
                    continue

                sentence_objs = locate_sentences(text, sentences)
                pairs = self._build_pairs(
                    text, sentence_objs, window_size, tokenizer, max_tokens
                )

                if is_val_doc(text, val_fraction):
                    self.val_samples.extend(pairs)
                    self.stats.docs_val += 1
                    self.stats.pairs_val += len(pairs)
                else:
                    self.samples.extend(pairs)

                self.stats.context_target_pairs += len(pairs)
                self.stats.docs_processed += 1
                pbar.update(1)

                if limit and self.stats.docs_processed >= limit:
                    break

            except Exception as e:
                print(f"Error processing document: {e}")
                continue

        pbar.close()
        self._print_stats()

        if not self.samples:
            raise RuntimeError(
                f"No train samples found in {train_file}. "
                f"Consider lowering min_length ({min_length}) or min_sentences "
                f"({min_sentences}), or raising val_fraction < 1."
            )

    def _build_pairs(
        self,
        text: str,
        sentence_objs: list[Sentence],
        window_size: int,
        tokenizer: PreTrainedTokenizer | None,
        max_tokens: int | None,
    ) -> list[DatasetOutput]:
        pairs: list[DatasetOutput] = []
        for i in range(1, len(sentence_objs)):
            start = max(0, i - window_size)
            context = text[
                sentence_objs[start].start_idx : sentence_objs[i - 1].end_idx
            ]
            target = text[sentence_objs[i].start_idx : sentence_objs[i].end_idx]

            if tokenizer and max_tokens and len(tokenizer.encode(context)) > max_tokens:
                self.stats.pairs_rejected_length += 1
                continue

            pairs.append(DatasetOutput(context=context, target=target))
        return pairs

    def _print_stats(self) -> None:
        s = self.stats
        print("\nDataset Processing Statistics:")
        print(f"  total docs seen             : {s.total_docs:,}")
        print(f"  docs processed              : {s.docs_processed:,}")
        print(f"  docs rejected (length)      : {s.docs_rejected_length:,}")
        print(f"  docs rejected (sentences)   : {s.docs_rejected_sentences:,}")
        print(f"  context/target pairs total  : {s.context_target_pairs:,}")
        print(f"  pairs rejected (length)     : {s.pairs_rejected_length:,}")
        print(f"  docs in val split           : {s.docs_val:,}")
        print(f"  pairs in val split          : {s.pairs_val:,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetOutput:
        return self.samples[idx]

    def val_view(self) -> Dataset:
        """Dataset view over the held-out val split (possibly empty)."""
        return _SampleList(self.val_samples)


def worker_init_fn(worker_id: int) -> None:
    """Hook kept for DataLoader compatibility; samples are built in __init__."""
    pass
