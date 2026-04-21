from dataclasses import dataclass

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
from transformers import PreTrainedTokenizer

from src.decoder.concept_extractor import ConceptExtractor
from src.decoder.models import ConceptDecoder


@dataclass
class DecoderMetrics:
    """Holds evaluation metrics for concept decoder."""

    bleu: float
    rouge: dict[str, float]
    perplexity: float
    concept_cosine_sim: float
    diversity: float


class ConceptMetrics:
    """Evaluates concept decoder performance.

    Uses the canonical ConceptExtractor (masked-mean pooled + L2-normalized
    encoder features) so concept_cosine_sim measures the right thing.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        self._smooth = SmoothingFunction().method1

    @torch.no_grad()
    def compute_metrics(
        self,
        extractor: ConceptExtractor,
        decoder: ConceptDecoder,
        original_texts: list[str],
        generated_texts: list[str],
    ) -> DecoderMetrics:
        """Compute all metrics comparing generated text to originals."""
        refs = [[t.split()] for t in original_texts]
        hyps = [t.split() for t in generated_texts]
        bleu = corpus_bleu(refs, hyps, smoothing_function=self._smooth)

        rouge_scores = {
            name: sum(
                self.rouge.score(orig, gen)[name].fmeasure
                for orig, gen in zip(original_texts, generated_texts, strict=False)
            )
            / max(len(original_texts), 1)
            for name in ["rouge1", "rouge2", "rougeL"]
        }

        perplexity = self._compute_perplexity(extractor, decoder, original_texts)

        orig_concepts = self._concepts(extractor, original_texts)
        gen_concepts = self._concepts(extractor, generated_texts)
        concept_sim = F.cosine_similarity(orig_concepts, gen_concepts).mean().item()

        diversity = self._compute_diversity(generated_texts)

        return DecoderMetrics(
            bleu=bleu,
            rouge=rouge_scores,
            perplexity=perplexity,
            concept_cosine_sim=concept_sim,
            diversity=diversity,
        )

    def _concepts(
        self, extractor: ConceptExtractor, texts: list[str]
    ) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        return extractor(enc["input_ids"], enc["attention_mask"])

    @torch.no_grad()
    def _compute_perplexity(
        self,
        extractor: ConceptExtractor,
        decoder: ConceptDecoder,
        texts: list[str],
    ) -> float:
        """Teacher-forced perplexity: exp(mean CE) of decoder predicting texts."""
        if not texts:
            return float("nan")
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        input_ids = enc["input_ids"]
        concepts = extractor(input_ids, enc["attention_mask"])
        logits = decoder(concepts, target_ids=input_ids)  # [B, L-1, V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=decoder.config.pad_token_id,
        )
        return float(torch.exp(loss).item())

    def _compute_diversity(self, texts: list[str]) -> float:
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        if not all_words:
            return 0.0
        return len(set(all_words)) / len(all_words)


class SampleGenerator:
    """Generates and displays sample decoder outputs."""

    def __init__(
        self,
        extractor: ConceptExtractor,
        decoder: ConceptDecoder,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int = 128,
    ):
        self.extractor = extractor
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self._smooth = SmoothingFunction().method1

    @torch.no_grad()
    def generate_samples(
        self, texts: list[str], num_samples: int = 3
    ) -> list[dict[str, str]]:
        """Generate samples for visualization from a list of source texts."""
        samples: list[dict] = []

        for text in texts[:num_samples]:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            concept = self.extractor(inputs["input_ids"], inputs["attention_mask"])
            generated = self.decoder.generate(concept, self.tokenizer)[0]

            samples.append(
                {
                    "original": text,
                    "generated": generated,
                    "bleu": sentence_bleu(
                        [text.split()],
                        generated.split(),
                        smoothing_function=self._smooth,
                    ),
                }
            )

        return samples


def format_metrics(metrics: DecoderMetrics) -> str:
    return (
        f"BLEU: {metrics.bleu:.4f}\n"
        f"ROUGE-1: {metrics.rouge['rouge1']:.4f}\n"
        f"ROUGE-2: {metrics.rouge['rouge2']:.4f}\n"
        f"ROUGE-L: {metrics.rouge['rougeL']:.4f}\n"
        f"Perplexity: {metrics.perplexity:.2f}\n"
        f"Concept Similarity: {metrics.concept_cosine_sim:.4f}\n"
        f"Diversity: {metrics.diversity:.4f}"
    )
