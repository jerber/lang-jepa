"""Inference / demo script for LANG-JEPA.

Load an encoder checkpoint (+ optional decoder), given a text query produce:
  1. The concept embedding (first few dims + norm).
  2. If --corpus provided, top-k nearest sentences from that corpus.
  3. If --decoder-checkpoint provided, a decoder-reconstructed sentence.

Usage:
  echo "The cat sat on the mat." | python scripts/infer.py \
      --encoder-checkpoint logs/lang_jepa/checkpoint-epoch5.pth \
      --corpus data/index.txt --top-k 5

  python scripts/infer.py --encoder-checkpoint <path> \
      --decoder-checkpoint outputs/decoder/best_decoder.pt \
      --text "Some probe text."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.common.config import LANGJEPAConfig
from src.decoder.concept_extractor import ConceptExtractor
from src.decoder.models import ConceptDecoder, DecoderConfig
from src.encoder.models import TextTransformer


def _load_encoder(
    checkpoint: Path, config_path: Path, device: torch.device
) -> tuple[ConceptExtractor, PreTrainedTokenizer]:
    config = LANGJEPAConfig.from_yaml(str(config_path))
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.data.tokenizer = tokenizer

    encoder = TextTransformer(config)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device)
    return ConceptExtractor(encoder, normalize=True), tokenizer


def _load_decoder(
    checkpoint: Path, tokenizer: PreTrainedTokenizer, device: torch.device
) -> ConceptDecoder:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config: DecoderConfig = ckpt["config"]
    decoder = ConceptDecoder(config=config, tokenizer=tokenizer).to(device)
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()
    return decoder


@torch.no_grad()
def _embed(
    extractor: ConceptExtractor,
    tokenizer: PreTrainedTokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out.append(extractor(enc["input_ids"], enc["attention_mask"]).cpu())
    return torch.cat(out, dim=0)


def _read_queries(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    if not sys.stdin.isatty():
        lines = [line.strip() for line in sys.stdin if line.strip()]
        if lines:
            return lines
    raise SystemExit("Provide --text or pipe query lines on stdin.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--encoder-checkpoint", type=Path, required=True)
    p.add_argument(
        "--encoder-config",
        type=Path,
        default=Path("src/encoder/configs/base_lang_config.yaml"),
    )
    p.add_argument("--decoder-checkpoint", type=Path, default=None)
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to a .txt with one sentence per line to search over.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--text", default=None, help="Query text. Else read from stdin.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling (temp/top-p) instead of greedy when decoding.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor, tokenizer = _load_encoder(
        args.encoder_checkpoint, args.encoder_config, device
    )
    decoder = (
        _load_decoder(args.decoder_checkpoint, tokenizer, device)
        if args.decoder_checkpoint
        else None
    )

    queries = _read_queries(args)
    q_embed = _embed(
        extractor, tokenizer, queries, device, args.max_length, args.batch_size
    )

    corpus_texts: list[str] | None = None
    corpus_embed: torch.Tensor | None = None
    if args.corpus:
        corpus_texts = [
            line.strip()
            for line in Path(args.corpus).read_text().splitlines()
            if line.strip()
        ]
        corpus_embed = _embed(
            extractor,
            tokenizer,
            corpus_texts,
            device,
            args.max_length,
            args.batch_size,
        )

    for qi, query in enumerate(queries):
        concept = q_embed[qi]
        print(f"\n=== Query: {query}")
        print(
            f"concept (first 8 dims): {concept[:8].tolist()} "
            f"... norm={concept.norm().item():.4f}"
        )

        if corpus_embed is not None and corpus_texts is not None:
            sims = F.cosine_similarity(concept.unsqueeze(0), corpus_embed, dim=-1)
            k = min(args.top_k, len(corpus_texts))
            top = torch.topk(sims, k=k)
            print(f"Top-{k} nearest in corpus:")
            for score, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                print(f"  {score:.4f}  {corpus_texts[idx]}")

        if decoder is not None:
            concept_gpu = concept.unsqueeze(0).to(device)
            decoded = decoder.generate(
                concept_gpu,
                tokenizer,
                do_sample=args.sample,
                temperature=args.temperature,
                top_p=args.top_p if args.sample else None,
            )[0]
            print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()
