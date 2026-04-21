"""Intrinsic evaluation of a trained LANG-JEPA encoder.

Three tasks:
  sts    — STS-B dev set, Spearman ρ between cosine(concept(s1), concept(s2))
           and human similarity scores. Gold-standard sentence-embedding metric.
  probe  — SST-2 linear probe. Train a single nn.Linear on frozen concept
           features, report val accuracy. Tests whether concepts contain
           sentiment-relevant information.
  au     — Alignment (positive-pair distance) & Uniformity (log-exp-dist of
           random pairs), per Wang & Isola 2020. Low alignment + low uniformity
           = high-quality contrastive representations.

Usage:
  python scripts/eval_representations.py --checkpoint logs/lang_jepa/checkpoint-epoch5.pth
  python scripts/eval_representations.py --checkpoint <path> --tasks sts,probe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.common.config import LANGJEPAConfig
from src.common.stats import spearman_rho
from src.decoder.concept_extractor import ConceptExtractor
from src.encoder.models import TextTransformer


def load_encoder(
    checkpoint_path: Path, config_path: Path, device: torch.device
) -> tuple[ConceptExtractor, PreTrainedTokenizer]:
    config = LANGJEPAConfig.from_yaml(str(config_path))
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.data.tokenizer = tokenizer

    encoder = TextTransformer(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device)
    return ConceptExtractor(encoder, normalize=True), tokenizer


@torch.no_grad()
def _embed_batch(
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


def sts_benchmark(
    extractor: ConceptExtractor,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
) -> dict[str, float]:
    ds = load_dataset("glue", "stsb", split="validation")
    s1 = list(ds["sentence1"])
    s2 = list(ds["sentence2"])
    labels = list(ds["label"])

    c1 = _embed_batch(extractor, tokenizer, s1, device, max_length, batch_size)
    c2 = _embed_batch(extractor, tokenizer, s2, device, max_length, batch_size)
    sims = F.cosine_similarity(c1, c2, dim=-1).tolist()

    return {"sts_spearman": spearman_rho(sims, labels), "n_pairs": float(len(labels))}


def linear_probe_sst2(
    extractor: ConceptExtractor,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
    train_size: int = 5000,
    probe_epochs: int = 30,
    probe_lr: float = 1e-3,
) -> dict[str, float]:
    train = load_dataset("glue", "sst2", split=f"train[:{train_size}]")
    val = load_dataset("glue", "sst2", split="validation")

    X_tr = _embed_batch(
        extractor, tokenizer, list(train["sentence"]), device, max_length, batch_size
    )
    y_tr = torch.tensor(train["label"], dtype=torch.long)
    X_val = _embed_batch(
        extractor, tokenizer, list(val["sentence"]), device, max_length, batch_size
    )
    y_val = torch.tensor(val["label"], dtype=torch.long)

    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    probe = nn.Linear(X_tr.shape[-1], 2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=probe_lr)

    for _ in range(probe_epochs):
        probe.train()
        perm = torch.randperm(len(X_tr), device=device)
        for b in range(0, len(X_tr), 128):
            idx = perm[b : b + 128]
            loss = F.cross_entropy(probe(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        acc = (probe(X_val).argmax(-1) == y_val).float().mean().item()

    return {"sst2_probe_acc": float(acc), "n_val": float(len(y_val))}


def alignment_uniformity(
    extractor: ConceptExtractor,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
    num_positives: int = 500,
    num_uniform: int = 1000,
    t: float = 2.0,
) -> dict[str, float]:
    """Wang & Isola 2020 diagnostics.

    alignment = E[(|| f(x) - f(y) ||^2)] over positive pairs (high-sim STS-B)
    uniformity = log E[exp(-t || f(x) - f(y) ||^2)] over random pairs

    Interpretation: low alignment → positive pairs map close. Low (more
    negative) uniformity → embeddings spread out on the hypersphere. Good
    reps have both low.
    """
    stsb = load_dataset("glue", "stsb", split="train")
    positives = [
        (r["sentence1"], r["sentence2"]) for r in stsb if r["label"] > 4.0
    ][:num_positives]
    if not positives:
        return {"alignment": float("nan"), "uniformity": float("nan"), "n_pos": 0.0}

    s1 = [p[0] for p in positives]
    s2 = [p[1] for p in positives]
    f1 = _embed_batch(extractor, tokenizer, s1, device, max_length, batch_size)
    f2 = _embed_batch(extractor, tokenizer, s2, device, max_length, batch_size)
    alignment = (f1 - f2).pow(2).sum(dim=-1).mean().item()

    # For uniformity: pool of random sentences to sample pairs from
    pool_texts = list(stsb["sentence1"])[:num_uniform]
    feats = _embed_batch(extractor, tokenizer, pool_texts, device, max_length, batch_size)
    # Pairwise squared distances among a subsample (closed-form on sphere: ||a-b||^2 = 2 - 2 a·b for unit vectors)
    sq_dists = 2.0 - 2.0 * (feats @ feats.T)
    sq_dists.fill_diagonal_(float("nan"))
    flat = sq_dists[~torch.isnan(sq_dists)]
    uniformity = torch.log(torch.exp(-t * flat).mean()).item()

    return {
        "alignment": float(alignment),
        "uniformity": float(uniformity),
        "n_pos": float(len(positives)),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True, help="Encoder checkpoint .pth")
    p.add_argument(
        "--encoder-config",
        type=Path,
        default=Path("src/encoder/configs/base_lang_config.yaml"),
    )
    p.add_argument(
        "--tasks",
        default="sts,probe,au",
        help="Comma-separated subset of {sts, probe, au}",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor, tokenizer = load_encoder(args.checkpoint, args.encoder_config, device)

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    results: dict[str, dict[str, float]] = {}

    if "sts" in tasks:
        print("Running STS-B...")
        results["sts"] = sts_benchmark(
            extractor, tokenizer, device, args.max_length, args.batch_size
        )
    if "probe" in tasks:
        print("Running SST-2 linear probe...")
        results["probe"] = linear_probe_sst2(
            extractor, tokenizer, device, args.max_length, args.batch_size
        )
    if "au" in tasks:
        print("Running Alignment & Uniformity...")
        results["au"] = alignment_uniformity(
            extractor, tokenizer, device, args.max_length, args.batch_size
        )

    print("\n=== Results ===")
    for task, metrics in results.items():
        print(f"[{task}]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
