# LANG-JEPA: Learning to Think in Latent Space

LANG-JEPA is an experimental language model that operates in "concept space" rather than "token space." Building on Meta AI's JEPA framework ([I-JEPA](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) for images, [V-JEPA](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) for video), it predicts the **semantic feature embedding of the next sentence** from prior context, rather than predicting next tokens. The hypothesis is that reasoning at the conceptual level produces representations closer to how humans understand language.

## How it works

Two training stages:

**1. Encoder + predictor** — learn next-sentence embeddings
```
context sentences  ──►  online encoder ──►  predictor  ──►  predicted concept
next sentence      ──►  EMA target encoder ──►  masked mean  ──►  target concept
                                                             │
                                        loss = smooth-L1(pred, target)
```

Key architectural choices (all load-bearing):
- The **target encoder** is a momentum-updated (EMA) copy of the online encoder, per JEPA / BYOL / DINO. Using the same encoder for both paths collapses trivially; the EMA provides the stop-gradient asymmetry that keeps training stable.
- Targets are **masked-mean pooled** (padding tokens excluded) and L2-normalized.
- Predictions and targets live in the **same dimensionality** (`pred_dim == embed_dim`). No target-side projection head — that would be a shared trainable path from predictions to targets and is the classic collapse accelerant in asymmetric contrastive methods.
- **Smooth-L1** loss on normalized features (I-JEPA default) — kinder to outliers than MSE.

**2. Decoder** — reconstruct text from concept
```
text  ──►  frozen ConceptExtractor  ──►  concept  ──►  decoder  ──►  generated text
                    (encoder + masked mean + L2 norm)
```
The decoder inverts the *exact* map the encoder was optimized to produce. This is what makes concepts interpretable: if the decoder can reconstruct text from them, the concept space is semantically meaningful.

## Installation

```bash
poetry shell
poetry install
```

## Quickstart (single GPU)

```bash
# 1. Train the encoder (defaults: pretrained RoBERTa-base init, 10k FineWeb-Edu docs)
python main_encoder.py

# 2. Train the decoder (auto-picks the latest encoder checkpoint)
python main_decoder.py

# Fast iteration mode — tiny run for sanity checking
python main_encoder.py --debug
python main_decoder.py --debug
```

Override any config field from the CLI:

```bash
# Encoder overrides use -o / --override (applied to base_lang_config.yaml)
python main_encoder.py \
    -o data.limit=1000 \
    -o optimization.lr=3e-4 \
    -o optimization.epochs=2 \
    -o logging.log_to_wandb=true

# For the decoder, -o still targets the encoder config (shared data / tokenizer),
# while -d / --dec-override targets decoder_config.yaml:
python main_decoder.py \
    -o data.limit=5000 \
    -d training.batch_size=16 \
    -d decoder.num_layers=6
```

## Multi-GPU training

```bash
# Single node, 8 GPUs
./scripts/launch_encoder.sh 8
./scripts/launch_decoder.sh 8

# With overrides
./scripts/launch_encoder.sh 8 -o data.streaming=true -o optimization.max_steps=50000
```

For cluster-scale runs, enable streaming (`data.streaming=true`) so the dataset isn't materialized in memory. Streaming requires `optimization.max_steps` to be set since the IterableDataset has no known length.

Gradient accumulation scales the effective batch size without touching `batch_size`:

```bash
./scripts/launch_encoder.sh 4 -o optimization.grad_accum_steps=8
# Effective batch = batch_size (32) * accum (8) * world_size (4) = 1024
```

## Evaluating a trained encoder

Three intrinsic tests that don't require the decoder:

```bash
python scripts/eval_representations.py \
    --checkpoint logs/lang_jepa/checkpoint-epoch5.pth \
    --tasks sts,probe,au
```

- `sts` — Spearman ρ on STS-B dev (the gold-standard sentence-embedding benchmark).
- `probe` — linear probe accuracy on SST-2 (sentiment).
- `au` — Alignment & Uniformity (Wang & Isola 2020) — diagnostic scalars for contrastive representation quality.

## Inference / demos

```bash
# Concept embedding + nearest neighbors
echo "The cat sat on the mat." | python scripts/infer.py \
    --encoder-checkpoint logs/lang_jepa/checkpoint-epoch5.pth \
    --corpus data/my_sentences.txt --top-k 5

# With decoder: also print reconstruction
python scripts/infer.py \
    --encoder-checkpoint logs/lang_jepa/checkpoint-epoch5.pth \
    --decoder-checkpoint outputs/decoder/best_decoder.pt \
    --text "The cat sat on the mat." \
    --sample --temperature 0.8 --top-p 0.9
```

## Tests

```bash
pytest tests/ -q
# Skip network-dependent tests (pad-token checks against real tokenizers):
pytest tests/ -q -m "not network"
```

## Configuration reference

Encoder config at `src/encoder/configs/base_lang_config.yaml`. Key knobs:

| Key | Default | Notes |
|---|---|---|
| `model.pretrained` | `true` | Use `AutoModel.from_pretrained` vs random init |
| `model.embed_dim` / `model.pred_dim` | 768 / 768 | Must be equal (no target-side projection) |
| `model.max_length` | 512 | Token limit; keep context window x avg-sentence-len under this |
| `data.window_size` | 8 | Sentences of context before the target |
| `data.val_fraction` | 0.0 | Hash-based document-level held-out split |
| `data.streaming` | `false` | True for cluster-scale runs |
| `optimization.loss_fn` | `smooth_l1` | `smooth_l1` (I-JEPA) or `cosine` |
| `optimization.momentum_start` / `momentum_end` | 0.996 / 1.0 | EMA target schedule |
| `optimization.grad_accum_steps` | 1 | Effective batch multiplier |
| `optimization.max_steps` | `null` | Required when streaming |

Decoder config at `src/decoder/configs/decoder_config.yaml`.

## Healthy-training signals

On a successful encoder run:

- `train/loss` decreases (smooth-L1 scale is small, roughly 0.01–0.1).
- `stats/cosine_similarity` rises but does **not** saturate at 1.0 within the first few hundred steps — that would indicate collapse.
- `val/embeddings/target_eff_rank` stays comfortably above single digits on the 768-dim target space.
- `val/embeddings/target_std_mean` stays above ~0.05.

On collapse (all embeddings collide), the above diagnostics crash: eff-rank → 1, std → 0, diversity → 0, cosine sim pinned at 1. The fixes are already in place (EMA target, masked pooling, no shared projection head); if training still collapses at scale, consider adding VICReg variance/covariance regularization (see Plan file).

## File structure

```
lang-jepa/
├── main_encoder.py              # Encoder training entry point
├── main_decoder.py              # Decoder training entry point
├── scripts/
│   ├── launch_encoder.sh        # torchrun wrapper
│   ├── launch_decoder.sh        # torchrun wrapper
│   ├── eval_representations.py  # STS + probe + A&U
│   └── infer.py                 # Concept + nearest neighbors + decode
├── src/
│   ├── common/
│   │   ├── cli.py               # Dot-path config overrides, --debug preset
│   │   ├── config.py            # Pydantic config schema
│   │   ├── distributed.py       # DDP setup helper
│   │   ├── logging.py           # CSVLogger, AverageMeter
│   │   ├── pooling.py           # masked_mean
│   │   ├── schedulers.py        # LR + WD cosine schedules
│   │   └── datasets/
│   │       ├── fineweb_edu.py   # Materialized TextDataset + hash val split
│   │       ├── streaming.py     # StreamingTextDataset (IterableDataset)
│   │       ├── sentences.py     # Sentence, locate_sentences, is_val_doc (shared)
│   │       └── utils/sentence_splitting.py
│   ├── encoder/
│   │   ├── models.py            # TextTransformer, TextPredictor
│   │   ├── ema.py               # EMAEncoder (momentum target)
│   │   ├── collator.py          # Context/target batch collator
│   │   ├── train.py             # Main training loop (DDP + grad accum + EMA)
│   │   ├── utils/optim.py       # init_optimizer + schedulers factory
│   │   ├── utils/checkpoints.py # save_checkpoint / load_checkpoint
│   │   ├── utils/monitor.py     # Diagnostics (eff-rank, std, covariance)
│   │   └── configs/base_lang_config.yaml
│   └── decoder/
│       ├── models.py            # ConceptDecoder + sampling
│       ├── concept_extractor.py # Canonical text→concept map
│       ├── decoder_dataset.py   # DecoderDataset + sharded DataLoader
│       ├── train.py             # Decoder training loop (DDP + grad accum)
│       ├── config.py            # Decoder Pydantic config
│       ├── utils/evaluation.py  # BLEU/ROUGE/perplexity/concept similarity
│       └── configs/decoder_config.yaml
└── tests/                        # pytest suite
```

## Dependencies

See `pyproject.toml`. Highlights: `torch ^2.5.1`, `transformers`, `datasets ^3.2.0`, `wtpsplit ^2.1.2` (sentence splitting), `pydantic ^2.10.3`, `wandb` (optional).
