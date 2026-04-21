import argparse
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.common.cli import (
    DECODER_DEBUG_OVERRIDES,
    ENCODER_DEBUG_OVERRIDES,
    add_common_cli,
    load_yaml_with_overrides,
)
from src.common.config import LANGJEPAConfig
from src.common.datasets.fineweb_edu import TextDataset
from src.common.distributed import setup_distributed
from src.decoder.concept_extractor import ConceptExtractor
from src.decoder.config import DecoderFullConfig
from src.decoder.decoder_dataset import make_loader, split_train_eval
from src.decoder.models import ConceptDecoder, DecoderConfig
from src.decoder.train import DecoderTrainer
from src.encoder.models import TextTransformer


def _find_latest_checkpoint(log_dir: Path) -> Path:
    pattern = re.compile(r"checkpoint-epoch(\d+)\.pth$")
    candidates = [
        (int(m.group(1)), p)
        for p in log_dir.glob("checkpoint-epoch*.pth")
        if (m := pattern.search(p.name))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint-epoch*.pth in {log_dir}. Train the encoder first "
            f"(python main_encoder.py) or pass --encoder-checkpoint."
        )
    return max(candidates)[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the concept decoder.")
    parser.add_argument(
        "--encoder-config", default="src/encoder/configs/base_lang_config.yaml"
    )
    parser.add_argument(
        "--decoder-config", default="src/decoder/configs/decoder_config.yaml"
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help="Path to encoder .pth. Defaults to latest in encoder log_dir.",
    )
    add_common_cli(parser)
    parser.add_argument(
        "--dec-override",
        "-d",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override for the DECODER yaml (-o overrides the encoder yaml). "
        "Example: -d training.batch_size=16 -d decoder.num_layers=6.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dist_info = setup_distributed()

    enc_raw = load_yaml_with_overrides(
        args.encoder_config,
        overrides=args.override,
        debug=args.debug,
        debug_preset=ENCODER_DEBUG_OVERRIDES,
    )
    dec_raw = load_yaml_with_overrides(
        args.decoder_config,
        overrides=args.dec_override,
        debug=args.debug,
        debug_preset=DECODER_DEBUG_OVERRIDES,
    )
    enc_cfg = LANGJEPAConfig(**enc_raw)
    dec_cfg = DecoderFullConfig(**dec_raw)

    tokenizer = AutoTokenizer.from_pretrained(enc_cfg.data.tokenizer_path)
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc_cfg.data.tokenizer = tokenizer

    ckpt_path = (
        Path(args.encoder_checkpoint)
        if args.encoder_checkpoint
        else _find_latest_checkpoint(Path(enc_cfg.logging.log_dir))
    )
    if dist_info.is_main:
        print(f"Loading encoder checkpoint: {ckpt_path}")

    encoder = TextTransformer(enc_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])

    extractor = ConceptExtractor(encoder, normalize=True)
    decoder_config = DecoderConfig.from_tokenizer(
        tokenizer=tokenizer,
        embed_dim=extractor.embed_dim,
        hidden_dim=dec_cfg.decoder.hidden_dim,
        num_layers=dec_cfg.decoder.num_layers,
        num_heads=dec_cfg.decoder.num_heads,
        dropout=dec_cfg.decoder.dropout,
        max_length=dec_cfg.decoder.max_length,
    )
    decoder = ConceptDecoder(config=decoder_config, tokenizer=tokenizer)

    if dist_info.is_main:
        print("Loading text dataset for decoder training...")
    dataset = TextDataset(
        train_file=enc_cfg.data.train_file,
        limit=enc_cfg.data.limit,
        min_length=enc_cfg.data.min_length,
        min_sentences=enc_cfg.data.min_sentences,
        window_size=enc_cfg.data.window_size,
    )
    texts = [sample.target for sample in dataset.samples]
    train_texts, eval_texts = split_train_eval(
        texts, eval_ratio=dec_cfg.training.eval_ratio
    )
    if dist_info.is_main:
        print(f"Decoder split: {len(train_texts)} train / {len(eval_texts)} eval")

    train_loader = make_loader(
        train_texts,
        tokenizer,
        max_length=dec_cfg.decoder.max_length,
        batch_size=dec_cfg.training.batch_size,
        num_workers=enc_cfg.data.num_workers,
        shuffle=True,
        rank=dist_info.rank,
        world_size=dist_info.world_size,
    )
    eval_loader = (
        make_loader(
            eval_texts,
            tokenizer,
            max_length=dec_cfg.decoder.max_length,
            batch_size=dec_cfg.training.batch_size,
            num_workers=enc_cfg.data.num_workers,
            shuffle=False,
        )
        if eval_texts and dist_info.is_main
        else None
    )

    trainer = DecoderTrainer(
        config=dec_cfg,
        extractor=extractor,
        decoder=decoder,
        train_loader=train_loader,
        eval_loader=eval_loader,
        dist_info=dist_info,
    )
    trainer.train()


if __name__ == "__main__":
    main()
