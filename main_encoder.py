import argparse

from transformers import AutoTokenizer

from src.common.cli import (
    ENCODER_DEBUG_OVERRIDES,
    add_common_cli,
    load_yaml_with_overrides,
)
from src.common.config import LANGJEPAConfig
from src.encoder.train import train


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LANG-JEPA encoder.")
    parser.add_argument(
        "--config",
        default="src/encoder/configs/base_lang_config.yaml",
        help="Path to encoder config yaml.",
    )
    add_common_cli(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    raw = load_yaml_with_overrides(
        args.config,
        overrides=args.override,
        debug=args.debug,
        debug_preset=ENCODER_DEBUG_OVERRIDES,
    )
    config = LANGJEPAConfig(**raw)

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)
    # RoBERTa has <pad>; only fall back to EOS for tokenizers that lack a pad token entirely.
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config.data.tokenizer = tokenizer
    train(config)
