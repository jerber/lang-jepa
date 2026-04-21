"""Shared CLI helpers: dot-path config overrides and debug-mode presets.

Design: load yaml into a dict, let the user mutate keys via `--override a.b=c`,
then hand off to Pydantic for validation. Keeps the yaml authoritative but
lets experiments run without editing it.
"""
from __future__ import annotations

import argparse
from typing import Any

import yaml


def _coerce_scalar(value: str) -> Any:
    """Best-effort string → (bool | int | float | None | str)."""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Allow YAML-style lists/dicts, e.g. "[1, 2, 3]"
    if value.startswith(("[", "{")):
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            pass
    return value


def apply_overrides(config_dict: dict, overrides: list[str]) -> dict:
    """Apply `a.b.c=value` overrides to a nested dict. Returns the same dict.

    Unknown keys are created along the way rather than raising — the final
    Pydantic validation will reject unsupported field names if needed.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--override requires key=value, got {item!r}")
        key, raw_val = item.split("=", 1)
        parts = key.split(".")
        node = config_dict
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce_scalar(raw_val)
    return config_dict


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    """Attach --override and --debug to a parser."""
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config overrides, e.g. -o data.limit=500 -o "
        "optimization.lr=3e-4. Can be passed multiple times.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tiny run for iteration: 50 docs, 1 epoch, batch 4, verbose logging.",
    )


ENCODER_DEBUG_OVERRIDES: list[str] = [
    "data.limit=50",
    "data.num_workers=0",
    "optimization.epochs=1",
    "data.batch_size=4",
    "logging.log_freq=1",
    "logging.checkpoint_freq=1",
    "logging.log_to_wandb=false",
]

DECODER_DEBUG_OVERRIDES: list[str] = [
    "training.batch_size=4",
    "training.num_epochs=1",
    "evaluation.eval_steps=5",
    "evaluation.save_steps=10",
    "evaluation.num_samples=2",
]


def load_yaml_with_overrides(
    yaml_path: str, overrides: list[str], debug: bool = False, *, debug_preset: list[str]
) -> dict:
    """Load yaml, inject debug preset (if --debug), then apply --override entries."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    if debug:
        raw = apply_overrides(raw, debug_preset)
    raw = apply_overrides(raw, overrides)
    return raw
