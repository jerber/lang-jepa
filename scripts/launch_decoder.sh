#!/usr/bin/env bash
# Thin torchrun wrapper for multi-GPU decoder training.
#
# Usage:
#   ./scripts/launch_decoder.sh 8 --encoder-checkpoint logs/lang_jepa/checkpoint-epoch5.pth
set -euo pipefail

NPROC="${1:-1}"
shift || true

MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  main_decoder.py "$@"
