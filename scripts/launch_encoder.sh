#!/usr/bin/env bash
# Thin torchrun wrapper for multi-GPU encoder training.
#
# Single-node, N GPUs:
#   ./scripts/launch_encoder.sh 8                     # uses defaults
#   ./scripts/launch_encoder.sh 8 --override data.limit=100000
#
# Single-node with custom port (useful when running multiple jobs on one host):
#   MASTER_PORT=29501 ./scripts/launch_encoder.sh 4
set -euo pipefail

NPROC="${1:-1}"
shift || true

MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  main_encoder.py "$@"
