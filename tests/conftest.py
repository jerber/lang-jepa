"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable without `poetry install -e .`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
