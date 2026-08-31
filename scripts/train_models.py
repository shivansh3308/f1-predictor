#!/usr/bin/env python3
"""Thin CLI wrapper over ``src.train``.

Deliberately contains no training logic of its own. All of it lives in
``src/train.py`` so it stays importable and testable (spec Section 6:
"Core logic lived in scripts/ and app/ with no library layer"). This file
exists only so the training run has an obvious, conventional entrypoint:

    python scripts/train_models.py --save

Every argument is forwarded to ``src.train.main``; run with ``--help`` for
the full list.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (``python scripts/train_models.py``),
# which does not put the project root on sys.path the way ``-m`` does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import main  # noqa: E402 - import must follow the sys.path fix

if __name__ == "__main__":
    main()
