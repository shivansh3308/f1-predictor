#!/usr/bin/env python3
"""Predict an arbitrary specified round.

    python -m app.predict_any 2023 5

Thin wrapper over `src.predict` + `src.render` via the shared
`app.app.predict_and_render` (spec Section 6 -- no logic of its own).
Equivalent to ``python -m app.app any 2023 5``; this file exists because
the spec's Section 3 layout names it directly.
"""

from __future__ import annotations

import argparse
import logging

from app.app import predict_and_render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict a specific race")
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    parser.add_argument("--top", type=int, default=10, help="How many finishing positions to show")
    parser.add_argument("--actual", action="store_true", help="Show real results alongside predictions")
    parser.add_argument("--fetch", action="store_true", help="Fetch the round if it is not cached locally")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    return predict_and_render(
        args.season,
        args.round,
        top_n=args.top,
        show_actual=args.actual,
        fetch=args.fetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
