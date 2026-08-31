#!/usr/bin/env python3
"""Predict the most recently completed round and compare against the result.

    python -m app.predict_latest_done

Thin wrapper: resolves the latest completed round from the live calendar,
then hands off to `src.predict` + `src.render` with the actual-results
column enabled (spec Section 6 -- no logic of its own). Equivalent to
``python -m app.app latest``.

Caveat worth remembering when reading the output: the shipped models are
trained on all of 2018-2025, so a round inside that range is being
predicted in-sample and will look better than the model's true
forward-looking skill. `scripts/eval_past.py` is the honest measurement.
"""

from __future__ import annotations

import argparse
import logging

from app import app_calendar
from app.app import EXIT_NO_DATA, predict_and_render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict the most recent race and compare to the result")
    parser.add_argument("--season", type=int, default=None, help="Restrict to a season")
    parser.add_argument("--top", type=int, default=10, help="How many finishing positions to show")
    parser.add_argument("--fetch", action="store_true", help="Fetch the round if it is not cached locally")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    latest = app_calendar.latest_completed_round(season=args.season)
    if latest is None:
        print("No completed round found for that season.")
        return EXIT_NO_DATA

    print(f"Most recent: {latest}  ({latest.event_date.date()})")
    return predict_and_render(
        latest.season,
        latest.round_number,
        top_n=args.top,
        show_actual=True,
        fetch=args.fetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
