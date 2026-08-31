#!/usr/bin/env python3
"""Predict the next unraced round.

    python -m app.predict_upcoming

Thin wrapper: resolves the next round from the live calendar, then hands
off to `src.predict` + `src.render` (spec Section 6 -- no logic of its
own). Equivalent to ``python -m app.app upcoming``.

Note the round must have completed qualifying for its grid to exist,
which in practice means the day before the race.
"""

from __future__ import annotations

import argparse
import logging

from app import app_calendar
from app.app import EXIT_NO_DATA, predict_and_render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict the next unraced race")
    parser.add_argument("--season", type=int, default=None, help="Restrict to a season")
    parser.add_argument("--top", type=int, default=10, help="How many finishing positions to show")
    parser.add_argument("--fetch", action="store_true", help="Fetch the round if it is not cached locally")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    upcoming = app_calendar.next_unraced_round(season=args.season)
    if upcoming is None:
        print("No upcoming round found -- the season may be over and next year's calendar unpublished.")
        return EXIT_NO_DATA

    print(f"Next up: {upcoming}  ({upcoming.event_date.date()})")
    return predict_and_render(
        upcoming.season,
        upcoming.round_number,
        top_n=args.top,
        fetch=args.fetch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
