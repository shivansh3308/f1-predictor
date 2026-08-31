#!/usr/bin/env python3
"""Incremental refresh as new races complete.

    python scripts/update_training_data.py            # check + fetch what's missing
    python scripts/update_training_data.py --dry-run  # just report
    python scripts/update_training_data.py --retrain  # also refit the models

Fetches only the rounds that are actually missing, rather than re-pulling
whole seasons: a full refetch of 2018-2025 is ~170 rounds and will trip
FastF1's 500-calls/hour limit, while a normal weekly update is one round.

"Missing" includes rounds that are on disk but were saved with degraded
data -- `data_fetch.is_round_complete` catches rounds persisted without
qualifying during a rate-limited run, which would otherwise never be
retried (see the task-8 commit).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_fetch, features  # noqa: E402

logger = logging.getLogger(__name__)


def find_missing_rounds(seasons: list[int]) -> list[tuple[int, int, str]]:
    """Completed rounds that are absent from data/raw, or cached incomplete."""
    missing: list[tuple[int, int, str]] = []
    for season, round_number, event_name in data_fetch.iter_completed_rounds(seasons):
        path = config.DATA_RAW_DIR / str(season) / f"{round_number:02d}.parquet"
        if not path.exists():
            missing.append((season, round_number, event_name))
        elif not data_fetch.is_round_complete(path):
            missing.append((season, round_number, f"{event_name} (cached without qualifying)"))
    return missing


def fetch_missing(missing: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Fetch each missing round. Returns the ones that could not be retrieved."""
    failed: list[tuple[int, int, str]] = []
    for season, round_number, event_name in missing:
        print(f"  fetching {season} R{round_number:<2} {event_name} ...", flush=True)
        round_df = data_fetch.fetch_round_raw(season, round_number)
        if round_df is None:
            failed.append((season, round_number, event_name))
            continue
        data_fetch.save_raw_round(round_df, season, round_number)
    return failed


def rebuild_feature_table() -> int:
    """Regenerate data/processed/features.parquet. Returns the row count."""
    table = features.build_feature_table()
    features.save_feature_table(table)
    return len(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh training data with newly completed races")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=config.SEASONS,
        help=f"Seasons to check (default: the training range, {config.SEASON_START}-{config.SEASON_END})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what is missing without fetching")
    parser.add_argument("--retrain", action="store_true", help="Refit and save the models afterwards")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    data_fetch.enable_fastf1_cache()

    print(f"Checking seasons {args.seasons[0]}-{args.seasons[-1]} for new completed races ...\n")
    missing = find_missing_rounds(args.seasons)

    if not missing:
        print("Already up to date -- every completed round in range is cached.")
    else:
        print(f"{len(missing)} round(s) missing:")
        for season, round_number, event_name in missing:
            print(f"  {season} R{round_number:<2} {event_name}")
        print()

        if args.dry_run:
            print("(--dry-run: nothing fetched)")
            return 0

        failed = fetch_missing(missing)
        fetched = len(missing) - len(failed)
        print(f"\nFetched {fetched} of {len(missing)} round(s).")
        if failed:
            print("Could not fetch (data may not be published yet):")
            for season, round_number, event_name in failed:
                print(f"  {season} R{round_number:<2} {event_name}")

        if fetched:
            print(f"Rebuilt feature table: {rebuild_feature_table()} rows.")

    # A season outside the configured training range is a deliberate choice,
    # not an oversight -- but it is worth surfacing rather than silently
    # ignoring races the user can see happening.
    live_season = pd.Timestamp.now().year
    if live_season not in args.seasons:
        print(
            f"\nNote: the {live_season} season is not in the configured range "
            f"({config.SEASON_START}-{config.SEASON_END}). To include it:\n"
            f"    python scripts/update_training_data.py --seasons {live_season}\n"
            f"  and set SEASON_END = {live_season} in src/config.py to train on it."
        )

    if args.retrain:
        print("\nRetraining models ...")
        from src.train import main as train_main

        # Explicit argv: without it, train's parser would try to parse this
        # script's own flags.
        train_main(["--save", "--skip-diagnostic"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
