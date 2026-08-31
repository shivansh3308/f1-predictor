"""Main CLI entrypoint.

    python -m app.app upcoming          # predict the next unraced round
    python -m app.app latest            # predict + compare the most recent race
    python -m app.app any 2023 5        # predict an arbitrary round
    python -m app.app calendar          # list a season's calendar

Holds no prediction, training or rendering logic of its own -- it resolves
which round the user means (via `app_calendar`) and hands off to
`src.predict` + `src.render` (spec Section 6). The per-variant entrypoints
in task 17 call the same functions defined here, so there is one
implementation of each command rather than one per file.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app import app_calendar
from src import config, data_fetch, features, predict, render

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_NO_DATA = 1
EXIT_NO_MODELS = 2


def _explain_missing_round(season: int, round_number: int) -> str:
    """Actionable guidance when a round has no data, rather than a bare error."""
    lines = [
        f"No data available for {season} round {round_number}.",
        "",
    ]
    if season not in config.SEASONS:
        lines += [
            f"The dataset currently covers {config.SEASON_START}-{config.SEASON_END}.",
            f"To include {season}:",
            f"    python -m src.data_fetch --seasons {season}",
            "    python -m src.features",
        ]
    else:
        lines += [
            "That season is in the dataset, but this round is not.",
            f"    python -m src.data_fetch --seasons {season}",
            "    python -m src.features",
        ]
    lines += [
        "",
        "Note: predicting a race needs its grid, which only exists once",
        "qualifying has run -- typically the day before the race.",
    ]
    return "\n".join(lines)


def _fetch_round(season: int, round_number: int) -> bool:
    """Pull a single round on demand. Returns True if it landed on disk."""
    logger.info("Fetching %d round %d ...", season, round_number)
    data_fetch.enable_fastf1_cache()
    round_df = data_fetch.fetch_round_raw(season, round_number)
    if round_df is None:
        return False
    data_fetch.save_raw_round(round_df, season, round_number)
    # Rebuild the processed table so the new round is usable immediately.
    features.save_feature_table(features.build_feature_table(require_target=False))
    return True


def predict_and_render(
    season: int,
    round_number: int,
    top_n: int = 10,
    show_actual: bool = False,
    fetch: bool = False,
) -> int:
    """Resolve a round, predict it, and print the report. Returns an exit code.

    Shared by every subcommand and by the task-17 entrypoints.
    """
    try:
        models = predict.load_models()
    except predict.ModelsNotTrainedError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NO_MODELS

    try:
        render.render_round(
            season,
            round_number,
            models=models,
            top_n=top_n,
            show_actual=show_actual,
        )
    except predict.RoundNotFoundError:
        if fetch and _fetch_round(season, round_number):
            render.render_round(
                season,
                round_number,
                models=models,
                top_n=top_n,
                show_actual=show_actual,
            )
            return EXIT_OK
        print(_explain_missing_round(season, round_number), file=sys.stderr)
        if not fetch:
            print("\nOr retry with --fetch to pull it now.", file=sys.stderr)
        return EXIT_NO_DATA

    return EXIT_OK


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_any(args: argparse.Namespace) -> int:
    """Predict an explicitly specified round."""
    return predict_and_render(
        args.season,
        args.round,
        top_n=args.top,
        show_actual=args.actual,
        fetch=args.fetch,
    )


def cmd_upcoming(args: argparse.Namespace) -> int:
    """Predict the next round that has not been run yet."""
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


def cmd_latest(args: argparse.Namespace) -> int:
    """Predict the most recently completed round and compare against the result."""
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


def cmd_calendar(args: argparse.Namespace) -> int:
    """List a season's calendar with completion status."""
    season = args.season if args.season is not None else app_calendar.current_season()
    calendar = app_calendar.get_calendar(season)
    if calendar.empty:
        print(f"No calendar available for {season}.")
        return EXIT_NO_DATA

    print(f"{season} season calendar\n")
    for _, row in calendar.iterrows():
        marker = "done" if row["has_run"] else "    "
        print(f"  [{marker}] R{row['round']:<2}  {row['event_date'].date()}  {row['event_name']}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1-predictor",
        description="Predict Formula 1 race outcomes from pre-race information.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m app.app upcoming\n"
            "  python -m app.app latest\n"
            "  python -m app.app any 2023 5 --top 5\n"
            "  python -m app.app calendar --season 2024\n"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show INFO-level logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser, *, with_fetch: bool = True) -> None:
        sub.add_argument("--top", type=int, default=10, help="How many finishing positions to show")
        if with_fetch:
            sub.add_argument(
                "--fetch",
                action="store_true",
                help="Fetch the round from FastF1 if it is not cached locally",
            )

    any_parser = subparsers.add_parser("any", help="Predict a specific round")
    any_parser.add_argument("season", type=int)
    any_parser.add_argument("round", type=int)
    any_parser.add_argument("--actual", action="store_true", help="Show real results alongside predictions")
    add_common(any_parser)
    any_parser.set_defaults(func=cmd_any)

    upcoming_parser = subparsers.add_parser("upcoming", help="Predict the next unraced round")
    upcoming_parser.add_argument("--season", type=int, default=None, help="Restrict to a season")
    add_common(upcoming_parser)
    upcoming_parser.set_defaults(func=cmd_upcoming)

    latest_parser = subparsers.add_parser("latest", help="Predict the most recent race and compare to the result")
    latest_parser.add_argument("--season", type=int, default=None, help="Restrict to a season")
    add_common(latest_parser)
    latest_parser.set_defaults(func=cmd_latest)

    calendar_parser = subparsers.add_parser("calendar", help="List a season's calendar")
    calendar_parser.add_argument("--season", type=int, default=None)
    calendar_parser.set_defaults(func=cmd_calendar)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
