"""Season calendar helpers: round lookup, and finding the next unraced round.

Backed by the live FastF1 schedule rather than the training data, which
matters: the models are trained on complete seasons (2018-2025), but the
season being *predicted* is whatever is running now. Keeping the calendar
independent of the training range is what lets `predict_upcoming` work
without retraining (see the season-range note in `src/config.py`).

Every lookup takes an optional ``now`` so behaviour is testable without
depending on the real clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src import data_fetch

logger = logging.getLogger(__name__)

CALENDAR_COLUMNS = ["season", "round", "event_name", "location", "country", "event_date", "has_run"]


class RoundNotFoundError(LookupError):
    """Raised when a requested round is not on the season's calendar."""


@dataclass(frozen=True)
class RoundRef:
    """A single identified round."""

    season: int
    round_number: int
    event_name: str
    event_date: pd.Timestamp

    def __str__(self) -> str:
        return f"{self.season} Round {self.round_number} -- {self.event_name}"


def _now(now: pd.Timestamp | None = None) -> pd.Timestamp:
    return now if now is not None else pd.Timestamp.now()


def get_calendar(season: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return the full calendar for a season, flagged with which rounds have run.

    Returns an empty frame (rather than raising) when the schedule is not
    published yet -- a future season is a normal state, not an error.
    """
    try:
        schedule = data_fetch.get_season_schedule(season)
    except Exception as exc:  # noqa: BLE001 - FastF1 raises a mix of types
        logger.warning("No calendar available for %d (%s)", season, exc)
        return pd.DataFrame(columns=CALENDAR_COLUMNS)

    if schedule.empty:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)

    calendar = pd.DataFrame(
        {
            "season": season,
            "round": schedule["RoundNumber"].astype(int),
            "event_name": schedule["EventName"].astype(str),
            "location": schedule["Location"].astype(str),
            "country": schedule["Country"].astype(str),
            "event_date": pd.to_datetime(schedule["EventDate"]),
        }
    )
    # Round 0 is pre-season testing, not a race.
    calendar = calendar[calendar["round"] > 0]
    calendar["has_run"] = calendar["event_date"] < _now(now)
    return calendar.sort_values("round").reset_index(drop=True)


def round_name(season: int, round_number: int) -> str:
    """Map a round number to its Grand Prix name."""
    calendar = get_calendar(season)
    match = calendar[calendar["round"] == round_number]
    if match.empty:
        available = calendar["round"].tolist()
        raise RoundNotFoundError(
            f"{season} has no round {round_number}."
            + (f" Available rounds: {min(available)}-{max(available)}." if available else " No calendar published.")
        )
    return str(match["event_name"].iloc[0])


def _to_ref(row: pd.Series) -> RoundRef:
    return RoundRef(
        season=int(row["season"]),
        round_number=int(row["round"]),
        event_name=str(row["event_name"]),
        event_date=row["event_date"],
    )


def current_season(now: pd.Timestamp | None = None) -> int:
    """The season currently in progress, or the most recent one.

    Falls back to the previous calendar year when the new year's schedule
    is not published yet -- in January, "this season" is usually still
    last season as far as available data goes.
    """
    year = _now(now).year
    if not get_calendar(year, now=now).empty:
        return year
    logger.info("No calendar published for %d yet -- falling back to %d", year, year - 1)
    return year - 1


def next_unraced_round(season: int | None = None, now: pd.Timestamp | None = None) -> RoundRef | None:
    """The next race that has not happened yet.

    When `season` is omitted, searches the current season and rolls over
    into the next one if the current season has finished. Returns None if
    no upcoming round can be found (e.g. the season ended and next year's
    calendar is not out).
    """
    seasons = [season] if season is not None else [current_season(now), current_season(now) + 1]

    for candidate in seasons:
        calendar = get_calendar(candidate, now=now)
        upcoming = calendar[~calendar["has_run"]]
        if not upcoming.empty:
            return _to_ref(upcoming.iloc[0])

    logger.warning("No upcoming round found (searched seasons: %s)", seasons)
    return None


def latest_completed_round(season: int | None = None, now: pd.Timestamp | None = None) -> RoundRef | None:
    """The most recently completed race.

    When `season` is omitted, searches the current season and falls back to
    the previous one if the current season has not started yet.
    """
    seasons = [season] if season is not None else [current_season(now), current_season(now) - 1]

    for candidate in seasons:
        calendar = get_calendar(candidate, now=now)
        completed = calendar[calendar["has_run"]]
        if not completed.empty:
            return _to_ref(completed.iloc[-1])

    logger.warning("No completed round found (searched seasons: %s)", seasons)
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Season calendar lookup")
    parser.add_argument("--season", type=int, default=None, help="Season to list (default: current)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    season = args.season if args.season is not None else current_season()
    calendar = get_calendar(season)

    if calendar.empty:
        print(f"No calendar available for {season}.")
        return

    print(f"{season} season calendar\n")
    for _, row in calendar.iterrows():
        marker = "done" if row["has_run"] else "    "
        print(f"  [{marker}] R{row['round']:<2}  {row['event_date'].date()}  {row['event_name']}")

    print()
    latest = latest_completed_round(season)
    upcoming = next_unraced_round(season)
    print(f"  most recent : {latest or 'none'}")
    print(f"  next up     : {upcoming or 'none -- season complete'}")


if __name__ == "__main__":
    main()
