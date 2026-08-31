"""FastF1 data fetching.

Enables the FastF1 on-disk cache, iterates season schedules, pulls per-round
Race + Qualifying results, and persists the combined raw pull to
``data/raw/<season>/<round>.parquet``.

This module is intentionally "dumb": it fetches and lightly reshapes what
FastF1 gives back, but does no feature engineering (rolling averages,
standings, DNF rates, leakage checks) — that all lives in ``src/features.py``
so there is one place where those derived columns are computed.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from src import config

logger = logging.getLogger(__name__)

_CACHE_ENABLED = False

# Retry policy for transient network errors (FastF1 sits on top of the
# official F1 timing API / Ergast, both of which rate-limit and occasionally
# blip). Session-not-found errors are NOT retried — those are permanent for
# a given (season, round) and should just be skipped.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0

# FastF1 enforces a shared "500 calls/h" limit across all APIs, using a
# rolling window (a deque of the last 500 call timestamps). Critically, it
# timestamps a call *before* checking the count, so a rejected call still
# occupies a slot in that window — hammering it with quick retries only
# keeps refreshing the window with new failures and pushes the reset further
# away. So instead of the normal short backoff, a rate-limit hit gets exactly
# one long cooldown (comfortably longer than the 60-minute window) and then
# one retry. No further requests are made during the cooldown, so the window
# is guaranteed to have cleared by the time it ends.
RATE_LIMIT_COOLDOWN_SECONDS = 65 * 60

# Columns pulled from a Qualifying session.load() result.
_QUALI_COLUMNS = ["DriverId", "Position", "Q1", "Q2", "Q3"]

# Columns pulled from a Race session.load() result.
_RACE_COLUMNS = [
    "DriverId",
    "Abbreviation",
    "TeamId",
    "TeamName",
    "GridPosition",
    "ClassifiedPosition",
    "Position",
    "Status",
    "Points",
    "Laps",
]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def enable_fastf1_cache(cache_dir: Path = config.FASTF1_CACHE_DIR) -> None:
    """Enable FastF1's on-disk cache. Idempotent — safe to call repeatedly.

    Must happen before any schedule/session fetch. Without it, every run
    re-downloads session data from scratch and risks getting rate-limited
    (spec Section 7, "Data integrity").
    """
    global _CACHE_ENABLED
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    _CACHE_ENABLED = True


def _ensure_cache_enabled() -> None:
    if not _CACHE_ENABLED:
        enable_fastf1_cache()


# ---------------------------------------------------------------------------
# Retry helper (rate limits / transient network errors)
# ---------------------------------------------------------------------------


# Errors FastF1/fuzzy-matching raise for a permanently invalid request (bad
# round number, session type that doesn't exist for that event, etc). These
# won't succeed on retry, so fail fast instead of burning time + requests.
_NON_RETRYABLE_EXCEPTIONS = (ValueError, KeyError)


def _with_retries(fn, *args, retries: int = MAX_RETRIES, backoff: float = RETRY_BACKOFF_SECONDS, **kwargs):
    """Call ``fn(*args, **kwargs)``, retrying transient errors with backoff.

    Permanent errors (`_NON_RETRYABLE_EXCEPTIONS` — invalid round, session
    that doesn't exist for this event, ...) are raised immediately.
    Rate-limit errors get one long cooldown + one retry (see
    `RATE_LIMIT_COOLDOWN_SECONDS`), not the short backoff below. Anything
    else (network blips) is retried with short backoff; on the final attempt
    it propagates to the caller, which treats it as "this round is
    unavailable".
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except _NON_RETRYABLE_EXCEPTIONS:
            raise
        except RateLimitExceededError as exc:
            logger.warning(
                "FastF1 rate limit hit (%s). Retrying immediately would only push the "
                "reset further out (rejected calls still count against the window), so "
                "cooling down for %d minutes instead of retrying quickly...",
                exc,
                RATE_LIMIT_COOLDOWN_SECONDS // 60,
            )
            time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
            return fn(*args, **kwargs)  # single retry; let this raise if still limited
        except Exception as exc:  # noqa: BLE001 - FastF1/requests raise a mix of types
            last_exc = exc
            if attempt == retries:
                break
            wait_s = backoff * attempt
            logger.warning(
                "Transient error on attempt %d/%d (%s: %s) — retrying in %.1fs",
                attempt,
                retries,
                type(exc).__name__,
                exc,
                wait_s,
            )
            time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Schedule / round iteration
# ---------------------------------------------------------------------------


def get_season_schedule(season: int) -> pd.DataFrame:
    """Return the event schedule for a season, excluding testing events."""
    _ensure_cache_enabled()
    return _with_retries(fastf1.get_event_schedule, season, include_testing=False)


def iter_completed_rounds(seasons: list[int] | None = None) -> list[tuple[int, int, str]]:
    """List ``(season, round_number, event_name)`` for every already-run race.

    A round counts as "completed" if its event date is in the past — this
    keeps in-progress or future rounds (which would fail or return partial
    data) out of fetch/train runs.
    """
    seasons = seasons if seasons is not None else config.SEASONS
    now = pd.Timestamp.now()
    completed: list[tuple[int, int, str]] = []
    for season in seasons:
        schedule = get_season_schedule(season)
        for _, event in schedule.iterrows():
            event_date = event["EventDate"]
            round_number = int(event["RoundNumber"])
            if round_number <= 0 or pd.isna(event_date) or event_date >= now:
                continue
            completed.append((season, round_number, str(event["EventName"])))
    return completed


# ---------------------------------------------------------------------------
# Per-round fetch
# ---------------------------------------------------------------------------


def _timedelta_to_seconds(value: pd.Timedelta) -> float | None:
    if pd.isna(value):
        return None
    return value.total_seconds()


class _IncompleteSessionError(RuntimeError):
    """Raised when a session `.load()` call succeeds but the results are unusable.

    Seen in practice: under load, FastF1 can silently fall back to a
    "livetiming mirror" that has the driver roster but none of the actual
    session results (grid, classified position, ...). No exception is
    raised in that case, so without this check the empty/garbage row would
    get persisted as if it were real data.
    """


def _validate_race_results(results: pd.DataFrame, season: int, round_number: int) -> None:
    if results.empty:
        raise _IncompleteSessionError(f"{season} round {round_number}: race results are empty")
    # DriverId and GridPosition are populated for every legitimately loaded
    # race, even one with a DNS/withdrawal (that shows up as one null, not
    # all of them). All-null here means the session "loaded" but is hollow.
    if results["DriverId"].eq("").all() or results["GridPosition"].isna().all():
        raise _IncompleteSessionError(
            f"{season} round {round_number}: race session loaded but results look empty "
            "(DriverId/GridPosition all missing) -- likely a partial data source fallback"
        )


def fetch_round_raw(season: int, round_number: int) -> pd.DataFrame | None:
    """Fetch and combine Race + Qualifying results for one round.

    Returns ``None`` (and logs a warning) if either session can't be loaded
    at all, e.g. the round was cancelled or FastF1 has no data for it. This
    is the "missing sessions" half of the graceful-handling requirement; the
    "rate limits" half is `_with_retries` around each load. Also returns
    ``None`` if the race session loads "successfully" but the results are
    hollow (see `_validate_race_results`) -- treated the same as unavailable.
    """
    _ensure_cache_enabled()

    try:
        race = _with_retries(fastf1.get_session, season, round_number, "R")
        _with_retries(race.load, laps=False, telemetry=False, weather=False, messages=False)
        _validate_race_results(race.results, season, round_number)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping %d round %d: race session unavailable (%s)", season, round_number, exc)
        return None

    try:
        quali = _with_retries(fastf1.get_session, season, round_number, "Q")
        _with_retries(quali.load, laps=False, telemetry=False, weather=False, messages=False)
        quali_results = quali.results[_QUALI_COLUMNS].copy()
    except Exception as exc:  # noqa: BLE001
        # Qualifying missing (e.g. cancelled) shouldn't sink the whole round —
        # grid position from the race session is still usable on its own.
        logger.warning(
            "%d round %d: qualifying session unavailable, proceeding without Q1/Q2/Q3 (%s)",
            season,
            round_number,
            exc,
        )
        quali_results = pd.DataFrame(columns=_QUALI_COLUMNS)

    race_results = race.results[_RACE_COLUMNS].copy()

    quali_results = quali_results.rename(columns={"Position": "quali_position"})
    for col in ("Q1", "Q2", "Q3"):
        if col in quali_results.columns:
            quali_results[col.lower() + "_time_s"] = quali_results[col].apply(_timedelta_to_seconds)
    quali_results = quali_results.drop(columns=["Q1", "Q2", "Q3"], errors="ignore")

    merged = race_results.merge(quali_results, on="DriverId", how="left")

    merged = merged.rename(
        columns={
            "DriverId": "driver_id",
            "Abbreviation": "driver_abbreviation",
            "TeamId": "constructor_id",
            "TeamName": "constructor_name",
            "GridPosition": "grid",
            "ClassifiedPosition": "classified_position",
            "Position": "finish_position",
            "Status": "status",
            "Points": "points",
            "Laps": "laps",
        }
    )

    event = race.event
    merged.insert(0, "season", season)
    merged.insert(1, "round", round_number)
    merged.insert(2, "event_name", str(event.get("EventName", "")))
    merged.insert(3, "circuit", str(event.get("Location", "")))
    merged.insert(4, "event_date", event.get("EventDate"))

    return merged


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _raw_round_path(season: int, round_number: int) -> Path:
    return config.DATA_RAW_DIR / str(season) / f"{round_number:02d}.parquet"


def save_raw_round(df: pd.DataFrame, season: int, round_number: int) -> Path:
    path = _raw_round_path(season, round_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def fetch_season_raw(season: int, force: bool = False) -> list[Path]:
    """Fetch every completed round of ``season`` and persist it to data/raw/.

    Rounds already on disk are skipped unless ``force=True``.
    """
    saved_paths: list[Path] = []
    for s, round_number, event_name in iter_completed_rounds([season]):
        path = _raw_round_path(s, round_number)
        if path.exists() and not force:
            logger.info("Round %d (%s) already cached at %s — skipping", round_number, event_name, path)
            saved_paths.append(path)
            continue

        logger.info("Fetching %d round %d: %s", s, round_number, event_name)
        round_df = fetch_round_raw(s, round_number)
        if round_df is None:
            continue
        saved_paths.append(save_raw_round(round_df, s, round_number))

    return saved_paths


def fetch_all_seasons_raw(seasons: list[int] | None = None, force: bool = False) -> list[Path]:
    seasons = seasons if seasons is not None else config.SEASONS
    all_paths: list[Path] = []
    for season in seasons:
        all_paths.extend(fetch_season_raw(season, force=force))
    return all_paths


def load_raw_season(season: int) -> pd.DataFrame:
    """Load and concatenate every cached round parquet for a season."""
    season_dir = config.DATA_RAW_DIR / str(season)
    if not season_dir.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(season_dir.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_raw(seasons: list[int] | None = None) -> pd.DataFrame:
    """Load and concatenate every cached round parquet across seasons."""
    seasons = seasons if seasons is not None else config.SEASONS
    frames = [load_raw_season(season) for season in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FastF1 race data into data/raw/")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=config.SEASONS,
        help=f"Seasons to fetch (default: {config.SEASON_START}-{config.SEASON_END})",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch rounds already cached on disk")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    enable_fastf1_cache()
    paths = fetch_all_seasons_raw(seasons=args.seasons, force=args.force)
    print(f"Saved {len(paths)} round file(s) to {config.DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
