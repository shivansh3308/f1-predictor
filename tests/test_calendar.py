"""Tests for app/app_calendar.py.

The FastF1 schedule is mocked throughout: these tests must not depend on
the network, on FastF1's rate limit, or on the real clock. Every lookup
takes an injectable ``now`` for exactly that reason.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import app_calendar as cal
from src import data_fetch

# A compact two-season fixture. 2030 runs Mar-Nov; 2031 starts the next March.
_SCHEDULES = {
    2030: [
        (1, "Opening Grand Prix", "2030-03-10", "Melbourne", "Australia"),
        (2, "Second Grand Prix", "2030-04-14", "Sakhir", "Bahrain"),
        (3, "Third Grand Prix", "2030-07-21", "Spa", "Belgium"),
        (4, "Finale Grand Prix", "2030-11-24", "Yas Island", "UAE"),
    ],
    2031: [
        (1, "New Year Grand Prix", "2031-03-09", "Melbourne", "Australia"),
    ],
}


@pytest.fixture(autouse=True)
def fake_schedule(monkeypatch):
    """Replace the live FastF1 lookup with the fixture above."""

    def _get_season_schedule(season: int) -> pd.DataFrame:
        rounds = _SCHEDULES.get(season)
        if rounds is None:
            raise ValueError(f"no schedule for {season}")
        # Include a round 0 testing entry to prove it gets filtered out.
        rows = [(0, "Pre-Season Testing", f"{season}-02-20", "Sakhir", "Bahrain")] + rounds
        return pd.DataFrame(
            {
                "RoundNumber": [r[0] for r in rows],
                "EventName": [r[1] for r in rows],
                "EventDate": pd.to_datetime([r[2] for r in rows]),
                "Location": [r[3] for r in rows],
                "Country": [r[4] for r in rows],
            }
        )

    monkeypatch.setattr(data_fetch, "get_season_schedule", _get_season_schedule)


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


# ---------------------------------------------------------------------------
# Calendar shape
# ---------------------------------------------------------------------------


def test_calendar_lists_every_race():
    calendar = cal.get_calendar(2030, now=ts("2030-06-01"))
    assert len(calendar) == 4
    assert calendar["round"].tolist() == [1, 2, 3, 4]


def test_calendar_excludes_preseason_testing():
    """Round 0 is testing, not a race."""
    assert 0 not in cal.get_calendar(2030, now=ts("2030-06-01"))["round"].tolist()


def test_calendar_flags_which_rounds_have_run():
    calendar = cal.get_calendar(2030, now=ts("2030-06-01"))
    assert calendar.set_index("round")["has_run"].to_dict() == {1: True, 2: True, 3: False, 4: False}


def test_calendar_for_unpublished_season_is_empty_not_an_error():
    """A future season with no schedule is a normal state."""
    calendar = cal.get_calendar(2099, now=ts("2030-06-01"))
    assert calendar.empty
    assert list(calendar.columns) == cal.CALENDAR_COLUMNS


# ---------------------------------------------------------------------------
# Round name lookup
# ---------------------------------------------------------------------------


def test_round_name_maps_number_to_grand_prix():
    assert cal.round_name(2030, 3) == "Third Grand Prix"


def test_round_name_rejects_unknown_round_with_a_useful_message():
    with pytest.raises(cal.RoundNotFoundError, match="Available rounds: 1-4"):
        cal.round_name(2030, 99)


def test_round_name_reports_missing_calendar():
    with pytest.raises(cal.RoundNotFoundError, match="No calendar published"):
        cal.round_name(2099, 1)


# ---------------------------------------------------------------------------
# Next unraced round
# ---------------------------------------------------------------------------


def test_next_unraced_round_mid_season():
    nxt = cal.next_unraced_round(2030, now=ts("2030-06-01"))
    assert (nxt.season, nxt.round_number, nxt.event_name) == (2030, 3, "Third Grand Prix")


def test_next_unraced_round_is_none_when_season_is_over():
    assert cal.next_unraced_round(2030, now=ts("2030-12-31")) is None


def test_next_unraced_round_before_season_starts_is_round_one():
    assert cal.next_unraced_round(2030, now=ts("2030-01-01")).round_number == 1


def test_next_unraced_round_rolls_into_the_following_season():
    """With the season finished, 'next up' must cross the year boundary."""
    nxt = cal.next_unraced_round(now=ts("2030-12-31"))
    assert (nxt.season, nxt.round_number) == (2031, 1)


def test_next_unraced_round_returns_none_when_nothing_is_scheduled():
    assert cal.next_unraced_round(now=ts("2031-12-31")) is None


# ---------------------------------------------------------------------------
# Latest completed round
# ---------------------------------------------------------------------------


def test_latest_completed_round_mid_season():
    latest = cal.latest_completed_round(2030, now=ts("2030-06-01"))
    assert (latest.round_number, latest.event_name) == (2, "Second Grand Prix")


def test_latest_completed_round_is_the_finale_after_the_season():
    assert cal.latest_completed_round(2030, now=ts("2030-12-31")).round_number == 4


def test_latest_completed_round_is_none_before_the_season_starts():
    assert cal.latest_completed_round(2030, now=ts("2030-01-01")) is None


def test_latest_completed_round_falls_back_to_the_previous_season():
    """In January, the most recent race is still last year's finale."""
    latest = cal.latest_completed_round(now=ts("2031-01-05"))
    assert (latest.season, latest.round_number) == (2030, 4)


# ---------------------------------------------------------------------------
# Current season
# ---------------------------------------------------------------------------


def test_current_season_is_the_calendar_year_when_published():
    assert cal.current_season(now=ts("2030-06-01")) == 2030


def test_current_season_falls_back_when_next_year_is_unpublished():
    assert cal.current_season(now=ts("2032-01-10")) == 2031


# ---------------------------------------------------------------------------
# RoundRef
# ---------------------------------------------------------------------------


def test_round_ref_formats_readably():
    ref = cal.next_unraced_round(2030, now=ts("2030-06-01"))
    assert str(ref) == "2030 Round 3 -- Third Grand Prix"
