"""Tests for app/app.py.

Everything below the CLI is mocked -- no models are loaded, no network is
touched. What is under test is the routing and the exit-code contract: did
the right round get resolved, did the right function get called, and does a
failure report itself as a failure rather than exiting 0.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from app import app, app_calendar
from src import predict


@pytest.fixture
def rendered(monkeypatch):
    """Capture render_round calls instead of printing a report."""
    calls: list[dict] = []

    def _render(season, round_number, **kwargs):
        calls.append({"season": season, "round": round_number, **kwargs})

    monkeypatch.setattr(app.render, "render_round", _render)
    monkeypatch.setattr(app.predict, "load_models", lambda: object())
    return calls


@pytest.fixture
def fake_calendar(monkeypatch):
    monkeypatch.setattr(
        app_calendar,
        "next_unraced_round",
        lambda season=None: app_calendar.RoundRef(2030, 7, "Next Grand Prix", pd.Timestamp("2030-06-01")),
    )
    monkeypatch.setattr(
        app_calendar,
        "latest_completed_round",
        lambda season=None: app_calendar.RoundRef(2030, 6, "Last Grand Prix", pd.Timestamp("2030-05-18")),
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_all_four_subcommands_are_registered():
    parser = app.build_parser()
    for command in ("any", "upcoming", "latest", "calendar"):
        assert parser.parse_args([command] if command != "any" else ["any", "2023", "1"])


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        app.build_parser().parse_args([])


def test_any_requires_season_and_round():
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["any", "2023"])


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_any_renders_the_requested_round(rendered):
    assert app.main(["any", "2023", "5"]) == app.EXIT_OK
    assert (rendered[0]["season"], rendered[0]["round"]) == (2023, 5)


def test_any_forwards_top_and_actual_flags(rendered):
    app.main(["any", "2023", "5", "--top", "3", "--actual"])
    assert rendered[0]["top_n"] == 3
    assert rendered[0]["show_actual"] is True


def test_upcoming_resolves_the_next_unraced_round(rendered, fake_calendar):
    assert app.main(["upcoming"]) == app.EXIT_OK
    assert (rendered[0]["season"], rendered[0]["round"]) == (2030, 7)


def test_latest_resolves_the_most_recent_round(rendered, fake_calendar):
    assert app.main(["latest"]) == app.EXIT_OK
    assert (rendered[0]["season"], rendered[0]["round"]) == (2030, 6)


def test_latest_always_shows_actual_results(rendered, fake_calendar):
    """Comparing against the real result is the entire point of this command."""
    app.main(["latest"])
    assert rendered[0]["show_actual"] is True


def test_upcoming_does_not_show_actual_results(rendered, fake_calendar):
    """An unraced round has no result to compare against."""
    app.main(["upcoming"])
    assert rendered[0].get("show_actual") is not True


# ---------------------------------------------------------------------------
# Exit codes -- a failure must not report success
# ---------------------------------------------------------------------------


def test_missing_models_exits_with_its_own_code(monkeypatch, capsys):
    def _raise():
        raise predict.ModelsNotTrainedError("no models here")

    monkeypatch.setattr(app.predict, "load_models", _raise)
    assert app.predict_and_render(2023, 5) == app.EXIT_NO_MODELS
    assert "no models here" in capsys.readouterr().err


def test_missing_round_exits_with_no_data_code(monkeypatch, capsys):
    monkeypatch.setattr(app.predict, "load_models", lambda: object())

    def _raise(*a, **k):
        raise predict.RoundNotFoundError("nope")

    monkeypatch.setattr(app.render, "render_round", _raise)
    assert app.predict_and_render(2026, 13) == app.EXIT_NO_DATA
    assert "No data available for 2026 round 13" in capsys.readouterr().err


def test_unfetched_season_message_says_to_fetch_it(monkeypatch):
    """Cause 1: the season was never pulled."""
    monkeypatch.setattr(app.data_fetch, "available_seasons", lambda: [2018, 2019])
    message = app._explain_missing_round(2026, 13)
    assert "has not been fetched" in message
    assert "python -m src.data_fetch --seasons 2026" in message


def test_future_round_message_says_it_has_not_run(monkeypatch):
    """Cause 2: the season is on disk but this race is still in the future.

    Telling someone to fetch data for a race that has not happened is
    actively misleading, so this branch must not do that.
    """
    monkeypatch.setattr(app.data_fetch, "available_seasons", lambda: [2026])
    monkeypatch.setattr(
        app_calendar,
        "get_calendar",
        lambda season: pd.DataFrame(
            {
                "round": [13],
                "event_name": ["Italian Grand Prix"],
                "event_date": [pd.Timestamp("2026-09-06")],
                "has_run": [False],
            }
        ),
    )
    message = app._explain_missing_round(2026, 13)
    assert "has not been run yet" in message
    assert "2026-09-06" in message
    assert "qualifying" in message, "should explain that a grid requires qualifying"
    assert "src.data_fetch" not in message, "must not tell the user to fetch an unraced race"


def test_uncached_round_message_says_to_fetch_the_season(monkeypatch):
    """Cause 3: season on disk, race has run, but this round is not cached."""
    monkeypatch.setattr(app.data_fetch, "available_seasons", lambda: [2026])
    monkeypatch.setattr(
        app_calendar,
        "get_calendar",
        lambda season: pd.DataFrame(
            {
                "round": [5],
                "event_name": ["Some Grand Prix"],
                "event_date": [pd.Timestamp("2026-05-03")],
                "has_run": [True],
            }
        ),
    )
    message = app._explain_missing_round(2026, 5)
    assert "not cached yet" in message
    assert "python -m src.data_fetch --seasons 2026" in message


def test_upcoming_with_no_scheduled_round_exits_no_data(monkeypatch, rendered):
    monkeypatch.setattr(app_calendar, "next_unraced_round", lambda season=None: None)
    assert app.main(["upcoming"]) == app.EXIT_NO_DATA
    assert not rendered, "nothing should be rendered when there is no round"


def test_latest_with_no_completed_round_exits_no_data(monkeypatch, rendered):
    monkeypatch.setattr(app_calendar, "latest_completed_round", lambda season=None: None)
    assert app.main(["latest"]) == app.EXIT_NO_DATA


def test_calendar_with_no_schedule_exits_no_data(monkeypatch):
    monkeypatch.setattr(app_calendar, "get_calendar", lambda season: pd.DataFrame())
    monkeypatch.setattr(app_calendar, "current_season", lambda: 2099)
    assert app.main(["calendar"]) == app.EXIT_NO_DATA


# ---------------------------------------------------------------------------
# On-demand fetch
# ---------------------------------------------------------------------------


def test_fetch_flag_retries_after_pulling_the_round(monkeypatch):
    monkeypatch.setattr(app.predict, "load_models", lambda: object())

    attempts = {"n": 0}

    def _render(*a, **k):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise predict.RoundNotFoundError("not cached yet")

    monkeypatch.setattr(app.render, "render_round", _render)
    monkeypatch.setattr(app, "_fetch_round", lambda s, r: True)

    assert app.predict_and_render(2026, 13, fetch=True) == app.EXIT_OK
    assert attempts["n"] == 2, "should render again after a successful fetch"


def test_predict_any_entrypoint_routes_to_the_requested_round(rendered):
    from app import predict_any

    assert predict_any.main(["2023", "5", "--top", "4"]) == app.EXIT_OK
    assert (rendered[0]["season"], rendered[0]["round"], rendered[0]["top_n"]) == (2023, 5, 4)


def test_predict_upcoming_entrypoint_uses_the_calendar(rendered, fake_calendar):
    from app import predict_upcoming

    assert predict_upcoming.main([]) == app.EXIT_OK
    assert (rendered[0]["season"], rendered[0]["round"]) == (2030, 7)


def test_predict_latest_done_entrypoint_compares_against_results(rendered, fake_calendar):
    from app import predict_latest_done

    assert predict_latest_done.main([]) == app.EXIT_OK
    assert (rendered[0]["season"], rendered[0]["round"]) == (2030, 6)
    assert rendered[0]["show_actual"] is True


def test_entrypoints_share_one_implementation_with_the_subcommands(rendered, fake_calendar):
    """The wrappers must not grow their own copy of the logic."""
    from app import predict_latest_done

    predict_latest_done.main([])
    app.main(["latest"])

    # `models` is a fresh mock object per call, so compare everything else.
    def routing(call: dict) -> dict:
        return {k: v for k, v in call.items() if k != "models"}

    assert routing(rendered[0]) == routing(rendered[1])


def test_fetch_flag_gives_up_cleanly_when_the_round_cannot_be_fetched(monkeypatch, capsys):
    monkeypatch.setattr(app.predict, "load_models", lambda: object())
    monkeypatch.setattr(
        app.render, "render_round", lambda *a, **k: (_ for _ in ()).throw(predict.RoundNotFoundError("nope"))
    )
    monkeypatch.setattr(app, "_fetch_round", lambda s, r: False)

    assert app.predict_and_render(2026, 13, fetch=True) == app.EXIT_NO_DATA
    # Already tried fetching, so it should not suggest --fetch again.
    assert "retry with --fetch" not in capsys.readouterr().err
