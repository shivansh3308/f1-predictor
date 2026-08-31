"""Tests for scripts/update_training_data.py.

FastF1 is mocked throughout -- an update script that hits the network in
tests would be slow and would burn the 500-calls/hour budget the script
itself is designed to conserve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import update_training_data as update  # noqa: E402
from src import config, data_fetch  # noqa: E402

_COMPLETED = [
    (2030, 1, "First Grand Prix"),
    (2030, 2, "Second Grand Prix"),
    (2030, 3, "Third Grand Prix"),
]


@pytest.fixture
def fake_raw_dir(tmp_path, monkeypatch):
    """Point DATA_RAW_DIR at a temp dir and stub the completed-round listing."""
    monkeypatch.setattr(config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(update.config, "DATA_RAW_DIR", tmp_path)
    monkeypatch.setattr(data_fetch, "iter_completed_rounds", lambda seasons: _COMPLETED)
    return tmp_path


def write_round(raw_dir: Path, season: int, round_number: int, complete: bool = True) -> Path:
    import pandas as pd

    path = raw_dir / str(season) / f"{round_number:02d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "quali_position": [1.0, 2.0] if complete else [None, None],
        }
    ).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Detecting what needs fetching
# ---------------------------------------------------------------------------


def test_all_rounds_missing_on_a_fresh_checkout(fake_raw_dir):
    assert len(update.find_missing_rounds([2030])) == 3


def test_cached_rounds_are_not_refetched(fake_raw_dir):
    for round_number in (1, 2, 3):
        write_round(fake_raw_dir, 2030, round_number)
    assert update.find_missing_rounds([2030]) == []


def test_only_the_new_round_is_reported(fake_raw_dir):
    """The normal weekly case: one race has happened since the last update."""
    write_round(fake_raw_dir, 2030, 1)
    write_round(fake_raw_dir, 2030, 2)
    missing = update.find_missing_rounds([2030])
    assert [(s, r) for s, r, _ in missing] == [(2030, 3)]


def test_round_cached_without_qualifying_is_refetched(fake_raw_dir):
    """A rate-limited run can persist a degraded round; it must not stick."""
    write_round(fake_raw_dir, 2030, 1)
    write_round(fake_raw_dir, 2030, 2, complete=False)
    write_round(fake_raw_dir, 2030, 3)

    missing = update.find_missing_rounds([2030])
    assert [(s, r) for s, r, _ in missing] == [(2030, 2)]
    assert "without qualifying" in missing[0][2], "the reason should be visible in the report"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def test_fetch_missing_reports_rounds_it_could_not_retrieve(fake_raw_dir, monkeypatch):
    """A race whose data is not published yet should be reported, not crash."""
    monkeypatch.setattr(data_fetch, "fetch_round_raw", lambda s, r: None)
    monkeypatch.setattr(update.data_fetch, "fetch_round_raw", lambda s, r: None)

    failed = update.fetch_missing(_COMPLETED)
    assert len(failed) == 3


def test_fetch_missing_saves_what_it_retrieves(fake_raw_dir, monkeypatch):
    import pandas as pd

    saved: list[tuple[int, int]] = []
    monkeypatch.setattr(update.data_fetch, "fetch_round_raw", lambda s, r: pd.DataFrame({"driver_id": ["a"]}))
    monkeypatch.setattr(update.data_fetch, "save_raw_round", lambda df, s, r: saved.append((s, r)))

    failed = update.fetch_missing(_COMPLETED[:2])
    assert failed == []
    assert saved == [(2030, 1), (2030, 2)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_dry_run_fetches_nothing(fake_raw_dir, monkeypatch, capsys):
    called = {"n": 0}

    def _fetch(*a, **k):
        called["n"] += 1
        return []

    monkeypatch.setattr(update, "fetch_missing", _fetch)
    monkeypatch.setattr(update.data_fetch, "enable_fastf1_cache", lambda: None)

    assert update.main(["--seasons", "2030", "--dry-run"]) == 0
    assert called["n"] == 0
    assert "nothing fetched" in capsys.readouterr().out


def test_reports_when_already_up_to_date(fake_raw_dir, monkeypatch, capsys):
    for round_number in (1, 2, 3):
        write_round(fake_raw_dir, 2030, round_number)
    monkeypatch.setattr(update.data_fetch, "enable_fastf1_cache", lambda: None)

    assert update.main(["--seasons", "2030"]) == 0
    assert "Already up to date" in capsys.readouterr().out


def test_flags_a_live_season_outside_the_configured_range(fake_raw_dir, monkeypatch, capsys):
    """Ignoring the current season is a deliberate choice, but should be visible."""
    import pandas as pd

    for round_number in (1, 2, 3):
        write_round(fake_raw_dir, 2030, round_number)
    monkeypatch.setattr(update.data_fetch, "enable_fastf1_cache", lambda: None)
    monkeypatch.setattr(update.pd, "Timestamp", pd.Timestamp)

    update.main(["--seasons", "2030"])
    out = capsys.readouterr().out
    current_year = pd.Timestamp.now().year
    assert f"the {current_year} season is not in the configured range" in out
    assert "SEASON_END" in out, "should say what to change to include it"
