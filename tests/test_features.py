"""Tests for src/features.py.

Covers the three things the rebuild spec (Section 5, task 7) calls out
explicitly:
  1. no leakage -- assert_no_leakage actually raises on a post-race column,
     and the real config.FEATURE_COLUMNS is clean
  2. row counts match expected driver-per-race counts (and withdrawal rows
     with no target are dropped, without corrupting other drivers' rows)
  3. no unexpected nulls in required features

Plus explicit no-lookahead tests for the three backward-shift computations
(rolling form, DNF rate, standings) -- these are the easiest place to
silently introduce leakage, so each gets a small hand-computed fixture
rather than trusting the implementation.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src import config, features

pytestmark = pytest.mark.filterwarnings("ignore:.*DataFrame concatenation.*:FutureWarning")


def make_row(
    *,
    season: int = 2030,
    round: int = 1,  # noqa: A002 - matches the raw column name
    driver_id: str = "drv_a",
    constructor_id: str = "team_x",
    grid: float | None = 1.0,
    finish_position: float | None = 1.0,
    status: str = "Finished",
    points: float = 25.0,
    sprint_points: float = 0.0,
    quali_position: float | None = 1.0,
) -> dict:
    """A raw driver-race row with sensible defaults, matching data_fetch.py's schema.

    Every test overrides only the fields it cares about.
    """
    return {
        "season": season,
        "round": round,
        "event_name": f"Test GP {round}",
        "circuit": "Testville",
        "event_date": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=round),
        "driver_id": driver_id,
        "driver_abbreviation": driver_id[:3].upper(),
        "constructor_id": constructor_id,
        "constructor_name": constructor_id,
        "grid": grid,
        "classified_position": str(int(finish_position)) if finish_position is not None else "W",
        "finish_position": finish_position,
        "status": status,
        "points": points,
        "sprint_points": sprint_points,
        "laps": 50.0,
        "quali_position": quali_position,
        "q1_time_s": 90.0,
        "q2_time_s": None,
        "q3_time_s": None,
    }


def build(rows: list[dict]) -> pd.DataFrame:
    return features.build_feature_table(pd.DataFrame(rows))


def get_row(table: pd.DataFrame, *, round: int, driver_id: str = "drv_a", season: int = 2030) -> pd.Series:  # noqa: A002
    match = table[(table["season"] == season) & (table["round"] == round) & (table["driver_id"] == driver_id)]
    assert len(match) == 1, (
        f"expected exactly 1 row for season={season} round={round} driver={driver_id}, got {len(match)}"
    )
    return match.iloc[0]


# ---------------------------------------------------------------------------
# 1. Leakage
# ---------------------------------------------------------------------------


def test_feature_columns_contain_no_post_race_columns():
    """Static check: the declared feature list itself must be clean."""
    leaked = set(config.FEATURE_COLUMNS) & set(config.POST_RACE_ONLY_COLUMNS)
    assert not leaked


def test_assert_no_leakage_raises_on_planted_leak():
    with pytest.raises(features.LeakageError):
        features.assert_no_leakage(["driver_id", "grid", "finish_position"])
    with pytest.raises(features.LeakageError):
        features.assert_no_leakage(["driver_id", "status"])


def test_assert_no_leakage_passes_on_real_feature_columns():
    features.assert_no_leakage(config.FEATURE_COLUMNS)  # must not raise


def test_build_feature_table_calls_leakage_assertion(monkeypatch):
    """build_feature_table must actually call the assertion, not just define it."""
    calls = []
    monkeypatch.setattr(features, "assert_no_leakage", lambda cols: calls.append(list(cols)))
    build([make_row(round=1), make_row(round=2, finish_position=2.0, points=18.0)])
    assert calls, "build_feature_table did not call assert_no_leakage"


# ---------------------------------------------------------------------------
# 2. No-lookahead: rolling average finish
# ---------------------------------------------------------------------------


def test_rolling_avg_finish_has_no_lookahead():
    rows = [
        make_row(round=1, finish_position=10.0, points=1.0),
        make_row(round=2, finish_position=5.0, points=10.0),
        make_row(round=3, finish_position=1.0, points=25.0),
    ]
    table = build(rows)

    r1 = get_row(table, round=1)
    r2 = get_row(table, round=2)
    r3 = get_row(table, round=3)

    assert math.isnan(r1["driver_rolling_avg_finish"]), "debut race must have no prior history"
    assert r2["driver_rolling_avg_finish"] == 10.0, "round 2 must only see round 1's result"
    assert r3["driver_rolling_avg_finish"] == pytest.approx((10.0 + 5.0) / 2), (
        "round 3 must average rounds 1-2 only, never its own finish (1.0)"
    )


def test_constructor_rolling_avg_finish_pools_both_cars():
    rows = [
        make_row(round=1, driver_id="drv_a", finish_position=2.0),
        make_row(round=1, driver_id="drv_b", finish_position=6.0),
        make_row(round=2, driver_id="drv_a", finish_position=1.0),
        make_row(round=2, driver_id="drv_b", finish_position=1.0),
    ]
    table = build(rows)
    r2a = get_row(table, round=2, driver_id="drv_a")
    r2b = get_row(table, round=2, driver_id="drv_b")
    # Both cars at round 2 should see the SAME constructor history from
    # round 1 (both cars' round-1 finishes), regardless of which car it is.
    assert r2a["constructor_rolling_avg_finish"] == pytest.approx((2.0 + 6.0) / 2)
    assert r2b["constructor_rolling_avg_finish"] == pytest.approx((2.0 + 6.0) / 2)


# ---------------------------------------------------------------------------
# 3. No-lookahead: DNF rate
# ---------------------------------------------------------------------------


def test_dnf_rate_has_no_lookahead():
    rows = [
        make_row(round=1, status="Retired", finish_position=15.0, points=0.0),
        make_row(round=2, status="Finished", finish_position=3.0, points=15.0),
        make_row(round=3, status="Finished", finish_position=2.0, points=18.0),
    ]
    table = build(rows)

    assert math.isnan(get_row(table, round=1)["driver_dnf_rate"])
    assert get_row(table, round=2)["driver_dnf_rate"] == 1.0, "round 2 must see round 1's DNF"
    assert get_row(table, round=3)["driver_dnf_rate"] == pytest.approx(0.5), (
        "round 3 must average rounds 1-2 (1 DNF of 2), not count its own (non-DNF) result"
    )


@pytest.mark.parametrize("status", ["+1 Lap", "+2 Laps", "Lapped", "Finished"])
def test_classified_finish_statuses_are_not_dnfs(status):
    rows = [
        make_row(round=1, status=status, finish_position=12.0, points=0.0),
        make_row(round=2, status="Finished", finish_position=1.0, points=25.0),
    ]
    table = build(rows)
    assert get_row(table, round=2)["driver_dnf_rate"] == 0.0


@pytest.mark.parametrize("status", ["Retired", "Accident", "Collision", "Engine", "Disqualified", "Did not start"])
def test_non_finish_statuses_are_dnfs(status):
    rows = [
        make_row(round=1, status=status, finish_position=18.0, points=0.0),
        make_row(round=2, status="Finished", finish_position=1.0, points=25.0),
    ]
    table = build(rows)
    assert get_row(table, round=2)["driver_dnf_rate"] == 1.0


# ---------------------------------------------------------------------------
# 4. No-lookahead: championship standings
# ---------------------------------------------------------------------------


def test_points_before_race_resets_each_season():
    rows = [
        make_row(season=2030, round=1, points=25.0, finish_position=1.0),
        make_row(season=2030, round=2, points=18.0, finish_position=2.0),
        make_row(season=2031, round=1, points=10.0, finish_position=4.0),
    ]
    table = build(rows)
    assert get_row(table, season=2030, round=1)["driver_points_before_race"] == 0.0
    assert get_row(table, season=2030, round=2)["driver_points_before_race"] == 25.0
    s2031_r1 = get_row(table, season=2031, round=1)
    assert s2031_r1["driver_points_before_race"] == 0.0, "must not carry 2030 points into a new season"


def test_sprint_points_count_toward_standings():
    rows = [
        make_row(round=1, points=25.0, sprint_points=0.0, finish_position=1.0),
        make_row(round=2, points=0.0, sprint_points=8.0, finish_position=10.0, status="Retired"),
        make_row(round=3, points=18.0, sprint_points=0.0, finish_position=2.0),
    ]
    table = build(rows)
    r3 = get_row(table, round=3)
    assert r3["driver_points_before_race"] == 33.0, "must include round 2's sprint points (25 + 8)"


def test_standing_before_race_ranks_by_prior_points():
    rows = [
        make_row(round=1, driver_id="leader", constructor_id="team_x", finish_position=1.0, points=25.0),
        make_row(round=1, driver_id="chaser", constructor_id="team_y", finish_position=2.0, points=18.0),
        make_row(round=2, driver_id="leader", constructor_id="team_x", finish_position=2.0, points=18.0),
        make_row(round=2, driver_id="chaser", constructor_id="team_y", finish_position=1.0, points=25.0),
    ]
    table = build(rows)
    r2_leader = get_row(table, round=2, driver_id="leader")
    r2_chaser = get_row(table, round=2, driver_id="chaser")
    assert r2_leader["driver_standing_before_race"] == 1.0, "leader's round-1 win puts them P1 entering round 2"
    assert r2_chaser["driver_standing_before_race"] == 2.0


# ---------------------------------------------------------------------------
# 4b. Pit-lane starts
# ---------------------------------------------------------------------------


def test_pit_lane_start_is_remapped_to_back_of_grid():
    """grid==0 (pit lane) must become worse than last, not better than pole."""
    rows = [
        make_row(round=1, driver_id="pole", grid=1.0, finish_position=1.0),
        make_row(round=1, driver_id="last", grid=3.0, finish_position=2.0),
        make_row(round=1, driver_id="pitlane", grid=0.0, finish_position=3.0),
    ]
    table = build(rows)
    pit = get_row(table, round=1, driver_id="pitlane")
    assert pit["grid"] == 4.0, "pit-lane start should sort behind the last real grid slot (3 -> 4)"
    assert pit["grid"] > get_row(table, round=1, driver_id="last")["grid"]
    assert pit["grid"] > get_row(table, round=1, driver_id="pole")["grid"]


def test_pit_lane_remap_is_per_round():
    """The remap uses each round's own grid size, not a global constant."""
    rows = [
        make_row(round=1, driver_id="a", grid=2.0),
        make_row(round=1, driver_id="pit1", grid=0.0),
        make_row(round=2, driver_id="a", grid=5.0),
        make_row(round=2, driver_id="pit1", grid=0.0),
    ]
    table = build(rows)
    assert get_row(table, round=1, driver_id="pit1")["grid"] == 3.0
    assert get_row(table, round=2, driver_id="pit1")["grid"] == 6.0


def test_normal_grid_positions_are_untouched():
    rows = [make_row(round=1, driver_id=f"d{i}", grid=float(i)) for i in range(1, 6)]
    table = build(rows)
    for i in range(1, 6):
        assert get_row(table, round=1, driver_id=f"d{i}")["grid"] == float(i)


# ---------------------------------------------------------------------------
# 5. Row counts / dropped rows
# ---------------------------------------------------------------------------


def test_row_count_matches_driver_count_per_round():
    rows = [
        make_row(round=1, driver_id=d, constructor_id="team_x", finish_position=float(i + 1))
        for i, d in enumerate(["a", "b", "c"])
    ]
    table = build(rows)
    assert len(table) == 3
    assert table.groupby(["season", "round"]).size().iloc[0] == 3


def test_withdrawal_row_is_dropped_but_others_unaffected():
    rows = [
        make_row(round=1, driver_id="a", finish_position=1.0, points=25.0),
        make_row(round=1, driver_id="b", finish_position=None, grid=None, status="Withdrew", points=0.0),
        make_row(round=2, driver_id="a", finish_position=1.0, points=25.0),
        make_row(round=2, driver_id="b", finish_position=1.0, points=25.0),
    ]
    table = build(rows)

    assert len(table[(table["round"] == 1) & (table["driver_id"] == "b")]) == 0, "withdrawal row must be dropped"
    assert len(table[(table["round"] == 1) & (table["driver_id"] == "a")]) == 1, "other driver must be unaffected"

    # driver b's round-2 rolling avg must skip straight to "no history" —
    # not corrupted by the dropped round-1 row.
    b_r2 = get_row(table, round=2, driver_id="b")
    assert math.isnan(b_r2["driver_rolling_avg_finish"])
    assert b_r2["driver_points_before_race"] == 0.0


def test_no_duplicate_driver_race_rows():
    rows = [
        make_row(round=r, driver_id=d, constructor_id="team_x" if d in ("a", "b") else "team_y")
        for r in (1, 2, 3)
        for d in ("a", "b", "c", "d")
    ]
    table = build(rows)
    assert table.duplicated(subset=["season", "round", "driver_id"]).sum() == 0


# ---------------------------------------------------------------------------
# 6. Required-feature nulls / target consistency
# ---------------------------------------------------------------------------


_ALWAYS_POPULATED = [
    "driver_id",
    "constructor_id",
    "grid",
    "circuit_id",
    "season",
    "driver_points_before_race",
    "driver_standing_before_race",
    "constructor_points_before_race",
    "constructor_standing_before_race",
]


def test_no_unexpected_nulls_in_required_features():
    rows = [
        make_row(round=r, driver_id=d, constructor_id="team_x" if d in ("a", "b") else "team_y")
        for r in (1, 2, 3)
        for d in ("a", "b", "c", "d")
    ]
    table = build(rows)
    for col in _ALWAYS_POPULATED:
        assert table[col].isnull().sum() == 0, f"unexpected null(s) in required column {col!r}"


def test_debut_nulls_are_limited_to_form_and_dnf_columns():
    """Every column *except* rolling-avg/DNF-rate must be fully populated even on a driver's debut."""
    rows = [make_row(round=1)]
    table = build(rows)
    row = table.iloc[0]
    allowed_null_on_debut = {
        "driver_rolling_avg_finish",
        "constructor_rolling_avg_finish",
        "driver_dnf_rate",
        "constructor_dnf_rate",
        "quali_position",
        "q1_time_s",
        "q2_time_s",
        "q3_time_s",
    }
    for col in table.columns:
        if col in allowed_null_on_debut:
            continue
        assert pd.notna(row[col]), f"unexpected null in {col!r} on a driver's first-ever race"


@pytest.mark.parametrize(
    ("finish_position", "expected_podium", "expected_winner"),
    [(1.0, 1, 1), (2.0, 1, 0), (3.0, 1, 0), (4.0, 0, 0), (15.0, 0, 0)],
)
def test_targets_derived_correctly_from_finish_position(finish_position, expected_podium, expected_winner):
    table = build([make_row(round=1, finish_position=finish_position)])
    row = table.iloc[0]
    assert row["podium_finish"] == expected_podium
    assert row["is_winner"] == expected_winner
    assert row["finish_position"] == finish_position


def test_build_feature_table_raises_on_empty_input():
    with pytest.raises(ValueError):
        features.build_feature_table(pd.DataFrame())


# ---------------------------------------------------------------------------
# 7. Integration check against the real cached dataset, if present.
#    Skipped (not failed) when data/raw hasn't been fetched, e.g. on a
#    fresh checkout or CI without network access to FastF1.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not any(config.DATA_RAW_DIR.glob("*/*.parquet")), reason="data/raw not populated -- run data_fetch first")
def test_real_dataset_invariants():
    from src import data_fetch

    raw = data_fetch.load_all_raw()
    table = features.build_feature_table(raw)

    n_rounds = table.groupby(["season", "round"]).ngroups
    assert table["is_winner"].sum() == n_rounds, "exactly one winner per round"
    assert table["podium_finish"].sum() == n_rounds * 3, "exactly three podium finishers per round"
    assert table.duplicated(subset=["season", "round", "driver_id"]).sum() == 0
    assert (table["grid"] == 0).sum() == 0, "pit-lane starts should have been remapped off 0"
    for col in _ALWAYS_POPULATED:
        assert table[col].isnull().sum() == 0


# ---------------------------------------------------------------------------
# 8. data_fetch round-completeness gate (guards against silently keeping a
#    round that was persisted with degraded data during a failed fetch).
# ---------------------------------------------------------------------------


def test_is_round_complete_rejects_round_with_no_quali(tmp_path):
    from src import data_fetch

    path = tmp_path / "01.parquet"
    pd.DataFrame([make_row(round=1, quali_position=None), make_row(round=1, driver_id="b", quali_position=None)]).to_parquet(
        path, index=False
    )
    assert data_fetch.is_round_complete(path) is False


def test_is_round_complete_accepts_normal_round(tmp_path):
    from src import data_fetch

    path = tmp_path / "01.parquet"
    pd.DataFrame([make_row(round=1, quali_position=1.0), make_row(round=1, driver_id="b", quali_position=None)]).to_parquet(
        path, index=False
    )
    assert data_fetch.is_round_complete(path) is True, "a single non-null quali entry means the session loaded"


def test_is_round_complete_rejects_unreadable_file(tmp_path):
    from src import data_fetch

    path = tmp_path / "broken.parquet"
    path.write_text("not a parquet file")
    assert data_fetch.is_round_complete(path) is False
