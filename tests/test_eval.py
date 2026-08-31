"""Tests for scripts/eval_past.py.

The backtest is the number the project is defended on, so the scoring
arithmetic is tested directly on hand-built races where the right answer
is obvious by inspection. A silently wrong baseline would invert the
headline conclusion, so the baseline is tested as carefully as the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import eval_past  # noqa: E402


def make_race(
    finish: list[int],
    grid: list[int],
    prob_win: list[float],
    prob_podium: list[float],
    pred_position: list[float],
) -> pd.DataFrame:
    n = len(finish)
    return pd.DataFrame(
        {
            "season": [2030] * n,
            "round": [1] * n,
            "event_name": ["Test GP"] * n,
            "driver_id": [f"d{i}" for i in range(n)],
            "finish_position": [float(f) for f in finish],
            "grid": [float(g) for g in grid],
            "prob_win": prob_win,
            "prob_podium": prob_podium,
            "pred_position": pred_position,
        }
    )


def perfect_race() -> pd.DataFrame:
    """Model predicts the exact result; grid order also matches."""
    return make_race(
        finish=[1, 2, 3, 4, 5],
        grid=[1, 2, 3, 4, 5],
        prob_win=[0.9, 0.05, 0.03, 0.01, 0.01],
        prob_podium=[0.95, 0.9, 0.8, 0.2, 0.1],
        pred_position=[1.0, 2.0, 3.0, 4.0, 5.0],
    )


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------


def test_perfect_prediction_scores_perfectly():
    outcome = eval_past._score_race(perfect_race())
    assert outcome.winner_correct is True
    assert outcome.podium_hits == 3
    assert outcome.position_mae == 0.0
    assert outcome.podium_hit_rate == 1.0


def test_winner_is_taken_from_highest_win_probability():
    race = perfect_race()
    race["prob_win"] = [0.01, 0.9, 0.03, 0.05, 0.01]  # d1 favoured, d0 actually wins
    outcome = eval_past._score_race(race)
    assert outcome.predicted_winner == "d1"
    assert outcome.actual_winner == "d0"
    assert outcome.winner_correct is False


def test_podium_hits_count_set_overlap_not_order():
    """Predicting the right three drivers in the wrong order is still 3 hits."""
    race = perfect_race()
    race["prob_podium"] = [0.8, 0.95, 0.9, 0.2, 0.1]  # same three, reordered
    assert eval_past._score_race(race).podium_hits == 3


def test_partial_podium_hits_are_counted():
    race = perfect_race()
    race["prob_podium"] = [0.95, 0.9, 0.1, 0.85, 0.05]  # d3 in place of d2
    assert eval_past._score_race(race).podium_hits == 2


def test_position_mae_uses_rank_not_raw_prediction():
    """Raw regressor output is unconstrained, so the metric must rank first."""
    race = perfect_race()
    # Same ordering, wildly different scale -- MAE should still be 0.
    race["pred_position"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert eval_past._score_race(race).position_mae == 0.0


def test_position_mae_reflects_misordering():
    race = perfect_race()
    race["pred_position"] = [2.0, 1.0, 3.0, 4.0, 5.0]  # top two swapped
    # Two drivers off by one place each, across five drivers.
    assert eval_past._score_race(race).position_mae == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# Baseline scoring -- a wrong baseline would invert the headline conclusion
# ---------------------------------------------------------------------------


def test_baseline_predicts_pole_sitter_as_winner():
    race = make_race(
        finish=[3, 1, 2, 4, 5],
        grid=[1, 2, 3, 4, 5],
        prob_win=[0.1] * 5,
        prob_podium=[0.5] * 5,
        pred_position=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    outcome = eval_past._score_race(race)
    assert outcome.baseline_winner == "d0", "pole sitter is grid position 1"
    assert outcome.baseline_winner_correct is False, "d1 actually won"


def test_baseline_podium_is_the_front_three_of_the_grid():
    race = make_race(
        finish=[1, 2, 3, 4, 5],
        grid=[5, 4, 3, 2, 1],  # grid exactly reversed from the result
        prob_win=[0.1] * 5,
        prob_podium=[0.5] * 5,
        pred_position=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    outcome = eval_past._score_race(race)
    # Grid front three are d4, d3, d2; actual podium is d0, d1, d2 -> 1 overlap.
    assert outcome.baseline_podium_hits == 1


def test_baseline_is_independent_of_model_predictions():
    """Changing model outputs must not move the baseline."""
    race = perfect_race()
    first = eval_past._score_race(race)
    race["prob_win"] = [0.01, 0.01, 0.01, 0.01, 0.96]
    race["prob_podium"] = [0.1, 0.1, 0.1, 0.1, 0.9]
    race["pred_position"] = [5.0, 4.0, 3.0, 2.0, 1.0]
    second = eval_past._score_race(race)

    assert first.baseline_winner_correct == second.baseline_winner_correct
    assert first.baseline_podium_hits == second.baseline_podium_hits
    assert first.baseline_position_mae == second.baseline_position_mae
    assert first.winner_correct != second.winner_correct, "model score should have moved"


def test_baseline_handles_pit_lane_start_remapped_to_back():
    """grid 0 was remapped to back-of-grid in features; pole must still be 1."""
    race = make_race(
        finish=[1, 2, 3],
        grid=[1, 2, 4],  # 4 = remapped pit lane on a 3-car grid
        prob_win=[0.8, 0.1, 0.1],
        prob_podium=[0.9, 0.8, 0.1],
        pred_position=[1.0, 2.0, 3.0],
    )
    assert eval_past._score_race(race).baseline_winner == "d0"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_summarize_reports_each_season_plus_overall():
    results = pd.DataFrame(
        {
            "season": [2030, 2030, 2031],
            "winner_correct": [True, False, True],
            "baseline_winner_correct": [True, True, False],
            "podium_hits": [3, 0, 3],
            "baseline_podium_hits": [3, 3, 0],
            "position_mae": [1.0, 3.0, 2.0],
            "baseline_position_mae": [2.0, 2.0, 2.0],
        }
    )
    summary = eval_past.summarize(results)

    assert summary["season"].tolist() == ["2030", "2031", "ALL"]
    assert summary[summary["season"] == "ALL"]["winner_acc"].iloc[0] == pytest.approx(2 / 3)
    assert summary[summary["season"] == "2030"]["winner_acc"].iloc[0] == pytest.approx(0.5)
    assert summary[summary["season"] == "ALL"]["podium_rate"].iloc[0] == pytest.approx(6 / 9)


def test_summary_renders_both_model_and_baseline():
    results = pd.DataFrame(
        {
            "season": [2030],
            "winner_correct": [True],
            "baseline_winner_correct": [False],
            "podium_hits": [2],
            "baseline_podium_hits": [1],
            "position_mae": [2.0],
            "baseline_position_mae": [3.0],
        }
    )
    text = eval_past.format_summary(eval_past.summarize(results))
    assert "model / grid base" in text
    assert "Gap vs simply predicting the grid order" in text
