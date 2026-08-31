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


# ---------------------------------------------------------------------------
# Holdout evaluation (scripts/eval_holdout.py)
# ---------------------------------------------------------------------------


def _fake_models(drivers: list[str], constructors: list[str]):
    from src import predict as predict_module

    categories = {"driver_id": drivers, "constructor_id": constructors, "circuit_id": ["c"]}
    artifact = {"model": None, "categories": categories}
    return predict_module.LoadedModels(podium=artifact, position=artifact, winner=artifact)


def test_unseen_categories_flags_new_drivers_and_teams():
    """A season with new entrants is harder for reasons unrelated to racing."""
    from scripts import eval_holdout

    rows = pd.DataFrame(
        {
            "driver_id": ["known_a", "rookie", "known_b"],
            "constructor_id": ["old_team", "new_team", "old_team"],
        }
    )
    models = _fake_models(["known_a", "known_b"], ["old_team"])
    unseen = eval_holdout.unseen_categories(rows, models)

    assert unseen["driver_id"] == ["rookie"]
    assert unseen["constructor_id"] == ["new_team"]


def test_unseen_categories_empty_when_everything_is_known():
    from scripts import eval_holdout

    rows = pd.DataFrame({"driver_id": ["a"], "constructor_id": ["t"]})
    unseen = eval_holdout.unseen_categories(rows, _fake_models(["a", "b"], ["t"]))
    assert unseen == {"driver_id": [], "constructor_id": []}


def _holdout_results(n_races: int = 12, model_wins: int = 8, base_wins: int = 9) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * n_races,
            "round_number": list(range(1, n_races + 1)),
            "event_name": [f"GP {i}" for i in range(1, n_races + 1)],
            "n_drivers": [22] * n_races,
            "predicted_winner": ["a"] * n_races,
            "actual_winner": ["a"] * n_races,
            "winner_correct": [True] * model_wins + [False] * (n_races - model_wins),
            "baseline_winner_correct": [True] * base_wins + [False] * (n_races - base_wins),
            "podium_hits": [2] * n_races,
            "baseline_podium_hits": [2] * n_races,
            "position_mae": [4.0] * n_races,
            "baseline_position_mae": [3.5] * n_races,
        }
    )


def test_holdout_report_states_the_race_count_caveat():
    """A 12-race sample cannot support a winner-accuracy conclusion; say so."""
    from scripts import eval_holdout

    text = eval_holdout.format_report(2026, _holdout_results(), {"driver_id": [], "constructor_id": []})
    assert "dominated by noise" in text
    assert "8/12 vs 9/12" in text
    assert "difference of 1 race" in text


def test_holdout_report_points_at_the_more_reliable_metric():
    from scripts import eval_holdout

    text = eval_holdout.format_report(2026, _holdout_results(), {"driver_id": [], "constructor_id": []})
    assert "264 driver-races" in text, "should scale by drivers, not races"
    assert "more reliable signal" in text


def test_holdout_report_lists_unseen_entrants_when_present():
    from scripts import eval_holdout

    text = eval_holdout.format_report(
        2026, _holdout_results(), {"driver_id": ["rookie"], "constructor_id": ["audi"]}
    )
    assert "rookie" in text and "audi" in text
    assert "not guessed" in text


# ---------------------------------------------------------------------------
# Winner calibration (scripts/eval_winners.py)
# ---------------------------------------------------------------------------


def make_calibration_frame(prob: list[float], won: list[int]) -> pd.DataFrame:
    from src import config

    return pd.DataFrame({"prob_win": prob, config.TARGET_WINNER: won})


def test_perfectly_calibrated_predictions_score_zero_error():
    from scripts import eval_winners

    # In the 0.4-0.6 band: 10 predictions at 0.5, exactly 5 of which win.
    frame = make_calibration_frame([0.5] * 10, [1] * 5 + [0] * 5)
    table = eval_winners.reliability_table(frame)
    assert eval_winners.expected_calibration_error(table) == pytest.approx(0.0)


def test_overconfident_predictions_show_a_negative_gap():
    from scripts import eval_winners

    # Model says 90%, only 3 of 10 actually win.
    frame = make_calibration_frame([0.9] * 10, [1] * 3 + [0] * 7)
    table = eval_winners.reliability_table(frame)
    assert table["gap"].iloc[0] < 0, "over-confidence should read as a negative gap"
    assert eval_winners.expected_calibration_error(table) == pytest.approx(0.6)


def test_underconfident_predictions_show_a_positive_gap():
    from scripts import eval_winners

    frame = make_calibration_frame([0.1] * 10, [1] * 5 + [0] * 5)
    table = eval_winners.reliability_table(frame)
    assert table["gap"].iloc[0] > 0


def test_reliability_table_counts_rows_and_wins_per_band():
    from scripts import eval_winners

    frame = make_calibration_frame([0.005] * 4 + [0.9] * 2, [0, 0, 0, 0, 1, 1])
    table = eval_winners.reliability_table(frame)
    assert table["n"].sum() == 6
    assert table["wins"].sum() == 2


def test_actionable_calibration_ignores_the_near_zero_band():
    """The overall ECE is dominated by no-hope drivers; this must not be."""
    from scripts import eval_winners

    # 1000 correctly-near-zero rows, plus 10 badly over-confident ones.
    frame = make_calibration_frame([0.001] * 1000 + [0.9] * 10, [0] * 1000 + [1] * 3 + [0] * 7)
    table = eval_winners.reliability_table(frame)

    overall = eval_winners.expected_calibration_error(table)
    top, n = eval_winners.actionable_calibration(table)

    assert n == 10
    assert top > overall * 10, "restricting to confident predictions should expose the error the aggregate hides"


def test_brier_score_rewards_confident_correct_predictions():
    from scripts import eval_winners

    confident = make_calibration_frame([0.95, 0.05], [1, 0])
    hedged = make_calibration_frame([0.55, 0.45], [1, 0])
    assert eval_winners.brier_score(confident) < eval_winners.brier_score(hedged)


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
