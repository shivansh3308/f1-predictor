"""Tests for src/predict.py.

Model loading is mocked (spec task 21), so these run without trained
artifacts on disk and without refitting anything. What matters here is the
contract `src/render.py` and the CLI depend on: the output schema, and that
win probabilities form a real per-race distribution summing to 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, predict


class FakeClassifier:
    """Returns fixed positive-class probabilities, in row order."""

    def __init__(self, probabilities: list[float]):
        self._probabilities = np.array(probabilities, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._probabilities[: len(X)]
        return np.column_stack([1 - p, p])


class FakeRegressor:
    def __init__(self, predictions: list[float]):
        self._predictions = np.array(predictions, dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predictions[: len(X)]


def make_table(n: int = 4, season: int = 2030, round_number: int = 1) -> pd.DataFrame:
    """A feature table with every column build_model_matrix requires."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "season": season,
                "round": round_number,
                "event_name": "Test Grand Prix",
                "driver_id": f"driver_{i}",
                "driver_abbreviation": f"D{i}",
                "constructor_id": f"team_{i % 2}",
                "grid": float(i + 1),
                "quali_position": float(i + 1),
                "q1_time_s": 90.0 + i,
                "q2_time_s": None,
                "q3_time_s": None,
                "driver_rolling_avg_finish": float(i + 1),
                "constructor_rolling_avg_finish": float(i + 1),
                "driver_points_before_race": float(20 - i),
                "driver_standing_before_race": float(i + 1),
                "constructor_points_before_race": float(20 - i),
                "constructor_standing_before_race": float(i % 2 + 1),
                "circuit_id": "test_circuit",
                "driver_dnf_rate": 0.1,
                "constructor_dnf_rate": 0.1,
                config.TARGET_POSITION: float(i + 1),
                config.TARGET_PODIUM: float(i < 3),
                config.TARGET_WINNER: float(i == 0),
            }
        )
    return pd.DataFrame(rows)


def make_models(
    podium: list[float] | None = None,
    position: list[float] | None = None,
    winner: list[float] | None = None,
    categories: dict[str, list] | None = None,
) -> predict.LoadedModels:
    """A LoadedModels bundle backed by fakes, shaped like a real artifact."""
    if categories is None:
        categories = {
            "driver_id": [f"driver_{i}" for i in range(4)],
            "constructor_id": ["team_0", "team_1"],
            "circuit_id": ["test_circuit"],
        }

    def artifact(model):
        return {"model": model, "categories": categories, "feature_columns": list(config.FEATURE_COLUMNS)}

    return predict.LoadedModels(
        podium=artifact(FakeClassifier(podium or [0.9, 0.7, 0.5, 0.1])),
        position=artifact(FakeRegressor(position or [1.0, 2.0, 3.0, 4.0])),
        winner=artifact(FakeClassifier(winner or [0.8, 0.3, 0.1, 0.05])),
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


def test_output_has_exactly_the_declared_columns_in_order():
    result = predict.predict_round(2030, 1, models=make_models(), table=make_table())
    assert list(result.columns) == predict.PREDICTION_COLUMNS


def test_output_has_one_row_per_driver():
    result = predict.predict_round(2030, 1, models=make_models(), table=make_table(n=4))
    assert len(result) == 4
    assert result["driver_id"].nunique() == 4


def test_output_carries_the_original_probability_table_columns():
    """The four columns the original project's probability table used."""
    result = predict.predict_round(2030, 1, models=make_models(), table=make_table())
    for column in ("driver_id", "constructor_id", "grid", "prob_win"):
        assert column in result.columns


def test_output_is_sorted_by_win_probability():
    result = predict.predict_round(2030, 1, models=make_models(), table=make_table())
    assert result["prob_win"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Win probabilities form a real distribution
# ---------------------------------------------------------------------------


def test_win_probabilities_sum_to_one():
    result = predict.predict_round(2030, 1, models=make_models(), table=make_table())
    assert result["prob_win"].sum() == pytest.approx(1.0)


def test_win_probabilities_sum_to_one_even_when_raw_scores_are_tiny():
    """Raw classifier output need not sum to anything in particular."""
    models = make_models(winner=[0.01, 0.008, 0.005, 0.002])
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    assert result["prob_win"].sum() == pytest.approx(1.0)


def test_win_probabilities_sum_to_one_even_when_raw_scores_exceed_one():
    models = make_models(winner=[0.9, 0.85, 0.8, 0.75])
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    assert result["prob_win"].sum() == pytest.approx(1.0)


def test_win_probabilities_stay_in_the_unit_interval():
    result = predict.predict_round(2030, 1, models=make_models(), table=make_table())
    assert (result["prob_win"] >= 0).all()
    assert (result["prob_win"] <= 1).all()


def test_normalization_preserves_the_ranking_of_the_raw_scores():
    models = make_models(winner=[0.2, 0.9, 0.1, 0.4])
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    assert result.iloc[0]["driver_id"] == "driver_1", "highest raw score should stay first"


def test_each_race_is_normalized_independently():
    """Two rounds in one table must each sum to 1, not share a total."""
    table = pd.concat([make_table(round_number=1), make_table(round_number=2)], ignore_index=True)
    for round_number in (1, 2):
        result = predict.predict_round(2030, round_number, models=make_models(), table=table)
        assert result["prob_win"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Model outputs are wired to the right columns
# ---------------------------------------------------------------------------


def test_podium_probability_comes_from_the_podium_model():
    models = make_models(podium=[0.11, 0.22, 0.33, 0.44])
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    by_driver = result.set_index("driver_id")["prob_podium"]
    assert by_driver["driver_0"] == pytest.approx(0.11)
    assert by_driver["driver_3"] == pytest.approx(0.44)


def test_predicted_position_comes_from_the_position_model():
    models = make_models(position=[7.5, 1.5, 3.5, 9.5])
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    by_driver = result.set_index("driver_id")["pred_position"]
    assert by_driver["driver_0"] == pytest.approx(7.5)
    assert by_driver["driver_1"] == pytest.approx(1.5)


def test_each_model_uses_its_own_saved_categories():
    """Artifacts are independent; one retrained separately must still work."""
    models = make_models()
    models.winner["categories"] = {
        "driver_id": [f"driver_{i}" for i in range(4)],
        "constructor_id": ["team_0", "team_1"],
        "circuit_id": ["a_different_circuit"],  # deliberately mismatched
    }
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    assert result["prob_win"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Finishing order
# ---------------------------------------------------------------------------


def test_finishing_order_ranks_by_predicted_position():
    models = make_models(position=[4.0, 1.0, 3.0, 2.0])
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    ordered = predict.predicted_finishing_order(result)
    assert ordered["driver_id"].tolist() == ["driver_1", "driver_3", "driver_2", "driver_0"]
    assert ordered["predicted_rank"].tolist() == [1, 2, 3, 4]


def test_finishing_order_respects_top_n():
    result = predict.predict_round(2030, 1, models=make_models(), table=make_table())
    assert len(predict.predicted_finishing_order(result, top_n=2)) == 2


def test_finishing_order_handles_unconstrained_regressor_output():
    """Raw predictions can fall outside 1..N; ranking must still work."""
    models = make_models(position=[0.4, 21.7, -1.2, 8.0])
    result = predict.predict_round(2030, 1, models=models, table=make_table())
    ordered = predict.predicted_finishing_order(result)
    assert ordered["predicted_rank"].tolist() == [1, 2, 3, 4]
    assert ordered.iloc[0]["driver_id"] == "driver_2", "lowest prediction should rank first"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unknown_round_names_the_rounds_that_exist():
    with pytest.raises(predict.RoundNotFoundError, match=r"\[1\]"):
        predict.predict_round(2030, 99, models=make_models(), table=make_table())


def test_unknown_season_names_the_seasons_that_exist():
    with pytest.raises(predict.RoundNotFoundError, match="not in the dataset"):
        predict.predict_round(1999, 1, models=make_models(), table=make_table())


def test_missing_model_artifacts_point_at_the_training_command(tmp_path):
    with pytest.raises(predict.ModelsNotTrainedError, match="scripts/train_models.py"):
        predict.load_models(
            podium_path=tmp_path / "nope.joblib",
            position_path=tmp_path / "nope.joblib",
            winner_path=tmp_path / "nope.joblib",
        )


def test_race_label_returns_the_event_name():
    assert predict.race_label(2030, 1, table=make_table()) == "Test Grand Prix"
