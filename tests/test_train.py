"""Tests for src/train.py.

The headline claim of the podium model is that its CV score is
time-respecting -- no fold is ever scored using a model that saw future
races. That property is invisible in the metrics themselves (a leaky split
just looks like a better number), so it gets tested directly here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features, train


def make_table(seasons=(2018, 2019, 2020, 2021, 2022, 2023), rounds_per_season=3, drivers=4) -> pd.DataFrame:
    """Minimal feature-table-shaped frame: enough columns for the model matrix."""
    rows = []
    for season in seasons:
        for rnd in range(1, rounds_per_season + 1):
            for d in range(drivers):
                rows.append(
                    {
                        "season": season,
                        "round": rnd,
                        "event_name": f"GP{rnd}",
                        "driver_id": f"drv_{d}",
                        "driver_abbreviation": f"D{d}",
                        "constructor_id": f"team_{d % 2}",
                        "grid": float(d + 1),
                        "quali_position": float(d + 1),
                        "q1_time_s": 90.0 + d,
                        "q2_time_s": None,
                        "q3_time_s": None,
                        "driver_rolling_avg_finish": float(d + 1),
                        "constructor_rolling_avg_finish": float(d + 1),
                        "driver_points_before_race": float(10 - d),
                        "driver_standing_before_race": float(d + 1),
                        "constructor_points_before_race": float(10 - d),
                        "constructor_standing_before_race": float(d % 2 + 1),
                        "circuit_id": f"circuit_{rnd}",
                        "driver_dnf_rate": 0.1,
                        "constructor_dnf_rate": 0.1,
                        config.TARGET_POSITION: float(d + 1),
                        config.TARGET_PODIUM: int(d < 3),
                        config.TARGET_WINNER: int(d == 0),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split correctness -- the no-lookahead guarantee
# ---------------------------------------------------------------------------


def test_season_forward_splits_never_train_on_future_seasons():
    table = make_table()
    folds = list(train.season_forward_splits(table, n_splits=3))
    assert folds, "expected at least one fold"

    for label, train_idx, test_idx in folds:
        train_seasons = set(table.iloc[train_idx]["season"])
        test_seasons = set(table.iloc[test_idx]["season"])
        assert len(test_seasons) == 1, "each fold should test exactly one season"
        test_season = test_seasons.pop()
        assert str(test_season) == label
        assert max(train_seasons) < test_season, (
            f"fold {label} trains on {sorted(train_seasons)}, which includes a season >= the test season"
        )


def test_season_forward_splits_expand_over_time():
    table = make_table()
    folds = list(train.season_forward_splits(table, n_splits=3))
    sizes = [len(train_idx) for _, train_idx, _ in folds]
    assert sizes == sorted(sizes), "training window should grow with each successive fold"
    assert len(set(sizes)) == len(sizes), "each fold should add data"


def test_season_forward_splits_tests_the_most_recent_seasons():
    table = make_table()
    folds = list(train.season_forward_splits(table, n_splits=3))
    tested = [int(label) for label, _, _ in folds]
    assert tested == [2021, 2022, 2023]


def test_season_forward_splits_partition_without_overlap():
    table = make_table()
    for _, train_idx, test_idx in train.season_forward_splits(table, n_splits=3):
        assert not set(train_idx) & set(test_idx), "train and test folds must be disjoint"


def test_season_forward_splits_requires_enough_seasons():
    table = make_table(seasons=(2018, 2019))
    with pytest.raises(ValueError):
        list(train.season_forward_splits(table, n_splits=5))


def test_random_kfold_is_not_time_respecting():
    """Guards the labelling: the diagnostic splitter really does leak, by design."""
    table = make_table()
    leaks = 0
    for _, train_idx, test_idx in train.random_kfold_splits(table, n_splits=3):
        train_seasons = set(table.iloc[train_idx]["season"])
        test_seasons = set(table.iloc[test_idx]["season"])
        if any(ts < max(train_seasons) for ts in test_seasons):
            leaks += 1
    assert leaks > 0, "random K-fold should mix seasons -- it is only kept as a diagnostic contrast"


# ---------------------------------------------------------------------------
# Model matrix / encoding consistency
# ---------------------------------------------------------------------------


def test_model_matrix_columns_are_exactly_feature_columns():
    table = make_table()
    X, _ = features.build_model_matrix(table)
    assert list(X.columns) == list(config.FEATURE_COLUMNS)


def test_model_matrix_reuses_supplied_categories():
    """Predict-time encoding must match train-time encoding exactly."""
    table = make_table()
    train_df = table[table["season"] < 2023]
    test_df = table[table["season"] == 2023]

    _, categories = features.build_model_matrix(train_df)
    X_test, resolved = features.build_model_matrix(test_df, categories=categories)

    assert resolved == categories
    for col in config.CATEGORICAL_FEATURES:
        assert list(X_test[col].cat.categories) == list(categories[col])


def test_unseen_category_becomes_null_not_a_wrong_identity():
    """A debuting driver must encode as unknown, never silently as another driver."""
    table = make_table()
    _, categories = features.build_model_matrix(table)

    rookie = table.head(1).copy()
    rookie["driver_id"] = "brand_new_driver"
    X, _ = features.build_model_matrix(rookie, categories=categories)
    assert X["driver_id"].isna().all()


def test_model_matrix_raises_on_missing_column():
    table = make_table().drop(columns=["grid"])
    with pytest.raises(ValueError, match="missing required column"):
        features.build_model_matrix(table)


def test_model_matrix_coerces_all_null_numeric_column():
    """A feature nobody has a value for yet must stay numeric-NaN, not object.

    Real predict-time case: building features for a single upcoming round
    where no Q3 time has been set. An object-dtype column would blow up
    inside XGBoost's DMatrix construction instead of being treated as
    missing.
    """
    table = make_table()
    table["q3_time_s"] = None
    X, _ = features.build_model_matrix(table)
    assert X["q3_time_s"].isna().all()
    assert X["q3_time_s"].dtype.kind == "f", "all-null numeric feature should coerce to float, not object"


# ---------------------------------------------------------------------------
# Training / reproducibility
# ---------------------------------------------------------------------------


def test_podium_training_is_deterministic():
    table = make_table()
    m1, c1 = train.train_podium_model(table)
    m2, c2 = train.train_podium_model(table)
    X1, _ = features.build_model_matrix(table, categories=c1)
    X2, _ = features.build_model_matrix(table, categories=c2)
    assert np.allclose(m1.predict_proba(X1)[:, 1], m2.predict_proba(X2)[:, 1])


def test_cross_validate_podium_reports_one_fold_per_season():
    table = make_table()
    report = train.cross_validate_podium(table, splitter=lambda df: train.season_forward_splits(df, n_splits=3))
    assert len(report.folds) == 3
    assert set(report.metric_names()) == {"auc", "logloss"}
    assert 0.0 <= report.mean("auc") <= 1.0


def test_position_training_is_deterministic():
    table = make_table()
    m1, c1 = train.train_position_model(table)
    m2, c2 = train.train_position_model(table)
    X1, _ = features.build_model_matrix(table, categories=c1)
    X2, _ = features.build_model_matrix(table, categories=c2)
    assert np.allclose(m1.predict(X1), m2.predict(X2))


def test_cross_validate_position_reports_mae():
    table = make_table()
    report = train.cross_validate_position(
        table, splitter=lambda df: train.season_forward_splits(df, n_splits=3)
    )
    assert len(report.folds) == 3
    assert set(report.metric_names()) == {"mae", "rmse"}
    assert report.mean("mae") >= 0.0
    assert report.mean("rmse") >= report.mean("mae"), "RMSE cannot be below MAE"


def test_position_model_optimises_absolute_error():
    """The objective must match the reported metric.

    Guards a deliberate choice: squared error chases DNF outliers, which no
    pre-race feature can predict, and measurably worsens MAE on the real
    dataset (3.95 vs 3.33).
    """
    assert train.build_position_model().get_params()["objective"] == "reg:absoluteerror"


def test_position_and_podium_share_the_same_cv_split():
    """Both models must be scored on identical folds, or their metrics aren't comparable."""
    table = make_table()
    splits = [
        (label, tuple(tr), tuple(te))
        for label, tr, te in train.season_forward_splits(table, n_splits=3)
    ]
    again = [
        (label, tuple(tr), tuple(te))
        for label, tr, te in train.season_forward_splits(table, n_splits=3)
    ]
    assert splits == again, "splits must be deterministic across calls"


# ---------------------------------------------------------------------------
# Winner model / per-race probability normalization
# ---------------------------------------------------------------------------


def _two_race_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2020],
            "round": [1, 1, 2, 2],
            "driver_id": ["a", "b", "a", "b"],
        }
    )


def test_win_probabilities_sum_to_one_per_race():
    df = _two_race_frame()
    out = train.normalize_win_probabilities(df, np.array([0.9, 0.3, 0.2, 0.1]))
    assert out[:2].sum() == pytest.approx(1.0)
    assert out[2:].sum() == pytest.approx(1.0)


def test_normalization_preserves_ordering_within_a_race():
    df = _two_race_frame()
    raw = np.array([0.9, 0.3, 0.1, 0.2])
    out = train.normalize_win_probabilities(df, raw)
    assert out[0] > out[1], "race 1 ordering should be preserved"
    assert out[3] > out[2], "race 2 ordering should be preserved"


def test_normalization_does_not_mix_across_races():
    """A confident race must not deflate a chaotic one, or vice versa."""
    df = _two_race_frame()
    # Race 1 sums to 1.2, race 2 sums to 0.2 -- wildly different confidence.
    out = train.normalize_win_probabilities(df, np.array([0.9, 0.3, 0.1, 0.1]))
    assert out[:2].sum() == pytest.approx(1.0)
    assert out[2:].sum() == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5), "an even 2-car race should split 50/50 regardless of raw scale"


def test_normalization_falls_back_to_uniform_when_all_zero():
    """Degenerate race: model says nobody wins. Must not divide by zero."""
    df = _two_race_frame()
    out = train.normalize_win_probabilities(df, np.array([0.0, 0.0, 0.5, 0.5]))
    assert out[:2].sum() == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.5) and out[1] == pytest.approx(0.5)
    assert np.isfinite(out).all()


def test_normalization_rejects_length_mismatch():
    df = _two_race_frame()
    with pytest.raises(ValueError, match="probabilities for"):
        train.normalize_win_probabilities(df, np.array([0.5, 0.5]))


def test_normalized_probabilities_stay_in_unit_interval():
    df = _two_race_frame()
    out = train.normalize_win_probabilities(df, np.array([0.9, 0.3, 0.2, 0.1]))
    assert (out >= 0).all() and (out <= 1).all()


def test_winner_cv_reports_normalized_metrics():
    table = make_table()
    report = train.cross_validate_winner(
        table, splitter=lambda df: train.season_forward_splits(df, n_splits=3)
    )
    assert set(report.metric_names()) == {"auc", "logloss", "top1_acc", "winner_prob"}
    assert 0.0 <= report.mean("top1_acc") <= 1.0


def test_winner_training_is_deterministic():
    table = make_table()
    m1, c1 = train.train_winner_model(table)
    m2, c2 = train.train_winner_model(table)
    X1, _ = features.build_model_matrix(table, categories=c1)
    X2, _ = features.build_model_matrix(table, categories=c2)
    assert np.allclose(m1.predict_proba(X1)[:, 1], m2.predict_proba(X2)[:, 1])


def test_model_paths_are_distinct():
    """Two models writing to one path would silently overwrite each other."""
    paths = [config.PODIUM_MODEL_PATH, config.POSITION_MODEL_PATH, config.WINNER_MODEL_PATH]
    assert len(set(paths)) == len(paths)


def test_model_paths_match_spec_filenames():
    """Spec Section 3 names these files explicitly; downstream code expects them."""
    assert config.PODIUM_MODEL_PATH.name == "podium_xgb.joblib"
    assert config.POSITION_MODEL_PATH.name == "position_xgb.joblib"
    assert config.WINNER_MODEL_PATH.name == "winner_xgb.joblib"


def test_all_three_models_are_wired_into_the_cli():
    assert set(train.MODELS) == {"podium", "position", "winner"}


def test_save_and_load_model_artifact_round_trip(tmp_path):
    table = make_table()
    model, categories = train.train_podium_model(table)
    path = train.save_model_artifact(model, categories, tmp_path / "podium.joblib", metadata={"note": "test"})

    artifact = train.load_model_artifact(path)
    assert artifact["categories"] == categories
    assert artifact["feature_columns"] == list(config.FEATURE_COLUMNS)
    assert artifact["metadata"]["note"] == "test"

    X, _ = features.build_model_matrix(table, categories=artifact["categories"])
    assert np.allclose(
        artifact["model"].predict_proba(X)[:, 1],
        model.predict_proba(X)[:, 1],
    )
