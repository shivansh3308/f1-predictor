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
