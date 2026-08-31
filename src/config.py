"""Single source of truth for project configuration.

Season range, filesystem paths, the canonical feature list, and target column
names all live here. Nothing outside this module should hardcode any of them
(see rebuild spec Section 7, "Style").
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

FASTF1_CACHE_DIR: Final[Path] = BASE_DIR / ".fastf1_cache"

DATA_DIR: Final[Path] = BASE_DIR / "data"
DATA_RAW_DIR: Final[Path] = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
FEATURE_TABLE_PATH: Final[Path] = DATA_PROCESSED_DIR / "features.parquet"

MODELS_DIR: Final[Path] = BASE_DIR / "models"
PODIUM_MODEL_PATH: Final[Path] = MODELS_DIR / "podium_xgb.joblib"
POSITION_MODEL_PATH: Final[Path] = MODELS_DIR / "position_xgb.joblib"
WINNER_MODEL_PATH: Final[Path] = MODELS_DIR / "winner_xgb.joblib"


def ensure_directories() -> None:
    """Create the cache/data/model directories if they don't exist yet.

    Safe to call repeatedly (e.g. at the top of any fetch/train entrypoint).
    """
    for directory in (
        FASTF1_CACHE_DIR,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training data range
# ---------------------------------------------------------------------------

SEASON_START: Final[int] = 2018
SEASON_END: Final[int] = 2025  # inclusive
# NOTE: deliberately stops at 2025 (complete seasons only) even though it's
# already 2026 as this is being written. Plan: finish the full pipeline
# (fetch/features/train/predict/render/tests) and reproduce the benchmark
# metrics against 2018-2025 first, matching the original spec. Only *after*
# that's validated, bump this to include 2026 as a second pass -- training
# on a still-in-progress season would skew rolling-average/standings
# features. `predict_upcoming.py` (task 15) is what reaches out live for
# whatever round is next regardless of this constant.
SEASONS: Final[list[int]] = list(range(SEASON_START, SEASON_END + 1))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: Final[int] = 42

# Number of splits for time-respecting cross-validation (see Section 7:
# random K-fold across seasons leaks future information into past
# predictions — use a time-series split or hold out full seasons instead).
N_CV_SPLITS: Final[int] = 5

# Window size (in races) for driver/constructor rolling-average-finish
# features. Computed with a strict backward shift — see src/features.py.
ROLLING_WINDOW: Final[int] = 5

# ---------------------------------------------------------------------------
# Feature set (spec Section 4)
# ---------------------------------------------------------------------------

# Minimum feature set confirmed by the original probability-table screenshot.
CORE_FEATURES: Final[list[str]] = [
    "driver_id",
    "constructor_id",
    "grid",
]

# Recommended additions — all computable strictly before lights-out.
EXTENDED_FEATURES: Final[list[str]] = [
    "quali_position",
    "q1_time_s",
    "q2_time_s",
    "q3_time_s",
    "driver_rolling_avg_finish",
    "constructor_rolling_avg_finish",
    "driver_points_before_race",
    "driver_standing_before_race",
    "constructor_points_before_race",
    "constructor_standing_before_race",
    "circuit_id",
    "season",
    "driver_dnf_rate",
    "constructor_dnf_rate",
]

# Canonical feature list — the single list that both training and prediction
# must use (spec Section 7: "Training and prediction must call the *same*
# function, or they will silently drift").
FEATURE_COLUMNS: Final[list[str]] = CORE_FEATURES + EXTENDED_FEATURES

# Categorical columns among FEATURE_COLUMNS that need encoding before being
# handed to XGBoost.
CATEGORICAL_FEATURES: Final[list[str]] = [
    "driver_id",
    "constructor_id",
    "circuit_id",
]

# Columns that must never appear in FEATURE_COLUMNS: they are only knowable
# after the race and would leak the target. Enforced by an assertion in
# src/features.py, not just documented here. Includes both the raw
# data_fetch.py column names (status, laps, points, classified_position) and
# the engineered target names, so the check catches leakage at either stage.
POST_RACE_ONLY_COLUMNS: Final[list[str]] = [
    "finish_position",
    "classified_position",
    "podium_finish",
    "is_winner",
    "race_time",
    "fastest_lap_time",
    "points",
    "sprint_points",
    "status",
    "laps",
    "laps_completed",
]

# ---------------------------------------------------------------------------
# Target column names
# ---------------------------------------------------------------------------

TARGET_PODIUM: Final[str] = "podium_finish"     # binary: top-3 finish
TARGET_POSITION: Final[str] = "finish_position"  # regression: 1..N
TARGET_WINNER: Final[str] = "is_winner"          # binary: race winner

TARGET_COLUMNS: Final[list[str]] = [TARGET_PODIUM, TARGET_POSITION, TARGET_WINNER]

# ---------------------------------------------------------------------------
# Benchmark metrics to reproduce (spec Section 1) — used by train.py /
# eval scripts to report deltas against the original run.
# ---------------------------------------------------------------------------

BENCHMARK_PODIUM_CV_AUC: Final[float] = 0.934
BENCHMARK_PODIUM_CV_LOGLOSS: Final[float] = 0.269
BENCHMARK_POSITION_CV_MAE: Final[float] = 3.626

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

# Shared across all three models. Deliberately modest depth/learning rate:
# ~3.4k rows is a small dataset, and the strongest feature (grid) is very
# predictive on its own, so a deep unregularised model overfits quickly.
XGB_PARAMS: Final[dict] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "enable_categorical": True,
    "tree_method": "hist",
}
