"""Prediction: turn a (season, round) into a per-driver probability table.

Loads the three serialized models and produces the format the original
project's probability table used -- ``driver_id, constructor_id, grid,
prob_win`` -- plus the podium probability and predicted finishing position
that ``src/render.py`` needs for the formatted report.

**Feature parity.** Features are built by calling the exact same
`features.build_feature_table` / `features.build_model_matrix` used in
training, and each model's categorical mapping is taken from its own saved
artifact. Recomputing either here would let training and prediction drift,
which the spec calls out as producing predictions "wrong in ways that are
very hard to debug" (Section 7).

**In-sample caveat.** The shipped models are trained on all of 2018-2025,
so predicting a round inside that range is in-sample and will look better
than the model's true forward-looking skill. That is fine for inspecting a
race or demoing the CLI, but it is *not* an evaluation. The honest
out-of-sample numbers come from the season-forward CV in `src/train.py` and
from the backtest in `scripts/eval_past.py` (task 18), which retrains
excluding the race being scored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config, data_fetch, features, train

logger = logging.getLogger(__name__)

# Columns of the returned table, in the order the original probability
# table used. Extra model outputs follow the four core columns.
PREDICTION_COLUMNS = [
    "driver_id",
    "driver_abbreviation",
    "constructor_id",
    "grid",
    "prob_win",
    "prob_podium",
    "pred_position",
]


class ModelsNotTrainedError(FileNotFoundError):
    """Raised when a model artifact is missing, with a pointer to the fix."""


class RoundNotFoundError(LookupError):
    """Raised when the requested (season, round) has no data available."""


@dataclass
class LoadedModels:
    """The three trained models plus the metadata saved alongside them."""

    podium: dict
    position: dict
    winner: dict

    @property
    def artifacts(self) -> dict[str, dict]:
        return {"podium": self.podium, "position": self.position, "winner": self.winner}


def load_models(
    podium_path: Path = config.PODIUM_MODEL_PATH,
    position_path: Path = config.POSITION_MODEL_PATH,
    winner_path: Path = config.WINNER_MODEL_PATH,
) -> LoadedModels:
    """Load all three model artifacts from disk.

    Raises `ModelsNotTrainedError` naming the missing file rather than
    letting joblib raise a bare FileNotFoundError -- the fix (run the
    training script) is not obvious from the raw error.
    """
    paths = {"podium": podium_path, "position": position_path, "winner": winner_path}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise ModelsNotTrainedError(
            f"Missing model artifact(s): {', '.join(missing)}. "
            "Train them first with:  python scripts/train_models.py --save"
        )

    return LoadedModels(**{name: train.load_model_artifact(path) for name, path in paths.items()})


def get_round_features(
    season: int,
    round_number: int,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the feature rows for one round.

    The feature table is built from the *full* history and then filtered,
    never built from the single round alone -- the rolling-form, standings
    and DNF-rate features are defined by what came before, so a round
    computed in isolation would silently produce a table of nulls.
    """
    if table is None:
        table = _load_or_build_table()

    round_rows = table[(table["season"] == season) & (table["round"] == round_number)]
    if round_rows.empty:
        available = table[table["season"] == season]["round"].unique()
        raise RoundNotFoundError(
            f"No data for {season} round {round_number}. "
            + (
                f"Season {season} has rounds {sorted(available.tolist())}."
                if len(available)
                else f"Season {season} is not in the dataset (have {sorted(table['season'].unique().tolist())})."
            )
        )
    return round_rows


def _load_or_build_table() -> pd.DataFrame:
    """Prefer the cached processed table; fall back to rebuilding from raw."""
    if config.FEATURE_TABLE_PATH.exists():
        return features.load_feature_table()
    logger.info("No processed feature table found -- rebuilding from data/raw/")
    return features.build_feature_table(data_fetch.load_all_raw())


def predict_round(
    season: int,
    round_number: int,
    models: LoadedModels | None = None,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict one round. Returns a table in `PREDICTION_COLUMNS` order.

    `prob_win` is normalized across the grid so the column sums to 1.0 --
    see `train.normalize_win_probabilities` for why the raw classifier
    output is not usable as-is.
    """
    models = models or load_models()
    round_rows = get_round_features(season, round_number, table=table)

    # Each model carries its own categorical mapping. They are identical in
    # practice (same training table) but using each artifact's own mapping
    # keeps this correct even if one model is retrained separately later.
    def _matrix(artifact: dict) -> pd.DataFrame:
        X, _ = features.build_model_matrix(round_rows, categories=artifact["categories"])
        return X

    prob_podium = models.podium["model"].predict_proba(_matrix(models.podium))[:, 1]
    pred_position = train.predict_position(models.position["model"], _matrix(models.position), round_rows["grid"])
    raw_win = models.winner["model"].predict_proba(_matrix(models.winner))[:, 1]

    result = round_rows[["driver_id", "driver_abbreviation", "constructor_id", "grid"]].copy()
    result["prob_win"] = train.normalize_win_probabilities(round_rows, raw_win)
    result["prob_podium"] = prob_podium
    result["pred_position"] = pred_position

    return (
        result[PREDICTION_COLUMNS]
        .sort_values("prob_win", ascending=False)
        .reset_index(drop=True)
    )


def predicted_finishing_order(predictions: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Rank drivers by predicted finishing position.

    The regressor's raw output is unconstrained (it can predict 0.78 or
    20.1) and can tie or skip values, so the displayed order comes from
    ranking those predictions rather than printing them as positions.
    """
    ordered = predictions.sort_values("pred_position").reset_index(drop=True)
    ordered.insert(0, "predicted_rank", range(1, len(ordered) + 1))
    return ordered.head(top_n) if top_n else ordered


def race_label(season: int, round_number: int, table: pd.DataFrame | None = None) -> str:
    """Human-readable 'Event Name' for a round, for report headers."""
    rows = get_round_features(season, round_number, table=table)
    return str(rows["event_name"].iloc[0]) if "event_name" in rows.columns else f"Round {round_number}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict a single race")
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    predictions = predict_round(args.season, args.round)
    print(predictions.to_string(index=False))
    print(f"\nprob_win sums to {predictions['prob_win'].sum():.6f}")


if __name__ == "__main__":
    main()
