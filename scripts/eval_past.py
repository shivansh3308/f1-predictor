#!/usr/bin/env python3
"""Backtest the models against completed races.

    python scripts/eval_past.py

**This is the headline number for the project** (spec task 18). CV metrics
are aggregate and abstract; "the model named the winner in N% of real
races" is the claim that has to survive questioning.

Honesty is the whole point of this script, so two things are done
deliberately:

1. **Walk-forward retraining.** The serialized models in ``models/`` were
   fit on all of 2018-2025, so scoring any race in that range with them
   would be in-sample and meaningless. Instead every test season gets a
   fresh set of models trained *only* on seasons strictly before it. That
   is the same discipline as the season-forward CV in ``src/train.py``,
   applied per race so the results can be reported as real race outcomes.

2. **A baseline to compare against.** "Picks the winner 50% of the time"
   means nothing on its own -- the pole-sitter wins a lot of races
   unaided. Every metric is therefore reported next to a grid-order
   baseline (pole wins, front three are the podium, everyone finishes
   where they started). The number worth defending is the *gap*.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, features, train  # noqa: E402

logger = logging.getLogger(__name__)

PODIUM_SIZE = 3


@dataclass
class RaceOutcome:
    """Model vs baseline scores for a single race."""

    season: int
    round_number: int
    event_name: str
    n_drivers: int

    predicted_winner: str
    actual_winner: str
    winner_correct: bool
    podium_hits: int
    position_mae: float

    baseline_winner: str
    baseline_winner_correct: bool
    baseline_podium_hits: int
    baseline_position_mae: float

    @property
    def podium_hit_rate(self) -> float:
        return self.podium_hits / PODIUM_SIZE

    @property
    def baseline_podium_hit_rate(self) -> float:
        return self.baseline_podium_hits / PODIUM_SIZE


def _score_race(race: pd.DataFrame) -> RaceOutcome:
    """Score one race, given a frame carrying predictions and actual results."""
    actual_podium = set(race.nsmallest(PODIUM_SIZE, "finish_position")["driver_id"])
    actual_winner = race.loc[race["finish_position"].idxmin(), "driver_id"]

    # --- model ---
    predicted_winner = race.loc[race["prob_win"].idxmax(), "driver_id"]
    predicted_podium = set(race.nlargest(PODIUM_SIZE, "prob_podium")["driver_id"])
    predicted_order = race["pred_position"].rank(method="first")
    position_mae = float(np.mean(np.abs(predicted_order - race["finish_position"])))

    # --- baseline: everyone finishes where they started ---
    baseline_winner = race.loc[race["grid"].idxmin(), "driver_id"]
    baseline_podium = set(race.nsmallest(PODIUM_SIZE, "grid")["driver_id"])
    baseline_order = race["grid"].rank(method="first")
    baseline_mae = float(np.mean(np.abs(baseline_order - race["finish_position"])))

    return RaceOutcome(
        season=int(race["season"].iloc[0]),
        round_number=int(race["round"].iloc[0]),
        event_name=str(race["event_name"].iloc[0]),
        n_drivers=len(race),
        predicted_winner=predicted_winner,
        actual_winner=actual_winner,
        winner_correct=predicted_winner == actual_winner,
        podium_hits=len(predicted_podium & actual_podium),
        position_mae=position_mae,
        baseline_winner=baseline_winner,
        baseline_winner_correct=baseline_winner == actual_winner,
        baseline_podium_hits=len(baseline_podium & actual_podium),
        baseline_position_mae=baseline_mae,
    )


def backtest(
    table: pd.DataFrame | None = None,
    n_splits: int = config.N_CV_SPLITS,
) -> pd.DataFrame:
    """Walk-forward backtest. Returns one row per race.

    For each test season, three fresh models are trained on the seasons
    before it, then used to predict every race of that season.
    """
    table = features.load_feature_table() if table is None else table
    outcomes: list[RaceOutcome] = []

    for label, train_idx, test_idx in train.season_forward_splits(table, n_splits=n_splits):
        train_df = table.iloc[train_idx]
        test_df = table.iloc[test_idx].copy()
        logger.info("Season %s: training on %d prior rows ...", label, len(train_df))

        X_train, categories = features.build_model_matrix(train_df)
        X_test, _ = features.build_model_matrix(test_df, categories=categories)

        podium_model = train.build_podium_model().fit(X_train, train_df[config.TARGET_PODIUM])
        position_model = train.build_position_model().fit(X_train, train_df[config.TARGET_POSITION])
        winner_model = train.build_winner_model().fit(X_train, train_df[config.TARGET_WINNER])

        test_df["prob_podium"] = podium_model.predict_proba(X_test)[:, 1]
        test_df["pred_position"] = position_model.predict(X_test)
        test_df["prob_win"] = train.normalize_win_probabilities(
            test_df, winner_model.predict_proba(X_test)[:, 1]
        )

        for _, race in test_df.groupby(["season", "round"]):
            outcomes.append(_score_race(race))

    return pd.DataFrame([vars(o) for o in outcomes])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-season and overall, model against baseline."""
    def _agg(frame: pd.DataFrame) -> dict:
        return {
            "races": len(frame),
            "winner_acc": frame["winner_correct"].mean(),
            "base_winner_acc": frame["baseline_winner_correct"].mean(),
            "podium_rate": frame["podium_hits"].mean() / PODIUM_SIZE,
            "base_podium_rate": frame["baseline_podium_hits"].mean() / PODIUM_SIZE,
            "position_mae": frame["position_mae"].mean(),
            "base_position_mae": frame["baseline_position_mae"].mean(),
        }

    rows = [{"season": str(season), **_agg(frame)} for season, frame in results.groupby("season")]
    rows.append({"season": "ALL", **_agg(results)})
    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame) -> str:
    lines = [
        "Backtest -- walk-forward (each season predicted by models trained only on earlier seasons)",
        "=" * 92,
        f"{'season':>7}  {'races':>5}  {'winner acc':>18}  {'podium hit rate':>18}  {'position MAE':>18}",
        f"{'':>7}  {'':>5}  {'model / grid base':>18}  {'model / grid base':>18}  {'model / grid base':>18}",
        "-" * 92,
    ]
    for _, row in summary.iterrows():
        divider = "=" if row["season"] == "ALL" else " "
        if row["season"] == "ALL":
            lines.append("-" * 92)
        lines.append(
            f"{row['season']:>7}  {int(row['races']):>5}  "
            f"{row['winner_acc']:>7.1%} / {row['base_winner_acc']:<8.1%}  "
            f"{row['podium_rate']:>7.1%} / {row['base_podium_rate']:<8.1%}  "
            f"{row['position_mae']:>7.2f} / {row['base_position_mae']:<8.2f}"
        )
        del divider

    overall = summary[summary["season"] == "ALL"].iloc[0]
    lines += [
        "=" * 92,
        "",
        "Gap vs simply predicting the grid order:",
        f"  winner accuracy   {overall['winner_acc'] - overall['base_winner_acc']:+.1%}",
        f"  podium hit rate   {overall['podium_rate'] - overall['base_podium_rate']:+.1%}",
        f"  position MAE      {overall['position_mae'] - overall['base_position_mae']:+.2f}  (negative is better)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest predictions against completed races")
    parser.add_argument("--splits", type=int, default=config.N_CV_SPLITS, help="How many seasons to back-test")
    parser.add_argument("--save", type=Path, default=None, help="Write per-race results to a CSV")
    parser.add_argument("--misses", action="store_true", help="List the races whose winner was called wrong")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    results = backtest(n_splits=args.splits)
    print()
    print(format_summary(summarize(results)))

    if args.misses:
        missed = results[~results["winner_correct"]]
        print(f"\nWinner called wrong in {len(missed)} of {len(results)} races:")
        for _, row in missed.iterrows():
            print(
                f"  {row['season']} R{row['round_number']:<2} {row['event_name'][:28]:30} "
                f"predicted {row['predicted_winner']:<16} actual {row['actual_winner']}"
            )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.save, index=False)
        print(f"\nPer-race results -> {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
