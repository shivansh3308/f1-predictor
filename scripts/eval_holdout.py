#!/usr/bin/env python3
"""Evaluate the shipped models on a season they were never trained on.

    python scripts/eval_holdout.py --season 2026

`eval_past.py` reconstructs out-of-sample performance by retraining
walk-forward. This is stronger evidence: it takes the models already
serialized in ``models/`` and scores them on a season that did not exist
when they were fit. Nothing is retrained, so there is no way for the
evaluation to be contaminated.

That makes it the closest thing to the real use case -- "here is a model,
here is a season it has never seen, how did it do?" -- and it is
especially informative across a regulation change, where the competitive
order the model learned no longer holds.

Every metric is reported against the same grid-order baseline used in
`eval_past.py`. The baseline is the control: if a season is simply harder
to predict, the baseline degrades too. If only the model degrades, the
problem is the model's priors, not the season.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_past import PODIUM_SIZE, _score_race  # noqa: E402
from src import config, data_fetch, features, predict, train  # noqa: E402

logger = logging.getLogger(__name__)


def predict_season(season: int, models: predict.LoadedModels | None = None) -> pd.DataFrame:
    """Score one season with the serialized models. Returns per-driver rows.

    Features are built across the full history, not the season alone --
    rolling form and standings are defined by what came before.
    """
    models = models or predict.load_models()

    seasons = sorted(set(config.SEASONS) | {season})
    table = features.build_feature_table(data_fetch.load_all_raw(seasons=seasons))
    rows = table[table["season"] == season].copy()
    if rows.empty:
        raise SystemExit(
            f"No data for {season}. Fetch it first:\n"
            f"    python -m src.data_fetch --seasons {season}"
        )

    def matrix(artifact: dict) -> pd.DataFrame:
        X, _ = features.build_model_matrix(rows, categories=artifact["categories"])
        return X

    rows["prob_podium"] = models.podium["model"].predict_proba(matrix(models.podium))[:, 1]
    rows["pred_position"] = models.position["model"].predict(matrix(models.position))
    rows["prob_win"] = train.normalize_win_probabilities(
        rows, models.winner["model"].predict_proba(matrix(models.winner))[:, 1]
    )
    return rows


def score_season(rows: pd.DataFrame) -> pd.DataFrame:
    outcomes = [_score_race(race) for _, race in rows.groupby(["season", "round"])]
    return pd.DataFrame([vars(o) for o in outcomes])


def unseen_categories(rows: pd.DataFrame, models: predict.LoadedModels) -> dict[str, list[str]]:
    """Drivers and constructors in this season that the models never saw.

    These are encoded as NaN rather than guessed at, so they are predicted
    almost entirely from grid position -- worth surfacing, because a season
    with new entrants is harder for reasons unrelated to the racing.
    """
    categories = models.winner["categories"]
    return {
        "driver_id": sorted(set(rows["driver_id"]) - set(categories["driver_id"])),
        "constructor_id": sorted(set(rows["constructor_id"]) - set(categories["constructor_id"])),
    }


def format_report(season: int, results: pd.DataFrame, unseen: dict[str, list[str]]) -> str:
    winner, base_winner = results["winner_correct"].mean(), results["baseline_winner_correct"].mean()
    podium = results["podium_hits"].mean() / PODIUM_SIZE
    base_podium = results["baseline_podium_hits"].mean() / PODIUM_SIZE
    mae, base_mae = results["position_mae"].mean(), results["baseline_position_mae"].mean()
    n_races = len(results)

    lines = [
        f"{season} holdout -- scored with models trained on "
        f"{config.SEASON_START}-{config.SEASON_END}, never retrained",
        "=" * 74,
        f"{'metric':>20}  {'model':>10}  {'grid baseline':>14}  {'gap':>9}",
        "-" * 74,
        f"{'winner accuracy':>20}  {winner:>9.1%}  {base_winner:>13.1%}  {winner - base_winner:>+8.1%}",
        f"{'podium hit rate':>20}  {podium:>9.1%}  {base_podium:>13.1%}  {podium - base_podium:>+8.1%}",
        f"{'position MAE':>20}  {mae:>10.2f}  {base_mae:>14.2f}  {mae - base_mae:>+9.2f}",
        "=" * 74,
        "",
        f"{n_races} races. Winner accuracy over so few races is dominated by noise "
        f"({int(results['winner_correct'].sum())}/{n_races} vs "
        f"{int(results['baseline_winner_correct'].sum())}/{n_races} is a difference of "
        f"{abs(int(results['winner_correct'].sum()) - int(results['baseline_winner_correct'].sum()))} race(s));",
        f"position MAE is measured over {int(results['n_drivers'].sum())} driver-races and is the more reliable signal.",
    ]

    if unseen["driver_id"] or unseen["constructor_id"]:
        lines += ["", "Not present in training data (encoded as unknown, not guessed):"]
        if unseen["driver_id"]:
            lines.append(f"  drivers      : {', '.join(unseen['driver_id'])}")
        if unseen["constructor_id"]:
            lines.append(f"  constructors : {', '.join(unseen['constructor_id'])}")

    lines += ["", "Per race:"]
    for _, row in results.iterrows():
        mark = "HIT " if row["winner_correct"] else "miss"
        lines.append(
            f"  R{row['round_number']:<2} {row['event_name'][:26]:28} {mark}  "
            f"predicted {row['predicted_winner']:<16} actual {row['actual_winner']}"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the shipped models on an unseen season")
    parser.add_argument("--season", type=int, required=True, help="Season to evaluate")
    parser.add_argument("--save", type=Path, default=None, help="Write per-race results to a CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    if args.season in config.SEASONS:
        print(
            f"Warning: {args.season} is inside the training range "
            f"({config.SEASON_START}-{config.SEASON_END}), so this is NOT a holdout. "
            f"Use scripts/eval_past.py for in-range seasons.\n"
        )

    models = predict.load_models()
    rows = predict_season(args.season, models=models)
    results = score_season(rows)

    print()
    print(format_report(args.season, results, unseen_categories(rows, models)))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.save, index=False)
        print(f"\nPer-race results -> {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
