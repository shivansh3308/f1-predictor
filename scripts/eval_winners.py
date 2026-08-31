#!/usr/bin/env python3
"""Winner-model evaluation: calibration.

    python scripts/eval_winners.py

Accuracy asks "was the top pick right?". Calibration asks a different and
arguably more useful question: **when the model says 40%, does that driver
actually win about 40% of the time?** A model can be poorly calibrated
while still ranking drivers correctly, and vice versa -- and a probability
that does not mean what it says is misleading in a way a wrong ranking is
not.

Uses the same walk-forward, out-of-sample predictions as
``scripts/eval_past.py`` (never the serialized models, which saw every
race), so the calibration reported here is the calibration a user would
actually experience on a future race.

Reported:
  * a reliability table -- predicted vs observed win rate, per bin
  * Expected Calibration Error (ECE), the bin-count-weighted mean gap
  * Brier score, against a base-rate baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_past import walk_forward_predictions  # noqa: E402
from src import config  # noqa: E402

logger = logging.getLogger(__name__)

# Deliberately uneven bins. Most drivers in most races sit near zero, so
# equal-width bins would put ~90% of rows in one bucket and tell us nothing
# about the high-confidence end, which is the end that matters.
DEFAULT_BINS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]


def reliability_table(
    predictions: pd.DataFrame,
    bins: list[float] | None = None,
    prob_col: str = "prob_win",
    outcome_col: str = config.TARGET_WINNER,
) -> pd.DataFrame:
    """Bin predictions and compare predicted probability to observed frequency."""
    bins = bins or DEFAULT_BINS
    df = predictions[[prob_col, outcome_col]].dropna().copy()
    df["bin"] = pd.cut(df[prob_col], bins=bins, include_lowest=True)

    grouped = df.groupby("bin", observed=True)
    table = pd.DataFrame(
        {
            "n": grouped[prob_col].size(),
            "mean_predicted": grouped[prob_col].mean(),
            "observed": grouped[outcome_col].mean(),
        }
    ).reset_index()
    table["wins"] = grouped[outcome_col].sum().values
    table["gap"] = table["observed"] - table["mean_predicted"]
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Bin-count-weighted mean absolute gap between predicted and observed."""
    if table["n"].sum() == 0:
        return float("nan")
    return float((table["n"] * table["gap"].abs()).sum() / table["n"].sum())


def brier_score(predictions: pd.DataFrame, prob_col: str = "prob_win") -> float:
    outcome = predictions[config.TARGET_WINNER]
    return float(np.mean((predictions[prob_col] - outcome) ** 2))


def base_rate_brier(predictions: pd.DataFrame) -> float:
    """Brier score of always predicting the overall win rate (1/grid size)."""
    rate = predictions[config.TARGET_WINNER].mean()
    return float(np.mean((rate - predictions[config.TARGET_WINNER]) ** 2))


# Below this, a prediction is "this driver almost certainly won't win" --
# true for most of the grid in most races, and not something anyone acts on.
ACTIONABLE_THRESHOLD = 0.20


def actionable_calibration(table: pd.DataFrame, threshold: float = ACTIONABLE_THRESHOLD) -> tuple[float, int]:
    """ECE restricted to bins above `threshold`, plus their row count.

    Overall ECE is dominated by the huge near-zero bin -- most drivers in
    most races are correctly given ~0% and contribute almost no error. That
    makes the headline number look excellent while saying nothing about the
    confident predictions, which are the ones anyone would act on. This
    reports the calibration of that end separately so the aggregate cannot
    flatter the model.
    """
    top = table[table["mean_predicted"] >= threshold]
    if top.empty or top["n"].sum() == 0:
        return float("nan"), 0
    ece = float((top["n"] * top["gap"].abs()).sum() / top["n"].sum())
    return ece, int(top["n"].sum())


def format_reliability(table: pd.DataFrame, ece: float, brier: float, base_brier: float) -> str:
    lines = [
        "Winner model calibration -- walk-forward, out-of-sample",
        "=" * 78,
        f"{'predicted band':>16}  {'n':>6}  {'wins':>5}  {'predicted':>10}  {'observed':>9}  {'gap':>8}",
        "-" * 78,
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{str(row['bin']):>16}  {int(row['n']):>6}  {int(row['wins']):>5}  "
            f"{row['mean_predicted']:>9.1%}  {row['observed']:>8.1%}  {row['gap']:>+7.1%}"
        )
    top_ece, top_n = actionable_calibration(table)
    lines += [
        "-" * 78,
        "",
        f"Expected Calibration Error : {ece:.4f}   (0 = perfectly calibrated)",
        f"  ...above {ACTIONABLE_THRESHOLD:.0%} predicted : {top_ece:.4f}   (n={top_n})",
        f"Brier score                : {brier:.4f}",
        f"Brier, base-rate baseline  : {base_brier:.4f}   "
        f"({'better' if brier < base_brier else 'WORSE'} than always predicting the base rate)",
        "",
        "Reading a row: of the N driver-races the model put in that band, this",
        "many actually won. A positive gap means the model was under-confident",
        "(it happened more often than predicted); negative means over-confident.",
        "",
        "Read the overall ECE with care: most driver-races sit in the bottom",
        "band, are correctly given ~0%, and contribute almost no error -- which",
        "drags the aggregate down regardless of how the confident predictions",
        "behave. The second line is the number that describes the predictions",
        "anyone would actually act on.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate winner-model calibration")
    parser.add_argument("--splits", type=int, default=config.N_CV_SPLITS, help="How many seasons to evaluate")
    parser.add_argument("--save", type=Path, default=None, help="Write the reliability table to a CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    predictions = walk_forward_predictions(n_splits=args.splits)
    table = reliability_table(predictions)
    ece = expected_calibration_error(table)

    print()
    print(
        format_reliability(
            table,
            ece,
            brier_score(predictions),
            base_rate_brier(predictions),
        )
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.save, index=False)
        print(f"\nReliability table -> {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
