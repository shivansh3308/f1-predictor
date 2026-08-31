"""Model training and cross-validation.

Task 9 scope: the podium classifier (`XGBClassifier`), reporting CV AUC and
LogLoss. Position and winner models follow in tasks 10-11.

**Cross-validation strategy.** The spec (Section 7, "Reproducibility") is
explicit that random K-fold across seasons leaks future information into
past predictions, and that if honest time-aware CV lands below the original
benchmark we report the lower number. So the headline metric here comes from
`season_forward_splits`: expanding-window, forward-chaining folds where each
fold trains only on seasons *strictly before* the season it is evaluated on.
That is the number to defend in an interview.

A random K-fold score is also computed, purely as a diagnostic, to quantify
how much of the original 0.934 was likely optimism from a leaky split. It is
clearly labelled as such and is never the reported headline.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

from src import config, features

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Metrics for a single CV fold."""

    label: str
    n_train: int
    n_test: int
    metrics: dict[str, float]


@dataclass
class CVReport:
    """Aggregated cross-validation result for one model."""

    model_name: str
    strategy: str
    folds: list[FoldResult] = field(default_factory=list)

    def mean(self, metric: str) -> float:
        return float(np.mean([f.metrics[metric] for f in self.folds]))

    def std(self, metric: str) -> float:
        return float(np.std([f.metrics[metric] for f in self.folds]))

    def metric_names(self) -> list[str]:
        return list(self.folds[0].metrics) if self.folds else []


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def season_forward_splits(
    df: pd.DataFrame,
    n_splits: int = config.N_CV_SPLITS,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield `(label, train_idx, test_idx)` expanding-window folds by season.

    Fold *i* trains on every season strictly before test season *i* and
    evaluates on that whole season. With 2018-2025 and `n_splits=5` the test
    seasons are 2021..2025, each trained on everything prior (2018-2020,
    then 2018-2021, and so on).

    This is deliberately stricter than `GroupKFold` on season: grouping alone
    would still let a fold train on 2025 to predict 2018, which is exactly
    the lookahead the spec rules out.
    """
    seasons = sorted(df["season"].unique())
    if len(seasons) <= n_splits:
        raise ValueError(
            f"Need more than n_splits={n_splits} seasons to build forward folds, got {len(seasons)}"
        )

    positions = np.arange(len(df))
    for test_season in seasons[-n_splits:]:
        train_mask = df["season"].to_numpy() < test_season
        test_mask = df["season"].to_numpy() == test_season
        yield str(test_season), positions[train_mask], positions[test_mask]


def random_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = config.N_CV_SPLITS,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield plain shuffled K-fold splits -- **diagnostic only**.

    Ignores time ordering entirely, so a fold can train on 2025 races to
    predict 2019 ones. Kept solely to measure how much optimism that
    introduces relative to `season_forward_splits`.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    for i, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        yield f"fold {i}", train_idx, test_idx


# ---------------------------------------------------------------------------
# Podium model
# ---------------------------------------------------------------------------


def build_podium_model() -> XGBClassifier:
    """Fresh, unfitted podium classifier with the project's fixed seed/params."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        **config.XGB_PARAMS,
    )


def build_position_model() -> XGBRegressor:
    """Fresh, unfitted finishing-position regressor.

    Uses ``reg:absoluteerror`` rather than the usual squared-error
    objective because the reported metric is MAE. Squared error chases
    outliers -- and in F1 the outliers are lap-1 crashes and mechanical
    DNFs, which are exactly the races no pre-race feature can predict.
    Optimising the metric we actually report avoids letting those
    unpredictable blowups drag the whole model around.
    """
    return XGBRegressor(
        objective="reg:absoluteerror",
        eval_metric="mae",
        **config.XGB_PARAMS,
    )


def _podium_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def _position_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _predict_raw(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def cross_validate(
    table: pd.DataFrame,
    *,
    model_name: str,
    target: str,
    model_factory,
    metric_fn,
    predict_fn=_predict_proba,
    splitter=season_forward_splits,
    strategy: str = "season-forward (time-respecting)",
) -> CVReport:
    """Cross-validate one model and return per-fold metrics.

    Shared by every model so the split, the encoding discipline and the
    reporting stay identical across them -- the alternative is three
    near-identical CV loops that drift apart (spec Section 6).

    Note the categories are rebuilt from *each fold's training data only*.
    Deriving them once from the full table would let the encoding itself
    carry information about drivers/teams that only appear in the test
    season -- a subtle leak that is easy to miss.
    """
    report = CVReport(model_name=model_name, strategy=strategy)
    y_all = table[target].to_numpy()

    for label, train_idx, test_idx in splitter(table):
        X_train, categories = features.build_model_matrix(table.iloc[train_idx])
        X_test, _ = features.build_model_matrix(table.iloc[test_idx], categories=categories)

        model = model_factory()
        model.fit(X_train, y_all[train_idx])
        y_pred = predict_fn(model, X_test)

        report.folds.append(
            FoldResult(
                label=label,
                n_train=len(train_idx),
                n_test=len(test_idx),
                metrics=metric_fn(y_all[test_idx], y_pred),
            )
        )

    return report


def cross_validate_podium(table: pd.DataFrame, **kwargs) -> CVReport:
    return cross_validate(
        table,
        model_name="podium",
        target=config.TARGET_PODIUM,
        model_factory=build_podium_model,
        metric_fn=_podium_metrics,
        predict_fn=_predict_proba,
        **kwargs,
    )


def cross_validate_position(table: pd.DataFrame, **kwargs) -> CVReport:
    return cross_validate(
        table,
        model_name="position",
        target=config.TARGET_POSITION,
        model_factory=build_position_model,
        metric_fn=_position_metrics,
        predict_fn=_predict_raw,
        **kwargs,
    )


def _fit_full(table: pd.DataFrame, target: str, model_factory):
    X, categories = features.build_model_matrix(table)
    model = model_factory()
    model.fit(X, table[target].to_numpy())
    return model, categories


def train_podium_model(table: pd.DataFrame) -> tuple[XGBClassifier, dict[str, list]]:
    """Fit the final podium model on the full table. Returns `(model, categories)`."""
    return _fit_full(table, config.TARGET_PODIUM, build_podium_model)


def train_position_model(table: pd.DataFrame) -> tuple[XGBRegressor, dict[str, list]]:
    """Fit the final position model on the full table. Returns `(model, categories)`."""
    return _fit_full(table, config.TARGET_POSITION, build_position_model)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model_artifact(
    model,
    categories: dict[str, list],
    path: Path,
    metadata: dict | None = None,
) -> Path:
    """Persist a model together with everything needed to reproduce its inputs.

    The categorical mapping travels *with* the model on purpose: reloading a
    model without it would re-derive category codes from whatever data is
    being predicted, silently remapping driver/constructor identities.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "categories": categories,
            "feature_columns": list(config.FEATURE_COLUMNS),
            "metadata": metadata or {},
        },
        path,
    )
    return path


def load_model_artifact(path: Path) -> dict:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(report: CVReport, benchmarks: dict[str, float] | None = None) -> str:
    lines = [
        f"{report.model_name} model -- {report.strategy}",
        "-" * 64,
    ]
    metric_names = report.metric_names()
    header = f"{'fold':>10}  {'n_train':>8}  {'n_test':>7}  " + "  ".join(f"{m:>9}" for m in metric_names)
    lines.append(header)
    for fold in report.folds:
        row = f"{fold.label:>10}  {fold.n_train:>8}  {fold.n_test:>7}  " + "  ".join(
            f"{fold.metrics[m]:>9.4f}" for m in metric_names
        )
        lines.append(row)
    lines.append("-" * 64)
    mean_row = f"{'MEAN':>10}  {'':>8}  {'':>7}  " + "  ".join(f"{report.mean(m):>9.4f}" for m in metric_names)
    lines.append(mean_row)
    std_row = f"{'std':>10}  {'':>8}  {'':>7}  " + "  ".join(f"{report.std(m):>9.4f}" for m in metric_names)
    lines.append(std_row)

    if benchmarks:
        lines.append("")
        lines.append("vs original benchmark:")
        for metric, target in benchmarks.items():
            actual = report.mean(metric)
            delta = actual - target
            lines.append(f"  {metric:>8}: {actual:.4f}  (original {target:.4f}, delta {delta:+.4f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


MODELS = ("podium", "position")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and cross-validate the F1 models")
    parser.add_argument("--save", action="store_true", help="Fit on all data and write models/*.joblib")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Which models to train (default: all implemented)",
    )
    parser.add_argument(
        "--skip-diagnostic",
        action="store_true",
        help="Skip the random K-fold comparison run",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    table = features.load_feature_table()
    print(f"Loaded feature table: {len(table)} rows, {table['season'].nunique()} seasons\n")

    if "podium" in args.models:
        honest = cross_validate_podium(table)
        print(
            format_report(
                honest,
                benchmarks={
                    "auc": config.BENCHMARK_PODIUM_CV_AUC,
                    "logloss": config.BENCHMARK_PODIUM_CV_LOGLOSS,
                },
            )
        )

        if not args.skip_diagnostic:
            leaky = cross_validate_podium(
                table,
                splitter=random_kfold_splits,
                strategy="random K-fold (DIAGNOSTIC ONLY -- leaks future races)",
            )
            print()
            print(format_report(leaky))
            print()
            print(
                f"Optimism from ignoring time order: "
                f"AUC {leaky.mean('auc') - honest.mean('auc'):+.4f}, "
                f"LogLoss {leaky.mean('logloss') - honest.mean('logloss'):+.4f}"
            )

        if args.save:
            model, categories = train_podium_model(table)
            path = save_model_artifact(
                model,
                categories,
                config.PODIUM_MODEL_PATH,
                metadata={
                    "cv_strategy": honest.strategy,
                    "cv_auc": honest.mean("auc"),
                    "cv_logloss": honest.mean("logloss"),
                    "n_rows": len(table),
                    "seasons": sorted(table["season"].unique().tolist()),
                },
            )
            print(f"\nSaved podium model -> {path}")

    if "position" in args.models:
        print("\n")
        honest_pos = cross_validate_position(table)
        print(format_report(honest_pos, benchmarks={"mae": config.BENCHMARK_POSITION_CV_MAE}))

        if not args.skip_diagnostic:
            leaky_pos = cross_validate_position(
                table,
                splitter=random_kfold_splits,
                strategy="random K-fold (DIAGNOSTIC ONLY -- leaks future races)",
            )
            print()
            print(format_report(leaky_pos))
            print()
            print(f"Optimism from ignoring time order: MAE {leaky_pos.mean('mae') - honest_pos.mean('mae'):+.4f}")

        if args.save:
            model, categories = train_position_model(table)
            path = save_model_artifact(
                model,
                categories,
                config.POSITION_MODEL_PATH,
                metadata={
                    "cv_strategy": honest_pos.strategy,
                    "cv_mae": honest_pos.mean("mae"),
                    "cv_rmse": honest_pos.mean("rmse"),
                    "n_rows": len(table),
                    "seasons": sorted(table["season"].unique().tolist()),
                },
            )
            print(f"\nSaved position model -> {path}")


if __name__ == "__main__":
    main()
