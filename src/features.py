"""Feature engineering — single source of truth.

Turns the raw per-driver-race pulls from ``src/data_fetch.py`` into the
model-ready feature table. Both training (``src/train.py``) and prediction
(``src/predict.py``) must call `build_feature_table` / `build_features_for_row`
from here rather than recomputing anything themselves, or the two will
silently drift (spec Section 7, "Style").

**Leakage discipline** (spec Section 4 / Section 7): every engineered
feature must be computable from information available strictly *before* the
race in question. Concretely:

- Rolling averages, standings, and DNF rates use a backward shift — a row's
  features are built from that driver/constructor's races *before* this
  round, never including this round's own result.
- `assert_no_leakage` is a hard runtime check (not just a comment) that
  raises immediately if a post-race-only column (see
  `config.POST_RACE_ONLY_COLUMNS`) ever ends up in the model feature set.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src import config, data_fetch

logger = logging.getLogger(__name__)

# Status strings that count as a classified finish (not a DNF) for the
# reliability-rate features. Determined empirically from the fetched data
# (see FastF1's `Status` values): a clean finish, or finishing N laps down
# but still classified, are not reliability failures. Everything else
# (Retired, Accident, Engine, Disqualified, Did not start, Withdrew, ...)
# counts as a DNF.
_CLASSIFIED_FINISH_STATUSES = {"Finished", "Lapped"}


class LeakageError(RuntimeError):
    """A post-race-only column was found in the model feature set."""


def assert_no_leakage(feature_columns: Iterable[str]) -> None:
    """Raise immediately if any post-race-only column is in `feature_columns`.

    Call this on the *actual* columns being handed to a model — in
    `build_feature_table` before it returns, and again defensively in
    `train.py`/`predict.py` right before `.fit()`/`.predict()`. Cheap, and
    exactly the kind of mistake that's invisible until the metrics look
    suspiciously good.
    """
    leaked = set(feature_columns) & set(config.POST_RACE_ONLY_COLUMNS)
    if leaked:
        raise LeakageError(
            f"Post-race-only column(s) {sorted(leaked)} found in the model feature set. "
            "These are only known after the race and must never be used as predictive "
            "features -- this is the exact mistake that produces a fake-good AUC."
        )


def _is_dnf(status: pd.Series) -> pd.Series:
    """True where `status` represents a DNF/reliability failure, not a finish.

    A missing status means the race has not been run yet (a prediction row),
    which is *unknown*, not a DNF -- returning True there would count every
    unraced round as a reliability failure against that driver.
    """
    is_classified = status.eq("Finished") | status.str.startswith("+", na=False) | status.eq("Lapped")
    return (~is_classified).where(status.notna())


def _constructor_round_level(df: pd.DataFrame, value_col: str, agg: str, out_col: str) -> pd.DataFrame:
    """Aggregate `value_col` to one row per (season, round, constructor_id).

    Required before any backward-shift over constructor history. Two
    teammates share a round, so shifting a plain per-driver-row series
    (`groupby("constructor_id")...shift(1)`) shifts across *rows*, not
    *rounds* -- when both cars finish the same round, that can pull one
    driver's teammate's SAME-round result into their own "prior" feature,
    which is leakage (that result isn't known until this race is run).
    Aggregating to one row per round first, then shifting, closes that gap.
    """
    result = df.groupby(["season", "round", "constructor_id"], as_index=False)[value_col].agg(agg)
    result = result.rename(columns={value_col: out_col}).sort_values(["season", "round"], kind="stable")
    return result


def _add_rolling_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Driver/constructor rolling-average finish position, backward-looking only.

    For each driver (constructor), row *i*'s value is the mean finish
    position over up to `config.ROLLING_WINDOW` races strictly before row
    *i* -- computed as `shift(1)` (drop the current race) then a rolling
    mean, within each driver/constructor group sorted chronologically. A
    driver/constructor's first appearance in the dataset has no prior races,
    so the value is NaN (honest "no history yet" signal; XGBoost handles
    NaN natively) rather than something that fakes a value.
    """
    df = df.sort_values(["season", "round"], kind="stable")

    df["driver_rolling_avg_finish"] = df.groupby("driver_id")["finish_position"].transform(
        lambda s: s.shift(1).rolling(window=config.ROLLING_WINDOW, min_periods=1).mean()
    )

    # Constructor form: average both cars' finish position within each round
    # first (one value per round), THEN shift/roll over rounds -- see
    # `_constructor_round_level` for why the naive per-row groupby is wrong.
    constructor_round = _constructor_round_level(df, "finish_position", "mean", "constructor_round_finish")
    constructor_round["constructor_rolling_avg_finish"] = constructor_round.groupby("constructor_id")[
        "constructor_round_finish"
    ].transform(lambda s: s.shift(1).rolling(window=config.ROLLING_WINDOW, min_periods=1).mean())

    df = df.merge(
        constructor_round[["season", "round", "constructor_id", "constructor_rolling_avg_finish"]],
        on=["season", "round", "constructor_id"],
        how="left",
    )

    return df


def _add_dnf_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Driver/constructor DNF rate, backward-looking only (same shift discipline as rolling form)."""
    df = df.sort_values(["season", "round"], kind="stable")
    df["_is_dnf"] = _is_dnf(df["status"])

    df["driver_dnf_rate"] = df.groupby("driver_id")["_is_dnf"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    # Same round-level-first requirement as constructor rolling form (see
    # `_constructor_round_level`): average both cars' DNF flag within the
    # round, then shift/expand over rounds.
    constructor_round = _constructor_round_level(df, "_is_dnf", "mean", "constructor_round_dnf")
    constructor_round["constructor_dnf_rate"] = constructor_round.groupby("constructor_id")[
        "constructor_round_dnf"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    df = df.merge(
        constructor_round[["season", "round", "constructor_id", "constructor_dnf_rate"]],
        on=["season", "round", "constructor_id"],
        how="left",
    )

    return df.drop(columns=["_is_dnf"])


def _add_standings_features(df: pd.DataFrame) -> pd.DataFrame:
    """Championship points/standing *entering* this race (i.e. through the previous round).

    Championships reset every season, so all of this groups by `season` in
    addition to driver/constructor. `points_before_race` = cumulative sum of
    points from strictly earlier rounds this season (`cumsum() - own points`
    is the backward-shift-free way to write that, since the group is already
    sorted chronologically). `standing_before_race` = rank of that value
    among all drivers/constructors active in the same (season, round)
    snapshot -- rank 1 is the championship leader at that point in time.
    """
    df = df.sort_values(["season", "round"], kind="stable")

    # Sprint weekends (2021+) award points in a separate session on top of
    # the Race -- `points` alone would undercount a driver's real
    # championship total there. `total_points` is what actually accumulates
    # in the standings; the split race/sprint columns are kept around only
    # as raw inputs.
    # fillna(0) on both: an unraced round has no points yet, and leaving it
    # NaN would poison the cumsum for every subsequent round in the season.
    df["total_points"] = df["points"].fillna(0.0) + df["sprint_points"].fillna(0.0)

    df["driver_points_before_race"] = df.groupby(["season", "driver_id"])["total_points"].transform(
        lambda s: s.cumsum() - s
    )
    df["driver_standing_before_race"] = df.groupby(["season", "round"])["driver_points_before_race"].rank(
        method="min", ascending=False
    )

    # Constructor points are the sum of both cars' points in a round. Build
    # a small per-(season, round, constructor) summary, compute the
    # before-this-round cumulative total there, then merge it back onto
    # every driver row for that constructor/round.
    constructor_round = (
        df.groupby(["season", "round", "constructor_id"], as_index=False)["total_points"].sum().rename(
            columns={"total_points": "constructor_round_points"}
        )
    )
    constructor_round = constructor_round.sort_values(["season", "round"], kind="stable")
    constructor_round["constructor_points_before_race"] = constructor_round.groupby(
        ["season", "constructor_id"]
    )["constructor_round_points"].transform(lambda s: s.cumsum() - s)
    constructor_round["constructor_standing_before_race"] = constructor_round.groupby(["season", "round"])[
        "constructor_points_before_race"
    ].rank(method="min", ascending=False)

    df = df.merge(
        constructor_round[
            ["season", "round", "constructor_id", "constructor_points_before_race", "constructor_standing_before_race"]
        ],
        on=["season", "round", "constructor_id"],
        how="left",
    )

    return df


def _normalize_pit_lane_starts(df: pd.DataFrame) -> pd.DataFrame:
    """Re-encode pit-lane starts (`grid == 0`) as back-of-grid.

    FastF1 reports a pit-lane start as grid position 0. Left as-is, that
    inverts the meaning of the single strongest feature in the model: 0
    sorts as *better than pole*, when a pit-lane start is in fact worse
    than last on the grid. Measured over 2018-2025, `grid == 0` rows
    average a 14.1 finish with a 0.000 podium rate -- i.e. they behave
    like the very back (grid 18-20 average ~14.6), nothing like pole
    (3.5 / 0.798).

    Remapping to `max(grid) + 1` within the round restores a monotonic
    "higher number = worse start" ordering, so the model can use a clean
    split instead of having to carve out 0 as a special case.
    """
    grid_max = df.groupby(["season", "round"])["grid"].transform("max")
    df["grid"] = df["grid"].mask(df["grid"] == 0, grid_max + 1)
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    finish = df["finish_position"]
    df[config.TARGET_POSITION] = finish
    # Left as NaN (not 0) where the race has not been run. A plain
    # `(finish <= 3).astype(int)` would turn every unraced row into a
    # confident "did not podium" label, which is a fabricated target that
    # would quietly corrupt training if such rows were ever included.
    df[config.TARGET_PODIUM] = (finish <= 3).astype("float").where(finish.notna())
    df[config.TARGET_WINNER] = (finish == 1).astype("float").where(finish.notna())
    return df


def build_feature_table(raw: pd.DataFrame | None = None, require_target: bool = True) -> pd.DataFrame:
    """Build the full model-ready feature table from raw per-round pulls.

    Parameters
    ----------
    raw:
        Raw driver-race rows as produced by `data_fetch.load_all_raw()`. If
        not given, loads all cached seasons from `config.DATA_RAW_DIR`.
    require_target:
        When True (the default, and what training uses), rows with no
        finishing position are dropped -- a driver who withdrew before the
        race has no target to learn from.

        Set False for *prediction*, where the race being predicted has not
        been run and so no row has a finishing position yet. Keeping those
        rows is the whole point: their features come from prior races, and
        their targets stay NaN rather than being fabricated as zeros.

    Returns
    -------
    One row per driver-race, containing identity columns, every column in
    `config.FEATURE_COLUMNS`, and every column in `config.TARGET_COLUMNS`.
    """
    if raw is None:
        raw = data_fetch.load_all_raw()
    if raw.empty:
        raise ValueError("No raw data found -- run data_fetch first (see src/data_fetch.py).")

    df = raw.copy()

    before = len(df)
    if require_target:
        df = df[df["finish_position"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        logger.info(
            "Dropped %d row(s) with no finish_position (withdrew/DNS before the race, no valid target)",
            dropped,
        )

    df = df.rename(columns={"circuit": "circuit_id"})

    df = _normalize_pit_lane_starts(df)
    df = _add_rolling_form_features(df)
    df = _add_dnf_rate_features(df)
    df = _add_standings_features(df)
    df = _add_targets(df)

    identity_columns = ["season", "round", "event_name", "driver_id", "driver_abbreviation", "constructor_id"]
    output_columns = identity_columns + config.FEATURE_COLUMNS + config.TARGET_COLUMNS
    # `season` and `constructor_id`/`driver_id` are both identity and feature
    # columns -- keep a single copy each, in the declared order.
    seen: set[str] = set()
    ordered_columns = [c for c in output_columns if not (c in seen or seen.add(c))]

    result = df[ordered_columns].sort_values(["season", "round", "driver_id"], kind="stable").reset_index(drop=True)

    assert_no_leakage(config.FEATURE_COLUMNS)

    return result


def build_model_matrix(
    df: pd.DataFrame,
    categories: dict[str, list] | None = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Return `(X, categories)` ready to hand to XGBoost.

    Selects exactly `config.FEATURE_COLUMNS` (so column order is identical
    every time) and casts `config.CATEGORICAL_FEATURES` to pandas
    ``category`` dtype for XGBoost's native categorical support.

    The `categories` mapping is the critical part for train/predict
    consistency. At training time, pass ``None`` and the categories are
    derived from the data and returned, to be persisted alongside the
    model. At prediction time, pass the persisted mapping back in so that
    e.g. ``driver_id`` encodes to the same internal code it did during
    training -- otherwise the model silently reads one driver's history as
    another's. Values unseen at training time become NaN, which XGBoost
    handles natively (a debuting driver simply has no learned identity).

    This function is the single place the model matrix is defined; both
    `src/train.py` and `src/predict.py` must call it rather than
    assembling columns themselves (spec Section 7).
    """
    assert_no_leakage(config.FEATURE_COLUMNS)

    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required column(s): {sorted(missing)}")

    X = df[config.FEATURE_COLUMNS].copy()

    resolved: dict[str, list] = {}
    for col in config.CATEGORICAL_FEATURES:
        if categories is None:
            values = sorted(X[col].dropna().unique().tolist())
        else:
            values = list(categories[col])
        X[col] = pd.Categorical(X[col], categories=values)
        resolved[col] = values

    # Every remaining feature must be numeric for XGBoost. An all-null
    # column arrives as object dtype rather than float -- which happens for
    # real at predict time on a single round where nobody has set a Q3 time
    # yet -- and would otherwise fail deep inside XGBoost's DMatrix
    # construction. Coerce explicitly so a missing value stays a missing
    # value (NaN), which XGBoost handles natively.
    for col in X.columns:
        if col not in config.CATEGORICAL_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, resolved


def save_feature_table(df: pd.DataFrame, path=config.FEATURE_TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_feature_table(path=config.FEATURE_TABLE_PATH) -> pd.DataFrame:
    """Load the processed feature table -- every season present on disk."""
    return pd.read_parquet(path)


def load_training_table(path=config.FEATURE_TABLE_PATH) -> pd.DataFrame:
    """The feature table restricted to `config.SEASONS`.

    The processed table deliberately contains every season that has been
    fetched, including any held out of training (2026). Anything that
    *fits* a model must go through here, or a held-out season would
    silently become training data and quietly invalidate its own
    evaluation.
    """
    table = load_feature_table(path)
    training = table[table["season"].isin(config.SEASONS)]
    held_out = sorted(set(table["season"]) - set(config.SEASONS))
    if held_out:
        logger.info("Excluding held-out season(s) %s from training", held_out)
    return training.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    table = build_feature_table()
    save_feature_table(table)
    print(f"Built feature table: {table.shape[0]} rows, {table.shape[1]} columns -> {config.FEATURE_TABLE_PATH}")
