# F1 Race Predictor — Rebuild Specification

Reconstructed from four screenshots of the original project (structure, training output, probability table, and formatted round output). This is a rebuild, not a redesign — the goal is to get back to a working system that reproduces the original metrics, with a handful of structural fixes noted in Section 6.

---

## 1. Project Summary & Goal

**What this is:** A Formula 1 race outcome predictor that pulls historical race data via FastF1, engineers features from qualifying/grid/driver/constructor history, and trains XGBoost models to predict three targets:

1. **Podium finish** (binary classification) — will this driver finish top 3?
2. **Finishing position** (regression) — what position will this driver finish in?
3. **Race winner** (binary/probabilistic classification) — win probability per driver, normalized across the grid.

**Original benchmark metrics to reproduce (from training output):**

| Model | Metric | Value |
|---|---|---|
| Podium | CV AUC | 0.934 |
| Podium | CV LogLoss | 0.269 |
| Position | CV MAE | 3.626 |

**Training data range:** seasons 2018–2025 (fetched season by season via FastF1).

**Output surface:** CLI. A formatted terminal report per race round showing predicted podium with probabilities, top-3 win probabilities, and predicted top-10 finishing order.

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | original used a `.venv311`, so 3.11 specifically |
| Data source | `fastf1` | official F1 timing/results API wrapper; has a local cache |
| Modeling | `xgboost` | `XGBClassifier` for podium/winner, `XGBRegressor` for position |
| ML tooling | `scikit-learn` | cross-validation, metrics (`roc_auc_score`, `log_loss`, `mean_absolute_error`), preprocessing |
| Data handling | `pandas`, `numpy` | |
| Model persistence | `joblib` | matches original `.joblib` artifacts |
| Terminal output | `rich` (recommended) or plain ANSI | the original output used color, emoji, and aligned columns |
| Env/config | `python-dotenv` or plain constants module | no API keys needed — FastF1 is open |
| Notebooks | `jupyter` | for the exploration folder |

**Caching note:** FastF1 requires an explicit cache directory (`.fastf1_cache/` in the original). Enable it before any data fetch — without it, every run re-downloads session data and is punishingly slow.

---

## 3. Target File Structure

Reconstructed from the original, with the fixes from Section 6 applied:

```
f1-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── .fastf1_cache/              # gitignored — FastF1 session cache
│
├── app/
│   ├── __init__.py
│   ├── app.py                  # main CLI entrypoint
│   ├── app_calendar.py         # season calendar / round lookup helpers
│   ├── predict_any.py          # predict for an arbitrary specified round
│   ├── predict_upcoming.py     # predict the next unraced round
│   ├── predict_latest_done.py  # predict + compare against the most recent completed race
│   ├── predict_podium_example.py
│   ├── predict_order_example.py
│   └── predict_example.py
│
├── src/                        # NEW — core logic extracted out of scripts
│   ├── __init__.py
│   ├── config.py               # season range, cache path, model paths, feature list
│   ├── data_fetch.py           # FastF1 session loading, season iteration
│   ├── features.py             # feature engineering — single source of truth
│   ├── train.py                # trains all three models, reports CV metrics
│   ├── predict.py              # loads models, produces predictions for a round
│   └── render.py               # the formatted terminal report (replaces prediction1-24.py)
│
├── data/
│   ├── raw/                    # cached raw session pulls
│   └── processed/              # engineered feature tables
│
├── models/
│   ├── podium_xgb.joblib
│   ├── position_xgb.joblib
│   └── winner_xgb.joblib
│
├── scripts/
│   ├── train_models.py         # thin CLI wrapper over src/train.py
│   ├── update_training_data.py # refresh data for new completed races
│   ├── eval_past.py            # backtest predictions against completed races
│   └── eval_winners.py         # winner-model-specific evaluation
│
├── notebooks/
│   └── exploration.ipynb
│
├── screenshots/                # output captures for the README
│
└── tests/
    ├── test_features.py
    └── test_predict.py
```

---

## 4. Feature Set

From the probability table screenshot, the model consumed at minimum:

- `driver_id` — categorical, encoded
- `constructor_id` — categorical, encoded (team identity carries most of the car-performance signal)
- `grid` — starting grid position (integer; this is almost certainly the single strongest predictor)

**Recommended additions for the rebuild** (these were likely present in the original given the 0.934 AUC, and should be included regardless):

- Qualifying position and Q1/Q2/Q3 session times
- Driver's rolling average finishing position (last N races)
- Constructor's rolling average finishing position (last N races)
- Driver's championship points/standing coming into the race
- Constructor's championship standing coming into the race
- Circuit identifier (some drivers/teams over- or under-perform at specific tracks)
- Season year (captures regulation-era shifts — 2022 ground-effect rules were a major discontinuity)
- DNF/reliability rate for the driver and constructor

**Critical constraint:** every feature must be computable using *only* information available before the race starts. No lap times from the race itself, no final classification data. Grid position and qualifying results are fine — they precede the race. This is the single easiest way to accidentally leak and produce a fake-good AUC.

---

## 5. Build Roadmap

Flat, sequential. One task per session.

1. Scaffold the repo: `requirements.txt`, `.gitignore` (must include `.fastf1_cache/`, `.venv*/`, `data/raw/`, `models/*.joblib`), and the directory tree from Section 3.
2. Implement `src/config.py` — season range (2018–2025), FastF1 cache path, model output paths, canonical feature list, target column names.
3. Implement `src/data_fetch.py` — enable FastF1 cache, load a season's race results, iterate seasons, persist raw pulls to `data/raw/`. Handle FastF1 rate limits and missing sessions gracefully.
4. Run a single-season fetch (e.g. 2023 only) and confirm the data shape before pulling all eight seasons.
5. Pull all seasons 2018–2025 into `data/raw/`.
6. Implement `src/features.py` — build the feature table per driver-race row. Include the no-leakage assertion: any column derived from post-race data raises immediately.
7. Write `tests/test_features.py` — assert no leakage, assert row counts match expected driver-per-race counts, assert no unexpected nulls in required features.
8. Generate `data/processed/` feature table from all seasons and eyeball it against a known race.
9. Implement `src/train.py` — podium model (`XGBClassifier`), reporting CV AUC and LogLoss. Target: reproduce approximately AUC 0.934 / LogLoss 0.269.
10. Extend `src/train.py` — position model (`XGBRegressor`), reporting CV MAE. Target: approximately 3.626.
11. Extend `src/train.py` — winner model. Note: this is not just "podium with k=1." Win probabilities must be **normalized across the grid so they sum to 1.0** per race, which the raw classifier output will not do. Implement that normalization explicitly.
12. Serialize all three models to `models/*.joblib`. Write `scripts/train_models.py` as a thin CLI wrapper.
13. Implement `src/predict.py` — load models, take a (season, round), build features for that grid, return a dataframe matching the probability-table format: `driver_id, constructor_id, grid, prob_win`.
14. Implement `src/render.py` — the formatted terminal report: header with season/round/GP name, predicted podium with per-driver probabilities, top-3 win probabilities as percentages, predicted top-10 finishing order.
15. Implement `app/app_calendar.py` — season calendar lookup, map round number to GP name, identify next unraced round.
16. Implement `app/app.py` as the main CLI entrypoint with subcommands wired to the predict variants.
17. Implement `app/predict_any.py`, `predict_upcoming.py`, `predict_latest_done.py` as thin wrappers over `src/predict.py` + `src/render.py`.
18. Implement `scripts/eval_past.py` — backtest across completed races: how often was the predicted winner correct, podium hit rate, mean position error. **This is the number that matters most for the resume** — CV metrics are in-sample-ish; backtest accuracy on real completed races is what you defend in an interview.
19. Implement `scripts/eval_winners.py` — winner-model-specific evaluation (calibration: when the model says 40%, does that driver win ~40% of the time?).
20. Implement `scripts/update_training_data.py` — incremental refresh as new races complete.
21. Write `tests/test_predict.py` — mock model loading, assert output schema and that win probabilities sum to 1.0.
22. Write the README: metrics table first (CV metrics *and* backtest results), then setup, then usage, then a screenshot of the formatted output.
23. Add a **Limitations** section to the README: what the model can't see (weather changes, crashes, safety cars, strategy calls, mechanical failures), and where it's systematically wrong.
24. **Push to GitHub before anything else is added.** This is task 24 only because it needs something to push — do it the moment task 12 completes if you can.

---

## 6. Fixes to Apply During the Rebuild

The original structure showed a few things worth correcting rather than faithfully reproducing:

**`prediction1.py` through `prediction24.py` should not exist.** Twenty-four near-identical files, one per round, plus `pretty_round.py`, `pretty_round19.py`, and `pretty_all_rounds.py` — this is copy-paste duplication. Replace all of it with one parameterized function: `render_round(season, round_number)`. A reviewer looking at your repo will notice 24 duplicate files immediately, and it reads as inexperience even if the modeling underneath is solid.

**Core logic lived in `scripts/` and `app/` with no library layer.** Extract the actual logic into `src/` so training, prediction, and rendering are importable and testable. The `app/` and `scripts/` files become thin CLI wrappers.

**There were no tests.** Add at minimum the two in the roadmap — the leakage assertion is the important one.

**Add backtest evaluation as a first-class output.** The original reported CV metrics. Backtest results on completed races are far more convincing and far harder to fake.

---

## 7. Development Constraints

**Data integrity**
- Enable the FastF1 cache before any fetch. Without it, re-runs are extremely slow and you risk rate limiting.
- Never use post-race information as a feature. Enforce this with an explicit assertion in `features.py`, not just a comment.
- Rolling averages must be computed with a strict backward window — no lookahead. Use expanding/rolling with a shift, and test it.

**Reproducibility**
- Fix random seeds for all XGBoost training and CV splits.
- Cross-validation must respect time ordering where it matters — random K-fold across seasons leaks future information into past predictions. Prefer time-series splits or at minimum hold out full seasons.
- Note: if time-aware CV drops the metrics below 0.934 AUC / 3.626 MAE, **report the lower honest number**. The original figures may have come from random K-fold, which is optimistic for time-series data. A defensible 0.89 beats an indefensible 0.934.

**Cost**
- $0. FastF1 is free and open. No paid APIs anywhere in this project.

**Version control — non-negotiable**
- `git init` and first commit before writing any real code.
- Push to a GitHub remote as soon as there is anything to push, and push at the end of every session.
- This project was lost once. Do not let that happen twice.

**Style**
- Type-hint public functions.
- Config values (seasons, feature lists, model paths) live in `src/config.py` — never hardcoded inline.
- One source of truth for feature engineering. Training and prediction must call the *same* function, or they will silently drift and your predictions will be wrong in ways that are very hard to debug.
