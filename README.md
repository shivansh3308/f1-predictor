# F1 Race Predictor

Predicts Formula 1 race outcomes from information available **before lights out** — qualifying, grid, and rolling driver/constructor form. Three XGBoost models: podium finish, finishing position, and race winner.

Built on [FastF1](https://docs.fastf1.dev/). 2018–2025, 173 races, 3,455 driver-race rows. No paid APIs.

---

## Results

Two different questions get two different answers, and the gap between them is the most interesting thing in this project.

### Cross-validation (season-forward, time-respecting)

Every fold trains only on seasons *strictly before* the season it is scored on — test seasons 2021–2025. This is stricter than grouping by season, which would still allow training on 2025 to predict 2018.

| Model | Metric | Result | Original benchmark |
|---|---|---|---|
| Podium | CV AUC | **0.9258** | 0.934 |
| Podium | CV LogLoss | **0.2491** | 0.269 ✅ better |
| Position | CV MAE | **3.3330** | 3.626 ✅ better |
| Position | CV RMSE | 4.5195 | — |
| Winner | CV AUC | **0.9468** | — |
| Winner | CV LogLoss | 0.1209 | — |
| Winner | Top-1 accuracy | 50.0% | — |

These reproduce the original project's figures closely — within ~1% on podium AUC, and better on both LogLoss and MAE — while using a split that doesn't leak the future.

### Backtest on real races — the number that actually matters

CV scores are aggregate and abstract. This is 114 completed races (2021–2025), each predicted by models trained only on earlier seasons, scored against a baseline of **simply predicting the grid order** (pole wins, front three are the podium, everyone finishes where they started).

| Metric | Model | Grid baseline | Gap |
|---|---|---|---|
| Winner accuracy | 50.0% | **54.4%** | −4.4% |
| Podium hit rate | 64.3% | **67.8%** | −3.5% |
| Position MAE | 3.48 | **3.42** | +0.06 |

**The model does not beat the grid-order baseline.** On real races, "assume everyone finishes where they qualified" is slightly better than the trained model on all three metrics.

This is reported rather than tuned away, because tuning against this backtest and then quoting it would destroy the only thing that makes it worth quoting. The baseline was verified independently before drawing the conclusion — recomputed straight from the feature table, the pole-sitter really does win 54.4% of these races.

The per-season split shows why:

| Season | Model | Baseline | |
|---|---|---|---|
| 2021 | 36.4% | 50.0% | |
| 2022 | 31.8% | 45.5% | |
| 2023 | **81.8%** | 63.6% | ← Verstappen won 19 of 22 |
| 2024 | 37.5% | 45.8% | |
| 2025 | 62.5% | 66.7% | |

The model wins big in 2023, a season one driver dominated, and loses whenever the competitive order shifted. In 2024 it predicted Verstappen **13 times in races someone else won** — it had learned 2023's pecking order from driver and constructor identity and carried that stale prior into McLaren's rise.

### Calibration — does "40%" mean 40%?

Not at the confident end. Same walk-forward, out-of-sample predictions:

| Predicted band | n | Predicted | Observed | Gap |
|---|---|---|---|---|
| 0.4–0.6 | 41 | 49.9% | 34.1% | −15.8% |
| 0.6–0.8 | 33 | 68.7% | 69.7% | +1.0% |
| 0.8–1.0 | 34 | 87.8% | **58.8%** | **−29.0%** |

When the model says a driver is 80–100% to win, they win **59%** of the time.

Overall Expected Calibration Error is **0.0144**, which looks excellent and is misleading: 1,663 of 2,275 driver-races sit in the bottom band, are correctly given ~0%, and contribute almost no error. Restricted to predictions above 20% — the ones anyone would act on — ECE is **0.1167**, roughly 8× worse.

Brier score is 0.0335 against a base-rate baseline of 0.0476, so the probabilities do carry real information. They are simply too confident.

---

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

Fetch the data and build the models (the first fetch pulls ~173 races and takes a while — FastF1 rate-limits at 500 calls/hour, and the script backs off and resumes on its own):

```bash
python -m src.data_fetch
python -m src.features
python scripts/train_models.py --save
```

---

## Usage

```bash
python -m app.app upcoming
```

```bash
python -m app.app latest --season 2025
```

```bash
python -m app.app any 2021 22 --top 10
```

```bash
python -m app.app calendar --season 2026
```

Add `--actual` to compare predictions against the real result, and `--fetch` to pull a round that isn't cached locally.

### Evaluation

```bash
python scripts/eval_past.py --misses
```

```bash
python scripts/eval_winners.py
```

### Keeping data current

Fetches only the rounds that are actually missing, rather than re-pulling whole seasons:

```bash
python scripts/update_training_data.py
```

---

## Sample output

`python -m app.app any 2021 22` — the 2021 Abu Dhabi title decider:

```
╭──────────────────────────────────────╮
│ 2021  Round 22  Abu Dhabi Grand Prix │
╰──────────────────────────────────────╯

Predicted podium
    Driver               Team      Grid  Podium prob
🥇  VER  Max Verstappen  Red Bull    P1        99.0%
🥈  HAM  Hamilton        Mercedes    P2        94.3%
🥉  SAI  Sainz           Ferrari     P5        38.1%

Win probability (top 3)
    Driver               Team        Win prob
1   VER  Max Verstappen  Red Bull       73.3%  ███████████████░░░░░
2   HAM  Hamilton        Mercedes       24.4%  █████░░░░░░░░░░░░░░░
3   TSU  Tsunoda         AlphaTauri      0.6%  ░░░░░░░░░░░░░░░░░░░░

Predicted top 10
Pos  Driver               Team        Grid
  1  VER  Max Verstappen  Red Bull      P1
  2  HAM  Hamilton        Mercedes      P2
  3  PER  Perez           Red Bull      P4
  4  NOR  Norris          McLaren       P3
  5  BOT  Bottas          Mercedes      P6
  6  SAI  Sainz           Ferrari       P5
  7  LEC  Leclerc         Ferrari       P7
  8  OCO  Ocon            Alpine        P9
  9  ALO  Alonso          Alpine       P11
 10  GAS  Gasly           AlphaTauri   P12
```

Output is colourised in a real terminal. With `--actual`, a fourth column shows the true finishing position and a signed delta.

---

## How it works

**Features** (`src/features.py`) — one row per driver-race, every value computable strictly before the race starts: grid, qualifying position, Q1/Q2/Q3 times, rolling driver and constructor form, championship points and standing entering the round, DNF rates, circuit, and season.

**No leakage** is enforced at runtime, not by convention. `assert_no_leakage` raises if any post-race column reaches the model, rolling features use a strict backward shift, and the test suite plants a leak to confirm the assertion actually fires.

**One source of truth** — training and prediction both call `features.build_model_matrix`, and each model's categorical mapping is persisted inside its own artifact. Re-deriving categories at predict time would silently remap driver identities.

**Validation** — `season-forward` folds for CV, and a separate walk-forward backtest that never scores a race with a model that saw it.

### Project layout

```
src/        core library — config, data_fetch, features, train, predict, render
app/        CLI entrypoints — app.py plus the predict_* variants
scripts/    train_models, eval_past, eval_winners, update_training_data
tests/      179 tests
data/       raw/ and processed/ (gitignored — regenerable)
models/     *.joblib artifacts (gitignored — regenerable)
```

The original project had `prediction1.py` through `prediction24.py` — 24 near-identical files, one per round, plus three `pretty_*.py` variants. All 27 are replaced by a single parameterized `render_round(season, round_number)`.

### Tests

```bash
python -m pytest tests/ -q
```

179 tests. The ones worth knowing about assert the properties that fail silently rather than loudly: that CV folds never train on future seasons, that rolling features can't see the race they describe, that win probabilities sum to 1.0 per race, and that the backtest's baseline is computed independently of the model.
