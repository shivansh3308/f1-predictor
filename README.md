# F1 Race Predictor

Predicts Formula 1 race outcomes from information available **before lights out** — qualifying, grid, and rolling driver/constructor form. Three XGBoost models: podium finish, finishing position, and race winner.

Built on [FastF1](https://docs.fastf1.dev/). Trained on 2018–2025 — 173 races, 3,455 driver-race rows — and evaluated on the in-progress 2026 season as a true holdout. No paid APIs.

---

## Results

Two different questions get two different answers, and the gap between them is the most interesting thing in this project.

### Cross-validation (season-forward, time-respecting)

Every fold trains only on seasons *strictly before* the season it is scored on — test seasons 2021–2025. This is stricter than grouping by season, which would still allow training on 2025 to predict 2018.

| Model | Metric | Result | Original benchmark |
|---|---|---|---|
| Podium | CV AUC | **0.9238** | 0.934 |
| Podium | CV LogLoss | **0.2486** | 0.269 ✅ better |
| Position | CV MAE | **3.2352** | 3.626 ✅ better |
| Position | CV RMSE | 4.5319 | — |
| Winner | CV AUC | **0.9480** | — |
| Winner | CV LogLoss | 0.1168 | — |
| Winner | Top-1 accuracy | 55.5% | — |

These reproduce the original project's figures closely — within ~1% on podium AUC, and better on both LogLoss and MAE — while using a split that doesn't leak the future.

### Backtest on real races — the number that actually matters

CV scores are aggregate and abstract. This is 114 completed races (2021–2025), each predicted by models trained only on earlier seasons, scored against a baseline of **simply predicting the grid order** (pole wins, front three are the podium, everyone finishes where they started).

| Metric | Model | Grid baseline | Gap |
|---|---|---|---|
| Winner accuracy | **55.3%** | 54.4% | +0.9% |
| Podium hit rate | 64.3% | **67.8%** | −3.5% |
| Position MAE | **3.39** | 3.42 | −0.03 |

The model edges the baseline on winner accuracy and position error, and still loses on podium selection. **The headline numbers are close enough to the baseline that the honest summary is "roughly comparable", not "better"** — see the next section, which is the more useful read.

The baseline was verified independently — recomputed straight from the feature table, the pole-sitter really does win 54.4% of these races.

Per season:

| Season | Model | Baseline | |
|---|---|---|---|
| 2021 | 50.0% | 50.0% | |
| 2022 | 40.9% | 45.5% | |
| 2023 | **86.4%** | 63.6% | ← Verstappen won 19 of 22 |
| 2024 | 37.5% | 45.8% | |
| 2025 | 62.5% | 66.7% | |

The model wins big in 2023, a season one driver dominated, and loses whenever the competitive order shifted. In 2024 it predicted Verstappen **13 times in races someone else won** — it had learned 2023's pecking order from driver and constructor identity and carried that stale prior into McLaren's rise.

### What actually improved, and what only looked like it did

An earlier version of this model used raw qualifying lap times and predicted absolute finishing position. Two changes were made, tuned only on 2021–2023, with 2024–2025 and 2026 left untouched until the end:

1. **Qualifying features became gap-to-pole instead of raw lap times.** A raw time is dominated ~30:1 by which circuit it was set at (mean Q3 ranges 71 s at Monaco to 109 s at Spa, while within-session driver spread is ~1.4 s), and regulation eras move it again — the fastest Silverstone Q3 shifts 16.7 s across seasons in this dataset.
2. **The position model now predicts places gained or lost relative to the grid**, not absolute position. Predicting absolute position makes the model relearn "grid ≈ finish" from scratch, which is exactly what the baseline already gets for free.

Splitting the result by whether a season was used for tuning is what makes this interpretable:

| | Tuning seasons (2021–23) | Untouched (2024–25) |
|---|---|---|
| Winner accuracy gain | **+9.1%** | **+0.0%** |
| Position MAE gain | −0.08 | **−0.10** |

**The winner-accuracy gain did not replicate at all.** It was an artefact of the seasons it was selected on — a false positive that would have been reported as a real result had the tuning and reporting sets not been separated. The MAE gain is real: it is slightly *larger* on untouched data than on tuning data, and it holds again on the sealed 2026 set below (4.25 → 3.83).

So of the two changes, one produced a genuine, replicated improvement and the other produced a convincing-looking number that evaporated on contact with unseen data. The gap-to-pole features were kept anyway — the raw times are a measured defect and the change does no harm — but no claim is made that they improved winner accuracy.

### 2026 holdout — a season the models have genuinely never seen

The backtest above reconstructs out-of-sample performance by retraining. This is stronger: the models serialized in `models/` were fit on 2018–2025 and saved *before* any 2026 race existed. Scoring the 12 completed 2026 rounds with them, unmodified, is the real use case — and 2026 brought a major regulation change, new power units, and two new constructors.

| Metric | Model | Grid baseline | Gap |
|---|---|---|---|
| Winner accuracy | 66.7% | 75.0% | −8.3% |
| Podium hit rate | **63.9%** | 61.1% | **+2.8%** |
| Position MAE | 3.83 | 3.47 | +0.36 |

**The useful comparison is against the earlier backtest, because the baseline acts as a control:**

| | 2021–2025 | 2026 | Change |
|---|---|---|---|
| Grid baseline MAE | 3.42 | 3.47 | +0.05 — flat |
| **Model MAE** | 3.39 | **3.83** | **+0.44** |

The baseline barely moved, so 2026 is not intrinsically harder to predict. Only the model degraded — which isolates the cause to its learned competitive order going stale, exactly the failure the Limitations section predicts for a regulation year. (The earlier version of the model scored 4.25 here; the grid-relative target improved it to 3.83 on a set that was sealed while that change was being made.) Podium selection remains the model's weakest metric, and is the one it loses on in every evaluation here.

Winner accuracy here should be read with care: 8/12 versus 9/12 is a difference of **one race**, with a 95% confidence interval of roughly 35–90%. Position MAE, measured over 264 driver-races, is the number carrying real information.

```bash
python scripts/eval_holdout.py --season 2026
```

### Calibration — does "40%" mean 40%?

Not at the confident end. Same walk-forward, out-of-sample predictions:

| Predicted band | n | Predicted | Observed | Gap |
|---|---|---|---|---|
| 0.2–0.4 | 53 | 28.9% | 39.6% | +10.7% |
| 0.4–0.6 | 34 | 51.8% | 41.2% | −10.7% |
| 0.6–0.8 | 40 | 70.1% | 62.5% | −7.6% |
| 0.8–1.0 | 34 | 88.5% | **61.8%** | **−26.8%** |

When the model says a driver is 80–100% to win, they win **62%** of the time.

Overall Expected Calibration Error is **0.0146**, which looks excellent and is misleading: 1,698 of 2,275 driver-races sit in the bottom band, are correctly given ~0%, and contribute almost no error. Restricted to predictions above 20% — the ones anyone would act on — ECE is **0.1330**, roughly 9× worse.

Brier score is 0.0314 against a base-rate baseline of 0.0476, so the probabilities do carry real information. They are simply too confident.

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

```bash
python -m app.app standings --season 2025
```

Add `--actual` to compare predictions against the real result, and `--fetch` to pull a round that isn't cached locally.

### Evaluation

```bash
python scripts/eval_past.py --misses
```

```bash
python scripts/eval_winners.py
```

Score the shipped models on a season they were never trained on:

```bash
python scripts/eval_holdout.py --season 2026
```

### Keeping data current

Fetches only the rounds that are actually missing, rather than re-pulling whole seasons:

```bash
python scripts/update_training_data.py
```

---

## Sample output

### Predicting a race

`python -m app.app any 2021 22` — the 2021 Abu Dhabi title decider, with Verstappen and Hamilton arriving level on points:

![Race prediction](screenshots/prediction.png)

### Comparing against the real result

`python -m app.app any 2024 1 --actual` adds the true finishing position and a signed delta, so you can see where the model was right and where it wasn't:

![Prediction vs actual](screenshots/prediction_vs_actual.png)

Verstappen called exactly; Perez predicted P5 and finished P2.

### Season calendar

`python -m app.app calendar --season 2026` — the live schedule, marking which rounds have run:

![Season calendar](screenshots/calendar.png)

---

## Limitations

### What the model cannot see

Every feature is fixed before lights out. Everything that decides a Grand Prix after that is invisible to it: **safety cars and red flags**, **weather changes mid-race**, **first-lap collisions**, **pit strategy and undercuts**, **tyre degradation**, **mechanical failures**, and **team orders**.

This is measurable, not hypothetical. Splitting the backtest by whether a driver actually finished:

| | Mean position error | n |
|---|---|---|
| Finished | **2.90** | 1,965 |
| Did not finish | **7.12** | 310 |

DNFs are 13.6% of driver-races and carry 2.5× the error. Excluding them, MAE falls from 3.48 to 2.90 — so roughly a sixth of the model's total error comes from outcomes no pre-race feature could have predicted. A retirement on lap 3 is not a modelling failure; it is a car breaking.

Error also rises steadily down the field, which is where chaos concentrates:

| Actual finish | Mean position error | n |
|---|---|---|
| 1–3 | 1.85 | 342 |
| 4–10 | 3.16 | 798 |
| 11–20 | 4.19 | 1,135 |

### Where it is systematically wrong

**It barely beats predicting the grid order.** Across 114 real races the model is ahead by 0.9% on winner accuracy and 0.03 on position error, and behind by 3.5% on podium hit rate. Those margins are small enough that "roughly comparable to assuming everyone finishes where they qualified" is the honest description. On the 2024-25 seasons held out of tuning it does not lead on any metric.

**It over-anchors on driver and constructor identity.** The clearest failure in the dataset: in 2024 it predicted Verstappen as winner in **13 races he did not win**. It had learned 2023's order — where he won 19 of 22 — and carried that prior into a season where McLaren became the quicker car. The model adapts to a shifting competitive order roughly a season late.

**It is over-confident exactly where confidence matters.** At a stated 80–100% win probability, the driver wins 61.8% of the time. Treat high probabilities as strong preferences, not as odds.

**It is weakest in transition years and strongest in dominant ones.** It beat the baseline decisively in 2023 and lost in every other backtested season. A model that only outperforms when one driver is winning most races is not adding much: that is the season you least need a model for.

**Upsets are invisible to it by construction.** In 14% of races the winner started outside the top three — the races most worth predicting are the ones where pre-race form is least informative.

**Regulation changes break the learned order.** 2022's ground-effect rules reset the competitive picture, and the model has no feature representing "the cars are different now". `season` is included, but it cannot anticipate a regime shift it has not yet seen.

**New drivers have no history.** A debut driver's rolling-form, DNF-rate and standings features are null, and their identity is an unseen category. XGBoost handles this natively rather than guessing, but such a driver is predicted almost purely from grid position.

### Scope

- **Training data is 2018–2025.** The in-progress 2026 season is deliberately held out rather than trained on. It is more valuable as an uncontaminated test set than as ~12 extra rounds of training data, and folding in a part-run season would also skew rolling-form and standings features. 2026 data *is* fetched and used for evaluation (`scripts/eval_holdout.py`), and `python -m app.app` can predict 2026 rounds directly.
- **Predicting a race inside the training range is in-sample** and will look better than the model's true skill. `scripts/eval_past.py` is the honest measurement.
- **Requires qualifying to have run.** A race has no grid until then, which in practice means predictions are only possible from the day before.
- **Sprint points are included** in championship standings, but sprint results themselves are not used as a form signal.

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
scripts/    train_models, eval_past, eval_holdout, eval_winners, update_training_data
tests/      191 tests
data/       raw/ and processed/ (gitignored — regenerable)
models/     *.joblib artifacts (gitignored — regenerable)
```

The original project had `prediction1.py` through `prediction24.py` — 24 near-identical files, one per round, plus three `pretty_*.py` variants. All 27 are replaced by a single parameterized `render_round(season, round_number)`.

### Tests

```bash
python -m pytest tests/ -q
```

191 tests. The ones worth knowing about assert the properties that fail silently rather than loudly: that CV folds never train on future seasons, that rolling features can't see the race they describe, that win probabilities sum to 1.0 per race, that the backtest's baseline is computed independently of the model, and that a held-out season never reaches the training table.
