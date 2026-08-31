# F1 Race Predictor

A Formula 1 race outcome predictor. Pulls historical race data via [FastF1](https://docs.fastf1.dev/),
engineers pre-race features (qualifying, grid, driver/constructor history), and trains XGBoost models to predict:

1. **Podium finish** (binary classification)
2. **Finishing position** (regression)
3. **Race winner** (probabilistic, normalized across the grid)

> Status: scaffolding only. See [`f1-predictor-rebuild-spec.md`](f1-predictor-rebuild-spec.md) for the full
> rebuild plan and roadmap. Metrics, setup, and usage sections will be filled in as the build progresses
> (spec Section 5, task 22).

## Project layout

See `f1-predictor-rebuild-spec.md` Section 3 for the target directory structure and Section 5 for the
build roadmap this repo is following, task by task.
