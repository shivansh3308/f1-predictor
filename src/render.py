"""Formatted terminal report for a single race.

This module is the replacement for the original project's ``prediction1.py``
through ``prediction24.py``, plus ``pretty_round.py``, ``pretty_round19.py``
and ``pretty_all_rounds.py`` -- 27 near-identical files, one per round.
All of it collapses into one parameterized function, `render_round(season,
round_number)` (spec Section 6).

Report layout (spec task 14):
  1. header with season / round / GP name
  2. predicted podium with per-driver probabilities
  3. top-3 win probabilities as percentages
  4. predicted top-10 finishing order
"""

from __future__ import annotations

import logging

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import predict

logger = logging.getLogger(__name__)

PODIUM_MEDALS = ["\U0001F947", "\U0001F948", "\U0001F949"]  # gold, silver, bronze

# Probability above which a prediction is shown as "confident" (green) and
# below which it reads as a long shot (dim). Purely presentational.
_HIGH_CONFIDENCE = 0.50
_LOW_CONFIDENCE = 0.05


# Slugs whose real-world capitalisation plain title-casing gets wrong
# (initialisms, internal capitals) or that are abbreviated in the source
# data. Checked against every constructor_id present in 2018-2025.
_NAME_OVERRIDES = {
    "rb": "RB",
    "alphatauri": "AlphaTauri",
    "alfa": "Alfa Romeo",
    "mclaren": "McLaren",
    "de_vries": "de Vries",
}


def prettify_slug(slug: str) -> str:
    """``max_verstappen`` -> ``Max Verstappen``; ``red_bull`` -> ``Red Bull``.

    FastF1 ids are lowercase snake_case. Names that title-casing would
    mangle are handled by `_NAME_OVERRIDES`.
    """
    key = str(slug)
    if key in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[key]
    return " ".join(word.capitalize() for word in key.split("_"))


def _probability_style(probability: float) -> str:
    if probability >= _HIGH_CONFIDENCE:
        return "bold green"
    if probability <= _LOW_CONFIDENCE:
        return "dim"
    return "yellow"


def build_header(season: int, round_number: int, event_name: str) -> Panel:
    title = Text()
    title.append(f"{season}", style="bold white")
    title.append(f"  Round {round_number}  ", style="white")
    title.append(f"{event_name}", style="bold cyan")
    return Panel(title, expand=False, border_style="bright_black")


def build_podium_table(predictions: pd.DataFrame) -> Table:
    """Predicted podium: the three drivers most likely to finish top-3.

    Ranked by `prob_podium` (the podium model), not by win probability --
    a driver can be a likely podium finisher without being a likely winner,
    and this section is answering the podium question.
    """
    table = Table(title="Predicted podium", title_justify="left", title_style="bold", box=None, pad_edge=False)
    table.add_column("", width=2)
    table.add_column("Driver", style="bold")
    table.add_column("Team", style="cyan")
    table.add_column("Grid", justify="right")
    table.add_column("Podium prob", justify="right")

    top3 = predictions.sort_values("prob_podium", ascending=False).head(3)
    for medal, (_, row) in zip(PODIUM_MEDALS, top3.iterrows()):
        table.add_row(
            medal,
            f"{row['driver_abbreviation']}  {prettify_slug(row['driver_id'])}",
            prettify_slug(row["constructor_id"]),
            f"P{int(row['grid'])}",
            Text(f"{row['prob_podium']:.1%}", style=_probability_style(row["prob_podium"])),
        )
    return table


def build_win_probability_table(predictions: pd.DataFrame, top_n: int = 3) -> Table:
    """Top-N win probabilities, as percentages of a distribution summing to 100%."""
    table = Table(
        title=f"Win probability (top {top_n})",
        title_justify="left",
        title_style="bold",
        box=None,
        pad_edge=False,
    )
    table.add_column("", width=2)
    table.add_column("Driver", style="bold")
    table.add_column("Team", style="cyan")
    table.add_column("Win prob", justify="right")
    table.add_column("", width=22)

    top = predictions.sort_values("prob_win", ascending=False).head(top_n)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        probability = row["prob_win"]
        filled = round(probability * 20)
        bar = Text("█" * filled + "░" * (20 - filled), style=_probability_style(probability))
        table.add_row(
            str(rank),
            f"{row['driver_abbreviation']}  {prettify_slug(row['driver_id'])}",
            prettify_slug(row["constructor_id"]),
            Text(f"{probability:.1%}", style=_probability_style(probability)),
            bar,
        )
    return table


def build_finishing_order_table(
    predictions: pd.DataFrame,
    top_n: int = 10,
    show_actual: bool = False,
) -> Table:
    """Predicted finishing order.

    Drivers are *ranked* by the regressor's output rather than showing it
    directly: the raw prediction is unconstrained (it can read 0.78 or
    20.1) and can tie, so it is not itself a position.
    """
    table = Table(
        title=f"Predicted top {top_n}",
        title_justify="left",
        title_style="bold",
        box=None,
        pad_edge=False,
    )
    table.add_column("Pos", justify="right", width=3)
    table.add_column("Driver", style="bold")
    table.add_column("Team", style="cyan")
    table.add_column("Grid", justify="right")
    if show_actual:
        table.add_column("Actual", justify="right")
        table.add_column("", width=6)

    ordered = predict.predicted_finishing_order(predictions, top_n=top_n)
    for _, row in ordered.iterrows():
        cells = [
            str(row["predicted_rank"]),
            f"{row['driver_abbreviation']}  {prettify_slug(row['driver_id'])}",
            prettify_slug(row["constructor_id"]),
            f"P{int(row['grid'])}",
        ]
        if show_actual:
            actual = row.get("finish_position")
            if pd.isna(actual):
                cells += [Text("--", style="dim"), Text("")]
            else:
                delta = int(actual) - int(row["predicted_rank"])
                marker = (
                    Text("exact", style="bold green")
                    if delta == 0
                    else Text(f"{delta:+d}", style="yellow" if abs(delta) <= 2 else "red")
                )
                cells += [f"P{int(actual)}", marker]
        table.add_row(*cells)
    return table


def render_round(
    season: int,
    round_number: int,
    predictions: pd.DataFrame | None = None,
    table: pd.DataFrame | None = None,
    models: predict.LoadedModels | None = None,
    top_n: int = 10,
    show_actual: bool = False,
    console: Console | None = None,
) -> None:
    """Print the full formatted report for one race.

    **This single function replaces all 27 per-round scripts in the
    original project.** Everything that varied between them is a
    parameter here.

    Parameters
    ----------
    predictions / table / models:
        Optional pre-computed inputs, so callers that already have them
        (backtests, the ``predict_latest_done`` wrapper) don't recompute.
    show_actual:
        Include the real finishing positions alongside the predictions.
        Only meaningful for a race that has already run.
    console:
        Injectable for testing; defaults to a normal stdout console.
    """
    console = console or Console()

    if predictions is None:
        predictions = predict.predict_round(season, round_number, models=models, table=table)

    if show_actual and "finish_position" not in predictions.columns:
        predictions = _attach_actual_results(predictions, season, round_number, table=table)

    event_name = predict.race_label(season, round_number, table=table)

    console.print()
    console.print(build_header(season, round_number, event_name))
    console.print()
    console.print(build_podium_table(predictions))
    console.print()
    console.print(build_win_probability_table(predictions))
    console.print()
    console.print(build_finishing_order_table(predictions, top_n=top_n, show_actual=show_actual))
    console.print()


def _attach_actual_results(
    predictions: pd.DataFrame,
    season: int,
    round_number: int,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge real finishing positions onto a prediction table, if the race has run.

    Degrades to an empty column rather than raising: `show_actual` is a
    display option, and a report is still worth printing when the results
    simply are not available (an upcoming race, or a partial table).
    """
    actual = predict.get_round_features(season, round_number, table=table)

    if "finish_position" not in actual.columns:
        logger.warning(
            "No finish_position available for %d round %d -- rendering without actual results",
            season,
            round_number,
        )
        return predictions.assign(finish_position=pd.NA)

    return predictions.merge(
        actual[["driver_id", "finish_position"]],
        on="driver_id",
        how="left",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render the formatted report for one race")
    parser.add_argument("season", type=int)
    parser.add_argument("round", type=int)
    parser.add_argument("--top", type=int, default=10, help="How many finishing positions to show")
    parser.add_argument("--actual", action="store_true", help="Show real results alongside predictions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    render_round(args.season, args.round, top_n=args.top, show_actual=args.actual)


if __name__ == "__main__":
    main()
