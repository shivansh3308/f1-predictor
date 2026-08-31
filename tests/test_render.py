"""Tests for src/render.py.

The central claim of this module is that ONE parameterized function
replaces the original project's 27 near-identical per-round scripts
(spec Section 6), so the tests focus on `render_round` working for
arbitrary rounds and on the formatting logic that would otherwise only be
checked by eye.
"""

from __future__ import annotations

import pandas as pd
import pytest
from rich.console import Console

from src import render


def make_predictions(n: int = 12) -> pd.DataFrame:
    """A prediction table shaped exactly like `predict.predict_round` returns."""
    return pd.DataFrame(
        {
            "driver_id": [f"driver_{i}" for i in range(n)],
            "driver_abbreviation": [f"D{i:02d}" for i in range(n)],
            "constructor_id": ["red_bull" if i % 2 else "mclaren" for i in range(n)],
            "grid": [float(i + 1) for i in range(n)],
            "prob_win": [1.0 / n] * n,
            "prob_podium": [(n - i) / n for i in range(n)],
            "pred_position": [float(i + 1) for i in range(n)],
        }
    )


def make_table(n: int = 12, with_results: bool = True) -> pd.DataFrame:
    """A feature-table slice for one round, as `predict.get_round_features` returns."""
    table = pd.DataFrame(
        {
            "season": [2023] * n,
            "round": [7] * n,
            "event_name": ["Test Grand Prix"] * n,
            "driver_id": [f"driver_{i}" for i in range(n)],
        }
    )
    if with_results:
        # Mostly matches the predicted order with a couple of swaps, so the
        # delta column exercises both "exact" and signed-miss rendering --
        # which is also what a real race looks like.
        positions = list(range(1, n + 1))
        if n >= 5:
            positions[3], positions[4] = positions[4], positions[3]
        if n >= 9:
            positions[7], positions[8] = positions[8], positions[7]
        table["finish_position"] = [float(p) for p in positions]
    return table


def render_to_text(**kwargs) -> str:
    console = Console(record=True, width=120, force_terminal=False)
    render.render_round(
        2023,
        7,
        predictions=make_predictions(),
        table=make_table(),
        console=console,
        **kwargs,
    )
    return console.export_text()


# ---------------------------------------------------------------------------
# Name formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("max_verstappen", "Max Verstappen"),
        ("red_bull", "Red Bull"),
        ("hamilton", "Hamilton"),
        ("aston_martin", "Aston Martin"),
        ("force_india", "Force India"),
    ],
)
def test_prettify_slug_handles_normal_slugs(slug, expected):
    assert render.prettify_slug(slug) == expected


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("mclaren", "McLaren"),   # plain title-casing gives "Mclaren"
        ("rb", "RB"),
        ("alphatauri", "AlphaTauri"),
        ("alfa", "Alfa Romeo"),
        ("de_vries", "de Vries"),
    ],
)
def test_prettify_slug_overrides_names_titlecasing_gets_wrong(slug, expected):
    assert render.prettify_slug(slug) == expected


# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------


def test_report_contains_all_four_required_sections():
    """Spec task 14 names exactly these four."""
    text = render_to_text()
    assert "2023" in text and "Round 7" in text and "Test Grand Prix" in text
    assert "Predicted podium" in text
    assert "Win probability" in text
    assert "Predicted top" in text


def test_podium_section_shows_three_drivers_with_medals():
    text = render_to_text()
    for medal in render.PODIUM_MEDALS:
        assert medal in text


def test_top_n_is_respected():
    assert "Predicted top 5" in render_to_text(top_n=5)
    assert "Predicted top 10" in render_to_text(top_n=10)


def test_finishing_order_lists_exactly_top_n_rows():
    text = render_to_text(top_n=6)
    order_section = text.split("Predicted top 6")[1]
    listed = [ln for ln in order_section.splitlines() if ln.strip().startswith(tuple("123456789"))]
    assert len(listed) == 6


def test_actual_column_only_appears_when_requested():
    assert "Actual" not in render_to_text()
    assert "Actual" in render_to_text(show_actual=True)


def test_render_round_works_for_any_round_without_per_round_code():
    """The whole point of replacing 27 per-round scripts with one function."""
    for round_number in (1, 9, 22, 24):
        console = Console(record=True, width=120, force_terminal=False)
        render.render_round(
            2023,
            round_number,
            predictions=make_predictions(),
            table=make_table().assign(**{"round": round_number, "event_name": f"GP {round_number}"}),
            console=console,
        )
        assert f"Round {round_number}" in console.export_text()


def test_probabilities_render_as_percentages():
    text = render_to_text()
    assert "%" in text
    # 12 drivers sharing the distribution evenly -> 8.3% each
    assert "8.3%" in text


def test_actual_column_shows_deltas_against_predicted_rank():
    text = render_to_text(show_actual=True, top_n=12)
    assert "exact" in text, "a correctly predicted position should be marked"
    assert "+" in text or "-" in text, "misses should show a signed delta"


def test_show_actual_degrades_when_results_are_unavailable():
    """An upcoming race has no results; the report should still render."""
    console = Console(record=True, width=120, force_terminal=False)
    render.render_round(
        2023,
        7,
        predictions=make_predictions(),
        table=make_table(with_results=False),
        console=console,
        show_actual=True,
    )
    text = console.export_text()
    assert "Predicted podium" in text, "report should still render without results"
    assert "--" in text, "missing results should display as a placeholder"


def test_handles_grid_smaller_than_requested_top_n():
    """A short grid must not raise or pad with empty rows."""
    console = Console(record=True, width=120, force_terminal=False)
    render.render_round(
        2023,
        7,
        predictions=make_predictions(n=4),
        table=make_table(n=4).assign(event_name="Short GP"),
        console=console,
        top_n=10,
    )
    text = console.export_text()
    order_section = text.split("Predicted top 10")[1]
    listed = [ln for ln in order_section.splitlines() if ln.strip().startswith(tuple("123456789"))]
    assert len(listed) == 4
