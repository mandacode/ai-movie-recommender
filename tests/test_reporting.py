"""Tests for metrics persistence and README table generation."""
from __future__ import annotations

import pytest

from src.evaluation import EvalResult
from src.reporting import (
    END_MARKER,
    START_MARKER,
    load_metrics,
    render_metrics_table,
    save_metrics,
    update_readme_section,
)


def _result(name: str, p: float, r: float, n: float, k: int = 10) -> EvalResult:
    return EvalResult(model=name, k=k, precision=p, recall=r, ndcg=n, n_users=42)


# --- persistence -----------------------------------------------------------

def test_save_metrics_writes_expected_json_shape(tmp_path):
    path = tmp_path / "experiments" / "metrics.json"
    save_metrics({"popularity_baseline": _result("popularity_baseline", 0.083, 0.041, 0.119)}, path)

    assert path.exists()  # parent dir created automatically
    data = load_metrics(path)
    assert data["popularity_baseline"] == {
        "k": 10,
        "precision@10": 0.083,
        "recall@10": 0.041,
        "ndcg@10": 0.119,
    }


def test_save_then_load_roundtrip_multiple_models(tmp_path):
    path = tmp_path / "metrics.json"
    results = {
        "popularity_baseline": _result("popularity_baseline", 0.083, 0.041, 0.119),
        "collaborative_filtering": _result("collaborative_filtering", 0.142, 0.091, 0.181),
    }
    save_metrics(results, path)

    loaded = load_metrics(path)
    assert set(loaded) == {"popularity_baseline", "collaborative_filtering"}
    assert loaded["collaborative_filtering"]["precision@10"] == 0.142


# --- markdown table --------------------------------------------------------

def test_render_metrics_table_single_model():
    metrics = {"popularity_baseline": {"k": 10, "precision@10": 0.083, "recall@10": 0.041, "ndcg@10": 0.119}}
    table = render_metrics_table(metrics)

    assert "| Model | Precision@10 | Recall@10 | NDCG@10 |" in table
    assert "| Popularity Baseline | 0.083 | 0.041 | 0.119 |" in table


def test_render_metrics_table_multiple_models_one_row_each():
    metrics = {
        "popularity_baseline": {"k": 10, "precision@10": 0.083, "recall@10": 0.041, "ndcg@10": 0.119},
        "collaborative_filtering": {"k": 10, "precision@10": 0.142, "recall@10": 0.091, "ndcg@10": 0.181},
    }
    table = render_metrics_table(metrics)

    assert "| Popularity Baseline | 0.083 | 0.041 | 0.119 |" in table
    assert "| Collaborative Filtering | 0.142 | 0.091 | 0.181 |" in table
    # header + separator + 2 data rows
    assert len(table.splitlines()) == 4


def test_render_metrics_table_empty():
    assert render_metrics_table({}) == "_No evaluation results yet._"


# --- README section update -------------------------------------------------

README = f"""# Title

Intro paragraph that must be preserved.

## Evaluation

{START_MARKER}
_old content_
{END_MARKER}

## Status

Footer that must be preserved.
"""


def test_update_readme_replaces_only_marked_section():
    updated = update_readme_section(README, "NEW TABLE")

    assert "NEW TABLE" in updated
    assert "_old content_" not in updated
    # markers themselves survive
    assert START_MARKER in updated and END_MARKER in updated


def test_update_readme_preserves_surrounding_content():
    updated = update_readme_section(README, "NEW TABLE")

    assert "# Title" in updated
    assert "Intro paragraph that must be preserved." in updated
    assert "## Status" in updated
    assert "Footer that must be preserved." in updated


def test_update_readme_is_idempotent():
    once = update_readme_section(README, "NEW TABLE")
    twice = update_readme_section(once, "NEW TABLE")
    assert once == twice


def test_update_readme_missing_markers_raises():
    with pytest.raises(ValueError):
        update_readme_section("# No markers here", "TABLE")


def test_end_to_end_json_to_readme(tmp_path):
    """Full chain: EvalResult → JSON → load → table → README section."""
    path = tmp_path / "metrics.json"
    save_metrics(
        {
            "popularity_baseline": _result("popularity_baseline", 0.083, 0.041, 0.119),
            "collaborative_filtering": _result("collaborative_filtering", 0.142, 0.091, 0.181),
        },
        path,
    )
    table = render_metrics_table(load_metrics(path))
    updated = update_readme_section(README, table)

    assert "Popularity Baseline" in updated
    assert "Collaborative Filtering" in updated
    assert "_old content_" not in updated
