"""Persist evaluation results and render them into the README.

Flow:  evaluate() → save_metrics() → experiments/metrics.json
                                          ↓
                    load_metrics() → render_metrics_table()
                                          ↓
                    update_readme_section() → README.md

GitHub renders static Markdown and never executes code, so the README must
contain a *pre-generated* table. `scripts/update_readme_metrics.py` regenerates
it from the JSON whenever metrics change.
"""
from __future__ import annotations

import json
from pathlib import Path

from .evaluation import EvalResult

# Markers delimiting the auto-generated block in the README. Everything between
# them is replaced on update; everything outside is left untouched.
START_MARKER = "<!-- EVALUATION_START -->"
END_MARKER = "<!-- EVALUATION_END -->"


def save_metrics(results: dict[str, EvalResult], path: str | Path) -> None:
    """Write ``{model_key: metrics}`` to ``path`` as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: result.to_dict() for key, result in results.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_metrics(path: str | Path) -> dict[str, dict]:
    """Read a metrics JSON file back into a plain dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Tokens that should stay upper-cased in display names instead of title-cased.
_ACRONYMS = {"cf": "CF", "svd": "SVD", "xgboost": "XGBoost", "ndcg": "NDCG"}


def _display_name(model_key: str) -> str:
    """``"popularity_baseline"`` → ``"Popularity Baseline"``; ``"cf_x"`` → ``"CF X"``."""
    return " ".join(_ACRONYMS.get(w, w.title()) for w in model_key.split("_"))


def render_metrics_table(metrics: dict[str, dict]) -> str:
    """Render a metrics dict as a GitHub-Markdown table.

    Columns: Model | Precision@K | Recall@K | NDCG@K. The K in the headers is
    taken from each model's own ``k`` field (assumed consistent across models).
    """
    if not metrics:
        return "_No evaluation results yet._"

    k = next(iter(metrics.values())).get("k", 10)
    headers = ["Model", f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"]

    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for model_key, m in metrics.items():
        rows.append(
            "| {name} | {p:.3f} | {r:.3f} | {n:.3f} |".format(
                name=_display_name(model_key),
                p=m[f"precision@{k}"],
                r=m[f"recall@{k}"],
                n=m[f"ndcg@{k}"],
            )
        )
    return "\n".join(rows)


def update_readme_section(readme_text: str, table_md: str) -> str:
    """Replace the content between the EVALUATION markers with ``table_md``.

    Only the marked block changes; the rest of the README is preserved exactly.
    Raises ``ValueError`` if either marker is missing.
    """
    start = readme_text.find(START_MARKER)
    end = readme_text.find(END_MARKER)
    if start == -1 or end == -1:
        raise ValueError(
            f"README is missing markers {START_MARKER!r} / {END_MARKER!r}"
        )
    if end < start:
        raise ValueError("END marker appears before START marker in README")

    before = readme_text[: start + len(START_MARKER)]
    after = readme_text[end:]
    return f"{before}\n{table_md}\n{after}"
