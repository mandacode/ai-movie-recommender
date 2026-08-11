"""Regenerate the README Evaluation table from experiments/metrics.json.

Usage:
    python scripts/update_readme_metrics.py

Reads the metrics JSON, renders a Markdown table, and swaps it into the
README between the EVALUATION markers — leaving the rest of the file untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reporting import (  # noqa: E402
    load_metrics,
    render_metrics_table,
    update_readme_section,
)

METRICS_PATH = ROOT / "experiments" / "metrics.json"
README_PATH = ROOT / "README.md"


def main() -> None:
    if not METRICS_PATH.exists():
        raise SystemExit(
            f"No metrics found at {METRICS_PATH}. Run `python main.py` first."
        )

    metrics = load_metrics(METRICS_PATH)
    table = render_metrics_table(metrics)
    updated = update_readme_section(README_PATH.read_text(encoding="utf-8"), table)
    README_PATH.write_text(updated, encoding="utf-8")

    print(f"Updated {README_PATH.relative_to(ROOT)} with {len(metrics)} model(s):")
    print(table)


if __name__ == "__main__":
    main()
