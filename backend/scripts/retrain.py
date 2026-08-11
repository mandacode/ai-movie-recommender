"""Trigger a batch retrain of the recommender.

Two-speed design:
  * fold-in personalises in real time on every like (no training);
  * this job folds accumulated interactions into the SVD *item* factors and is
    meant to run on a schedule (e.g. nightly cron), not per like.

It POSTs to the running API's retrain endpoint, so the live model is rebuilt
and hot-swapped in place.

Cron example (nightly at 03:00):
    0 3 * * *  cd /path/backend && ./.venv/bin/python scripts/retrain.py

Usage:
    python scripts/retrain.py [--url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()

    req = urllib.request.Request(f"{args.url}/api/admin/retrain", method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        result = json.load(r)
    print(f"retrained on {result['trained_on']:,} interactions, {result['movies']:,} movies")


if __name__ == "__main__":
    main()
