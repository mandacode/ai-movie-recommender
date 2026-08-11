"""Fetch TMDB movie overviews into a resumable SQLite cache.

Design (senior default for a static, offline, single-machine dataset):
  * SQLite cache keyed by movieId → only missing movies are fetched, so the
    job is fully resumable after interruptions / rate limits.
  * Concurrent fetches with backoff on HTTP 429.

Usage:
    python scripts/fetch_tmdb.py                    # fetch all missing
    python scripts/fetch_tmdb.py --limit 30         # small test batch
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
TOKEN = os.environ.get("TMDB_READ_ACCESS_TOKEN")


def fetch_one(tmdb_id: int) -> tuple[str, str | None, str | None, str | None]:
    """Return (status, title, overview, release_date). status in ok/notfound/error."""
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {TOKEN}", "accept": "application/json"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.load(r)
                return ("ok", d.get("title"), d.get("overview"), d.get("release_date"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ("notfound", None, None, None)
            if e.code == 429:  # rate limited → exponential backoff
                time.sleep(2 ** attempt)
                continue
            return ("error", None, None, None)
        except Exception:
            time.sleep(1 + attempt)
    return ("error", None, None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default="datasets/ml-32m-sample")
    ap.add_argument("--db", default="datasets/tmdb_cache.db")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap fetches (for testing)")
    args = ap.parse_args()

    if not TOKEN:
        raise SystemExit("TMDB_READ_ACCESS_TOKEN not set in .env")

    import pandas as pd

    links = pd.read_csv(Path(args.sample_dir) / "links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    con = sqlite3.connect(args.db)
    con.execute(
        """CREATE TABLE IF NOT EXISTS tmdb (
            movieId INTEGER PRIMARY KEY, tmdbId INTEGER, title TEXT,
            overview TEXT, release_date TEXT, status TEXT)"""
    )
    con.commit()
    done = {r[0] for r in con.execute("SELECT movieId FROM tmdb WHERE status IN ('ok','notfound')")}

    todo = [(int(m), int(t)) for m, t in zip(links.movieId, links.tmdbId) if int(m) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"cache: {len(done)} gotowe | do pobrania: {len(todo)}  (workers={args.workers})")
    if not todo:
        con.close()
        return

    n_ok = 0
    t0 = time.time()
    batch: list[tuple] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, t): (m, t) for m, t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            m, t = futs[fut]
            status, title, overview, date = fut.result()
            batch.append((m, t, title, overview, date, status))
            n_ok += status == "ok"
            if len(batch) >= 200:
                con.executemany("INSERT OR REPLACE INTO tmdb VALUES (?,?,?,?,?,?)", batch)
                con.commit()
                batch = []
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  z opisem={n_ok}  {rate:.0f} req/s", end="\r")
    if batch:
        con.executemany("INSERT OR REPLACE INTO tmdb VALUES (?,?,?,?,?,?)", batch)
        con.commit()
    con.close()
    print(f"\ngotowe: pobrano {len(todo)}, z opisem {n_ok} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
