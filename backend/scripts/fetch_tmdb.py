"""Fetch TMDB movie metadata into a resumable SQLite cache.

Design (senior default for a static, offline, single-machine dataset):
  * SQLite cache keyed by movieId → only missing movies are fetched, so the
    job is fully resumable after interruptions / rate limits.
  * Concurrent fetches with backoff on HTTP 429.
  * One call per movie with `append_to_response=credits` → poster, backdrop,
    overview, runtime, director and top cast in a single request.

Usage:
    python scripts/fetch_tmdb.py                                  # ml-latest-small
    python scripts/fetch_tmdb.py --sample-dir datasets/ml-32m-sample
    python scripts/fetch_tmdb.py --limit 30                       # small test batch
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

COLUMNS = ["movieId", "tmdbId", "title", "overview", "release_date",
           "poster_path", "backdrop_path", "runtime", "director", "cast_top", "status"]


def fetch_one(tmdb_id: int) -> dict:
    url = (f"https://api.themoviedb.org/3/movie/{tmdb_id}"
           f"?append_to_response=credits")
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {TOKEN}", "accept": "application/json"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.load(r)
                crew = d.get("credits", {}).get("crew", [])
                cast = d.get("credits", {}).get("cast", [])
                director = next((c["name"] for c in crew if c.get("job") == "Director"), None)
                top_cast = ", ".join(c["name"] for c in cast[:5])
                return {
                    "status": "ok", "title": d.get("title"), "overview": d.get("overview"),
                    "release_date": d.get("release_date"), "poster_path": d.get("poster_path"),
                    "backdrop_path": d.get("backdrop_path"), "runtime": d.get("runtime"),
                    "director": director, "cast": top_cast,
                }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"status": "notfound"}
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            return {"status": "error"}
        except Exception:
            time.sleep(1 + attempt)
    return {"status": "error"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default="datasets/ml-latest-small")
    ap.add_argument("--db", default="datasets/tmdb_cache.db")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not TOKEN:
        raise SystemExit("TMDB_READ_ACCESS_TOKEN not set in .env")

    import pandas as pd

    links = pd.read_csv(Path(args.sample_dir) / "links.csv").dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    con = sqlite3.connect(args.db)
    con.execute(f"CREATE TABLE IF NOT EXISTS tmdb ({', '.join(f'{c} TEXT' for c in COLUMNS)}, "
                f"PRIMARY KEY (movieId))")
    existing = {r[1] for r in con.execute("PRAGMA table_info(tmdb)")}
    for col in COLUMNS:  # migrate older caches
        if col not in existing:
            con.execute(f"ALTER TABLE tmdb ADD COLUMN {col} TEXT")
    con.commit()

    # Rich data requires the credits fields → treat rows without `runtime` as todo.
    done = {r[0] for r in con.execute(
        "SELECT movieId FROM tmdb WHERE status='notfound' OR runtime IS NOT NULL")}
    todo = [(int(m), int(t)) for m, t in zip(links.movieId, links.tmdbId) if int(m) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"cache: {len(done)} gotowe | do pobrania: {len(todo)}  (workers={args.workers})")
    if not todo:
        con.close()
        return

    placeholders = ",".join("?" * len(COLUMNS))
    n_ok = 0
    t0 = time.time()
    batch: list[tuple] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, t): (m, t) for m, t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            m, t = futs[fut]
            r = fut.result()
            n_ok += r["status"] == "ok"
            batch.append((m, t, r.get("title"), r.get("overview"), r.get("release_date"),
                          r.get("poster_path"), r.get("backdrop_path"), r.get("runtime"),
                          r.get("director"), r.get("cast"), r["status"]))
            if len(batch) >= 200:
                con.executemany(f"INSERT OR REPLACE INTO tmdb VALUES ({placeholders})", batch)
                con.commit()
                batch = []
                print(f"  {i}/{len(todo)}  ok={n_ok}  {i/(time.time()-t0):.0f} req/s", end="\r")
    if batch:
        con.executemany(f"INSERT OR REPLACE INTO tmdb VALUES ({placeholders})", batch)
        con.commit()
    con.close()
    print(f"\ngotowe: pobrano {len(todo)}, ok {n_ok} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
