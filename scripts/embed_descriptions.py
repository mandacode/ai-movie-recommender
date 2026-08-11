"""Embed TMDB overviews into vectors for content features + semantic search.

Reads overviews from the SQLite cache, encodes them with a small sentence
embedding model (all-MiniLM-L6-v2, 384-dim, CPU-friendly), and saves an
(movieId-aligned) embedding matrix to a .npz file.

Usage:
    python scripts/embed_descriptions.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "datasets" / "tmdb_cache.db"
OUT = ROOT / "datasets" / "ml-32m-sample" / "desc_emb.npz"
MODEL = "all-MiniLM-L6-v2"


def main() -> None:
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT movieId, overview FROM tmdb WHERE overview IS NOT NULL AND overview != ''"
    ).fetchall()
    con.close()
    if not rows:
        raise SystemExit("No overviews in cache — run scripts/fetch_tmdb.py first.")

    movie_ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [r[1] for r in rows]
    print(f"embedding {len(texts):,} overviews with {MODEL} …")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    emb = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, movie_ids=movie_ids, embeddings=emb)
    print(f"saved {emb.shape} → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
