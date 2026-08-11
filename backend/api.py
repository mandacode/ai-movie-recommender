"""Mandaflix backend API (FastAPI).

Serves the streaming frontend: profiles, personalised recommendations, search,
movie detail and similar titles. All ranking comes from the fitted recommender
(SVD) — no algorithm internals (scores, neighbours) are exposed, per the design.

Run (dev):
    ./.venv/bin/uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from catalog import Catalog

ROOT = Path(__file__).resolve().parent
FRONTEND_DIST = ROOT.parent / "frontend" / "dist"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["catalog"] = Catalog()  # loads data + fits SVD once at startup
    yield
    state.clear()


app = FastAPI(title="Mandaflix API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def cat() -> Catalog:
    return state["catalog"]


@app.get("/api/users")
def users():
    return cat().users()


@app.get("/api/genres")
def genres():
    return ["All", *cat().genres]


@app.get("/api/recommendations")
def recommendations(user_id: int, k: int = 20, genre: str | None = None):
    return cat().recommendations(user_id, k=k, genre=genre)


@app.get("/api/search")
def search(q: str):
    return cat().search(q)


@app.get("/api/popular")
def popular(k: int = 24):
    return cat().popular(k=k)


@app.get("/api/movies/{movie_id}")
def movie(movie_id: int, user_id: int | None = None):
    m = cat().movie(movie_id, user_id)
    if m is None:
        raise HTTPException(status_code=404, detail="movie not found")
    return m


@app.get("/api/movies/{movie_id}/similar")
def similar(movie_id: int, user_id: int, k: int = 7):
    return cat().similar(movie_id, user_id, k=k)


class LikeIn(BaseModel):
    user_id: int
    movie_id: int
    liked: bool = True


@app.post("/api/likes")
def set_like(body: LikeIn):
    cat().set_like(body.user_id, body.movie_id, body.liked)
    return {"ok": True}


@app.get("/api/users/{user_id}/likes")
def user_likes(user_id: int):
    return cat().likes(user_id)


@app.get("/api/admin/metrics")
def admin_metrics():
    p = ROOT / "experiments" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@app.get("/api/admin/compare")
def admin_compare(user_id: int, k: int = 8):
    return cat().compare(user_id, k=k)


@app.get("/api/admin/search-metrics")
def admin_search_metrics():
    return cat().search_metrics()


@app.get("/api/admin/golden-metrics")
def admin_golden_metrics():
    return cat().golden_metrics()


@app.post("/api/admin/retrain")
def admin_retrain():
    """Batch retrain (nightly job): refit item factors on ratings + likes."""
    return cat().retrain()


# Serve the built frontend (production) if it exists; dev uses the Vite server.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
