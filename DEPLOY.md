# Deploying Mandaflix

The app is a single origin: FastAPI serves both `/api/*` and the built React
frontend. Docker Compose runs it alongside Postgres (pgvector). A Cloudflare
Tunnel fronts it at `https://mandaflix.projects.krystianjarmul.dev`.

```
Cloudflare Tunnel ──► app (uvicorn :8000) ──► db (pgvector)
   mandaflix.projects.krystianjarmul.dev        API + built frontend
```

## 1. Ship the code + data to the server

The seed needs the MovieLens CSVs and the TMDB cache. These live under
`backend/datasets/ml-latest-small/` (CSVs + `desc_emb.npz`) and
`backend/datasets/tmdb_cache.db`. The large `ml-32m*` dirs are **not** needed
(and are excluded by `.dockerignore`). Copy the repo including those data files
(e.g. `rsync -a --exclude .venv --exclude node_modules ./ server:~/mandaflix`).

## 2. Build + start

```bash
cd ~/mandaflix
docker compose up -d db          # start Postgres first
docker compose run --rm app python scripts/seed_db.py   # one-time seed
docker compose up -d app         # start the API + frontend
curl -s localhost:8000/api/genres   # smoke test
```

The app listens on `127.0.0.1:8000` only (not the LAN) — the tunnel reaches it.

## 3. Cloudflare Tunnel

Add an ingress rule to the **existing** tunnel config (the one already serving
cannabase), typically `~/.cloudflared/config.yml`:

```yaml
ingress:
  - hostname: mandaflix.projects.krystianjarmul.dev
    service: http://localhost:8000
  # ... existing cannabase rule(s) ...
  - service: http_status:404       # keep the catch-all last
```

Route the DNS record and reload:

```bash
cloudflared tunnel route dns <TUNNEL_NAME> mandaflix.projects.krystianjarmul.dev
sudo systemctl restart cloudflared      # or: docker restart cloudflared
```

Visit https://mandaflix.projects.krystianjarmul.dev.

## 4. Nightly retrain (optional)

Fold-in personalises in real time; a scheduled job folds accumulated likes into
the item factors. Cron on the host:

```cron
0 3 * * *  cd ~/mandaflix && docker compose exec -T app python scripts/retrain.py
```

## Notes

- **Secrets**: runtime needs none. `TMDB_*` keys are only for *re-fetching*
  posters (`scripts/fetch_tmdb.py`), not for serving.
- **Image size**: ~2 GB (CPU-only torch + the MiniLM embedding model).
- **Scaling to ml-32m**: seed from a larger MovieLens dump (`DATASET_DIR`),
  after fetching TMDB metadata + embeddings for it. Postgres/pgvector handle the
  scale; only the one-time enrichment + a precomputed-factors start need adding.
```
