# syntax=docker/dockerfile:1

# ── 1. Build the React frontend into static files ──────────────────────────
FROM node:22-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build          # → /fe/dist

# ── 2. Python backend serving API + the built frontend ─────────────────────
FROM python:3.12-slim
WORKDIR /app/backend

# CPU-only torch first (keeps the image ~2GB instead of pulling CUDA wheels).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
# FastAPI serves ../frontend/dist relative to the backend dir.
COPY --from=frontend /fe/dist /app/frontend/dist

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
