# Multi-stage: build web dashboard (Next.js) then serve via FastAPI.
# Cloud Run: set PORT=8080, DATABASE_URL and API keys via env or Secret Manager.

# Stage 1: build frontend
FROM node:20-alpine AS frontend
WORKDIR /app/web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npm run build

# Stage 2: API + static frontend
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY migrations ./migrations
COPY models/ ./models/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

COPY --from=frontend /app/web/out /app/static

ENV DASHBOARD_STATIC_DIR=/app/static
ENV PORT=8080
EXPOSE 8080

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
