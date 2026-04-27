# ── Bloom MultiAgent — Backend Dockerfile ──────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/ /app/backend/
COPY prompts/ /app/prompts/

RUN mkdir -p /app/pdfs /app/data/.index_cache /app/data/.curricula

ENV BLOOM_PDF_DIR=/app/pdfs \
    BLOOM_CACHE_FILE=/app/data/.index_cache/ma_rag_index.json \
    BLOOM_DATA_FILE=/app/data/.mission_data.json \
    BLOOM_CURRICULA_DIR=/app/data/.curricula

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
