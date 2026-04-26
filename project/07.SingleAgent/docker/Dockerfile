FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# Copy prompts (read at runtime by agent.py via relative path)
COPY prompts/ ./prompts/

# Data directory is mounted as a volume; create stub so app starts even without mount
RUN mkdir -p /app/data/.index_cache

# Environment defaults (override via docker-compose or -e flags)
ENV RAG_DATA_DIR=/app/data
ENV RAG_CACHE_FILE=/app/data/.index_cache/rag_index.json
ENV OPENAI_MODEL=gpt-4o-mini

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
