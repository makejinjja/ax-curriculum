FROM python:3.11-slim

# System deps for sentence-transformers (torch CPU-only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY 05_2.Schemas.py   schemas.py
COPY 05_3.Auth.py      auth.py
COPY 05_4.Indexing.py  indexing.py
COPY 05_5.Retrieval.py retrieval.py
COPY 05_6.Main.py      main.py
# Keep originals too so importlib.import_module("05_2.Schemas") still works
COPY 05_2.Schemas.py   05_2.Schemas.py
COPY 05_3.Auth.py      05_3.Auth.py
COPY 05_4.Indexing.py  05_4.Indexing.py
COPY 05_5.Retrieval.py 05_5.Retrieval.py
COPY 05_6.Main.py      05_6.Main.py

# Runtime directories (overridable via volume mounts)
RUN mkdir -p /app/pdfs /app/data/.index_cache

ENV BLOOM_DATA_FILE=/app/data/.mission_data.json
ENV BLOOM_PDF_DIR=/app/pdfs
ENV BLOOM_CACHE_FILE=/app/data/.index_cache/mission_rag_v7_index.json

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "05_6.Main.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
