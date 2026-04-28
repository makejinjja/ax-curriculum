# ── Bloom Single Agent — Backend Dockerfile ──────────────────
FROM python:3.11-slim

WORKDIR /app

# 빌드 의존성 (sentence-transformers / torch 컴파일용)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (레이어 캐싱)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 소스 코드 복사
COPY backend/ /app/backend/
COPY prompts/ /app/prompts/

# 데이터 디렉토리 생성
RUN mkdir -p /app/pdfs /app/data/.index_cache

# 환경변수 기본값 (docker-compose로 오버라이드 가능)
ENV BLOOM_PDF_DIR=/app/pdfs \
    BLOOM_CACHE_FILE=/app/data/.index_cache/sa_rag_index.json \
    BLOOM_DATA_FILE=/app/data/.mission_data.json

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
