"""
05_4.Indexing.py — RAG 인덱싱 파이프라인

PDF → 텍스트 추출 → 청크 분할 → 임베딩(text-embedding-3-small) → BM25 인덱스
SHA-256 해시 기반 캐시로 변경된 PDF만 재처리한다.
"""
from __future__ import annotations
import hashlib
import json
import re

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from schemas import (  # type: ignore[import]
    PDF_FILES, PDF_DIR, CACHE_FILE,
    MAX_CHUNK_CHARS, CHUNK_OVERLAP,
)


# ── 파일 해시 ────────────────────────────────────────────────

def _file_hash(path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ── PDF → 텍스트 ─────────────────────────────────────────────

def _extract_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ── 텍스트 → 청크 ────────────────────────────────────────────

def _chunk(text: str, source: str) -> list[dict]:
    """
    단락 단위로 나눈 뒤 MAX_CHUNK_CHARS 크기로 슬라이딩 윈도우 적용.
    CHUNK_OVERLAP 만큼 이전 청크와 중첩해 문맥 연속성을 유지한다.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[dict] = []
    idx = 0
    for para in paragraphs:
        para = para.strip()
        if len(para) < 80:
            continue
        start = 0
        while start < len(para):
            segment = para[start: start + MAX_CHUNK_CHARS]
            if len(segment) >= 80:
                chunks.append({"text": segment, "source": source, "chunk_index": idx})
                idx += 1
            start += MAX_CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


# ── 인덱스 빌드 ──────────────────────────────────────────────

def build_index(
    client: OpenAI,
    force_rebuild: bool = False,
    progress_callback=None,
) -> tuple[list[dict], list[list[float]], BM25Okapi | None]:
    """
    Returns:
        chunks     — 청크 메타데이터 리스트
        embeddings — 청크별 임베딩 벡터 리스트
        bm25       — BM25Okapi 인덱스 (청크가 없으면 None)

    progress_callback(current, total, message) — UI 진행 상황 콜백 (선택)
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 현재 PDF 해시 수집
    hashes: dict[str, str] = {}
    for fname, _ in PDF_FILES:
        p = PDF_DIR / fname
        if p.exists():
            hashes[fname] = _file_hash(p)

    # 캐시 유효성 검사
    if not force_rebuild and CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cache.get("hashes") == hashes:
            chunks: list[dict] = cache["chunks"]
            embeddings: list[list[float]] = cache["embeddings"]
            tokenized = [c["text"].split() for c in chunks]
            bm25 = BM25Okapi(tokenized) if tokenized else None
            if progress_callback:
                progress_callback(1, 1, f"캐시 로드 완료 — {len(chunks)}개 청크")
            return chunks, embeddings, bm25

    # 전체 재빌드
    all_chunks: list[dict] = []
    pdf_count = sum(1 for fname, _ in PDF_FILES if (PDF_DIR / fname).exists())
    processed = 0

    for fname, label in PDF_FILES:
        p = PDF_DIR / fname
        if not p.exists():
            continue
        processed += 1
        if progress_callback:
            progress_callback(processed, pdf_count, f"파싱 중: {label}")
        text = _extract_text(p)
        all_chunks.extend(_chunk(text, f"{fname} ({label})"))

    if not all_chunks:
        return [], [], None

    # 임베딩 생성 (배치 512)
    texts = [c["text"] for c in all_chunks]
    embeddings = []
    batch_size = 512
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(texts), batch_size)):
        if progress_callback:
            progress_callback(i + 1, total_batches, f"임베딩 배치 {i+1}/{total_batches}")
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[start: start + batch_size],
        )
        embeddings.extend([item.embedding for item in resp.data])

    # 캐시 저장
    CACHE_FILE.write_text(
        json.dumps({"hashes": hashes, "chunks": all_chunks, "embeddings": embeddings},
                   ensure_ascii=False),
        encoding="utf-8",
    )

    tokenized = [c["text"].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else None

    if progress_callback:
        progress_callback(1, 1, f"인덱싱 완료 — {len(all_chunks)}개 청크")

    return all_chunks, embeddings, bm25
