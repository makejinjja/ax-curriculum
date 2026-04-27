from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

from schemas import (
    PDF_DIR, CACHE_FILE, PDF_FILES, EMOTION_SOURCE_WEIGHT,
)

# ── Cross-encoder (지연 로딩) ──────────────────────────────────
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ── PDF 파싱 ──────────────────────────────────────────────────
def _extract_text(pdf_path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── 인덱스 빌드 ───────────────────────────────────────────────
def build_index(
    client,
    force_rebuild: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict], np.ndarray, BM25Okapi]:
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if CACHE_FILE.exists() and not force_rebuild:
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    all_chunks: list[dict] = []
    for label, filename in PDF_FILES.items():
        pdf_path = PDF_DIR / filename
        if not pdf_path.exists():
            log(f"[RAG] PDF 없음: {filename}")
            continue

        raw_text = _extract_text(pdf_path)
        current_hash = _sha256(raw_text)

        if label in cache and cache[label].get("hash") == current_hash:
            log(f"[RAG] 캐시 사용: {label}")
            for chunk in cache[label]["chunks"]:
                chunk["label"] = label
                all_chunks.append(chunk)
            continue

        log(f"[RAG] 임베딩 생성: {label}")
        texts = _chunk_text(raw_text)
        response = client.embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        vectors = [r.embedding for r in response.data]

        chunk_data = [
            {"text": t, "embedding": v, "source": filename, "label": label}
            for t, v in zip(texts, vectors)
        ]
        cache[label] = {"hash": current_hash, "chunks": chunk_data}
        all_chunks.extend(chunk_data)

    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not all_chunks:
        log("[RAG] 경고: 로드된 청크 없음")
        return [], np.zeros((0, 1536), dtype=np.float32), BM25Okapi([["placeholder"]])

    embeddings = np.array([c["embedding"] for c in all_chunks], dtype=np.float32)
    tokenized  = [c["text"].lower().split() for c in all_chunks]
    bm25       = BM25Okapi(tokenized)

    log(f"[RAG] 인덱스 완료: {len(all_chunks)} 청크")
    return all_chunks, embeddings, bm25


# ── 하이브리드 검색 ───────────────────────────────────────────
def search_rag(
    client,
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    bm25: BM25Okapi,
    emotion_type: str = "중립",
    k: int = 5,
    extra_boost: dict[str, float] | None = None,
) -> list[dict]:
    if not chunks:
        return []

    q_resp = client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    )
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)

    # 코사인 유사도
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    cos_scores = (embeddings @ q_vec) / (np.linalg.norm(q_vec) * norms.squeeze())

    # BM25 점수
    bm25_scores = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)

    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    hybrid = 0.5 * _norm(cos_scores) + 0.5 * _norm(bm25_scores)

    # 감정 부스트
    emotion_boost = EMOTION_SOURCE_WEIGHT.get(emotion_type, {})
    combined_boost = {**emotion_boost, **(extra_boost or {})}
    for i, chunk in enumerate(chunks):
        label = chunk.get("label", "")
        if label in combined_boost:
            hybrid[i] *= combined_boost[label]

    # 상위 20개 후보 → cross-encoder 재랭킹
    top20_idx = np.argsort(hybrid)[::-1][:20]
    pairs = [(query, chunks[i]["text"]) for i in top20_idx]
    ce_scores = _get_cross_encoder().predict(pairs)

    ranked = sorted(
        zip(top20_idx, ce_scores), key=lambda x: x[1], reverse=True
    )[:k]

    return [
        {**chunks[i], "score": float(s)}
        for i, s in ranked
    ]


def build_context(top_chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(top_chunks, 1):
        src = c.get("source", "unknown")
        parts.append(f"[출처 {i}: {src}]\n{c['text']}")
    return "\n\n".join(parts)
