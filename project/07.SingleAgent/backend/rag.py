"""
rag.py — RAG 인덱싱 & 하이브리드 검색 파이프라인

05_Advanced_RAG에서 이식:
  - PDF → 청크 → 임베딩(text-embedding-3-small) → BM25
  - 하이브리드 검색 (코사인 0.5 + BM25 0.5)
  - Cross-Encoder 리랭킹 (ms-marco-MiniLM-L-6-v2)
  - SHA-256 해시 기반 증분 캐시
"""
from __future__ import annotations
import hashlib
import json
import re

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from schemas import (
    PDF_FILES, PDF_DIR, CACHE_FILE,
    MAX_CHUNK_CHARS, CHUNK_OVERLAP,
    EMOTION_SOURCE_WEIGHT,
)

# ── Cross-Encoder 싱글톤 ─────────────────────────────────────
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ── 유틸 ────────────────────────────────────────────────────

def _file_hash(path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _extract_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk(text: str, source: str) -> list[dict]:
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


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


# ── 인덱스 빌드 ──────────────────────────────────────────────

def build_index(
    client: OpenAI,
    force_rebuild: bool = False,
    progress_callback=None,
) -> tuple[list[dict], list[list[float]], BM25Okapi | None]:
    """
    Returns:
        (chunks, embeddings, bm25)
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    for fname, _ in PDF_FILES:
        p = PDF_DIR / fname
        if p.exists():
            hashes[fname] = _file_hash(p)

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

    texts = [c["text"] for c in all_chunks]
    embeddings = []
    batch_size = 512
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(texts), batch_size)):
        if progress_callback:
            progress_callback(i + 1, total_batches, f"임베딩 배치 {i + 1}/{total_batches}")
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[start: start + batch_size],
        )
        embeddings.extend([item.embedding for item in resp.data])

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


# ── 검색 ─────────────────────────────────────────────────────

def search_rag(
    client: OpenAI,
    query: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    bm25: BM25Okapi | None = None,
    emotion_type: str | None = None,
    k: int = 4,
    extra_boost: dict[str, float] | None = None,
) -> list[dict]:
    """하이브리드 검색 + Cross-Encoder 리랭킹."""
    if not chunks:
        return []

    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding

    weights = dict(EMOTION_SOURCE_WEIGHT.get(emotion_type or "", {}))
    if extra_boost:
        for lbl, mult in extra_boost.items():
            weights[lbl] = weights.get(lbl, 1.0) * mult

    cosine_scores: list[float] = []
    for emb, chunk in zip(embeddings, chunks):
        score = _cosine(q_emb, emb)
        for keyword, multiplier in weights.items():
            if keyword in chunk.get("source", ""):
                score *= multiplier
                break
        cosine_scores.append(score)

    if bm25 and query:
        bm25_raw = bm25.get_scores(query.split())
        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        cos_max  = max(cosine_scores) if max(cosine_scores) > 0 else 1.0
        hybrid_scores = [
            0.5 * (c / cos_max) + 0.5 * (b / bm25_max)
            for c, b in zip(cosine_scores, bm25_raw)
        ]
    else:
        hybrid_scores = cosine_scores

    top20_idx = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)[:20]
    candidates = [chunks[i] for i in top20_idx]

    if query:
        ce = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = ce.predict(pairs)
        reranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in reranked[:k]]

    return candidates[:k]


def build_context(top_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks
    )
