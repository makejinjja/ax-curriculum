"""
rag.py — Internal RAG index (BM25 + cosine hybrid, adapted from 05_Advanced_RAG)

Scans RAG_DATA_DIR for .pdf and .txt files, chunks and embeds them,
then serves hybrid BM25+cosine search. Index is SHA-256 cached.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

try:
    from pypdf import PdfReader
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

_BASE = Path(__file__).parent.parent
DATA_DIR   = Path(os.environ.get("RAG_DATA_DIR",  str(_BASE / "data")))
CACHE_FILE = Path(os.environ.get("RAG_CACHE_FILE", str(DATA_DIR / ".index_cache" / "rag_index.json")))

MAX_CHUNK_CHARS = 600
CHUNK_OVERLAP   = 100


# ── helpers ───────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf" and _PDF_OK:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, source: str) -> list[dict]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[dict] = []
    idx = 0
    for para in paragraphs:
        para = para.strip()
        if len(para) < 60:
            continue
        start = 0
        while start < len(para):
            segment = para[start: start + MAX_CHUNK_CHARS]
            if len(segment) >= 60:
                chunks.append({"text": segment, "source": source, "chunk_index": idx})
                idx += 1
            start += MAX_CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    av, bv = np.array(a), np.array(b)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


# ── index class ───────────────────────────────────────────────

class RAGIndex:
    def __init__(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        bm25: BM25Okapi | None,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = bm25

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, client: OpenAI, query: str, k: int = 5) -> list[dict]:
        if not self.chunks:
            return []

        resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
        q_emb = resp.data[0].embedding

        cos_scores = [_cosine(q_emb, e) for e in self.embeddings]

        if self.bm25:
            raw = self.bm25.get_scores(query.split())
            mx = max(raw) or 1.0
            bm25_scores = [s / mx for s in raw]
        else:
            bm25_scores = cos_scores

        hybrid = [0.5 * c + 0.5 * b for c, b in zip(cos_scores, bm25_scores)]
        top_idx = sorted(range(len(hybrid)), key=lambda i: hybrid[i], reverse=True)[:k]
        return [{**self.chunks[i], "score": round(hybrid[i], 4)} for i in top_idx]


# ── singleton index ───────────────────────────────────────────

_index: RAGIndex | None = None


def get_index(client: OpenAI, force: bool = False) -> RAGIndex:
    global _index
    if _index is not None and not force:
        return _index

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt")))
    hashes = {p.name: _file_hash(p) for p in files if p.exists()}

    # Load from cache if hashes match
    if not force and CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if cache.get("hashes") == hashes:
                chunks = cache["chunks"]
                embs   = cache["embeddings"]
                tok    = [c["text"].split() for c in chunks]
                _index = RAGIndex(chunks, embs, BM25Okapi(tok) if tok else None)
                return _index
        except Exception:
            pass

    # Full rebuild
    all_chunks: list[dict] = []
    for p in files:
        all_chunks.extend(_chunk(_extract_text(p), p.name))

    if not all_chunks:
        _index = RAGIndex([], [], None)
        return _index

    texts = [c["text"] for c in all_chunks]
    embs: list[list[float]] = []
    for start in range(0, len(texts), 512):
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[start: start + 512],
        )
        embs.extend([it.embedding for it in resp.data])

    CACHE_FILE.write_text(
        json.dumps({"hashes": hashes, "chunks": all_chunks, "embeddings": embs}, ensure_ascii=False),
        encoding="utf-8",
    )

    tok = [c["text"].split() for c in all_chunks]
    _index = RAGIndex(all_chunks, embs, BM25Okapi(tok) if tok else None)
    return _index
