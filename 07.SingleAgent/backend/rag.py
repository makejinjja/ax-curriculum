"""
rag.py — RAG 파이프라인 (bloom_v10 / 05_Advanced_RAG 전체 이식)

포함:
  - build_index: SHA-256 증분 캐시, BM25
  - _expand_query: 감정+카테고리 조합 쿼리 확장
  - _hyde_query: HyDE (가상 논문 문장으로 쿼리 확장)
  - _generate_multi_queries: 쿼리 3변형 생성
  - retrieve: 하이브리드(코사인+BM25) + Cross-Encoder 리랭킹
  - multi_query_retrieve: 멀티쿼리 중복 제거 검색
  - classify_emotion: LLM 감정 분류
  - get_mission: 전체 미션 생성 파이프라인 (돌발 15%, 가중 난이도)
  - parse_mission / normalize_*
  - summarize_mission: 10자 동사형 요약
  - get_insight: 미션 완료 후 심리학 인사이트
  - get_motivational_nudge: 미션 거절 시 동기면담 넛지
  - analyze_coverage: 논문 활용 분석 + 약점 논문 부스트
"""
from __future__ import annotations
import hashlib
import json
import random
import re

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from schemas import (
    PDF_FILES, PDF_DIR, CACHE_FILE,
    MAX_CHUNK_CHARS, CHUNK_OVERLAP,
    EMOTION_SOURCE_WEIGHT, EMOTION_PURPOSE, EMOTION_CAT_QUERY,
    MOOD_QUERY_MAP, CAT_PRIMARY_SOURCE,
    CLASSIFY_PROMPT, WILDCARD_PROMPT, make_mission_prompt,
    normalize_difficulty, normalize_category,
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
    embeddings: list[list[float]] = []
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


# ── 쿼리 확장 ────────────────────────────────────────────────

def _time_query(minutes: int) -> str:
    if minutes <= 10:
        return "brief intervention micro-habit short activity immediate"
    elif minutes <= 40:
        return "moderate duration activity engagement exercise"
    return "extended activity deep work immersive exercise"


def expand_query(
    mood: str,
    minutes: int,
    emotion_type: str | None = None,
    target_cat: str | None = None,
) -> str:
    if emotion_type and target_cat:
        combo_kw = EMOTION_CAT_QUERY.get((emotion_type, target_cat))
        if combo_kw:
            return f"{combo_kw} {_time_query(minutes)} psychological intervention evidence-based"
    mood_kw = next(
        (v for k, v in MOOD_QUERY_MAP.items() if k in mood),
        "emotion regulation mood improvement well-being intervention",
    )
    return f"{mood_kw} {_time_query(minutes)} psychological intervention evidence-based"


# ── HyDE ────────────────────────────────────────────────────

def hyde_query(client: OpenAI, query: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "심리학 논문 스타일로 아래 주제에 맞는 개입법 설명을 영어로 100자 이내로 써라. "
                "논문 본문처럼 써야 하며 다른 말은 하지 마라."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


# ── 멀티쿼리 ────────────────────────────────────────────────

def _generate_multi_queries(client: OpenAI, query: str) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "아래 검색 쿼리를 다른 표현으로 3가지 변형해라. "
                "각 변형은 줄바꿈으로 구분하고 번호 없이 쿼리 텍스트만 출력해라."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
        max_tokens=150,
    )
    lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
    return lines[:3]


# ── 핵심 검색 ────────────────────────────────────────────────

def retrieve(
    query_emb: list[float],
    chunks: list[dict],
    embeddings: list[list[float]],
    k: int = 4,
    emotion_type: str | None = None,
    query_text: str = "",
    bm25: BM25Okapi | None = None,
    extra_boost: dict | None = None,
) -> list[dict]:
    weights = dict(EMOTION_SOURCE_WEIGHT.get(emotion_type, {}) if emotion_type else {})
    if extra_boost:
        for lbl, mult in extra_boost.items():
            weights[lbl] = weights.get(lbl, 1.0) * mult

    cosine_scores: list[float] = []
    for emb, chunk in zip(embeddings, chunks):
        score = _cosine(query_emb, emb)
        for keyword, multiplier in weights.items():
            if keyword in chunk.get("source", ""):
                score *= multiplier
                break
        cosine_scores.append(score)

    if bm25 and query_text:
        bm25_raw = bm25.get_scores(query_text.split())
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

    if query_text:
        ce = _get_cross_encoder()
        pairs = [(query_text, c["text"]) for c in candidates]
        ce_scores = ce.predict(pairs)
        reranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in reranked[:k]]

    return candidates[:k]


def multi_query_retrieve(
    client: OpenAI,
    query: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    k: int,
    emotion_type: str | None,
    bm25: BM25Okapi | None = None,
    extra_boost: dict | None = None,
) -> list[dict]:
    queries = [query] + _generate_multi_queries(client, query)
    seen: set[tuple] = set()
    merged: list[dict] = []

    for q in queries:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small", input=[q]
        ).data[0].embedding
        results = retrieve(q_emb, chunks, embeddings,
                           k=k, emotion_type=emotion_type,
                           query_text=q, bm25=bm25, extra_boost=extra_boost)
        for chunk in results:
            uid = (chunk["source"], chunk["chunk_index"])
            if uid not in seen:
                seen.add(uid)
                merged.append(chunk)

    return merged[:k]


def build_context(top_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks
    )


# ── 감정 분류 ────────────────────────────────────────────────

def classify_emotion(client: OpenAI, mood: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user",   "content": mood},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    raw = resp.choices[0].message.content.strip()
    for key in EMOTION_PURPOSE:
        if key in raw:
            return key
    return "중립"


# ── 미션 생성 (전체 파이프라인) ──────────────────────────────

def get_mission(
    client: OpenAI,
    mood: str,
    time_str: str,
    minutes: int,
    chunks: list[dict],
    embeddings: list[list[float]],
    emotion_type: str,
    user_data: dict,
    bm25: BM25Okapi | None = None,
) -> tuple[str, bool, list[str]]:
    """
    Returns (raw_text, is_wildcard, sources)

    파이프라인:
      1. 15% 확률 돌발 미션
      2. 가중 난이도 랜덤 (하50/중30/상15/최상5)
      3. 감정+카테고리 쿼리 확장
      4. HyDE 쿼리 확장
      5. 멀티쿼리 + 하이브리드 검색 + Cross-Encoder 리랭킹
      6. 카테고리 주 논문 + 약점 논문 부스트
      7. 최근 미션 중복 제거
      8. make_mission_prompt로 LLM 생성
    """
    is_wildcard = random.random() < 0.15

    if is_wildcard:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": WILDCARD_PROMPT},
                {"role": "user",   "content": f"가용 시간: {time_str}"},
            ],
            temperature=1.0,
            max_tokens=200,
        )
        return resp.choices[0].message.content, True, []

    forced_difficulty = random.choices(
        ["하", "중", "상", "최상"], weights=[50, 30, 15, 5]
    )[0]

    emotion_to_default_cat = {
        "부정적": "건강", "중립": "생산성", "긍정적": "재미",
        "집중됨": "생산성", "지루함": "재미",
    }
    target_cat = emotion_to_default_cat.get(emotion_type)

    extra_boost: dict[str, float] = {}
    primary_src = CAT_PRIMARY_SOURCE.get(target_cat or "")
    if primary_src:
        extra_boost[primary_src] = 2.0
    for weak_src in user_data.get("weak_paper_boost", []):
        extra_boost[weak_src] = extra_boost.get(weak_src, 1.0) * 2.0

    query     = expand_query(mood, minutes, emotion_type=emotion_type, target_cat=target_cat)
    hyde_text = hyde_query(client, query)

    top = multi_query_retrieve(
        client, hyde_text, chunks, embeddings,
        k=4, emotion_type=emotion_type, bm25=bm25,
        extra_boost=extra_boost or None,
    )
    context = build_context(top)
    sources = list(dict.fromkeys(c["source"] for c in top))

    recent     = user_data.get("mission_history", [])[-5:]
    recent_str = "\n".join(f"- {m}" for m in recent) if recent else "없음"
    purpose    = EMOTION_PURPOSE.get(emotion_type, "기분전환·회복")
    system_msg = make_mission_prompt(emotion_type, purpose, recent_str, forced_difficulty)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": (
                f"현재 기분: {mood}\n가용 시간: {time_str}\n\n[심리학 논문 근거]\n{context}"
            )},
        ],
        temperature=0.9,
        max_tokens=350,
    )
    return resp.choices[0].message.content, False, sources


def parse_mission(text: str, is_wildcard: bool, sources: list[str] | None = None) -> dict:
    blocks: dict[str, str] = {}
    current_tag: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        header = re.match(r"^\[(.+?)\]", line.strip())
        if header:
            if current_tag:
                blocks[current_tag] = "\n".join(current_lines).strip()
            current_tag = header.group(1)
            current_lines = []
        elif current_tag:
            content = re.sub(r"^-\s*", "", line.strip())
            if content:
                current_lines.append(content)

    if current_tag:
        blocks[current_tag] = "\n".join(current_lines).strip()

    return {
        "mission":     blocks.get("미션", ""),
        "category":    "돌발" if is_wildcard else normalize_category(blocks.get("카테고리", "")),
        "difficulty":  "하"   if is_wildcard else normalize_difficulty(blocks.get("난이도", "")),
        "basis":       blocks.get("근거", ""),
        "effect":      blocks.get("효과", ""),
        "is_wildcard": is_wildcard,
        "sources":     sources or [],
    }


# ── 미션 요약 ────────────────────────────────────────────────

def summarize_mission(client: OpenAI, mission_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "미션 텍스트를 10자 이내 짧은 동사형 한국어로 요약해라. 다른 말 없이 요약문만 출력."},
            {"role": "user",   "content": mission_text},
        ],
        temperature=0.0,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip()


# ── 인사이트 ─────────────────────────────────────────────────

def get_insight(
    client: OpenAI,
    mission_text: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    bm25: BM25Okapi | None = None,
) -> str:
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[mission_text]
    ).data[0].embedding
    top = retrieve(q_emb, chunks, embeddings, k=2, query_text=mission_text, bm25=bm25)
    context = build_context(top)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "심리학자처럼 방금 완료한 미션이 어떤 심리학 이론/기법과 연결되는지 1~2줄로 설명해라. "
                "형식: '방금 하신 행동은 [이론명]의 [기법명] 기법입니다. [효과]'"
            )},
            {"role": "user", "content": f"완료한 미션: {mission_text}\n\n[논문 근거]\n{context}"},
        ],
        temperature=0.5,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


# ── 동기면담 넛지 ────────────────────────────────────────────

def get_motivational_nudge(
    client: OpenAI,
    mood: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> str:
    mi_chunks = [c for c in chunks if "동기면담" in c.get("source", "")]
    mi_embs   = [e for c, e in zip(chunks, embeddings) if "동기면담" in c.get("source", "")]

    if not mi_chunks:
        return "괜찮아요, 언제든 준비되면 다시 도전해볼 수 있어요."

    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[mood]
    ).data[0].embedding
    top     = retrieve(q_emb, mi_chunks, mi_embs, k=2, query_text=mood, bm25=None)
    context = build_context(top)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "동기면담(Motivational Interviewing) 전문가처럼 "
                "미션을 거절한 사람에게 공감하며 부드럽게 동기를 북돋는 말 1~2줄을 한국어로 해라. "
                "강요하거나 설득하지 말고, 자율성을 존중하는 말투로."
            )},
            {"role": "user", "content": f"현재 기분: {mood}\n\n[동기면담 근거]\n{context}"},
        ],
        temperature=0.7,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


# ── 논문 커버리지 분석 ───────────────────────────────────────

def analyze_coverage(
    client: OpenAI,
    user_data: dict,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> dict:
    """
    Returns:
        {"scores": {label: avg}, "sorted": [(label, score)], "weak": [label1, label2]}
    Updates user_data["weak_paper_boost"] in-place.
    """
    fruits = [f for f in user_data.get("fruits", []) if f.get("success")][-30:]
    if not fruits:
        return {}

    combined = " ".join(f.get("full_mission") or f.get("mission", "") for f in fruits)
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[combined[:2000]]
    ).data[0].embedding

    label_scores: dict[str, list[float]] = {}
    for c, e in zip(chunks, embeddings):
        src = c.get("source", "")
        m   = re.search(r"\(([^)]+)\)$", src)
        label = m.group(1) if m else src
        label_scores.setdefault(label, []).append(_cosine(q_emb, e))

    avg_scores  = {lbl: sum(sc) / len(sc) for lbl, sc in label_scores.items()}
    sorted_pairs = sorted(avg_scores.items(), key=lambda x: x[1])
    weak = [lbl for lbl, _ in sorted_pairs[:2]]
    user_data["weak_paper_boost"] = weak

    return {"scores": avg_scores, "sorted": sorted_pairs, "weak": weak}
