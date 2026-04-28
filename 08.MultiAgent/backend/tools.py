"""
tools.py — 툴 실행 함수 모음 (스키마 없음)

각 전문 에이전트가 자신의 TOOL_DEFINITIONS를 직접 정의하고,
실제 실행은 여기 함수를 호출한다.
"""
from __future__ import annotations
import os
import re

from openai import OpenAI

from rag import (
    get_mission, parse_mission,
    get_insight as _get_insight,
    get_motivational_nudge as _get_motivational_nudge,
    classify_emotion,
    multi_query_retrieve, hyde_query, build_context,
)
from schemas import VALID_CATEGORIES, VALID_DIFFICULTIES, DIFFICULTY_MIN_MINUTES


def tool_generate_mission(
    client: OpenAI,
    rag_state: dict,
    user_data: dict,
    mood: str,
    minutes: int,
) -> dict:
    chunks     = rag_state.get("chunks", [])
    embeddings = rag_state.get("embeddings", [])
    bm25       = rag_state.get("bm25")

    if not chunks:
        return {"error": "RAG 인덱스가 아직 로드되지 않았습니다."}

    emotion_type = classify_emotion(client, mood)
    time_str     = f"{minutes}분"

    raw, is_wildcard, sources = get_mission(
        client, mood, time_str, minutes,
        chunks, embeddings, emotion_type, user_data, bm25=bm25,
    )
    mission = parse_mission(raw, is_wildcard, sources)

    return {
        "mission_text": raw,
        "mission":      mission,
        "emotion_type": emotion_type,
        "is_wildcard":  is_wildcard,
        "sources":      sources,
    }


def tool_get_insight(
    client: OpenAI,
    rag_state: dict,
    mission_text: str,
) -> dict:
    chunks     = rag_state.get("chunks", [])
    embeddings = rag_state.get("embeddings", [])
    bm25       = rag_state.get("bm25")

    if not chunks:
        return {"error": "RAG 인덱스가 로드되지 않았습니다."}

    insight = _get_insight(client, mission_text, chunks, embeddings, bm25)
    return {"insight": insight}


def tool_get_motivational_nudge(
    client: OpenAI,
    rag_state: dict,
    mood: str,
) -> dict:
    chunks     = rag_state.get("chunks", [])
    embeddings = rag_state.get("embeddings", [])

    if not chunks:
        return {"nudge": "괜찮아요, 언제든 준비되면 다시 도전해볼 수 있어요."}

    nudge = _get_motivational_nudge(client, mood, chunks, embeddings)
    return {"nudge": nudge}


def tool_rag_search(
    client: OpenAI,
    rag_state: dict,
    query: str,
    emotion_type: str,
    k: int = 4,
) -> dict:
    chunks     = rag_state.get("chunks", [])
    embeddings = rag_state.get("embeddings", [])
    bm25       = rag_state.get("bm25")

    if not chunks:
        return {"error": "RAG 인덱스가 로드되지 않았습니다."}

    k         = min(max(k, 1), 8)
    hyde_text = hyde_query(client, query)
    results   = multi_query_retrieve(
        client, hyde_text, chunks, embeddings,
        k=k, emotion_type=emotion_type, bm25=bm25,
    )
    context = build_context(results)
    sources = list(dict.fromkeys(c["source"] for c in results))
    return {"context": context, "sources": sources, "chunk_count": len(results)}


def tool_web_search(query: str, max_results: int = 3) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key or api_key.startswith("tvly-..."):
        return {
            "error":    "TAVILY_API_KEY가 설정되지 않았습니다.",
            "fallback": "RAG 검색 결과만으로 응답합니다.",
        }
    try:
        from tavily import TavilyClient
        tc = TavilyClient(api_key=api_key)
        max_results = min(max(max_results, 1), 5)
        results = tc.search(query=query, max_results=max_results)
        items = [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", "")[:600],
            }
            for r in results.get("results", [])
        ]
        return {"results": items, "query": query, "count": len(items)}
    except ImportError:
        return {"error": "tavily-python 패키지 미설치. pip install tavily-python"}
    except Exception as exc:
        return {"error": str(exc)}


def tool_validate_mission(
    mission_text: str,
    available_minutes: int,
    category: str,
    difficulty: str,
) -> dict:
    errors: list[str] = []

    for tag in ["[미션]", "[카테고리]", "[난이도]", "[근거]", "[효과]"]:
        if tag not in mission_text:
            errors.append(f"필수 태그 누락: {tag}")

    if category not in VALID_CATEGORIES:
        errors.append(f"잘못된 카테고리 '{category}'. 허용: {', '.join(sorted(VALID_CATEGORIES))}")

    if difficulty not in VALID_DIFFICULTIES:
        errors.append(f"잘못된 난이도 '{difficulty}'. 허용: {', '.join(sorted(VALID_DIFFICULTIES))}")

    if difficulty in DIFFICULTY_MIN_MINUTES:
        min_needed = DIFFICULTY_MIN_MINUTES[difficulty]
        if min_needed > available_minutes:
            errors.append(
                f"시간 불일치: '{difficulty}'는 최소 {min_needed}분 필요, 가용 {available_minutes}분. "
                "난이도를 낮추거나 미션 시간을 줄이세요."
            )

    m = re.search(r"\[미션\]\s*\n?-?\s*(.+)", mission_text)
    if m and len(m.group(1).strip()) < 5:
        errors.append("미션 내용이 너무 짧습니다 (최소 5자).")

    if errors:
        return {"valid": False, "errors": errors, "suggestion": "오류를 수정하거나 미션을 재생성하세요."}

    return {
        "valid":             True,
        "message":           "미션이 모든 검증을 통과했습니다.",
        "category":          category,
        "difficulty":        difficulty,
        "available_minutes": available_minutes,
    }
