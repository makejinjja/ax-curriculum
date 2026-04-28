"""
tools.py — 에이전트 툴 정의 및 실행기

도구 (05_Advanced_RAG 기능 전체 반영):
  generate_mission     — 전체 파이프라인 (HyDE·멀티쿼리·돌발·가중난이도·중복제거)
  get_insight          — 미션 완료 후 심리학 인사이트
  get_motivational_nudge — 미션 거절 시 동기면담 넛지
  rag_search           — 직접 RAG 검색 (에이전트 자유 사용)
  web_search           — Tavily 외부 검색
  validate_mission     — 미션 규칙 검증
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

# ── 툴 스키마 정의 ───────────────────────────────────────────
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "generate_mission",
            "description": (
                "사용자의 기분과 가용 시간을 바탕으로 미션을 생성한다. "
                "HyDE·멀티쿼리·하이브리드 검색·Cross-Encoder 리랭킹·돌발 미션(15%)·"
                "가중 난이도 랜덤·최근 미션 중복 제거가 모두 포함된 전체 파이프라인을 실행한다. "
                "미션이 필요할 때 반드시 이 도구를 먼저 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "사용자의 현재 기분 (자유 텍스트)",
                    },
                    "minutes": {
                        "type": "integer",
                        "description": "가용 시간 (분)",
                    },
                },
                "required": ["mood", "minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_insight",
            "description": (
                "완료한 미션이 어떤 심리학 이론·기법과 연결되는지 1~2줄로 설명한다. "
                "사용자가 미션을 성공적으로 완료했다고 보고할 때 호출한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mission_text": {
                        "type": "string",
                        "description": "완료한 미션 텍스트",
                    },
                },
                "required": ["mission_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_motivational_nudge",
            "description": (
                "미션을 거절하거나 어렵다고 할 때 동기면담 기반 공감 메시지를 생성한다. "
                "강요 없이 자율성을 존중하는 말투로 격려한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "사용자의 현재 기분",
                    },
                },
                "required": ["mood"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "내부 심리학 논문 지식베이스에서 HyDE·멀티쿼리 방식으로 검색한다. "
                "generate_mission으로 충분하지 않을 때 추가 근거를 찾는 데 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리 (영어 키워드 권장)",
                    },
                    "emotion_type": {
                        "type": "string",
                        "enum": ["부정적", "중립", "긍정적", "집중됨", "지루함"],
                        "description": "사용자의 감정 유형",
                    },
                    "k": {
                        "type": "integer",
                        "description": "반환할 청크 수 (기본값: 4, 최대: 8)",
                    },
                },
                "required": ["query", "emotion_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Tavily로 웹 검색한다. "
                "RAG에 없는 최신 정보나 구체적 활동 사례가 필요할 때 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "웹 검색 쿼리"},
                    "max_results": {"type": "integer", "description": "반환 결과 수 (기본 3, 최대 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_mission",
            "description": (
                "생성된 미션의 형식·시간 제약·카테고리·난이도를 검증한다. "
                "generate_mission 결과를 최종 제시하기 전에 호출한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mission_text":       {"type": "string",  "description": "검증할 미션 전체 텍스트 (태그 포함)"},
                    "available_minutes":  {"type": "integer", "description": "사용자의 가용 시간 (분)"},
                    "category":           {"type": "string",  "description": "미션 카테고리 (건강/생산성/재미/성장/돌발)"},
                    "difficulty":         {"type": "string",  "description": "미션 난이도 (하/중/상/최상)"},
                },
                "required": ["mission_text", "available_minutes", "category", "difficulty"],
            },
        },
    },
]


# ── 툴 실행 디스패처 ────────────────────────────────────────

def execute_tool(
    tool_name: str,
    args: dict,
    *,
    client: OpenAI,
    rag_state: dict,
    user_data: dict,
    available_minutes: int,
) -> dict:
    if tool_name == "generate_mission":
        return _tool_generate_mission(client, rag_state, user_data, **args)
    elif tool_name == "get_insight":
        return _tool_get_insight(client, rag_state, **args)
    elif tool_name == "get_motivational_nudge":
        return _tool_get_motivational_nudge(client, rag_state, **args)
    elif tool_name == "rag_search":
        return _tool_rag_search(client, rag_state, **args)
    elif tool_name == "web_search":
        return _tool_web_search(**args)
    elif tool_name == "validate_mission":
        args.setdefault("available_minutes", available_minutes)
        return _tool_validate_mission(**args)
    else:
        return {"error": f"알 수 없는 툴: {tool_name}"}


# ── generate_mission ─────────────────────────────────────────

def _tool_generate_mission(
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
        "mission_text":  raw,
        "mission":       mission,
        "emotion_type":  emotion_type,
        "is_wildcard":   is_wildcard,
        "sources":       sources,
    }


# ── get_insight ──────────────────────────────────────────────

def _tool_get_insight(
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


# ── get_motivational_nudge ───────────────────────────────────

def _tool_get_motivational_nudge(
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


# ── rag_search ───────────────────────────────────────────────

def _tool_rag_search(
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


# ── web_search ───────────────────────────────────────────────

def _tool_web_search(query: str, max_results: int = 3) -> dict:
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
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:600]}
            for r in results.get("results", [])
        ]
        return {"results": items, "query": query, "count": len(items)}
    except ImportError:
        return {"error": "tavily-python 패키지 미설치. pip install tavily-python"}
    except Exception as exc:
        return {"error": str(exc)}


# ── validate_mission ─────────────────────────────────────────

def _tool_validate_mission(
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
        "valid": True,
        "message": "미션이 모든 검증을 통과했습니다.",
        "category": category, "difficulty": difficulty,
        "available_minutes": available_minutes,
    }
