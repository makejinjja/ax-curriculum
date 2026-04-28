"""
tools.py — 에이전트 툴 정의 및 실행기

도구:
  rag_search     — 내부 심리학 논문 하이브리드 검색
  web_search     — Tavily 외부 웹 검색
  validate_mission — 생성된 미션 규칙 검증
"""
from __future__ import annotations
import json
import os
import re

from openai import OpenAI

from rag import search_rag, build_context
from schemas import VALID_CATEGORIES, VALID_DIFFICULTIES, DIFFICULTY_MIN_MINUTES

# ── 툴 스키마 정의 ───────────────────────────────────────────
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "내부 심리학 논문 지식베이스에서 관련 연구를 검색한다. "
                "미션 생성을 위한 과학적 근거를 찾을 때 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리 (영어 키워드 권장, 예: 'behavioral activation low mood depression')",
                    },
                    "emotion_type": {
                        "type": "string",
                        "enum": ["부정적", "중립", "긍정적", "집중됨", "지루함"],
                        "description": "사용자의 감정 유형 (논문 부스트 가중치에 사용)",
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
                "Tavily를 사용해 웹에서 추가 정보를 검색한다. "
                "내부 RAG에 없는 최신 정보나 구체적 활동 사례가 필요할 때 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "웹 검색 쿼리",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "반환할 결과 수 (기본값: 3, 최대: 5)",
                    },
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
                "생성된 미션의 형식(필수 태그), 시간 제약, 카테고리·난이도 규칙을 검증한다. "
                "미션 생성 후 반드시 호출해야 하며, 검증 실패 시 오류 메시지를 참고해 수정한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mission_text": {
                        "type": "string",
                        "description": "검증할 미션 전체 텍스트 (태그 포함)",
                    },
                    "available_minutes": {
                        "type": "integer",
                        "description": "사용자의 가용 시간 (분)",
                    },
                    "category": {
                        "type": "string",
                        "description": "미션 카테고리 (건강/생산성/재미/성장/돌발 중 하나)",
                    },
                    "difficulty": {
                        "type": "string",
                        "description": "미션 난이도 (하/중/상/최상 중 하나)",
                    },
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
    available_minutes: int,
) -> dict:
    if tool_name == "rag_search":
        return _rag_search(client, rag_state, **args)
    elif tool_name == "web_search":
        return _web_search(**args)
    elif tool_name == "validate_mission":
        args.setdefault("available_minutes", available_minutes)
        return _validate_mission(**args)
    else:
        return {"error": f"알 수 없는 툴: {tool_name}"}


# ── rag_search 구현 ──────────────────────────────────────────

def _rag_search(
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
        return {"error": "RAG 인덱스가 아직 로드되지 않았습니다. 잠시 후 다시 시도하세요."}

    k = min(max(k, 1), 8)
    results = search_rag(
        client=client,
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        bm25=bm25,
        emotion_type=emotion_type,
        k=k,
    )
    context = build_context(results)
    sources = list(dict.fromkeys(c["source"] for c in results))
    return {
        "context": context,
        "sources": sources,
        "chunk_count": len(results),
    }


# ── web_search 구현 ──────────────────────────────────────────

def _web_search(query: str, max_results: int = 3) -> dict:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return {
            "error": "TAVILY_API_KEY가 설정되지 않았습니다. .env에 추가하세요.",
            "fallback": "RAG 검색 결과만으로 미션을 생성합니다.",
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
        return {"error": "tavily-python 패키지가 설치되지 않았습니다. pip install tavily-python"}
    except Exception as exc:
        return {"error": str(exc)}


# ── validate_mission 구현 ────────────────────────────────────

def _validate_mission(
    mission_text: str,
    available_minutes: int,
    category: str,
    difficulty: str,
) -> dict:
    errors: list[str] = []

    # 1. 필수 태그 확인
    for tag in ["[미션]", "[카테고리]", "[난이도]", "[근거]", "[효과]"]:
        if tag not in mission_text:
            errors.append(f"필수 태그 누락: {tag}")

    # 2. 카테고리 규칙
    if category not in VALID_CATEGORIES:
        errors.append(
            f"잘못된 카테고리: '{category}'. "
            f"허용값: {', '.join(sorted(VALID_CATEGORIES))}"
        )

    # 3. 난이도 규칙
    if difficulty not in VALID_DIFFICULTIES:
        errors.append(
            f"잘못된 난이도: '{difficulty}'. "
            f"허용값: {', '.join(sorted(VALID_DIFFICULTIES))}"
        )

    # 4. 시간 제약 (난이도별 최소 필요 시간이 가용 시간을 초과하는지 확인)
    if difficulty in DIFFICULTY_MIN_MINUTES:
        min_needed = DIFFICULTY_MIN_MINUTES[difficulty]
        if min_needed > available_minutes:
            errors.append(
                f"시간 불일치: '{difficulty}' 난이도는 최소 {min_needed}분이 필요하지만 "
                f"가용 시간은 {available_minutes}분입니다. "
                f"더 낮은 난이도를 선택하거나 미션 시간을 줄이세요."
            )

    # 5. 미션 내용 최소 길이
    m = re.search(r"\[미션\]\s*\n?-?\s*(.+)", mission_text)
    if m and len(m.group(1).strip()) < 5:
        errors.append("미션 내용이 너무 짧습니다 (최소 5자).")

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "suggestion": "위 오류를 수정하거나 미션을 재생성하세요.",
        }

    return {
        "valid": True,
        "message": "미션이 모든 검증을 통과했습니다.",
        "category": category,
        "difficulty": difficulty,
        "available_minutes": available_minutes,
    }
