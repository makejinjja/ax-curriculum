"""
search_agent.py — SearchAgent

담당: RAG 검색(rag_search) + 웹 검색(web_search)
오케스트레이터가 delegate_search_agent 툴을 통해 호출한다.
"""
from __future__ import annotations
from pathlib import Path

from openai import OpenAI

from agents import BaseSpecialistAgent
from tools import tool_rag_search, tool_web_search

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "search_agent.txt"


class SearchAgent(BaseSpecialistAgent):

    @property
    def _TOOL_DEFS(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "rag_search",
                    "description": (
                        "내부 심리학 논문 지식베이스에서 HyDE·멀티쿼리 방식으로 검색한다. "
                        "추가 심리학 근거가 필요할 때 사용한다."
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
                            "query":       {"type": "string",  "description": "웹 검색 쿼리"},
                            "max_results": {"type": "integer", "description": "반환 결과 수 (기본 3, 최대 5)"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def _dispatch(self, tool_name: str, args: dict, **ctx) -> dict:
        if tool_name == "rag_search":
            return tool_rag_search(self.client, self.rag_state, **args)
        elif tool_name == "web_search":
            return tool_web_search(**args)
        return {"error": f"알 수 없는 툴: {tool_name}"}

    def _system_prompt(self) -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text(encoding="utf-8").strip()
        return (
            "당신은 검색 전문가입니다. "
            "rag_search로 내부 지식베이스를 검색하고, "
            "필요시 web_search로 최신 정보를 보완하세요. "
            "검색 결과를 간결하게 요약해서 반환하세요."
        )

    def run(
        self,
        query: str,
        emotion_type: str = "중립",
        use_web: bool = False,
    ) -> dict:
        task = f"검색 쿼리: {query}\n감정 유형: {emotion_type}"
        if use_web:
            task += "\nRAG 검색 후 웹 검색도 수행하세요."
        else:
            task += "\nRAG 검색만 수행하세요."

        return self._run(task)
