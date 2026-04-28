"""
agents/__init__.py — BaseSpecialistAgent

전문 에이전트의 공통 ReAct 서브루프를 제공한다.
서브클래스는 _TOOL_DEFS, _dispatch(), _system_prompt()를 구현한다.
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod

from openai import OpenAI

MAX_SPECIALIST_ITERATIONS = 5


class BaseSpecialistAgent(ABC):
    def __init__(self, client: OpenAI, rag_state: dict) -> None:
        self.client    = client
        self.rag_state = rag_state

    # ── 서브클래스가 구현해야 하는 것들 ─────────────────────────

    @property
    @abstractmethod
    def _TOOL_DEFS(self) -> list[dict]:
        """OpenAI tool schema 목록"""

    @abstractmethod
    def _dispatch(self, tool_name: str, args: dict, **ctx) -> dict:
        """툴 이름과 인자를 받아 실행 결과를 반환"""

    @abstractmethod
    def _system_prompt(self) -> str:
        """이 전문 에이전트의 시스템 프롬프트"""

    # ── 공통 ReAct 서브루프 ──────────────────────────────────────

    def _run(self, task: str, **ctx) -> dict:
        """
        task: 오케스트레이터가 전달한 자연어 태스크 문자열
        ctx:  _dispatch에 그대로 전달되는 키워드 인자 (user_data, available_minutes 등)

        Returns:
            {"result": str, "tool_calls_made": list[str]}
        """
        messages: list[dict] = [
            {"role": "system",  "content": self._system_prompt()},
            {"role": "user",    "content": task},
        ]
        tool_calls_made: list[str] = []

        for _ in range(MAX_SPECIALIST_ITERATIONS):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self._TOOL_DEFS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1200,
            )
            msg    = response.choices[0].message
            finish = response.choices[0].finish_reason
            messages.append(self._msg_to_dict(msg))

            if finish == "stop" or not msg.tool_calls:
                return {
                    "result":          msg.content or "",
                    "tool_calls_made": tool_calls_made,
                }

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_calls_made.append(name)
                result = self._dispatch(name, args, **ctx)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result, ensure_ascii=False),
                })

        return {
            "result":          "최대 반복 횟수에 도달했습니다.",
            "tool_calls_made": tool_calls_made,
        }

    @staticmethod
    def _msg_to_dict(msg) -> dict:
        d: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": tc.type,
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d
