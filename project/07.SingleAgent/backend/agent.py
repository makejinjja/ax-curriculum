"""
agent.py — OpenAI tool-calling(ReAct) 기반 Single Agent

흐름:
  1. 시스템 프롬프트 로드
  2. 사용자 메시지 + 대화 이력으로 루프 진입
  3. LLM이 툴 호출 → execute_tool → 결과를 메시지에 추가 → 재호출
  4. LLM이 툴 없이 응답하면(finish_reason=stop) 루프 종료
  5. validate_mission 실패 횟수가 MAX_GENERATION_RETRIES에 달하면
     루프를 계속하되 에이전트에게 강제 종료를 지시
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from openai import OpenAI

from schemas import MAX_AGENT_ITERATIONS, MAX_GENERATION_RETRIES
from tools import TOOL_DEFINITIONS, execute_tool

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"


def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return "당신은 심리학 기반 미션 생성 전문 에이전트 Bloom입니다."


def _assistant_dict(msg) -> dict:
    """OpenAI ChatCompletionMessage → dict (API 재전송 형식)."""
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


def _parse_mission(text: str) -> dict | None:
    """최종 응답 텍스트에서 [태그] 블록을 파싱해 mission dict를 반환."""
    if "[미션]" not in text:
        return None

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

    if not blocks.get("미션"):
        return None

    return {
        "mission":    blocks.get("미션", ""),
        "category":   blocks.get("카테고리", ""),
        "difficulty": blocks.get("난이도", ""),
        "basis":      blocks.get("근거", ""),
        "effect":     blocks.get("효과", ""),
    }


class SingleAgent:
    def __init__(self, client: OpenAI, rag_state: dict) -> None:
        self.client        = client
        self.rag_state     = rag_state
        self.system_prompt = _load_system_prompt()

    def chat(
        self,
        user_message: str,
        available_minutes: int,
        conversation_history: list[dict],
    ) -> dict:
        """
        Returns:
            {
              "response":        str,
              "mission":         dict | None,
              "tool_calls_made": list[str],
            }
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            *conversation_history,
            {
                "role": "user",
                "content": f"{user_message}\n\n가용 시간: {available_minutes}분",
            },
        ]

        tool_calls_made: list[str] = []
        validation_failures = 0
        iteration = 0

        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=900,
            )

            msg         = response.choices[0].message
            finish      = response.choices[0].finish_reason
            messages.append(_assistant_dict(msg))

            # 툴 호출 없이 종료 → 최종 응답
            if finish == "stop" or not msg.tool_calls:
                final_text = msg.content or ""
                return {
                    "response":        final_text,
                    "mission":         _parse_mission(final_text),
                    "tool_calls_made": tool_calls_made,
                }

            # 툴 실행 루프
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                tool_calls_made.append(tool_name)

                # validate_mission 최대 실패 초과 시 강제 종료 유도
                if (
                    tool_name == "validate_mission"
                    and validation_failures >= MAX_GENERATION_RETRIES
                ):
                    result = {
                        "valid":   True,
                        "message": (
                            "최대 재생성 횟수에 도달했습니다. "
                            "현재 미션을 최종 결과로 출력하세요."
                        ),
                    }
                else:
                    result = execute_tool(
                        tool_name,
                        tool_args,
                        client=self.client,
                        rag_state=self.rag_state,
                        available_minutes=available_minutes,
                    )

                # 검증 실패 횟수 추적
                if tool_name == "validate_mission" and not result.get("valid"):
                    validation_failures += 1

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result, ensure_ascii=False),
                })

        return {
            "response": (
                "미션 생성 중 최대 반복 횟수에 도달했습니다. "
                "잠시 후 다시 시도해 주세요."
            ),
            "mission":         None,
            "tool_calls_made": tool_calls_made,
        }
