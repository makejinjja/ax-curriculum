"""
psychology_agent.py — PsychologyAgent

담당: 미션 완료 인사이트(get_insight) + 동기면담 격려(get_motivational_nudge)
오케스트레이터가 delegate_psychology_agent 툴을 통해 호출한다.
"""
from __future__ import annotations
from pathlib import Path

from openai import OpenAI

from agents import BaseSpecialistAgent
from tools import tool_get_insight, tool_get_motivational_nudge

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "psychology_agent.txt"


class PsychologyAgent(BaseSpecialistAgent):

    @property
    def _TOOL_DEFS(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_insight",
                    "description": (
                        "완료한 미션이 어떤 심리학 이론·기법과 연결되는지 1~2줄로 설명한다. "
                        "사용자가 미션을 완료했다고 보고할 때 호출한다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mission_text": {"type": "string", "description": "완료한 미션 텍스트"},
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
                        "미션을 거절하거나 어렵다고 할 때 동기면담 기반 공감 메시지를 생성한다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mood": {"type": "string", "description": "사용자의 현재 기분"},
                        },
                        "required": ["mood"],
                    },
                },
            },
        ]

    def _dispatch(self, tool_name: str, args: dict, **ctx) -> dict:
        if tool_name == "get_insight":
            return tool_get_insight(self.client, self.rag_state, **args)
        elif tool_name == "get_motivational_nudge":
            return tool_get_motivational_nudge(self.client, self.rag_state, **args)
        return {"error": f"알 수 없는 툴: {tool_name}"}

    def _system_prompt(self) -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text(encoding="utf-8").strip()
        return (
            "당신은 심리학 전문가입니다. "
            "미션 완료 시 get_insight로 인사이트를 제공하고, "
            "미션 거절 시 get_motivational_nudge로 격려 메시지를 제공하세요."
        )

    def run(
        self,
        action: str,          # "insight" | "nudge"
        mission_text: str = "",
        mood: str = "",
    ) -> dict:
        """
        action: "insight" → get_insight 호출
                "nudge"   → get_motivational_nudge 호출
        """
        if action == "insight":
            task = f"다음 미션을 완료했습니다. 심리학 인사이트를 제공해주세요:\n{mission_text}"
        else:
            task = f"사용자가 미션을 거절했습니다. 기분: {mood}\n동기면담 격려 메시지를 제공해주세요."

        return self._run(task)
