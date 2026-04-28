"""
mission_agent.py — MissionAgent

담당: 미션 생성(generate_mission) + 검증(validate_mission)
오케스트레이터가 delegate_mission_agent 툴을 통해 호출한다.
"""
from __future__ import annotations
from pathlib import Path

from openai import OpenAI

from agents import BaseSpecialistAgent
from tools import tool_generate_mission, tool_validate_mission

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "mission_agent.txt"


class MissionAgent(BaseSpecialistAgent):

    @property
    def _TOOL_DEFS(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_mission",
                    "description": (
                        "사용자의 기분과 가용 시간을 바탕으로 미션을 생성한다. "
                        "HyDE·멀티쿼리·하이브리드 검색·돌발 미션(15%)·가중 난이도 랜덤·"
                        "최근 미션 중복 제거가 포함된 전체 파이프라인을 실행한다."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mood":    {"type": "string",  "description": "사용자의 현재 기분 (자유 텍스트)"},
                            "minutes": {"type": "integer", "description": "가용 시간 (분)"},
                        },
                        "required": ["mood", "minutes"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_mission",
                    "description": "생성된 미션의 형식·시간·카테고리·난이도를 검증한다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mission_text":      {"type": "string",  "description": "검증할 미션 전체 텍스트 (태그 포함)"},
                            "available_minutes": {"type": "integer", "description": "사용자의 가용 시간 (분)"},
                            "category":          {"type": "string",  "description": "미션 카테고리"},
                            "difficulty":        {"type": "string",  "description": "미션 난이도"},
                        },
                        "required": ["mission_text", "available_minutes", "category", "difficulty"],
                    },
                },
            },
        ]

    def _dispatch(self, tool_name: str, args: dict, **ctx) -> dict:
        user_data         = ctx.get("user_data", {})
        available_minutes = ctx.get("available_minutes", 30)

        if tool_name == "generate_mission":
            return tool_generate_mission(self.client, self.rag_state, user_data, **args)
        elif tool_name == "validate_mission":
            args.setdefault("available_minutes", available_minutes)
            return tool_validate_mission(**args)
        return {"error": f"알 수 없는 툴: {tool_name}"}

    def _system_prompt(self) -> str:
        if _PROMPT_PATH.exists():
            return _PROMPT_PATH.read_text(encoding="utf-8").strip()
        return (
            "당신은 미션 생성 전문가입니다. "
            "generate_mission으로 미션을 생성하고 validate_mission으로 검증하세요. "
            "결과를 [미션][카테고리][난이도][근거][효과] 형식으로 반환하세요."
        )

    def run(
        self,
        mood: str,
        minutes: int,
        user_data: dict,
    ) -> dict:
        """
        Returns:
            {"result": str, "tool_calls_made": list[str], "last_mission": dict | None}
        """
        task = f"기분: {mood}\n가용 시간: {minutes}분\n미션을 생성하고 검증하세요."
        out  = self._run(task, user_data=user_data, available_minutes=minutes)

        # _run()이 tool result를 버리므로, tool_calls 결과에서 직접 캡처된 미션 확인
        # _run_with_mission()으로 재실행하여 generate_mission 결과를 캡처
        last_mission: dict | None = out.get("last_mission")
        return {**out, "last_mission": last_mission}

    def _run(self, task: str, **ctx) -> dict:
        """generate_mission 결과 캡처 + validate_mission 재시도 한계 적용."""
        import json as _json
        from agents import MAX_SPECIALIST_ITERATIONS
        from schemas import MAX_GENERATION_RETRIES

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",   "content": task},
        ]
        tool_calls_made: list[str] = []
        last_mission: dict | None = None
        validation_failures = 0

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
                    "last_mission":    last_mission,
                }

            for tc in msg.tool_calls:
                name = tc.function.name
                args = _json.loads(tc.function.arguments)
                tool_calls_made.append(name)

                # validate_mission 최대 실패 초과 시 강제 통과
                if name == "validate_mission" and validation_failures >= MAX_GENERATION_RETRIES:
                    result = {
                        "valid":   True,
                        "message": "최대 재시도 횟수 초과. 현재 미션을 최종 결과로 사용합니다.",
                    }
                else:
                    result = self._dispatch(name, args, **ctx)

                if name == "generate_mission" and result.get("mission"):
                    last_mission = result["mission"]

                if name == "validate_mission" and not result.get("valid"):
                    validation_failures += 1

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      _json.dumps(result, ensure_ascii=False),
                })

        return {
            "result":          "최대 반복 횟수에 도달했습니다.",
            "tool_calls_made": tool_calls_made,
            "last_mission":    last_mission,
        }
