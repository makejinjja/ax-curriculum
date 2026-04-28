"""
orchestrator.py — OrchestratorAgent

흐름:
  1. 시스템 프롬프트 로드
  2. 사용자 메시지 + 대화 이력으로 루프 진입
  3. LLM이 delegate_* 툴 호출 → 해당 전문 에이전트 실행 → 결과를 메시지에 추가 → 재호출
  4. LLM이 툴 없이 응답하면(finish_reason=stop) 루프 종료
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from openai import OpenAI

from agents.mission_agent    import MissionAgent
from agents.psychology_agent import PsychologyAgent
from agents.search_agent     import SearchAgent
from schemas import MAX_AGENT_ITERATIONS

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "orchestrator.txt"

# ── 오케스트레이터 delegate 툴 정의 ────────────────────────────

ORCHESTRATOR_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "delegate_mission_agent",
            "description": (
                "MissionAgent에게 미션 생성·검증을 위임한다. "
                "사용자가 기분을 말하고 미션을 원할 때 호출한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mood":    {"type": "string",  "description": "사용자의 현재 기분"},
                    "minutes": {"type": "integer", "description": "가용 시간 (분)"},
                },
                "required": ["mood", "minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_psychology_agent",
            "description": (
                "PsychologyAgent에게 심리학 응답을 위임한다. "
                "미션 완료 인사이트(action=insight) 또는 동기면담 격려(action=nudge) 중 하나."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["insight", "nudge"],
                        "description": "insight: 완료 인사이트 / nudge: 거절 시 격려",
                    },
                    "mission_text": {"type": "string", "description": "완료된 미션 텍스트 (insight 시 필요)"},
                    "mood":         {"type": "string", "description": "현재 기분 (nudge 시 필요)"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_search_agent",
            "description": (
                "SearchAgent에게 RAG/웹 검색을 위임한다. "
                "추가 심리학 근거나 최신 정보가 필요할 때 사용한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":        {"type": "string",  "description": "검색 쿼리"},
                    "emotion_type": {
                        "type": "string",
                        "enum": ["부정적", "중립", "긍정적", "집중됨", "지루함"],
                        "description": "사용자의 감정 유형",
                    },
                    "use_web": {"type": "boolean", "description": "웹 검색도 수행할지 여부 (기본 false)"},
                },
                "required": ["query", "emotion_type"],
            },
        },
    },
]


def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    return "당신은 심리학 기반 미션 오케스트레이터 Bloom입니다."


def _assistant_dict(msg) -> dict:
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


class OrchestratorAgent:
    def __init__(self, client: OpenAI, rag_state: dict) -> None:
        self.client         = client
        self.rag_state      = rag_state
        self.system_prompt  = _load_system_prompt()
        self._mission_agent = MissionAgent(client, rag_state)
        self._psych_agent   = PsychologyAgent(client, rag_state)
        self._search_agent  = SearchAgent(client, rag_state)

    def chat(
        self,
        user_message: str,
        available_minutes: int,
        conversation_history: list[dict],
        user_data: dict,
    ) -> dict:
        """
        Returns:
            {
              "response":        str,
              "mission":         dict | None,
              "tool_calls_made": list[str],
              "user_data":       dict,
            }
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            *conversation_history,
            {
                "role":    "user",
                "content": f"{user_message}\n\n가용 시간: {available_minutes}분",
            },
        ]

        tool_calls_made: list[str] = []
        last_tool_mission: dict | None = None

        for _ in range(MAX_AGENT_ITERATIONS):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=ORCHESTRATOR_TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1500,
            )

            msg    = response.choices[0].message
            finish = response.choices[0].finish_reason
            messages.append(_assistant_dict(msg))

            if finish == "stop" or not msg.tool_calls:
                final_text = msg.content or ""
                mission    = _parse_mission(final_text) or last_tool_mission
                return {
                    "response":        final_text,
                    "mission":         mission,
                    "tool_calls_made": tool_calls_made,
                    "user_data":       user_data,
                }

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                tool_calls_made.append(name)

                result = self._dispatch_delegate(name, args, available_minutes, user_data)

                # MissionAgent가 반환한 미션을 캐시 + mission_history 업데이트
                if name == "delegate_mission_agent":
                    # 우선순위: 구조화된 last_mission → 텍스트 파싱
                    parsed = result.get("last_mission") or _parse_mission(result.get("result", ""))
                    if parsed:
                        last_tool_mission = parsed
                        m_text = parsed.get("mission", "")
                        if m_text:
                            hist = user_data.setdefault("mission_history", [])
                            hist.append(m_text)
                            if len(hist) > 20:
                                user_data["mission_history"] = hist[-20:]

                # 전문 에이전트 내부 tool_calls_made도 상위에 병합
                sub_calls = result.pop("tool_calls_made", [])
                tool_calls_made.extend(sub_calls)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result, ensure_ascii=False),
                })

        return {
            "response":        "미션 생성 중 최대 반복 횟수에 도달했습니다. 잠시 후 다시 시도해 주세요.",
            "mission":         None,
            "tool_calls_made": tool_calls_made,
            "user_data":       user_data,
        }

    def _dispatch_delegate(
        self,
        name: str,
        args: dict,
        available_minutes: int,
        user_data: dict,
    ) -> dict:
        if name == "delegate_mission_agent":
            mood    = args.get("mood", "")
            minutes = args.get("minutes", available_minutes)
            return self._mission_agent.run(mood, minutes, user_data)

        elif name == "delegate_psychology_agent":
            action       = args.get("action", "nudge")
            mission_text = args.get("mission_text", "")
            mood         = args.get("mood", "")
            return self._psych_agent.run(action, mission_text, mood)

        elif name == "delegate_search_agent":
            query        = args.get("query", "")
            emotion_type = args.get("emotion_type", "중립")
            use_web      = args.get("use_web", False)
            return self._search_agent.run(query, emotion_type, use_web)

        return {"error": f"알 수 없는 delegate 툴: {name}"}
