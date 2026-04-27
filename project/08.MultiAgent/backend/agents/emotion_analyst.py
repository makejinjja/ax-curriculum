from __future__ import annotations
import json

from .base import BaseAgent
from schemas import EmotionAnalysis


class EmotionAnalystAgent(BaseAgent):
    prompt_filename = "emotion_analyst.txt"

    def analyze(self, user_message: str) -> EmotionAnalysis:
        raw = self._chat(
            user_content=f"사용자 메시지:\n{user_message}",
            temperature=0.3,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "emotion_type": "중립",
                "intensity": 3,
                "triggers": [],
                "needs": ["휴식"],
                "summary": "감정 분석 실패, 기본값 사용",
            }
        return EmotionAnalysis(**data)
