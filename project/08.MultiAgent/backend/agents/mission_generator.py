from __future__ import annotations

from .base import BaseAgent
from schemas import EmotionAnalysis, ResearchResult


class MissionGeneratorAgent(BaseAgent):
    prompt_filename = "mission_generator.txt"

    def generate(
        self,
        emotion: EmotionAnalysis,
        research: ResearchResult,
        available_minutes: int,
        feedback: str = "",
    ) -> str:
        user_content = (
            f"# 감정 분석\n"
            f"- 감정 유형: {emotion.emotion_type} (강도: {emotion.intensity}/5)\n"
            f"- 요약: {emotion.summary}\n"
            f"- 필요: {', '.join(emotion.needs)}\n\n"
            f"# 심리학 연구 결과\n"
            f"- 핵심 발견: {'; '.join(research.key_findings[:3])}\n"
            f"- 추천 활동: {', '.join(research.recommended_activities[:3])}\n"
            f"- 근거: {research.psychological_basis}\n"
            f"- 예상 효과: {research.expected_effect}\n\n"
            f"# 제약 조건\n"
            f"- 가용 시간: {available_minutes}분\n"
        )
        if feedback:
            user_content += f"\n# 이전 시도 피드백 (반드시 반영)\n{feedback}\n"

        return self._chat(user_content=user_content, temperature=0.8, max_tokens=512)
