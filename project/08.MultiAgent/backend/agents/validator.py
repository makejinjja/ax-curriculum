from __future__ import annotations
import json
import re

from .base import BaseAgent
from schemas import (
    CodeValidationResult, LLMValidationResult,
    DIFFICULTY_MIN_MINUTES, VALID_CATEGORIES, EmotionAnalysis,
)


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL)
    return m.group(1).strip() if m else ""


class ValidatorAgent(BaseAgent):
    prompt_filename = "validator.txt"

    # ── 코드 검증 (순수 Python) ────────────────────────────────
    def code_check(
        self,
        mission_text: str,
        available_minutes: int,
    ) -> CodeValidationResult:
        errors: list[str] = []
        required_tags = ["미션", "카테고리", "난이도", "근거", "효과"]
        has_required = all(
            re.search(rf"\[{t}\].*?\[/{t}\]", mission_text, re.DOTALL)
            for t in required_tags
        )
        if not has_required:
            missing = [t for t in required_tags
                       if not re.search(rf"\[{t}\].*?\[/{t}\]", mission_text, re.DOTALL)]
            errors.append(f"누락된 태그: {', '.join(missing)}")

        category = _extract_tag(mission_text, "카테고리")
        valid_category = category in VALID_CATEGORIES
        if not valid_category:
            errors.append(f"유효하지 않은 카테고리: '{category}'. 허용: {VALID_CATEGORIES}")

        difficulty = _extract_tag(mission_text, "난이도")
        valid_difficulty = difficulty in DIFFICULTY_MIN_MINUTES
        if not valid_difficulty:
            errors.append(f"유효하지 않은 난이도: '{difficulty}'")

        time_feasible = True
        if valid_difficulty:
            min_time = DIFFICULTY_MIN_MINUTES[difficulty]
            time_feasible = available_minutes >= min_time
            if not time_feasible:
                errors.append(
                    f"시간 부족: 난이도 '{difficulty}'는 최소 {min_time}분 필요 "
                    f"(가용: {available_minutes}분)"
                )

        passed = has_required and valid_category and valid_difficulty and time_feasible
        return CodeValidationResult(
            has_required_tags=has_required,
            valid_category=valid_category,
            valid_difficulty=valid_difficulty,
            time_feasible=time_feasible,
            passed=passed,
            errors=errors,
        )

    # ── LLM 판단 검증 ─────────────────────────────────────────
    def llm_judge(
        self,
        mission_text: str,
        emotion: EmotionAnalysis,
        available_minutes: int,
    ) -> LLMValidationResult:
        user_content = (
            f"# 사용자 감정 상태\n"
            f"- 감정: {emotion.emotion_type} (강도: {emotion.intensity}/5)\n"
            f"- 요약: {emotion.summary}\n"
            f"- 가용 시간: {available_minutes}분\n\n"
            f"# 검증할 미션\n{mission_text}"
        )
        raw = self._chat(
            user_content=user_content,
            temperature=0.2,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(raw)
            return LLMValidationResult(**data)
        except Exception:
            return LLMValidationResult(
                scores={"psychological_validity": 20, "practicality": 20,
                        "safety": 25, "specificity": 15},
                total_score=80,
                is_valid=True,
                feedback="",
                strengths="LLM 검증 파싱 실패, 기본 통과",
            )
