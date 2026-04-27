from __future__ import annotations
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from agents import (
    EmotionAnalystAgent, RAGResearcherAgent,
    MissionGeneratorAgent, ValidatorAgent,
)
from schemas import (
    EmotionAnalysis, ResearchResult, ValidationResult,
    CodeValidationResult, LLMValidationResult, CurriculumRecord,
    PROMPTS_DIR, CURRICULA_DIR, MAX_GENERATION_RETRIES,
)


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_mission(text: str) -> dict | None:
    mission = _extract_tag(text, "미션")
    if not mission:
        return None
    return {
        "mission":    mission,
        "category":   _extract_tag(text, "카테고리"),
        "difficulty": _extract_tag(text, "난이도"),
        "basis":      _extract_tag(text, "근거"),
        "effect":     _extract_tag(text, "효과"),
    }


def _save_curriculum(
    record: CurriculumRecord,
) -> None:
    CURRICULA_DIR.mkdir(parents=True, exist_ok=True)
    path = CURRICULA_DIR / f"{record.id}.json"
    path.write_text(
        json.dumps(record.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class Orchestrator:
    def __init__(self, client):
        self.client = client
        self.emotion_agent  = EmotionAnalystAgent(client)
        self.rag_agent      = RAGResearcherAgent(client)
        self.mission_agent  = MissionGeneratorAgent(client)
        self.validator      = ValidatorAgent(client)
        self._load_orchestrator_prompt()

    def _load_orchestrator_prompt(self):
        path = PROMPTS_DIR / "orchestrator.txt"
        self.orchestrator_prompt = path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def run(
        self,
        user_message: str,
        available_minutes: int,
        chunks: list[dict],
        embeddings: "np.ndarray",
        bm25: BM25Okapi,
        username: str = "user",
    ) -> dict:
        trace: list[dict] = []

        # ── Step 1: 감정 분석 ──────────────────────────────────
        emotion = self.emotion_agent.analyze(user_message)
        trace.append({
            "agent": "EmotionAnalyst",
            "output": emotion.model_dump(),
        })

        # ── Step 2: RAG 연구 ───────────────────────────────────
        research = self.rag_agent.research(
            emotion, chunks, embeddings, bm25
        )
        trace.append({
            "agent": "RAGResearcher",
            "output": {
                "key_findings": research.key_findings,
                "recommended_activities": research.recommended_activities,
                "psychological_basis": research.psychological_basis,
                "chunks_used": len(research.raw_chunks),
            },
        })

        # ── Step 3: 미션 생성 + 검증 루프 ─────────────────────
        mission_text = ""
        final_validation: ValidationResult | None = None
        feedback = ""

        for attempt in range(1, MAX_GENERATION_RETRIES + 1):
            mission_text = self.mission_agent.generate(
                emotion, research, available_minutes, feedback=feedback
            )
            trace.append({
                "agent": "MissionGenerator",
                "attempt": attempt,
                "output": mission_text,
            })

            # 코드 검증
            code_result = self.validator.code_check(mission_text, available_minutes)
            trace.append({
                "agent": "Validator(code)",
                "attempt": attempt,
                "passed": code_result.passed,
                "errors": code_result.errors,
            })

            llm_result: LLMValidationResult | None = None
            if code_result.passed:
                llm_result = self.validator.llm_judge(mission_text, emotion, available_minutes)
                trace.append({
                    "agent": "Validator(llm)",
                    "attempt": attempt,
                    "total_score": llm_result.total_score,
                    "is_valid": llm_result.is_valid,
                })

            overall_valid = (
                code_result.passed and
                llm_result is not None and
                llm_result.is_valid
            )
            final_validation = ValidationResult(
                code_result=code_result,
                llm_result=llm_result,
                overall_valid=overall_valid,
                attempt=attempt,
            )

            if overall_valid:
                break

            # 피드백 조합
            fb_parts: list[str] = []
            if code_result.errors:
                fb_parts.append("코드 검증 오류: " + "; ".join(code_result.errors))
            if llm_result and not llm_result.is_valid and llm_result.feedback:
                fb_parts.append("LLM 평가 피드백: " + llm_result.feedback)
            feedback = "\n".join(fb_parts)

        # ── Step 4: 최종 응답 합성 ─────────────────────────────
        mission_dict = _parse_mission(mission_text)
        final_response = self._compose_response(
            user_message, emotion, research, mission_text, final_validation
        )
        trace.append({"agent": "Orchestrator", "output": "최종 응답 생성 완료"})

        # ── Step 5: 커리큘럼 저장 (미션 성공 시) ─────────────
        curriculum_id: str | None = None
        if mission_dict and final_validation and final_validation.overall_valid:
            curriculum_id = str(uuid.uuid4())
            record = CurriculumRecord(
                id=curriculum_id,
                username=username,
                created_at=datetime.now(timezone.utc).isoformat(),
                emotion=emotion,
                research=research,
                mission=mission_dict,
                validation=final_validation,
                final_response=final_response,
                available_minutes=available_minutes,
            )
            _save_curriculum(record)
            trace.append({"agent": "Orchestrator", "output": f"커리큘럼 저장: {curriculum_id}"})

        return {
            "response":      final_response,
            "mission":       mission_dict,
            "curriculum_id": curriculum_id,
            "agent_trace":   trace,
            "emotion":       emotion.model_dump(),
            "validation":    final_validation.model_dump() if final_validation else None,
        }

    def _compose_response(
        self,
        user_message: str,
        emotion: EmotionAnalysis,
        research: ResearchResult,
        mission_text: str,
        validation: ValidationResult | None,
    ) -> str:
        if not self.orchestrator_prompt:
            return mission_text

        user_content = (
            f"# 사용자 메시지\n{user_message}\n\n"
            f"# 감정 분석\n"
            f"- 유형: {emotion.emotion_type} (강도: {emotion.intensity}/5)\n"
            f"- 요약: {emotion.summary}\n\n"
            f"# 심리학적 근거\n{research.psychological_basis}\n\n"
            f"# 생성된 미션\n{mission_text}\n\n"
            f"# 검증 결과\n"
            f"{'통과' if validation and validation.overall_valid else '최선 시도'}"
        )
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.orchestrator_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or mission_text
