"""
coverage.py — 요구사항 반영도(Requirement Coverage) 평가

testset case["requirements"] 리스트의 각 항목을 체크해
충족 비율을 score로 반환한다.

지원 requirement 키:
  mission_nonempty          미션 텍스트가 비어있지 않음
  basis_nonempty            근거 텍스트가 비어있지 않음
  effect_nonempty           효과 텍스트가 비어있지 않음
  emotion_type_match        분류된 감정이 expected.emotion_type과 일치
  category_allowed          미션 카테고리가 expected.category 목록 안에 있음
  difficulty_allowed        난이도가 expected.difficulty 목록 안에 있음
  time_feasible             난이도 기반 예상 소요 시간 ≤ input.minutes
  category_matches_emotion  감정 유형에 적합한 카테고리인지 휴리스틱 검사
"""
from __future__ import annotations
from .base import BaseEvaluator, EvalResult

# 난이도 → 예상 소요 시간(분) 매핑
DIFF_EST_MINUTES: dict[str, int] = {
    "하":   5,
    "중":   15,
    "상":   30,
    "최상": 60,
    "돌발": 5,
    "도전": 5,
}

# 감정 유형 → 적합한 카테고리
EMOTION_ALLOWED_CATS: dict[str, list[str]] = {
    "부정적": ["건강", "재미", "성장", "생산성", "돌발"],
    "중립":   ["생산성", "성장", "재미", "건강", "돌발"],
    "긍정적": ["재미", "성장", "생산성", "돌발"],
    "집중됨": ["생산성", "성장", "돌발"],
    "지루함": ["재미", "생산성", "성장", "돌발"],
}


class CoverageEvaluator(BaseEvaluator):
    """
    요구사항 반영도 평가.

    score = 충족된 요구사항 수 / 전체 요구사항 수
    알 수 없는 키는 체크에서 제외한다.
    """
    name = "coverage"

    def __init__(self, pass_threshold: float = 0.8):
        self.pass_threshold = pass_threshold

    def evaluate(self, case: dict, pipeline_output: dict) -> EvalResult:
        case_id      = case.get("id", "unknown")
        requirements = case.get("requirements", [])
        expected     = case.get("expected", {})

        parsed       = pipeline_output.get("parsed_mission", {})
        emotion_type = pipeline_output.get("emotion_type", "")
        minutes      = case.get("input", {}).get("minutes", 60)

        if not requirements:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=1.0,
                passed=True,
                notes="requirements 없음 — 스킵",
            )

        checks: dict[str, bool | None] = {}

        for req in requirements:

            # ── 텍스트 필드 비어있지 않음 ─────────────────────
            if req == "mission_nonempty":
                checks[req] = bool(parsed.get("mission", "").strip())

            elif req == "basis_nonempty":
                checks[req] = bool(parsed.get("basis", "").strip())

            elif req == "effect_nonempty":
                checks[req] = bool(parsed.get("effect", "").strip())

            # ── 감정 유형 일치 ────────────────────────────────
            elif req == "emotion_type_match":
                expected_et = expected.get("emotion_type")
                if expected_et:
                    checks[req] = (emotion_type == expected_et)
                else:
                    checks[req] = None  # ground truth 없으면 스킵

            # ── 카테고리 허용 목록 ────────────────────────────
            elif req == "category_allowed":
                allowed = expected.get("category", [])
                actual  = parsed.get("category", "")
                checks[req] = (actual in allowed) if allowed else None

            # ── 난이도 허용 목록 ──────────────────────────────
            elif req == "difficulty_allowed":
                allowed = expected.get("difficulty", [])
                actual  = parsed.get("difficulty", "")
                checks[req] = (actual in allowed) if allowed else None

            # ── 시간 실현 가능성 ──────────────────────────────
            elif req == "time_feasible":
                diff = parsed.get("difficulty", "하")
                est  = DIFF_EST_MINUTES.get(diff, 15)
                checks[req] = est <= minutes

            # ── 감정 유형 ↔ 카테고리 적합성 ──────────────────
            elif req == "category_matches_emotion":
                allowed = EMOTION_ALLOWED_CATS.get(emotion_type, [])
                actual  = parsed.get("category", "")
                checks[req] = (actual in allowed) if allowed else True

            else:
                checks[req] = None  # 알 수 없는 키: 스킵

        # 실제 평가된 항목만 집계
        evaluated = {k: v for k, v in checks.items() if v is not None}
        passed_n  = sum(1 for v in evaluated.values() if v)
        total_n   = len(evaluated)

        score = round(passed_n / total_n, 4) if total_n else 1.0

        # 미반영·실패 항목 메모
        skipped = [k for k, v in checks.items() if v is None]
        failed  = [k for k, v in evaluated.items() if not v]
        notes   = ""
        if failed:
            notes += f"실패: {', '.join(failed)}"
        if skipped:
            notes += (" | " if notes else "") + f"스킵: {', '.join(skipped)}"

        return EvalResult(
            evaluator=self.name,
            case_id=case_id,
            score=score,
            passed=score >= self.pass_threshold,
            details={
                "checks":    {k: (bool(v) if v is not None else "skipped") for k, v in checks.items()},
                "passed_n":  passed_n,
                "total_n":   total_n,
            },
            notes=notes,
        )
