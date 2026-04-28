"""
rule.py — Rule 기반 구조 검사

testset case["rules"] dict에 정의된 규칙을 LLM 없이 판정한다.

지원 규칙 키:
  max_time_minutes          난이도 예상 소요 ≤ 값
  allowed_categories        카테고리가 목록 안에 있음
  allowed_difficulties      난이도가 목록 안에 있음
  min_mission_length        미션 텍스트 길이 ≥ 값 (자)
  max_mission_length        미션 텍스트 길이 ≤ 값 (자)
  max_fruits                현재 data["fruits"] 개수 < 값
  combo_wildcard_no_increment  돌발 미션이면 콤보 변화 없음
  max_category_ratio        fruits 중 단일 카테고리 비율 ≤ 값 (0~1)

score = 통과 규칙 수 / 전체 규칙 수
"""
from __future__ import annotations
from collections import Counter
from .base import BaseEvaluator, EvalResult

DIFF_EST_MINUTES: dict[str, int] = {
    "하":   5,
    "중":   15,
    "상":   30,
    "최상": 60,
    "돌발": 5,
    "도전": 5,
}


class RuleEvaluator(BaseEvaluator):
    name = "rule"

    def evaluate(self, case: dict, pipeline_output: dict) -> EvalResult:
        case_id = case.get("id", "unknown")
        rules   = case.get("rules", {})

        if not rules:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=1.0,
                passed=True,
                notes="rules 없음 — 스킵",
            )

        parsed       = pipeline_output.get("parsed_mission", {})
        data         = pipeline_output.get("data_state", {})
        combo_before = pipeline_output.get("combo_before", 0)
        combo_after  = pipeline_output.get("combo_after",  0)

        mission    = parsed.get("mission", "")
        category   = parsed.get("category", "")
        difficulty = parsed.get("difficulty", "하")
        is_wildcard= parsed.get("is_wildcard", False)
        fruits     = data.get("fruits", [])

        checks: dict[str, bool] = {}

        # ── 시간 제한 ──────────────────────────────────────────
        if "max_time_minutes" in rules:
            est = DIFF_EST_MINUTES.get(difficulty, 15)
            checks["time_within_limit"] = est <= rules["max_time_minutes"]

        # ── 카테고리 유효성 ────────────────────────────────────
        if "allowed_categories" in rules:
            checks["category_valid"] = category in rules["allowed_categories"]

        # ── 난이도 유효성 ──────────────────────────────────────
        if "allowed_difficulties" in rules:
            checks["difficulty_valid"] = difficulty in rules["allowed_difficulties"]

        # ── 미션 텍스트 길이 ───────────────────────────────────
        if "min_mission_length" in rules:
            checks["mission_min_length"] = len(mission) >= rules["min_mission_length"]

        if "max_mission_length" in rules:
            checks["mission_max_length"] = len(mission) <= rules["max_mission_length"]

        # ── 열매 개수 상한 ─────────────────────────────────────
        if "max_fruits" in rules:
            checks["fruits_not_full"] = len(fruits) < rules["max_fruits"]

        # ── 돌발 미션은 콤보 변화 없음 ─────────────────────────
        if rules.get("combo_wildcard_no_increment"):
            if is_wildcard:
                checks["combo_no_change_on_wildcard"] = (combo_after == combo_before)
            else:
                checks["combo_no_change_on_wildcard"] = True  # 일반 미션은 규칙 해당 없음

        # ── 카테고리 편향 방지 (그룹 규칙) ────────────────────
        if "max_category_ratio" in rules and fruits:
            cats      = [f.get("category") for f in fruits if f.get("category")]
            if cats:
                cnt       = Counter(cats)
                max_ratio = max(v / len(cats) for v in cnt.values())
                checks["category_balance"] = max_ratio <= rules["max_category_ratio"]

        if not checks:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=1.0,
                passed=True,
                notes="유효한 규칙 없음 — 스킵",
            )

        passed_n = sum(1 for v in checks.values() if v)
        total_n  = len(checks)
        score    = round(passed_n / total_n, 4)

        failed = [k for k, v in checks.items() if not v]
        notes  = f"실패: {', '.join(failed)}" if failed else ""

        return EvalResult(
            evaluator=self.name,
            case_id=case_id,
            score=score,
            passed=score == 1.0,
            details={
                "checks":   checks,
                "passed_n": passed_n,
                "total_n":  total_n,
            },
            notes=notes,
        )
