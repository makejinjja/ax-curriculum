"""
retrieval.py — Retrieval 품질 평가
지표: Precision@k, Recall@k, Hit@k, MRR

ground truth: testset case["expected"]["relevant_doc_ids"]
  — 해당 쿼리에서 검색돼야 할 논문 레이블 부분 문자열 목록
  — 예: ["운동·정신건강", "CBT(인지행동치료)"]

retrieved_sources: pipeline_output["retrieved_sources"]
  — 실제 검색된 청크의 source 문자열 목록 (순위 순)
  — 예: ["Benefits_of_Exercise... (운동·정신건강)", ...]
"""
from __future__ import annotations
from .base import BaseEvaluator, EvalResult


def _match(source: str, relevant_ids: set[str]) -> bool:
    """source 문자열에 relevant_id 중 하나라도 포함되면 hit."""
    return any(r in source for r in relevant_ids)


class RetrievalEvaluator(BaseEvaluator):
    """
    Retrieval 평가 지표:
      - Precision@k : top-k 중 관련 문서 비율
      - Recall@k    : 전체 관련 문서 중 top-k에서 찾은 비율
      - Hit@k       : top-k 안에 관련 문서가 1개라도 있으면 1
      - MRR         : 첫 번째 관련 문서의 역순위 평균

    주 점수(score): Precision@3 (k_values 중간값)
    """
    name = "retrieval"

    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [1, 3, 5]

    def evaluate(self, case: dict, pipeline_output: dict) -> EvalResult:
        case_id      = case.get("id", "unknown")
        expected     = case.get("expected", {})
        relevant_ids = set(expected.get("relevant_doc_ids", []))
        retrieved    = pipeline_output.get("retrieved_sources", [])

        # ground truth 없으면 스킵
        if not relevant_ids:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=1.0,
                passed=True,
                notes="relevant_doc_ids 없음 — 스킵",
            )

        # 검색 결과 없으면 0점
        if not retrieved:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=0.0,
                passed=False,
                details={f"precision@{k}": 0.0 for k in self.k_values},
                notes="검색 결과 없음 (RAG 인덱스 비어있음)",
            )

        metrics: dict[str, float] = {}

        for k in self.k_values:
            top_k  = retrieved[:k]
            hits   = sum(1 for s in top_k if _match(s, relevant_ids))
            metrics[f"precision@{k}"] = round(hits / k, 4)
            metrics[f"recall@{k}"]    = round(hits / len(relevant_ids), 4)
            metrics[f"hit@{k}"]       = 1 if hits > 0 else 0

        # MRR
        mrr = 0.0
        for rank, src in enumerate(retrieved, 1):
            if _match(src, relevant_ids):
                mrr = 1.0 / rank
                break
        metrics["mrr"] = round(mrr, 4)

        # 주 점수: k_values 중간값의 Precision@k
        mid_k  = self.k_values[len(self.k_values) // 2]
        score  = metrics.get(f"precision@{mid_k}", 0.0)
        passed = score >= 0.33  # 1/3 이상 적중

        return EvalResult(
            evaluator=self.name,
            case_id=case_id,
            score=score,
            passed=passed,
            details=metrics,
        )
