"""
base.py — 공통 데이터 클래스 및 추상 평가기
"""
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class EvalResult:
    """단일 평가기의 케이스 결과."""
    evaluator: str
    case_id:   str
    score:     float        # 0.0 ~ 1.0 정규화 점수
    passed:    bool = True
    details:   dict[str, Any] = field(default_factory=dict)
    notes:     str = ""

    def to_dict(self) -> dict:
        return {
            "evaluator": self.evaluator,
            "case_id":   self.case_id,
            "score":     round(self.score, 4),
            "passed":    self.passed,
            "details":   self.details,
            "notes":     self.notes,
        }


class BaseEvaluator(ABC):
    """모든 평가기의 공통 인터페이스."""
    name: str = "base"

    @abstractmethod
    def evaluate(self, case: dict, pipeline_output: dict) -> EvalResult:
        """
        Args:
            case:            testset 케이스 dict
            pipeline_output: run_pipeline() 결과 dict
        Returns:
            EvalResult
        """
        ...
