"""
faithfulness.py — 생성 결과의 검색 근거 충실도 평가 (LLM judge)

RAGAS Faithfulness 방식:
  1. 생성 미션 + 근거 텍스트에서 핵심 claim 추출
  2. 각 claim이 검색된 컨텍스트로 뒷받침되는지 판정
  3. score = 지지된 claim 수 / 전체 claim 수

단순 버전(claim 분해 없이): LLM에게 0~1 점수를 직접 부여하도록 유도.
"""
from __future__ import annotations
import re
from openai import OpenAI
from .base import BaseEvaluator, EvalResult

# ── 1단계: claim 추출 ────────────────────────────────────────
CLAIM_EXTRACTION_PROMPT = """아래 [생성된 미션 텍스트]에서 검증 가능한 핵심 주장(claim)을 최대 5개 추출해라.
각 claim은 한 줄, 번호 없이 출력해라.

[생성된 미션 텍스트]
{text}"""

# ── 2단계: claim 검증 ────────────────────────────────────────
CLAIM_VERIFY_PROMPT = """아래 [컨텍스트]가 [Claim]을 뒷받침하면 "YES", 뒷받침하지 않으면 "NO"만 출력해라.

[컨텍스트]
{context}

[Claim]
{claim}"""

# ── 단일 점수 방식 (fallback) ────────────────────────────────
SINGLE_SCORE_PROMPT = """당신은 RAG Faithfulness 평가자입니다.

[검색된 논문 컨텍스트]와 [생성된 미션/근거]를 비교해 충실도를 평가하세요.

평가 기준:
- 미션 제안이 컨텍스트의 심리학적 근거에서 도출됐으면 높은 점수
- 컨텍스트와 무관한 내용이 많으면 낮은 점수

출력 형식 (반드시 지켜라):
점수: <0.0~1.0>
이유: <한 줄>"""


class FaithfulnessEvaluator(BaseEvaluator):
    """
    LLM 기반 Faithfulness 평가.
    mode="claim" : claim 분해 후 개별 검증 (정확하지만 API 2회 호출)
    mode="single": 단일 점수 요청 (빠름, fallback)
    """
    name = "faithfulness"

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        mode: str = "claim",
        pass_threshold: float = 0.6,
    ):
        self.client          = client
        self.model           = model
        self.mode            = mode
        self.pass_threshold  = pass_threshold

    # ── claim 분해 모드 ───────────────────────────────────────
    def _extract_claims(self, text: str) -> list[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(text=text)},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        return lines[:5]

    def _verify_claim(self, context: str, claim: str) -> bool:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": CLAIM_VERIFY_PROMPT.format(
                    context=context[:1200], claim=claim
                )},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        return resp.choices[0].message.content.strip().upper().startswith("YES")

    def _score_claim_mode(self, context: str, answer_text: str) -> tuple[float, dict]:
        claims = self._extract_claims(answer_text)
        if not claims:
            return 0.0, {"claims": [], "verified": []}

        verified = [self._verify_claim(context, c) for c in claims]
        score    = sum(verified) / len(verified)
        return round(score, 4), {
            "claims":   claims,
            "verified": verified,
            "supported": int(sum(verified)),
            "total":     len(claims),
        }

    # ── 단일 점수 모드 ───────────────────────────────────────
    def _score_single_mode(self, context: str, answer_text: str) -> tuple[float, dict]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SINGLE_SCORE_PROMPT},
                {"role": "user", "content": (
                    f"[검색된 논문 컨텍스트]\n{context[:1500]}\n\n"
                    f"[생성된 미션/근거]\n{answer_text}"
                )},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw    = resp.choices[0].message.content.strip()
        score  = 0.0
        reason = ""
        for line in raw.split("\n"):
            if line.startswith("점수:"):
                m = re.search(r"[\d.]+", line)
                if m:
                    score = max(0.0, min(1.0, float(m.group())))
            if line.startswith("이유:"):
                reason = line.split(":", 1)[1].strip()
        return round(score, 4), {"raw_response": raw, "reason": reason}

    # ── 공통 진입점 ──────────────────────────────────────────
    def evaluate(self, case: dict, pipeline_output: dict) -> EvalResult:
        case_id     = case.get("id", "unknown")
        context     = pipeline_output.get("retrieved_context", "")
        mission     = pipeline_output.get("generated_mission", "")
        basis       = pipeline_output.get("generated_basis", "")
        effect      = pipeline_output.get("generated_effect", "")
        is_wildcard = pipeline_output.get("is_wildcard", False)

        if not context:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=1.0 if is_wildcard else 0.0,
                passed=is_wildcard,
                notes="컨텍스트 없음 (RAG 인덱스 비어있거나 돌발 미션)",
            )
        if not mission:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=0.0,
                passed=False,
                notes="미션 텍스트 없음",
            )

        answer_text = f"미션: {mission}\n근거: {basis}\n효과: {effect}"

        try:
            if self.mode == "claim":
                score, details = self._score_claim_mode(context, answer_text)
            else:
                score, details = self._score_single_mode(context, answer_text)
        except Exception as e:
            return EvalResult(
                evaluator=self.name,
                case_id=case_id,
                score=0.0,
                passed=False,
                notes=f"LLM 호출 오류: {e}",
            )

        return EvalResult(
            evaluator=self.name,
            case_id=case_id,
            score=score,
            passed=score >= self.pass_threshold,
            details=details,
        )
