from __future__ import annotations
import json

import numpy as np
from rank_bm25 import BM25Okapi

from .base import BaseAgent
from schemas import EmotionAnalysis, ResearchResult
from rag import search_rag, build_context


class RAGResearcherAgent(BaseAgent):
    prompt_filename = "rag_researcher.txt"

    def research(
        self,
        emotion: EmotionAnalysis,
        chunks: list[dict],
        embeddings: "np.ndarray",
        bm25: BM25Okapi,
        k: int = 5,
    ) -> ResearchResult:
        query = f"{emotion.emotion_type} 감정 상태: {emotion.summary}. 필요: {', '.join(emotion.needs)}"
        top_chunks = search_rag(
            self.client, query, chunks, embeddings, bm25,
            emotion_type=emotion.emotion_type, k=k,
        )
        context = build_context(top_chunks)

        user_content = (
            f"감정 분석 결과:\n"
            f"- 감정 유형: {emotion.emotion_type} (강도: {emotion.intensity}/5)\n"
            f"- 요인: {', '.join(emotion.triggers)}\n"
            f"- 필요: {', '.join(emotion.needs)}\n\n"
            f"심리학 논문 검색 결과:\n{context}"
        )
        raw = self._chat(
            user_content=user_content,
            temperature=0.5,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "key_findings": [],
                "recommended_activities": [],
                "psychological_basis": "심리학적 근거를 찾을 수 없습니다.",
                "expected_effect": "긍정적 효과 기대",
            }
        result = ResearchResult(**data)
        result.raw_chunks = top_chunks
        return result
