import os
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

class FinancialImpact(BaseModel):
    is_impactful: bool
    impact_area: str
    impact_summary: str

class NewsAnalyzer:
    def __init__(self):
        # Assumes OPENAI_API_KEY is already loaded via dotenv
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 환경 변수에 설정되어 있지 않습니다.")
        self.client = OpenAI(api_key=api_key)

    def analyze(self, title: str, full_text: str) -> Optional[dict]:
        """
        주어진 뉴스 본문을 분석하여 돈에 영향이 있는지 판단합니다.
        영향이 있다면 {impact_area, impact_summary} 형태의 딕셔너리를 반환하고,
        없다면 None을 반환합니다.
        """
        system_prompt = (
            "당신은 날카로운 금융/경제 분석가입니다. 사용자가 전달한 뉴스 제목과 본문을 읽고, "
            "이 뉴스가 '돈(관련 주식의 가격 상승/하락, 부동산 시장, 가상화폐 등 자산 가치 변동)'에 "
            "직접적이고 유의미한 영향을 미칠지 분석하세요.\n"
            "단순한 회사 소개나 제품 출시, 행사 관련 뉴스 등 자산 변화와 거리가 멀다면 is_impactful을 false로 설정하세요.\n"
            "영향이 있다면 어떤 분야(관련 주식 섹터, 부동산 지역 등)인지 간결하게 적고, "
            "어떤 원리로 가격 등에 영향을 미치는지 2~3줄로 직관적으로 요약하세요."
        )

        # 모델 컨텍스트 제한을 방지하기 위해 앞 3000자 정도만 사용
        user_content = f"제목: {title}\n\n본문: {full_text[:3000]}"

        try:
            # Pydantic을 활용한 OpenAI Structured Output 적용 (보다 정확한 형식 추출)
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format=FinancialImpact,
            )
            
            result = completion.choices[0].message.parsed
            
            if result.is_impactful:
                return {
                    "area": result.impact_area,
                    "summary": result.impact_summary
                }
            else:
                return None
                
        except Exception as e:
            print(f"\n[AI 분석 중 오류 발생: {e}]")
            return None
