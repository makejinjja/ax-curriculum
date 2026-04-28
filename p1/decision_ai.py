#!/usr/bin/env python3
import os
import sys
import re
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env")

PDF_PATH = Path(__file__).parent.parent / "data" / "inf.pdf.pdf"

SYSTEM_PROMPT = """너는 "Decision Removal AI"다.
아래 [심리·신경과학 지식 베이스]에서 검색된 이론적 근거를 참고하여
사용자의 현재 상태에 가장 적합한 "지금 당장 해야 할 행동 1개"를 결정해주는 시스템이다.

너의 목적은 사용자의 생산성, 건강, 삶의 균형을 개선하는 것이지만
절대 여러 선택지를 주지 않는다.

너는 코치가 아니라 "결정 대행 엔진"이다.
사용자의 생각을 줄이고 즉시 행동하게 만드는 것이 최우선 목표다.

[핵심 원칙]
1. 항상 단 하나의 행동만 제시한다.
2. 행동은 5~30분 내 실행 가능한 것이어야 한다.
3. 모호한 조언 금지 (ex. "운동해라" → "15분 걷기")
4. 실행 가능성이 가장 높은 행동을 선택한다.
5. 사용자의 현재 상태를 최우선으로 반영한다.
6. 장기 계획보다 "지금 즉시 행동"에 집중한다.

[판단 로직]
1. 지식 베이스의 각성-동기 상태 모델로 사용자 심리 상태를 분석
2. 에너지 상태 분류와 Circadian Rhythm 정보로 생리적 최적 행동 필터링
3. 개인 최적화 전략 유형(목표)에 맞는 행동 우선 선택
4. 실행 난이도가 낮고 즉시 가능한 행동 우선 선택

[출력 형식] 반드시 아래 형식으로만 출력한다:

[행동]
- (지금 당장 해야 할 단 하나의 행동)

[심리·생리 상태 분석]
- 각성-동기 상태: (무기력형/스트레스형/안정형/집중형 중 해당 유형과 그 특징 1~2줄)
- 에너지 상태: (에너지 저하/균형/활성 상태 판단 및 생리적 근거 1줄)
- 시간대 최적화: (현재 시간대의 Circadian Rhythm 특성과 적합 활동 1줄)

[근거]
- 상태 이론: (각성-동기 모델 또는 에너지 상태 분류에서 이 행동이 도출되는 신경과학적 이유 2~3줄)
- 목표 연계: (사용자 목표 유형(건강/생산성/돈/균형)의 전략 특성과 행동의 연결고리 1~2줄)
- 시간 근거: (지금 이 시간대에 이 행동이 생리적으로 최적인 이유 1줄)

[실행 방법]
- (아주 간단한 실행 방법 1~2줄)

[소요 시간]
- (예: 10분 / 20분 등)

[금지 사항]
- 여러 개의 선택지 제시 금지
- 추상적인 표현 금지
- 동기부여 문장 금지
- 사용자의 상태를 무시한 추천 금지"""


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_by_sections(text: str) -> list[dict]:
    section_pattern = re.compile(
        r"(?m)^(?P<title>[1-4]\.\s+.+?(?:\(.+?\))?)\s*\n(?P<body>.*?)(?=^[1-4]\.|^탭\s*[1-4]|\Z)",
        re.DOTALL,
    )

    tab_pattern = re.compile(
        r"탭\s*(?P<num>[1-4])\s*\n(?P<content>.*?)(?=탭\s*[1-4]|\Z)",
        re.DOTALL,
    )

    tab_labels = {
        "1": "각성-동기 상태 모델",
        "2": "개인 최적화 전략 유형",
        "3": "에너지 상태 분류",
        "4": "Circadian Rhythm 기반 시간대별 최적화",
    }

    chunks = []
    for tab_match in tab_pattern.finditer(text):
        tab_num = tab_match.group("num")
        tab_content = tab_match.group("content")
        tab_label = tab_labels.get(tab_num, f"탭{tab_num}")

        for sec_match in section_pattern.finditer(tab_content):
            title = sec_match.group("title").strip()
            body = sec_match.group("body").strip()
            if len(body) < 30:
                continue
            chunks.append({
                "tab": tab_label,
                "title": title,
                "body": body,
                "embed_text": f"[{tab_label}] {title}\n{body}",
            })

    if not chunks:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 80]
        for i, para in enumerate(paragraphs):
            chunks.append({
                "tab": "전체",
                "title": f"단락 {i+1}",
                "body": para,
                "embed_text": para,
            })

    return chunks


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


def retrieve(query_emb, chunk_embs, chunks, k=4) -> list[dict]:
    scored = sorted(
        zip([cosine_sim(query_emb, e) for e in chunk_embs], chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c for _, c in scored[:k]]


def build_context(items: list[dict]) -> str:
    parts = []
    for item in items:
        parts.append(f"[{item['tab']}] {item['title']}\n{item['body']}")
    return "\n\n---\n\n".join(parts)


def build_query(state: dict) -> str:
    mood_map = {
        "무기력": "무기력형 Low Arousal Low Motivation 행동 개시 어려움 도파민 저활성 에너지 저하",
        "스트레스": "스트레스형 High Arousal Negative Emotion 불안 편도체 과활성 전전두엽 저하",
        "안정": "안정형 Moderate Arousal Positive Emotion 세로토닌 균형 회복 부교감신경",
        "집중됨": "집중형 Optimal Arousal High Engagement Flow 도파민 노르에피네프린 전전두엽 활성",
    }
    goal_map = {
        "건강": "건강 중심형 Energy Maximizer 수면 운동 식단 회복 Self-Regulation",
        "생산성": "생산성 중심형 Time Optimizer Deep Work 집중 시간블록 전두엽",
        "돈": "돈 중심형 Capital Maximizer ROI 레버리지 도파민 보상 시스템",
        "균형": "균형 중심형 Well-being Optimizer 삶의 만족 자율성 관계 Hedonic Adaptation",
    }
    energy_map = {
        "낮음": "에너지 저하 Low Energy State 피로 회복 우선 자율신경 불균형",
        "보통": "에너지 균형 Balanced State 항상성 Homeostasis 교감 부교감 균형",
        "높음": "에너지 활성 High Energy State 도파민 보상 높은 집중력 행동 선순환",
    }

    hour = _parse_hour(state.get("현재 시간", "12:00"))
    time_ctx = _time_context(hour)

    mood_ctx = mood_map.get(state.get("기분", ""), state.get("기분", ""))
    goal_ctx = goal_map.get(state.get("목표", ""), state.get("목표", ""))
    energy_ctx = energy_map.get(state.get("에너지 상태", ""), state.get("에너지 상태", ""))

    return (
        f"{mood_ctx} {goal_ctx} {energy_ctx} {time_ctx} "
        f"상황:{state.get('현재 상황','')} "
        f"가용시간:{state.get('사용 가능 시간','')}"
    )


def _parse_hour(time_str: str) -> int:
    try:
        return int(time_str.split(":")[0])
    except Exception:
        return 12


def _time_context(hour: int) -> str:
    if 5 <= hour < 10:
        return "아침 코르티솔 각성 반응 의사결정 집중 작업 개시"
    if 10 <= hour < 13:
        return "오전 고집중 작업 구간 인지 속도 논리 학습"
    if 13 <= hour < 15:
        return "점심 후 졸음 Post-lunch Dip 낮잠 회복 생산성"
    if 15 <= hour < 18:
        return "오후 사회적 상호작용 감정 인식 의사소통"
    if 18 <= hour < 21:
        return "저녁 운동 최적화 체온 최고점 근력 유연성"
    return "밤 창의적 사고 뇌 억제 약화 Glymphatic 수면 준비"


def get_input(prompt, options=None):
    while True:
        value = input(prompt).strip()
        if not value:
            print("  입력이 필요합니다.")
            continue
        if options and value not in options:
            print(f"  선택지: {' / '.join(options)}")
            continue
        return value


def collect_state() -> dict:
    print("\n=== Decision Removal AI (RAG) ===\n")
    time_now = get_input("현재 시간 (예: 14:30): ")
    print("\n에너지 상태")
    energy = get_input("  낮음 / 보통 / 높음: ", ["낮음", "보통", "높음"])
    print("\n기분")
    mood = get_input("  무기력 / 스트레스 / 안정 / 집중됨: ", ["무기력", "스트레스", "안정", "집중됨"])
    print("\n현재 상황")
    location = get_input("  집 / 회사 / 이동중: ", ["집", "회사", "이동중"])
    print("\n목표")
    goal = get_input("  건강 / 생산성 / 돈 / 균형: ", ["건강", "생산성", "돈", "균형"])
    available_time = get_input("\n사용 가능 시간 (분, 예: 20): ")
    return {
        "현재 시간": time_now,
        "에너지 상태": energy,
        "기분": mood,
        "현재 상황": location,
        "목표": goal,
        "사용 가능 시간": f"{available_time}분",
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("지식 베이스 로딩 중...", end="", flush=True)
    raw_text = extract_pdf_text(PDF_PATH)
    chunks = chunk_by_sections(raw_text)
    chunk_embeddings = get_embeddings(client, [c["embed_text"] for c in chunks])
    print(f" {len(chunks)}개 섹션 인덱싱 완료")

    try:
        state = collect_state()

        query = build_query(state)
        query_emb = get_embeddings(client, [query])[0]
        top_chunks = retrieve(query_emb, chunk_embeddings, chunks, k=4)
        context = build_context(top_chunks)

        user_msg = (
            "[사용자 현재 상태]\n"
            + "\n".join(f"- {k}: {v}" for k, v in state.items())
            + f"\n\n[심리·신경과학 지식 베이스 - 관련 이론]\n{context}"
        )

        print("\n결정 중...\n")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=600,
        )

        print(response.choices[0].message.content)
        print()

    except KeyboardInterrupt:
        print("\n\n종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
