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

PDF_PATH = Path(__file__).parent.parent / "data" / "knowledge.txt.pdf"

SYSTEM_PROMPT = """너는 "Decision Removal AI"다.
아래 [지식 베이스]에서 검색된 행동 데이터를 참고하여
사용자의 현재 상태에 가장 적합한 "지금 당장 해야 할 행동 1개"를 결정해주는 시스템이다.

[핵심 원칙]
1. 항상 단 하나의 행동만 제시한다.
2. 행동은 5~30분 내 실행 가능한 것이어야 한다.
3. 모호한 조언 금지
4. 실행 가능성이 가장 높은 행동을 선택한다.
5. 사용자의 현재 상태를 최우선으로 반영한다.
6. 장기 계획보다 "지금 즉시 행동"에 집중한다.

[출력 형식] 반드시 아래 형식으로만 출력한다:

[행동]
- (지금 당장 해야 할 단 하나의 행동)

[이유]
- (이 행동이 최적인 이유를 한 줄로 설명)

[실행 방법]
- (아주 간단한 실행 방법 1~2줄)

[소요 시간]
- (예: 10분 / 20분 등)

[금지 사항]
- 여러 개의 선택지 제시 금지
- 추상적인 표현 금지
- 동기부여 문장 금지
- 설명이 길어지는 것 금지"""


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_chunks(text: str) -> list[dict]:
    def field(name, src):
        m = re.search(rf"\[{name}\]\s*(.*?)(?=\[|\Z)", src, re.DOTALL)
        return m.group(1).strip() if m else ""

    actions = []
    for chunk in re.split(r"={10,}", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        action = field("Action", chunk)
        if not action:
            continue
        actions.append({
            "action":           action,
            "category":         field("Category", chunk),
            "best_for":         field("Best For", chunk),
            "why_now":          field("Why Now", chunk),
            "expected_outcome": field("Expected Outcome", chunk),
            "how":              field("How", chunk),
            "keywords":         field("Keywords", chunk),
        })
    return actions


def embed_text(item: dict) -> str:
    return (
        f"행동: {item['action']}\n"
        f"카테고리: {item['category']}\n"
        f"적합조건: {item['best_for']}\n"
        f"이유: {item['why_now']}\n"
        f"키워드: {item['keywords']}"
    )


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query_emb, chunk_embs, chunks, k=3) -> list[dict]:
    scored = sorted(
        zip([cosine_sim(query_emb, e) for e in chunk_embs], chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c for _, c in scored[:k]]


def build_context(items: list[dict]) -> str:
    parts = []
    for item in items:
        parts.append(
            f"[행동] {item['action']}\n"
            f"[적합 조건] {item['best_for']}\n"
            f"[이유] {item['why_now']}\n"
            f"[실행법] {item['how']}\n"
            f"[예상 결과] {item['expected_outcome']}"
        )
    return "\n---\n".join(parts)


def get_input(prompt, options=None):
    while True:
        val = input(prompt).strip()
        if not val:
            print("  입력이 필요합니다.")
            continue
        if options and val not in options:
            print(f"  선택지: {' / '.join(options)}")
            continue
        return val


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
    minutes = get_input("\n사용 가능 시간 (분, 예: 20): ")
    return {
        "현재 시간": time_now,
        "에너지 상태": energy,
        "기분": mood,
        "현재 상황": location,
        "목표": goal,
        "사용 가능 시간": f"{minutes}분",
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("지식 베이스 로딩 중...", end="", flush=True)
    text = extract_pdf_text(PDF_PATH)
    chunks = parse_chunks(text)
    chunk_embeddings = get_embeddings(client, [embed_text(c) for c in chunks])
    print(f" {len(chunks)}개 행동 인덱싱 완료")

    try:
        state = collect_state()

        query = (
            f"에너지:{state['에너지 상태']} 기분:{state['기분']} "
            f"상황:{state['현재 상황']} 목표:{state['목표']} "
            f"가용시간:{state['사용 가능 시간']}"
        )
        query_emb = get_embeddings(client, [query])[0]
        top = retrieve(query_emb, chunk_embeddings, chunks, k=3)
        context = build_context(top)

        user_msg = (
            "[사용자 현재 상태]\n"
            + "\n".join(f"- {k}: {v}" for k, v in state.items())
            + f"\n\n[지식 베이스 - 관련 행동 후보]\n{context}"
        )

        print("\n결정 중...\n")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        print(response.choices[0].message.content)
        print()

    except KeyboardInterrupt:
        print("\n\n종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
