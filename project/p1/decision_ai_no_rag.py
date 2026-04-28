#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SYSTEM_PROMPT = """너는 "Decision Removal AI"다.
사용자의 현재 상태를 기반으로 "지금 당장 해야 할 행동 1개"를 결정해주는 시스템이다.

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
1. 사용자의 에너지와 시간 기반으로 가능한 행동 필터링
2. 목표와 가장 직접적으로 연결되는 행동 선택
3. 실행 난이도가 낮고 즉시 가능한 행동 우선 선택
4. 행동 완료 시 체감 효과가 높은 것 선택

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
- 설명이 길어지는 것 금지
- 사용자의 상태를 무시한 추천 금지"""


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


def collect_state():
    print("\n=== Decision Removal AI ===\n")

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


def build_user_message(state):
    lines = [f"- {k}: {v}" for k, v in state.items()]
    return "\n".join(lines)


def get_decision(state):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    user_message = build_user_message(state)

    print("\n결정 중...\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content


def main():
    try:
        state = collect_state()
        decision = get_decision(state)
        print(decision)
        print()
    except KeyboardInterrupt:
        print("\n\n종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
