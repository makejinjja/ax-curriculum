#!/usr/bin/env python3
"""
decision_mood_mission.py — 기분전환 미션 + 열매 나무 시스템
"""
import os
import sys
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DATA_FILE = Path(__file__).parent / ".mission_data.json"

# ANSI
R   = "\033[91m"
BR  = "\033[38;5;130m"
SV  = "\033[97m"
GD  = "\033[93m"
GN  = "\033[32m"
BLD = "\033[1m"
RST = "\033[0m"

DIFF = {
    "하":   {"sym": "(R)", "col": R,  "card": "체력 포션 카드"},
    "중":   {"sym": "(B)", "col": BR, "card": "목재 카드"},
    "상":   {"sym": "(S)", "col": SV, "card": "다람쥐 카드"},
    "최상": {"sym": "(G)", "col": GD, "card": "골드 카드"},
}

# 난이도 확률 변경 이력:
# 기존: GPT 자체 판단 (지시 없음, 추정 하~40% 중~40% 상~15% 최상~5%)
# 변경: 하 50%, 중 30%, 상 15%, 최상 5% 로 프롬프트에 명시

# 시간대별 미션 소요 시간 범위:
# 10분 선택 → 5~15분, 30분 선택 → 20~40분, 1시간 선택 → 45~70분

SYSTEM_PROMPT = """너는 기분전환 미션 AI다.
사용자의 기분과 가용 시간을 보고 기분을 긍정적으로 바꿀 랜덤 미션 1개를 제안한다.

규칙:
- 기분이 이미 좋아도 미션 제공
- 매번 다른 미션 (최대한 랜덤하게, 이전과 겹치지 않게)
- 미션 소요 시간은 반드시 가용 시간과 일치해야 함:
  - 가용 시간 10분 → 실제 소요 5~15분인 미션만 제안
  - 가용 시간 30분 → 실제 소요 20~40분인 미션만 제안
  - 가용 시간 1시간 → 실제 소요 45~70분인 미션만 제안
- 난이도는 다음 확률로 결정해라: 하 50%, 중 30%, 상 15%, 최상 5%
- 난이도 기준:
  - 하: 매우 쉬운 즉각적 행동 (물 한 잔, 스트레칭 1분 등)
  - 중: 약간의 노력 필요 (10분 산책, 좋아하는 노래 듣기 등)
  - 상: 집중과 노력 필요 (25분 운동, 일기 쓰기 등)
  - 최상: 강한 의지력 필요 (찬물 샤워, 완전 디지털 디톡스 등)

반드시 아래 형식으로만 출력:

[미션]
- (구체적 행동 1개, 10~30자 이내)

[난이도]
- 하 또는 중 또는 상 또는 최상

[효과]
- (성공 시 심리·신체 효과 1~2줄)"""


# ── 데이터 저장 ─────────────────────────────────────────────

def load_data() -> dict:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return {"fruits": [], "cards": []}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 나무 UI ─────────────────────────────────────────────────

def fruit_sym(difficulty: str) -> str:
    info = DIFF.get(difficulty, {"col": "", "sym": "(?)"})
    return f"{info['col']}{info['sym']}{RST}"


def render_tree(fruits: list):
    def s(i):
        return fruit_sym(fruits[i]["difficulty"]) if i < len(fruits) else "   "

    print(f"\n{GN}{'━'*38}{RST}")
    print(f"  {BLD}🌳 나무   [{len(fruits)}/10 열매]{RST}")
    print(f"{GN}{'━'*38}{RST}")
    print(f"                {s(0)}")
    print(f"             {s(1)}   {s(2)}")
    print(f"          {s(3)}   {s(4)}   {s(5)}")
    print(f"       {s(6)}  {s(7)}  {s(8)}  {s(9)}")
    print(f"          ━━━━━━━━━━━━━")
    print(f"               ┃┃┃")
    print(f"           ~~~~~~~~~~~")
    print(f"{GN}{'━'*38}{RST}")
    print(f"  {R}(R){RST}=하  {BR}(B){RST}=중  {SV}(S){RST}=상  {GD}(G){RST}=최상")


# ── GPT 미션 ─────────────────────────────────────────────────

def get_mission(client: OpenAI, mood: str, time_str: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"현재 기분: {mood}\n가용 시간: {time_str}"},
        ],
        temperature=0.9,
        max_tokens=200,
    )
    return resp.choices[0].message.content


def normalize_difficulty(text: str) -> str:
    for d in ["최상", "상", "중", "하"]:
        if d in text:
            return d
    return "하"


def parse_mission(text: str) -> dict:
    def extract(tag):
        m = re.search(rf"\[{tag}\]\s*\n-\s*(.+?)(?=\n\[|\Z)", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    raw_diff = extract("난이도")
    return {
        "mission":    extract("미션"),
        "difficulty": normalize_difficulty(raw_diff),
        "effect":     extract("효과"),
    }


def print_mission(m: dict):
    info = DIFF.get(m["difficulty"], {"col": "", "sym": ""})
    print(f"\n{'─'*38}")
    print(f"  {BLD}📋 미션{RST}")
    print(f"  {m['mission']}")
    print(f"\n  {BLD}난이도{RST}: {info['col']}{m['difficulty']} {info['sym']}{RST}")
    print(f"\n  {BLD}효과{RST}")
    print(f"  {m['effect']}")
    print(f"{'─'*38}")


# ── 카드 UI ─────────────────────────────────────────────────

def show_card(card_name: str, difficulty: str):
    col = DIFF.get(difficulty, {"col": ""})["col"]
    print(f"\n{col}┌───────────────────┐{RST}")
    print(f"{col}│                   │{RST}")
    print(f"{col}│   ✦  카드 획득  ✦  │{RST}")
    print(f"{col}│                   │{RST}")
    print(f"{col}│  {card_name:<17} │{RST}")
    print(f"{col}│                   │{RST}")
    print(f"{col}└───────────────────┘{RST}")


def show_cards(cards: list):
    print(f"\n  {BLD}보유 카드{RST}")
    if not cards:
        print("  없음")
        return
    for card, cnt in Counter(c["card"] for c in cards).items():
        print(f"  • {card}  x{cnt}")


# ── 열매 쪼개기 ──────────────────────────────────────────────

def split_fruit_menu(data: dict):
    fruits = data["fruits"]
    if not fruits:
        print("\n  쪼갤 열매가 없습니다.")
        return

    print("\n  쪼갤 열매 번호를 선택하세요 (0 = 취소)")
    for i, f in enumerate(fruits, 1):
        d    = f["difficulty"]
        info = DIFF.get(d, {"col": "", "sym": "(?)"})
        print(f"  {i}. {info['col']}{info['sym']}{RST} [{d}] {f.get('mission','')[:22]}")

    while True:
        sel = input("  > ").strip()
        if sel == "0":
            return
        if sel.isdigit() and 1 <= int(sel) <= len(fruits):
            fruit = fruits.pop(int(sel) - 1)
            d     = fruit["difficulty"]
            card  = DIFF[d]["card"]
            data["cards"].append({"card": card, "difficulty": d})
            save_data(data)
            show_card(card, d)
            print(f"\n  {card}를 획득했습니다!")
            return
        print(f"  1~{len(fruits)} 또는 0을 입력하세요.")


# ── 입력 수집 ────────────────────────────────────────────────

def collect_input() -> dict:
    print(f"\n{'='*38}")
    print(f"  {BLD}기분전환 미션 AI{RST}")
    print(f"{'='*38}")

    now = datetime.now()
    print(f"\n  현재 시간: {now.strftime('%H:%M')}")

    print("\n지금 기분을 자유롭게 입력하세요")
    while True:
        mood = input("  > ").strip()
        if mood:
            break
        print("  입력이 필요합니다.")

    print("\n사용 가능 시간")
    opts = ["10분", "30분", "1시간"]
    for i, t in enumerate(opts, 1):
        print(f"  {i}. {t}")
    while True:
        sel = input("  선택 (1/2/3): ").strip()
        if sel in ("1", "2", "3"):
            time_str = opts[int(sel) - 1]
            break
        print("  1, 2, 3 중에 선택하세요.")

    return {"mood": mood, "time": time_str}


# ── 미션 루프 ────────────────────────────────────────────────

def mission_loop(client: OpenAI, mood: str, time_str: str) -> dict | None:
    while True:
        print("\n  미션 생성 중...", end="", flush=True)
        raw = get_mission(client, mood, time_str)
        m   = parse_mission(raw)
        print(f"\r{' '*20}\r", end="")

        if not m["mission"]:
            print("  미션 생성에 실패했습니다. 재시도합니다.")
            continue

        print_mission(m)

        print("\n  1. 수락   2. 다른 미션   3. 종료")
        sel = input("  선택: ").strip()
        if sel == "1":
            return m
        if sel == "3":
            return None


# ── 시작 메뉴 (나무 보기 / 쪼개기) ─────────────────────────────

def start_menu(data: dict):
    print(f"\n{'='*38}")
    print(f"  {BLD}기분전환 미션 AI{RST}")
    print(f"{'='*38}")
    print(f"\n  현재 시간: {datetime.now().strftime('%H:%M')}")

    render_tree(data["fruits"])
    show_cards(data["cards"])

    print(f"\n  1. 미션 시작")
    print(f"  2. 열매 쪼개기")
    print(f"  3. 종료")

    while True:
        sel = input("  선택 (1/2/3): ").strip()
        if sel == "1":
            return "mission"
        if sel == "2":
            split_fruit_menu(data)
            render_tree(data["fruits"])
            show_cards(data["cards"])
            return start_menu(data)  # 쪼개기 후 메뉴로 복귀
        if sel == "3":
            return "exit"
        print("  1, 2, 3 중에 선택하세요.")


# ── main ─────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    data   = load_data()

    action = start_menu(data)
    if action == "exit":
        print("\n  종료합니다.\n")
        sys.exit(0)

    # [2] 열매 10개 시 미션 진행 차단
    if len(data["fruits"]) >= 10:
        print(f"\n  {BLD}🌳 나무가 가득 찼습니다! 열매를 먼저 쪼개야 미션을 시작할 수 있습니다.{RST}")
        while len(data["fruits"]) >= 10:
            split_fruit_menu(data)
            render_tree(data["fruits"])
            show_cards(data["cards"])
            if len(data["fruits"]) < 10:
                print("\n  자리가 생겼습니다. 미션을 시작합니다!")
                break
            print("\n  아직 나무가 가득 찼습니다. 열매를 더 쪼개주세요.")

    state = collect_input()

    mission = mission_loop(client, state["mood"], state["time"])
    if not mission:
        print("\n  종료합니다.\n")
        sys.exit(0)

    print(f"\n  미션을 시작하세요!")
    print(f"  완료 후 Enter를 눌러 성공을 기록합니다.")
    input(f"\n  ▶ 미션 성공 → Enter ")

    data["fruits"].append({
        "difficulty": mission["difficulty"],
        "mission":    mission["mission"],
    })
    save_data(data)
    info = DIFF[mission["difficulty"]]
    print(f"\n  열매 획득! {info['col']}{info['sym']}{RST} [{mission['difficulty']}]")

    render_tree(data["fruits"])
    show_cards(data["cards"])

    if data["fruits"]:
        print("\n  열매를 쪼개시겠습니까? (y/n)")
        if input("  > ").strip().lower() == "y":
            split_fruit_menu(data)
            render_tree(data["fruits"])
            show_cards(data["cards"])

    print("\n  오늘도 잘 하셨습니다! 👍\n")


if __name__ == "__main__":
    main()
