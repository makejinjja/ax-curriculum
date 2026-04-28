#!/usr/bin/env python3
"""
mood_mission_app.py — 기분전환 미션 Streamlit UI
"""
import os
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DATA_FILE = Path(__file__).parent / ".mission_data.json"

DIFF = {
    "하":   {"color": "#e74c3c", "bg": "#fdecea", "card": "체력 포션 카드",  "emoji": "🍎", "label": "하"},
    "중":   {"color": "#cd7f32", "bg": "#fef3e2", "card": "목재 카드",       "emoji": "🟤", "label": "중"},
    "상":   {"color": "#7f8c8d", "bg": "#f2f3f4", "card": "다람쥐 카드",     "emoji": "🩶", "label": "상"},
    "최상": {"color": "#d4ac0d", "bg": "#fefbd8", "card": "골드 카드",       "emoji": "⭐", "label": "최상"},
}

FRUIT_POSITIONS = [
    (210, 82),
    (170, 128), (250, 124),
    (130, 174), (210, 164), (282, 170),
    (98,  216), (165, 209), (240, 206), (305, 213),
]

SYSTEM_PROMPT = """너는 기분전환 미션 AI다.
사용자의 기분과 가용 시간을 보고 기분을 긍정적으로 바꿀 랜덤 미션 1개를 제안한다.

규칙:
- 기분이 이미 좋아도 미션 제공
- 매번 다른 미션 (최대한 랜덤하게)
- 가용 시간 내에 완료 가능해야 함
- 난이도 기준:
  - 하: 매우 쉬운 즉각적 행동 (물 한 잔, 스트레칭 1분 등)
  - 중: 약간의 노력 필요 (10분 산책, 노래 듣기 등)
  - 상: 집중과 노력 필요 (25분 운동, 일기 쓰기 등)
  - 최상: 강한 의지력 필요 (찬물 샤워, 디지털 디톡스 등)

반드시 아래 형식으로만 출력:

[미션]
- (구체적 행동 1개, 10~30자 이내)

[난이도]
- 하 또는 중 또는 상 또는 최상

[효과]
- (성공 시 심리·신체 효과 1~2줄)"""


# ── 페이지 설정 ───────────────────────────────────────────────

st.set_page_config(
    page_title="🌳 기분전환 미션",
    page_icon="🌳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.stApp { background: #f7f9f7; }

section[data-testid="stSidebar"] { display: none; }

h1 { color: #2d5a27 !important; text-align: center; }

.mission-card {
    background: white;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border-left: 5px solid #4caf50;
}
.mission-title {
    font-size: 1.3em;
    font-weight: 700;
    color: #1a3a16;
    margin-bottom: 8px;
}
.effect-box {
    background: #f0f7ee;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 14px;
    color: #3a6b35;
    font-size: 0.95em;
    line-height: 1.6;
}
.diff-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82em;
    font-weight: 700;
    margin-bottom: 16px;
}
.card-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
}
.card-item {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    min-width: 130px;
    flex: 1;
}
.card-emoji { font-size: 2em; margin-bottom: 6px; }
.card-name { font-weight: 600; font-size: 0.9em; color: #333; margin-bottom: 4px; }
.card-count { font-size: 1.5em; font-weight: 700; color: #2d5a27; }

.new-card-banner {
    background: linear-gradient(135deg, #fff9c4, #fff3cd);
    border: 2px solid #f1c40f;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin: 16px 0;
    animation: glow 1s ease-in-out;
}
@keyframes glow {
    0% { box-shadow: 0 0 0 rgba(241,196,15,0); }
    50% { box-shadow: 0 0 20px rgba(241,196,15,0.5); }
    100% { box-shadow: 0 0 0 rgba(241,196,15,0); }
}

.tree-label {
    text-align: center;
    font-size: 1em;
    color: #5a8a55;
    margin-bottom: 4px;
    font-weight: 500;
}

div[data-testid="stButton"] > button {
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Noto Sans KR', sans-serif;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)


# ── 데이터 ───────────────────────────────────────────────────

def load_data() -> dict:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return {"fruits": [], "cards": []}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── GPT ──────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        st.stop()
    return OpenAI(api_key=api_key)


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


def fetch_mission(mood: str, time_str: str) -> dict:
    client = get_client()
    with st.spinner("미션 생성 중..."):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"현재 기분: {mood}\n가용 시간: {time_str}"},
            ],
            temperature=0.9,
            max_tokens=200,
        )
    return parse_mission(resp.choices[0].message.content)


# ── SVG 나무 ─────────────────────────────────────────────────

def build_fruit(x: int, y: int, color: str) -> str:
    return f"""
    <circle cx="{x}" cy="{y+2}" r="13" fill="rgba(0,0,0,0.12)"/>
    <circle cx="{x}" cy="{y}" r="13" fill="{color}" stroke="rgba(0,0,0,0.2)" stroke-width="1.2"/>
    <ellipse cx="{x+4}" cy="{y-5}" rx="4" ry="3" fill="rgba(255,255,255,0.25)"/>
    <rect x="{x-1}" y="{y-22}" width="3" height="9" fill="#5d4037" rx="1"/>
    <ellipse cx="{x+6}" cy="{y-22}" rx="6" ry="3.5" fill="#66bb6a"
             transform="rotate(-25 {x+6} {y-22})"/>
    """


def tree_svg(fruits: list) -> str:
    fruit_els = ""
    for i, (fx, fy) in enumerate(FRUIT_POSITIONS):
        if i < len(fruits):
            d = fruits[i]["difficulty"]
            color = DIFF.get(d, {"color": "#888"})["color"]
            fruit_els += build_fruit(fx, fy, color)
        else:
            fruit_els += f'<circle cx="{fx}" cy="{fy}" r="11" fill="none" stroke="rgba(100,150,100,0.25)" stroke-width="1.5" stroke-dasharray="5 3"/>'

    return f"""
    <svg width="420" height="310" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="c1" cx="45%" cy="35%">
          <stop offset="0%" stop-color="#81c784"/>
          <stop offset="100%" stop-color="#2e7d32"/>
        </radialGradient>
        <radialGradient id="c2" cx="45%" cy="35%">
          <stop offset="0%" stop-color="#66bb6a"/>
          <stop offset="100%" stop-color="#1b5e20"/>
        </radialGradient>
        <filter id="shadow">
          <feDropShadow dx="0" dy="3" stdDeviation="4" flood-opacity="0.15"/>
        </filter>
      </defs>

      <!-- Ground -->
      <ellipse cx="210" cy="296" rx="80" ry="12" fill="#a5d6a7" opacity="0.4"/>

      <!-- Trunk -->
      <rect x="188" y="245" width="44" height="52" fill="#6d4c41" rx="4" filter="url(#shadow)"/>
      <rect x="195" y="245" width="10" height="52" fill="rgba(255,255,255,0.08)" rx="2"/>

      <!-- Canopy layers -->
      <ellipse cx="210" cy="218" rx="115" ry="68" fill="url(#c1)" filter="url(#shadow)"/>
      <ellipse cx="210" cy="178" rx="95"  ry="62" fill="url(#c1)"/>
      <ellipse cx="210" cy="143" rx="76"  ry="52" fill="url(#c2)"/>
      <ellipse cx="210" cy="112" rx="57"  ry="42" fill="#388e3c"/>
      <ellipse cx="210" cy="84"  rx="38"  ry="30" fill="#2e7d32"/>

      <!-- Highlight -->
      <ellipse cx="190" cy="120" rx="22" ry="35" fill="rgba(255,255,255,0.06)"/>

      <!-- Fruits -->
      {fruit_els}
    </svg>
    """


# ── 세션 상태 초기화 ─────────────────────────────────────────

def init_state():
    defaults = {
        "phase":     "input",    # input | mission | in_progress | result
        "mood":      "",
        "time_str":  "10분",
        "mission":   None,
        "data":      load_data(),
        "new_card":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── UI 컴포넌트 ──────────────────────────────────────────────

def render_mission_card(m: dict):
    d    = m["difficulty"]
    info = DIFF.get(d, {"color": "#4caf50", "bg": "#f0f9f0", "emoji": "🌿", "label": d})
    st.markdown(f"""
    <div class="mission-card">
        <div class="diff-badge" style="background:{info['bg']};color:{info['color']};">
            {info['emoji']} 난이도 {info['label']}
        </div>
        <div class="mission-title">📋 {m['mission']}</div>
        <div class="effect-box">✨ {m['effect']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_tree_section():
    fruits = st.session_state.data["fruits"]
    st.markdown(f'<div class="tree-label">🌳 나무 · {len(fruits)} / 10 열매</div>', unsafe_allow_html=True)
    st.markdown(tree_svg(fruits), unsafe_allow_html=True)

    if fruits:
        st.markdown("**열매 색상 안내**")
        cols = st.columns(4)
        labels = [("🔴 하", "#e74c3c"), ("🟤 중", "#cd7f32"), ("⚪ 상", "#7f8c8d"), ("🟡 최상", "#d4ac0d")]
        for col, (label, color) in zip(cols, labels):
            col.markdown(f"<span style='color:{color};font-weight:600'>{label}</span>", unsafe_allow_html=True)


def render_cards_section():
    cards = st.session_state.data["cards"]
    st.markdown("---")
    st.markdown("### 🃏 보유 카드")

    if not cards:
        st.markdown("<span style='color:#aaa'>아직 카드가 없습니다.</span>", unsafe_allow_html=True)
        return

    counts = Counter(c["card"] for c in cards)
    diff_by_card = {info["card"]: (d, info) for d, info in DIFF.items()}

    html = '<div class="card-grid">'
    for card_name, cnt in counts.items():
        d, info = diff_by_card.get(card_name, ("하", DIFF["하"]))
        html += f"""
        <div class="card-item">
            <div class="card-emoji">{info['emoji']}</div>
            <div class="card-name">{card_name}</div>
            <div class="card-count">x{cnt}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_split_section():
    fruits = st.session_state.data["fruits"]
    if not fruits:
        return

    st.markdown("---")
    st.markdown("### 🔓 열매 쪼개기")

    options = {
        f"열매 {i+1} — [{f['difficulty']}] {DIFF[f['difficulty']]['emoji']} {f['mission'][:18]}": i
        for i, f in enumerate(fruits)
    }
    selected_label = st.selectbox("쪼갤 열매 선택", list(options.keys()))
    if st.button("쪼개기 🔨", type="primary"):
        idx   = options[selected_label]
        fruit = st.session_state.data["fruits"].pop(idx)
        d     = fruit["difficulty"]
        card  = DIFF[d]["card"]
        st.session_state.data["cards"].append({"card": card, "difficulty": d})
        save_data(st.session_state.data)
        st.session_state.new_card = {"card": card, "difficulty": d}
        st.rerun()


# ── 페이지 렌더링 ────────────────────────────────────────────

def page_input():
    st.markdown("## 🌳 기분전환 미션 AI")
    st.markdown(f"<span style='color:#888;font-size:0.9em'>현재 시간: {datetime.now().strftime('%H:%M')}</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    mood = st.text_area(
        "지금 기분을 자유롭게 입력하세요",
        placeholder="예: 머리가 무겁고 의욕이 없어요. 오후 내내 집중이 안 돼요.",
        height=100,
    )

    time_str = st.radio(
        "사용 가능 시간",
        ["10분", "30분", "1시간"],
        horizontal=True,
    )

    st.markdown("")
    if st.button("✨ 미션 받기", type="primary", use_container_width=True):
        if not mood.strip():
            st.warning("기분을 입력해 주세요.")
        else:
            st.session_state.mood     = mood.strip()
            st.session_state.time_str = time_str
            st.session_state.mission  = None
            st.session_state.phase    = "mission"
            st.rerun()


def page_mission():
    st.markdown("## 📋 오늘의 미션")
    st.markdown(f"**기분:** {st.session_state.mood}")
    st.markdown(f"**가용 시간:** {st.session_state.time_str}")
    st.markdown("---")

    if st.session_state.mission is None:
        m = fetch_mission(st.session_state.mood, st.session_state.time_str)
        if not m["mission"]:
            st.error("미션 생성에 실패했습니다. 다시 시도해 주세요.")
            if st.button("다시 시도"):
                st.rerun()
            return
        st.session_state.mission = m

    render_mission_card(st.session_state.mission)

    st.markdown("")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("✅ 수락", type="primary", use_container_width=True):
            st.session_state.phase = "in_progress"
            st.rerun()
    with col2:
        if st.button("🔄 다른 미션", use_container_width=True):
            st.session_state.mission = None
            st.rerun()
    with col3:
        if st.button("✕ 종료"):
            st.session_state.phase = "input"
            st.rerun()


def page_in_progress():
    st.markdown("## 🏃 미션 진행 중")
    st.markdown("---")

    render_mission_card(st.session_state.mission)

    st.info("미션을 완료했으면 아래 버튼을 눌러주세요!")
    st.markdown("")

    if st.button("🎉 미션 성공!", type="primary", use_container_width=True):
        data = st.session_state.data
        if len(data["fruits"]) >= 10:
            st.warning("🌳 나무가 가득 찼습니다! 열매를 먼저 쪼개주세요.")
        else:
            data["fruits"].append({
                "difficulty": st.session_state.mission["difficulty"],
                "mission":    st.session_state.mission["mission"],
            })
            save_data(data)
            st.session_state.phase   = "result"
            st.session_state.new_card = None
            st.rerun()

    if st.button("← 이전으로"):
        st.session_state.phase = "mission"
        st.rerun()


def page_result():
    st.markdown("## 🌳 나의 나무")
    st.markdown("---")

    # 새 열매 획득 메시지
    latest = st.session_state.data["fruits"][-1] if st.session_state.data["fruits"] else None
    if latest:
        d    = latest["difficulty"]
        info = DIFF[d]
        st.success(f"열매 획득! {info['emoji']} **{d}** 난이도 열매가 나무에 달렸습니다.")

    # 새 카드 획득 메시지
    if st.session_state.new_card:
        nc   = st.session_state.new_card
        info = DIFF[nc["difficulty"]]
        st.markdown(f"""
        <div class="new-card-banner">
            <div style="font-size:2.5em">{info['emoji']}</div>
            <div style="font-size:1.2em;font-weight:700;color:#8b6914;margin:8px 0">카드 획득!</div>
            <div style="font-size:1em;color:#555">{nc['card']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.new_card = None

    render_tree_section()
    render_cards_section()
    render_split_section()

    st.markdown("---")
    if st.button("🌟 새 미션 받기", type="primary", use_container_width=True):
        st.session_state.phase   = "input"
        st.session_state.mission = None
        st.rerun()


# ── 진입점 ───────────────────────────────────────────────────

init_state()

phase = st.session_state.phase
if phase == "input":
    page_input()
elif phase == "mission":
    page_mission()
elif phase == "in_progress":
    page_in_progress()
elif phase == "result":
    page_result()
