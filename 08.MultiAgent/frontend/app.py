"""
app.py — Bloom Multi-Agent Streamlit 프론트엔드
"""
from __future__ import annotations
import os
import re
from collections import Counter
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Bloom Multi-Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
* { font-family: 'Noto Sans KR', sans-serif !important; }

[data-testid="stSidebar"] { background-color: #0a0a0a; }
[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1a1a2e; color: #ffffff; border: 1px solid #444;
    border-radius: 8px; width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #333; color: #fff;
}

.chat-user {
    background: #2d6a4f; color: #ffffff;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px; margin: 6px 0;
    max-width: 75%; margin-left: auto;
    box-shadow: 0 2px 6px rgba(0,0,0,.15);
}
.chat-agent {
    background: #f4f6f8; color: #1a1a1a;
    border-radius: 14px 14px 14px 4px;
    padding: 12px 16px; margin: 6px 0;
    max-width: 85%;
    box-shadow: 0 2px 6px rgba(0,0,0,.08);
}
.mission-card {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-left: 5px solid #2d6a4f;
    border-radius: 12px; padding: 20px 24px; margin: 12px 0;
}
.wildcard-card {
    background: linear-gradient(135deg, #fffbea 0%, #ffeaa7 100%);
    border-left: 5px solid #f39c12;
    border-radius: 12px; padding: 20px 24px; margin: 12px 0;
}
.mission-title { font-size: 18px; font-weight: 700; color: #1b5e20; margin-bottom: 12px; }
.mission-row { margin: 6px 0; font-size: 14px; color: #2d3436; }
.insight-box {
    background: #f0f4ff; border-left: 4px solid #3498db;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
    font-size: 14px; color: #2c3e50;
}
.combo-banner {
    background: #111; color: #f1c40f; text-align: center;
    padding: 8px 12px; border-radius: 8px; font-weight: 700;
    font-size: 14px; margin: 8px 0; letter-spacing: 1px;
}
.tool-badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 10px; font-size: 11px; font-weight: 500;
    background: #e3f2fd; color: #1565c0; margin-right: 4px;
}
.delegate-badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 10px; font-size: 11px; font-weight: 500;
    background: #ede7f6; color: #4527a0; margin-right: 4px;
}
.stat-box {
    background: #f9f9f9; border: 1px solid #ddd; border-radius: 10px;
    padding: 14px; text-align: center;
}
.stat-num { font-size: 2rem; font-weight: 700; color: #111; }
.stat-lbl { font-size: 0.78rem; color: #777; margin-top: 2px; }
.tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin-right: 4px;
}
.ts { font-size: 10px; color: #aaa; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── 상수 ────────────────────────────────────────────────────
CAT_COLOR = {
    "건강": "#e74c3c", "생산성": "#3498db",
    "재미": "#9b59b6", "성장": "#27ae60", "돌발": "#f39c12",
}
CAT_SYM  = {"건강": "💪", "생산성": "📋", "재미": "🎮", "성장": "📚", "돌발": "⚡"}
DIFF_SYM = {"하": "🌱", "중": "🌿", "상": "🌟", "최상": "🏆", "돌발": "⚡"}
DIFF_COLOR = {
    "하": "#27ae60", "중": "#8B4513", "상": "#bdc3c7",
    "최상": "#f1c40f", "돌발": "#f39c12",
}

# Delegate tools (orchestrator level) + specialist sub-tools
DELEGATE_TOOLS = {
    "delegate_mission_agent":    "🎯 미션에이전트",
    "delegate_psychology_agent": "🧠 심리에이전트",
    "delegate_search_agent":     "🔍 검색에이전트",
}
TOOL_ICONS = {
    "generate_mission":           "🎯 미션생성",
    "validate_mission":           "✅ 검증",
    "get_insight":                "🔬 인사이트",
    "get_motivational_nudge":     "💬 격려",
    "rag_search":                 "🔍 RAG",
    "web_search":                 "🌐 웹",
}
MAX_FRUITS = 30


# ── 세션 상태 초기화 ─────────────────────────────────────────
def _init():
    defaults = {
        "token":           None,
        "username":        None,
        "session_id":      None,
        "messages":        [],
        "current_mission": None,
        "awaiting_result": False,
        "user_data":       {"fruits": [], "cards": [], "combo_count": 0, "last_category": None},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init()


# ── API 헬퍼 ─────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state['token']}"}


def api_login(username: str, password: str) -> str | None:
    try:
        r = requests.post(
            f"{BACKEND_URL}/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("access_token")
        st.error(r.json().get("detail", "로그인 실패"))
    except requests.exceptions.ConnectionError:
        st.error("백엔드 서버에 연결할 수 없습니다.")
    except Exception as exc:
        st.error(str(exc))
    return None


def api_chat(message: str, available_minutes: int) -> dict | None:
    try:
        r = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "message":           message,
                "session_id":        st.session_state["session_id"],
                "available_minutes": available_minutes,
            },
            headers=_headers(),
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            st.session_state["token"] = None
            st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
        else:
            st.error(f"서버 오류 ({r.status_code}): {r.text[:200]}")
    except requests.exceptions.Timeout:
        st.error("응답 시간이 초과되었습니다. 다시 시도하세요.")
    except requests.exceptions.ConnectionError:
        st.error("백엔드 서버에 연결할 수 없습니다.")
    except Exception as exc:
        st.error(str(exc))
    return None


def api_health() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "offline", "rag_index_loaded": False}


# ── 미션 카드 렌더 ───────────────────────────────────────────

def _render_mission_card(mission: dict) -> str:
    cat   = mission.get("category", "")
    diff  = mission.get("difficulty", "")
    is_wc = mission.get("is_wildcard", False) or cat == "돌발"
    css   = "wildcard-card" if is_wc else "mission-card"
    title = "⚡ 돌발 미션!" if is_wc else "🎯 미션"
    return f"""
<div class="{css}">
  <div class="mission-title">{title}: {mission.get('mission', '')}</div>
  <div class="mission-row">
    {CAT_SYM.get(cat,'🏷️')} 카테고리: <strong>{cat}</strong>
    &nbsp;|&nbsp;
    {DIFF_SYM.get(diff,'⭐')} 난이도: <strong>{diff}</strong>
  </div>
  <div class="mission-row">📚 근거: {mission.get('basis','')}</div>
  <div class="mission-row">✨ 효과: {mission.get('effect','')}</div>
</div>
"""


def _format_agent_content(text: str, mission: dict | None, tools: list[str]) -> str:
    parts = []

    if mission and "[미션]" in text:
        before = re.split(r"\[미션\]", text, maxsplit=1)[0].strip()
        after_match = re.search(r"\[효과\][^\[]*", text, re.DOTALL)
        after = ""
        if after_match:
            raw = text[after_match.end():].strip()
            if raw:
                after = raw
        if before:
            parts.append(f"<p>{before.replace(chr(10),'<br>')}</p>")
        parts.append(_render_mission_card(mission))
        if after:
            parts.append(f"<p>{after.replace(chr(10),'<br>')}</p>")
    else:
        parts.append(f"<p>{text.replace(chr(10),'<br>')}</p>")

    if "get_insight" in tools:
        insight_match = re.search(r"(🔬[^\n]+|인사이트[^\n]+)", text)
        if not insight_match:
            insight_match = re.search(r"(이론|기법|연구|심리)[^\n]{10,}", text)
        if insight_match:
            parts.append(f'<div class="insight-box">🔬 {insight_match.group(0).strip()}</div>')

    return "".join(parts)


def _render_tool_badges(tools: list[str]) -> str:
    """Delegate 툴은 보라색 배지, 전문가 서브 툴은 파란색 배지로 분리 표시."""
    badges = []
    seen = set()
    for t in tools:
        if t in seen:
            continue
        seen.add(t)
        if t in DELEGATE_TOOLS:
            badges.append(f'<span class="delegate-badge">{DELEGATE_TOOLS[t]}</span>')
        elif t in TOOL_ICONS:
            badges.append(f'<span class="tool-badge">{TOOL_ICONS[t]}</span>')
    return "".join(badges)


# ── 시각화 ───────────────────────────────────────────────────

def render_tree_chart(fruits: list):
    if not fruits:
        st.info("아직 열매가 없습니다. 첫 번째 미션을 완료해보세요!")
        return

    n   = len(fruits)
    cap = 30 if n > 20 else (20 if n > 10 else 10)
    label = "🌲 완전한 나무" if cap == 30 else ("🌿 성장 나무" if cap == 20 else "🌱 새싹 나무")

    rows = [1, 2, 3, 4, 5, 5, 5, 5]
    x_list, y_list, colors, texts, hover = [], [], [], [], []
    idx = 0
    for row_idx, row_count in enumerate(rows):
        y = 8 - row_idx
        start_x = -(row_count - 1) / 2
        for col in range(row_count):
            if idx >= cap:
                break
            x_list.append(start_x + col)
            y_list.append(y)
            if idx < n:
                cat = fruits[idx].get("category", "건강")
                colors.append(CAT_COLOR.get(cat, "#999"))
                texts.append(CAT_SYM.get(cat, "●"))
                hover.append(f"[{cat}] {fruits[idx].get('mission', '')[:20]}")
            else:
                colors.append("#eeeeee")
                texts.append("○")
                hover.append("빈 자리")
            idx += 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_list, y=y_list, mode="markers+text",
        marker=dict(size=28, color=colors, line=dict(width=1, color="#333")),
        text=texts, textposition="middle center",
        hovertext=hover, hoverinfo="text",
    ))
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.3, y1=0.5,
                  line=dict(color="#8B4513", width=8))
    fig.update_layout(
        title=dict(text=f"{label}  [{n}/{cap} 열매]", x=0.5, font=dict(size=14)),
        xaxis=dict(visible=False, range=[-3.5, 3.5]),
        yaxis=dict(visible=False, range=[-0.8, 9]),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(t=40, b=10, l=10, r=10),
        height=320, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_category_pie(fruits: list):
    cats = [f.get("category", "기타") for f in fruits]
    if not cats:
        return
    cnt = Counter(cats)
    fig = px.pie(
        values=list(cnt.values()), names=list(cnt.keys()),
        color=list(cnt.keys()),
        color_discrete_map=CAT_COLOR,
        hole=0.4,
    )
    fig.update_layout(
        title="카테고리 분포", height=260,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="#fff",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def render_success_gauge(total: int, success: int):
    rate = int(success / total * 100) if total else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#111"},
            "steps": [
                {"range": [0,  40],  "color": "#f5f5f5"},
                {"range": [40, 70],  "color": "#e0e0e0"},
                {"range": [70, 100], "color": "#c0c0c0"},
            ],
        },
        title={"text": "완료율"},
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20),
                      paper_bgcolor="#fff")
    st.plotly_chart(fig, use_container_width=True)


# ── 로그인 페이지 ─────────────────────────────────────────────

def render_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## 🌱 Bloom Multi-Agent")
        st.caption("오케스트레이터 + 전문 에이전트 협업 미션 생성")
        st.divider()
        with st.form("login"):
            username = st.text_input("사용자명", placeholder="bloom")
            password = st.text_input("비밀번호", type="password", placeholder="bloom1234")
            if st.form_submit_button("로그인", use_container_width=True, type="primary"):
                token = api_login(username, password)
                if token:
                    st.session_state.update({"token": token, "username": username})
                    st.rerun()
        health = api_health()
        if health.get("status") == "ok":
            rag_ok = health.get("rag_index_loaded", False)
            st.success(f"서버 정상 | RAG {'로드됨' if rag_ok else '로딩 중'}")
        else:
            st.warning("백엔드 서버에 연결할 수 없습니다.")


# ── 사이드바 ──────────────────────────────────────────────────

def render_sidebar(user_data: dict) -> str:
    fruits = user_data.get("fruits", [])
    cards  = user_data.get("cards", [])
    combo  = user_data.get("combo_count", 0)
    last   = user_data.get("last_category")

    with st.sidebar:
        st.markdown(f"### 안녕하세요, **{st.session_state['username']}** 님!")
        st.divider()

        progress_val = min(len(fruits) / MAX_FRUITS, 1.0)
        st.progress(progress_val, text=f"🌳 나무 성장 {len(fruits)}/{MAX_FRUITS}")

        col1, col2 = st.columns(2)
        col1.metric("🍎 열매", len(fruits))
        col2.metric("🃏 카드", len(cards))

        if combo >= 2 and last:
            cat_lbl = {"건강": "건강", "생산성": "생산성", "재미": "재미", "성장": "성장"}.get(last, last)
            st.markdown(
                f'<div class="combo-banner">🔥 {cat_lbl} {combo}연속 콤보!</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        page = st.radio(
            "메뉴",
            ["💬 채팅", "🌿 열매 & 카드", "✨ 내 여정"],
            label_visibility="collapsed",
        )
        st.divider()

        if st.button("🔄 새 대화 시작", use_container_width=True):
            st.session_state.update({
                "messages": [], "session_id": None,
                "current_mission": None, "awaiting_result": False,
            })
            st.rerun()

        if st.button("🚪 로그아웃", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    return page


# ── 채팅 탭 ───────────────────────────────────────────────────

def render_chat_tab():
    st.markdown("## 🌱 Bloom Multi-Agent — 오늘의 미션")
    st.caption("오케스트레이터가 전문 에이전트들과 협력하여 맞춤 미션을 생성해드립니다.")

    available_minutes = st.slider("⏱ 가용 시간 (분)", 5, 120, 30, 5)

    if st.session_state.get("awaiting_result"):
        st.divider()
        st.markdown("### 📝 미션 결과를 알려주세요")
        memo = st.text_input("메모 (선택)", placeholder="오늘 미션 후 소감...", key="memo_input")
        c1, c2 = st.columns(2)

        def _record(success: bool):
            if success:
                mission = st.session_state.get("current_mission", {})
                mission_text = mission.get("mission", "미션") if mission else "미션"
                msg = f"미션을 완료했어요! '{mission_text}'" + (f" 메모: {memo}" if memo else "")
            else:
                msg = "미션을 완료하지 못했습니다."
            with st.spinner("결과 기록 중..."):
                resp = api_chat(msg, available_minutes)
            if resp:
                if resp.get("user_data"):
                    st.session_state["user_data"] = resp["user_data"]
                    if resp["user_data"].get("combo_card"):
                        st.balloons()
                        st.success(f"🔥 {resp['user_data']['combo_card']} 획득!")
                tools = resp.get("tool_calls_made", [])
                if success and "get_insight" in tools:
                    st.success("🎉 열매 획득! 미션 완료를 기록했습니다.")
                elif not success:
                    st.info("😊 시도한 용기에 도전 카드를 드립니다!")
                st.session_state["messages"].append({
                    "role": "agent", "content": resp["response"],
                    "tools": tools, "ts": datetime.now().strftime("%H:%M"),
                    "mission": resp.get("mission"),
                })
            st.session_state.update({"awaiting_result": False, "current_mission": None})
            st.rerun()

        with c1:
            if st.button("✅ 성공!", use_container_width=True, type="primary"):
                _record(True)
        with c2:
            if st.button("❌ 포기 / 실패", use_container_width=True):
                _record(False)
        st.divider()

    for msg in st.session_state["messages"]:
        ts = msg.get("ts", "")
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["content"]}<div class="ts">{ts}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            tools      = msg.get("tools", [])
            tools_html = _render_tool_badges(tools)
            content_html = _format_agent_content(msg["content"], msg.get("mission"), tools)
            st.markdown(
                f'<div class="chat-agent">'
                f'{tools_html}{"<br>" if tools_html else ""}'
                f'{content_html}'
                f'<div class="ts">{ts}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    if not st.session_state.get("awaiting_result"):
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "메시지",
                    placeholder="예) 오늘 무기력하고 스트레스받았어요. 30분 있어요.",
                    label_visibility="collapsed",
                )
            with col2:
                send = st.form_submit_button("전송", type="primary", use_container_width=True)

            if send and user_input.strip():
                ts_now = datetime.now().strftime("%H:%M")
                st.session_state["messages"].append(
                    {"role": "user", "content": user_input, "ts": ts_now}
                )
                with st.spinner("멀티 에이전트가 미션을 생성하는 중..."):
                    resp = api_chat(user_input, available_minutes)

                if resp:
                    if not st.session_state["session_id"]:
                        st.session_state["session_id"] = resp.get("session_id")
                    if resp.get("user_data"):
                        st.session_state["user_data"] = resp["user_data"]

                    tools = resp.get("tool_calls_made", [])
                    st.session_state["messages"].append({
                        "role":    "agent",
                        "content": resp["response"],
                        "tools":   tools,
                        "ts":      datetime.now().strftime("%H:%M"),
                        "mission": resp.get("mission"),
                    })
                    if resp.get("mission"):
                        st.session_state["current_mission"] = resp["mission"]
                        st.session_state["awaiting_result"] = True

                st.rerun()


# ── 열매 & 카드 탭 ────────────────────────────────────────────

def render_cards_tab(user_data: dict):
    fruits = user_data.get("fruits", [])
    cards  = user_data.get("cards", [])

    st.markdown("## 🌿 열매 & 카드")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"열매 {len(fruits)}/{MAX_FRUITS}개")
        render_tree_chart(fruits)

        if fruits:
            st.markdown("#### 카테고리 분포")
            render_category_pie(fruits)

    with c2:
        st.subheader(f"카드 {len(cards)}장")
        if cards:
            cnt = Counter(c["card"] for c in cards)
            for card_name, n in cnt.most_common():
                diff  = next((c["difficulty"] for c in cards if c["card"] == card_name), "하")
                color = DIFF_COLOR.get(diff, "#333")
                st.markdown(f"""
                <div style="border:2px solid {color};border-radius:10px;
                            padding:10px 16px;margin:6px 0;
                            box-shadow:3px 3px 0 {color}40;">
                  <span style="font-size:1.05rem;font-weight:700;">{card_name}</span>
                  <span style="color:{color};margin-left:8px;font-weight:600;">×{n}장</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("보유 카드가 없습니다. 미션을 완료하면 카드를 획득할 수 있어요!")


# ── 내 여정 탭 ────────────────────────────────────────────────

def render_journey_tab(user_data: dict):
    fruits    = user_data.get("fruits", [])
    cards     = user_data.get("cards", [])
    successes = [f for f in fruits if f.get("success", True)]
    total     = len(fruits)
    s_count   = len(successes)

    st.markdown("## ✨ 내가 걸어온 길")

    col1, col2, col3, col4 = st.columns(4)
    for col, num, lbl in [
        (col1, total,           "전체 미션"),
        (col2, s_count,         "완료"),
        (col3, total - s_count, "미완료"),
        (col4, len(cards),      "보유 카드"),
    ]:
        col.markdown(
            f'<div class="stat-box"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    if not successes:
        st.info("완료한 미션이 없습니다. 첫 미션에 도전해보세요!")
        return

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        render_category_pie(successes)
    with c2:
        render_success_gauge(total, s_count)

    st.divider()
    st.subheader("완료 미션 타임라인")
    for f in reversed(successes[-15:]):
        cat   = f.get("category", "")
        color = CAT_COLOR.get(cat, "#999")
        sym   = CAT_SYM.get(cat, "")
        ts    = f.get("timestamp", "")
        short = f.get("mission", "")

        st.markdown(f"""
        <div style="border-left:4px solid {color};padding:8px 14px;margin:6px 0;border-radius:0 8px 8px 0;">
          <div style="font-size:.78rem;color:#888;">{ts}</div>
          <span class="tag" style="background:{color}20;color:{color};border:1px solid {color};">
            {sym} {cat}
          </span>
          <strong style="margin-left:6px;">{short}</strong>
        </div>
        """, unsafe_allow_html=True)


# ── 메인 ──────────────────────────────────────────────────────

if st.session_state.get("token"):
    user_data = st.session_state.get("user_data", {
        "fruits": [], "cards": [], "combo_count": 0, "last_category": None,
    })
    page = render_sidebar(user_data)

    if page == "💬 채팅":
        render_chat_tab()
    elif page == "🌿 열매 & 카드":
        render_cards_tab(user_data)
    elif page == "✨ 내 여정":
        render_journey_tab(user_data)
else:
    render_login()
