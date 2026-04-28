"""
app.py — Bloom Single Agent Streamlit 프론트엔드

배포: Streamlit Cloud (BACKEND_URL 환경변수로 백엔드 URL 지정)
"""
from __future__ import annotations
import os
import re
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bloom Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
* { font-family: 'Noto Sans KR', sans-serif !important; }

[data-testid="stSidebar"] { background-color: #0d0d0d; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1a1a2e; color: #ffffff; border: 1px solid #333;
    border-radius: 8px; width: 100%;
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
.mission-title { font-size: 18px; font-weight: 700; color: #1b5e20; margin-bottom: 12px; }
.mission-row { margin: 6px 0; font-size: 14px; color: #2d3436; }
.tool-badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 10px; font-size: 11px; font-weight: 500;
    background: #e3f2fd; color: #1565c0; margin-right: 4px;
}
.ts { font-size: 10px; color: #aaa; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── 세션 상태 초기화 ─────────────────────────────────────────
def _init():
    defaults = {
        "token":           None,
        "username":        None,
        "session_id":      None,
        "messages":        [],
        "current_mission": None,
        "awaiting_result": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

_init()


# ── API 헬퍼 ────────────────────────────────────────────────

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
            timeout=90,
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


# ── 미션 HTML 렌더 ───────────────────────────────────────────

def _render_mission_card(mission: dict) -> str:
    cat_icons  = {"건강": "💪", "생산성": "📋", "재미": "🎮", "성장": "📚", "돌발": "⚡"}
    diff_icons = {"하": "🌱", "중": "🌿", "상": "🌟", "최상": "🏆"}
    cat  = mission.get("category", "")
    diff = mission.get("difficulty", "")
    return f"""
<div class="mission-card">
  <div class="mission-title">🎯 {mission.get('mission', '')}</div>
  <div class="mission-row">{cat_icons.get(cat,'🏷️')} 카테고리: <strong>{cat}</strong>
      &nbsp;|&nbsp; {diff_icons.get(diff,'⭐')} 난이도: <strong>{diff}</strong></div>
  <div class="mission-row">📚 근거: {mission.get('basis','')}</div>
  <div class="mission-row">✨ 효과: {mission.get('effect','')}</div>
</div>
"""


def _format_agent_content(text: str, mission: dict | None) -> str:
    """응답 텍스트에서 [태그] 블록을 미션 카드로 교체."""
    if mission and "[미션]" in text:
        # 미션 블록 이전 텍스트 추출 (공감 메시지)
        before = re.split(r"\[미션\]", text, maxsplit=1)[0].strip()
        # 미션 블록 이후 텍스트 (수락 유도 문장)
        after_match = re.search(r"\[효과\][^\[]*", text, re.DOTALL)
        after = ""
        if after_match:
            raw_after = text[after_match.end():].strip()
            after = raw_after if raw_after else ""
        return (
            (f"<p>{before}</p>" if before else "")
            + _render_mission_card(mission)
            + (f"<p>{after}</p>" if after else "")
        )
    return f"<p>{text.replace(chr(10), '<br>')}</p>"


# ── 로그인 페이지 ────────────────────────────────────────────

def render_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## 🌱 Bloom Agent")
        st.caption("심리학 기반 일일 미션 생성 에이전트")
        st.divider()

        with st.form("login"):
            username = st.text_input("사용자명", placeholder="admin")
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


# ── 메인 채팅 UI ─────────────────────────────────────────────

def render_chat():
    TOOL_ICONS = {"rag_search": "🔍 RAG", "web_search": "🌐 웹", "validate_mission": "✅ 검증"}

    # ── 사이드바 ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 안녕하세요, **{st.session_state['username']}** 님!")
        st.divider()

        st.markdown("#### ⏱ 가용 시간")
        available_minutes = st.slider("분", 5, 120, 30, 5, label_visibility="collapsed")
        st.caption(f"현재 설정: **{available_minutes}분**")

        st.divider()
        st.markdown("#### 🤖 에이전트 도구")
        st.markdown("- 🔍 RAG 심리학 논문 검색\n- 🌐 Tavily 웹 검색\n- ✅ 미션 검증")

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

    # ── 메인 영역 ─────────────────────────────────────────────
    st.markdown("## 🌱 Bloom Agent")
    st.caption("현재 기분을 알려주시면 맞춤 미션을 생성해드립니다.")

    # 미션 결과 기록 UI
    if st.session_state.get("awaiting_result"):
        st.divider()
        st.markdown("### 📝 미션 결과를 기록해주세요")
        memo = st.text_input("메모 (선택)", placeholder="오늘 미션 후 소감...", key="memo_input")
        col1, col2 = st.columns(2)

        def _record(success: bool):
            msg = ("미션 완료! " + (f"메모: {memo}" if memo else "")) if success else "미션을 완료하지 못했습니다."
            with st.spinner("결과 기록 중..."):
                resp = api_chat(msg, available_minutes)
            if resp:
                st.session_state["messages"].append({
                    "role": "agent", "content": resp["response"],
                    "tools": resp.get("tool_calls_made", []),
                    "ts": datetime.now().strftime("%H:%M"),
                    "mission": resp.get("mission"),
                })
            st.session_state.update({"awaiting_result": False, "current_mission": None})
            st.rerun()

        with col1:
            if st.button("✅ 성공", use_container_width=True, type="primary"):
                _record(True)
        with col2:
            if st.button("❌ 실패 / 포기", use_container_width=True):
                _record(False)
        st.divider()

    # 대화 메시지 렌더링
    for msg in st.session_state["messages"]:
        ts = msg.get("ts", "")
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["content"]}<div class="ts">{ts}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            tools_html = "".join(
                f'<span class="tool-badge">{TOOL_ICONS.get(t, t)}</span>'
                for t in msg.get("tools", [])
            )
            content_html = _format_agent_content(msg["content"], msg.get("mission"))
            st.markdown(
                f'<div class="chat-agent">'
                f'{tools_html}{"<br>" if tools_html else ""}'
                f'{content_html}'
                f'<div class="ts">{ts}</div></div>',
                unsafe_allow_html=True,
            )

    # 입력창
    st.divider()
    if not st.session_state.get("awaiting_result"):
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "메시지",
                    placeholder="예) 오늘 좀 무기력하고 스트레스를 받았어요. 30분 있어요.",
                    label_visibility="collapsed",
                )
            with col2:
                send = st.form_submit_button("전송", type="primary", use_container_width=True)

            if send and user_input.strip():
                ts_now = datetime.now().strftime("%H:%M")
                st.session_state["messages"].append({
                    "role": "user", "content": user_input, "ts": ts_now,
                })

                with st.spinner("에이전트가 미션을 생성하는 중..."):
                    resp = api_chat(user_input, available_minutes)

                if resp:
                    if not st.session_state["session_id"]:
                        st.session_state["session_id"] = resp.get("session_id")

                    st.session_state["messages"].append({
                        "role":    "agent",
                        "content": resp["response"],
                        "tools":   resp.get("tool_calls_made", []),
                        "ts":      datetime.now().strftime("%H:%M"),
                        "mission": resp.get("mission"),
                    })

                    if resp.get("mission"):
                        st.session_state["current_mission"] = resp["mission"]
                        st.session_state["awaiting_result"] = True

                st.rerun()


# ── 진입점 ──────────────────────────────────────────────────
if st.session_state.get("token"):
    render_chat()
else:
    render_login()
