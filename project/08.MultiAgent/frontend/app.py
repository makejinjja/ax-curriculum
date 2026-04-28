"""
app.py — Bloom MultiAgent Streamlit 프론트엔드
"""
from __future__ import annotations
import os
import re
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Bloom MultiAgent",
    page_icon="🌿",
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
.trace-item {
    background: #1e1e2e; color: #cdd6f4;
    border-radius: 8px; padding: 8px 12px; margin: 4px 0;
    font-size: 12px; font-family: monospace;
}
.trace-agent { color: #89b4fa; font-weight: 600; }
.score-bar {
    height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #2d6a4f, #40c074);
    margin: 2px 0;
}
.ts { font-size: 10px; color: #aaa; margin-top: 4px; }
.curriculum-item {
    background: #f8f9fa; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0;
    border-left: 3px solid #2d6a4f;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ── 세션 초기화 ───────────────────────────────────────────────
def _init():
    defaults = {
        "token":           None,
        "username":        None,
        "session_id":      None,
        "messages":        [],
        "current_mission": None,
        "awaiting_result": False,
        "show_trace":      False,
        "last_trace":      [],
        "last_emotion":    None,
        "last_validation": None,
        "curriculum_list": [],
        "selected_curriculum": None,
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


def api_curriculum_list() -> list:
    try:
        r = requests.get(f"{BACKEND_URL}/curriculum", headers=_headers(), timeout=10)
        if r.status_code == 200:
            return r.json().get("curricula", [])
    except Exception:
        pass
    return []


def api_curriculum_detail(cid: str) -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/curriculum/{cid}", headers=_headers(), timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── UI 컴포넌트 ───────────────────────────────────────────────
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
    if mission and "[미션]" in text:
        before = re.split(r"\[미션\]", text, maxsplit=1)[0].strip()
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


def _render_trace(trace: list[dict]):
    if not trace:
        return
    st.markdown("**에이전트 실행 추적**")
    for item in trace:
        agent = item.get("agent", "")
        attempt = f" (시도 {item['attempt']})" if "attempt" in item else ""
        out = item.get("output", "")
        if isinstance(out, dict):
            out_str = ", ".join(f"{k}: {v}" for k, v in list(out.items())[:4])
        else:
            out_str = str(out)[:120]
        passed = item.get("passed")
        badge = ""
        if passed is True:
            badge = " ✅"
        elif passed is False:
            badge = " ❌"
        st.markdown(
            f'<div class="trace-item">'
            f'<span class="trace-agent">[{agent}{attempt}]{badge}</span> {out_str}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_validation_scores(validation: dict | None):
    if not validation:
        return
    llm = validation.get("llm_result")
    if not llm:
        return
    scores = llm.get("scores", {})
    total = llm.get("total_score", 0)
    labels = {
        "psychological_validity": "심리학적 타당성",
        "practicality": "실용성",
        "safety": "안전성",
        "specificity": "구체성",
    }
    st.markdown(f"**검증 점수: {total}/100** {'✅' if llm.get('is_valid') else '❌'}")
    for key, label in labels.items():
        score = scores.get(key, 0)
        pct = score / 25 * 100
        st.markdown(f"{label}: **{score}/25**")
        st.progress(int(pct))
    if llm.get("strengths"):
        st.caption(f"강점: {llm['strengths']}")


# ── 로그인 페이지 ─────────────────────────────────────────────
def render_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("## 🌿 Bloom MultiAgent")
        st.caption("심리학 기반 멀티에이전트 미션 생성 시스템")
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


# ── 메인 채팅 UI ──────────────────────────────────────────────
def render_chat():
    # ── 사이드바 ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 안녕하세요, **{st.session_state['username']}** 님!")
        st.divider()

        st.markdown("#### ⏱ 가용 시간")
        available_minutes = st.slider("분", 5, 120, 30, 5, label_visibility="collapsed")
        st.caption(f"현재 설정: **{available_minutes}분**")

        st.divider()
        st.markdown("#### 🤖 에이전트 파이프라인")
        st.markdown(
            "1. 😊 EmotionAnalyst\n"
            "2. 🔍 RAGResearcher\n"
            "3. 🎯 MissionGenerator\n"
            "4. ✅ Validator (코드 + LLM)\n"
            "5. 🎼 Orchestrator"
        )

        st.divider()
        show_trace = st.toggle("에이전트 추적 보기", value=st.session_state["show_trace"])
        st.session_state["show_trace"] = show_trace

        st.divider()

        # 커리큘럼 이력
        st.markdown("#### 📚 미션 이력")
        if st.button("이력 새로고침", use_container_width=True):
            st.session_state["curriculum_list"] = api_curriculum_list()

        curricula = st.session_state.get("curriculum_list", [])
        if curricula:
            for c in curricula[:10]:
                dt = c.get("created_at", "")[:10]
                score_str = f" | {c['score']}점" if c.get("score") else ""
                label = f"{dt} | {c['emotion']} | {c['mission'][:20]}...{score_str}"
                if st.button(label, key=f"cur_{c['id']}", use_container_width=True):
                    detail = api_curriculum_detail(c["id"])
                    st.session_state["selected_curriculum"] = detail
        else:
            st.caption("이력이 없습니다.")

        st.divider()
        if st.button("🔄 새 대화 시작", use_container_width=True):
            st.session_state.update({
                "messages": [], "session_id": None,
                "current_mission": None, "awaiting_result": False,
                "last_trace": [], "last_emotion": None, "last_validation": None,
            })
            st.rerun()

        if st.button("🚪 로그아웃", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── 메인 영역 ─────────────────────────────────────────────
    col_chat, col_detail = st.columns([3, 1])

    with col_chat:
        st.markdown("## 🌿 Bloom MultiAgent")
        st.caption("현재 기분을 알려주시면 맞춤 미션을 생성해드립니다.")

        # 미션 결과 기록 UI
        if st.session_state.get("awaiting_result"):
            st.divider()
            st.markdown("### 📝 미션 결과를 기록해주세요")
            memo = st.text_input("메모 (선택)", placeholder="오늘 미션 후 소감...", key="memo_input")
            c1, c2 = st.columns(2)

            def _record(success: bool):
                msg = ("미션 완료! " + (f"메모: {memo}" if memo else "")) if success else "미션을 완료하지 못했습니다."
                with st.spinner("결과 기록 중..."):
                    resp = api_chat(msg, available_minutes)
                if resp:
                    st.session_state["messages"].append({
                        "role": "agent", "content": resp["response"],
                        "ts": datetime.now().strftime("%H:%M"),
                        "mission": resp.get("mission"),
                        "trace": resp.get("agent_trace", []),
                        "emotion": resp.get("emotion"),
                        "validation": resp.get("validation"),
                    })
                st.session_state.update({"awaiting_result": False, "current_mission": None})
                st.rerun()

            with c1:
                if st.button("✅ 성공", use_container_width=True, type="primary"):
                    _record(True)
            with c2:
                if st.button("❌ 실패 / 포기", use_container_width=True):
                    _record(False)
            st.divider()

        # 대화 렌더링
        for msg in st.session_state["messages"]:
            ts = msg.get("ts", "")
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{msg["content"]}<div class="ts">{ts}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                content_html = _format_agent_content(msg["content"], msg.get("mission"))
                st.markdown(
                    f'<div class="chat-agent">{content_html}<div class="ts">{ts}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.session_state["show_trace"] and msg.get("trace"):
                    with st.expander("에이전트 추적", expanded=False):
                        _render_trace(msg["trace"])
                        _render_validation_scores(msg.get("validation"))

        # 입력창
        st.divider()
        if not st.session_state.get("awaiting_result"):
            with st.form("chat_form", clear_on_submit=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    user_input = st.text_input(
                        "메시지",
                        placeholder="예) 오늘 좀 무기력하고 스트레스를 받았어요. 30분 있어요.",
                        label_visibility="collapsed",
                    )
                with c2:
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
                            "role":       "agent",
                            "content":    resp["response"],
                            "ts":         datetime.now().strftime("%H:%M"),
                            "mission":    resp.get("mission"),
                            "trace":      resp.get("agent_trace", []),
                            "emotion":    resp.get("emotion"),
                            "validation": resp.get("validation"),
                        })
                        st.session_state["last_trace"]      = resp.get("agent_trace", [])
                        st.session_state["last_emotion"]    = resp.get("emotion")
                        st.session_state["last_validation"] = resp.get("validation")

                        if resp.get("mission"):
                            st.session_state["current_mission"] = resp["mission"]
                            st.session_state["awaiting_result"] = True

                        # 커리큘럼 이력 갱신
                        if resp.get("curriculum_id"):
                            st.session_state["curriculum_list"] = api_curriculum_list()

                    st.rerun()

    # ── 오른쪽 상세 패널 ──────────────────────────────────────
    with col_detail:
        st.markdown("### 📊 분석 결과")

        # 감정 분석
        emotion = st.session_state.get("last_emotion")
        if emotion:
            st.markdown("**감정 분석**")
            emo_type = emotion.get("emotion_type", "")
            intensity = emotion.get("intensity", 0)
            st.metric("감정 유형", emo_type)
            st.progress(intensity / 5, text=f"강도: {intensity}/5")
            st.caption(emotion.get("summary", ""))
            triggers = emotion.get("triggers", [])
            if triggers:
                st.caption("요인: " + ", ".join(triggers))
            st.divider()

        # 검증 점수
        validation = st.session_state.get("last_validation")
        if validation:
            _render_validation_scores(validation)
            st.divider()

        # 선택된 커리큘럼 상세
        selected = st.session_state.get("selected_curriculum")
        if selected:
            st.markdown("**선택한 미션 상세**")
            m = selected.get("mission", {})
            st.markdown(f"**{m.get('mission', '')}**")
            st.caption(f"카테고리: {m.get('category','')}")
            st.caption(f"난이도: {m.get('difficulty','')}")
            st.caption(f"근거: {m.get('basis','')}")
            st.caption(f"효과: {m.get('effect','')}")
            if st.button("닫기", key="close_curriculum"):
                st.session_state["selected_curriculum"] = None
                st.rerun()


# ── 진입점 ───────────────────────────────────────────────────
if st.session_state.get("token"):
    render_chat()
else:
    render_login()
