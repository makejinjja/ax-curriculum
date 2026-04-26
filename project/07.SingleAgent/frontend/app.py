"""
app.py — Streamlit frontend for the AX Compass Single Agent

Deploy to Streamlit Cloud:
  Main file path : project/07.SingleAgent/frontend/app.py
  Secrets        : BACKEND_URL, DEMO_USER, DEMO_PASSWORD

Local development:
  streamlit run app.py
  Set BACKEND_URL in .env or st.secrets
"""
from __future__ import annotations
import os
import json
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────

def _backend_url() -> str:
    try:
        return st.secrets["BACKEND_URL"]
    except Exception:
        return os.environ.get("BACKEND_URL", "http://localhost:8000")


def _default_creds() -> tuple[str, str]:
    try:
        return st.secrets.get("DEMO_USER", "admin"), st.secrets.get("DEMO_PASSWORD", "password")
    except Exception:
        return os.environ.get("DEMO_USER", "admin"), os.environ.get("DEMO_PASSWORD", "password")


BACKEND = _backend_url()

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="AX Compass — Curriculum Agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Chat bubbles */
.user-bubble {
    background: #1a1a1a; color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; margin: 4px 0;
    max-width: 80%; float: right; clear: both;
}
.agent-bubble {
    background: #f4f4f5; color: #18181b;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px; margin: 4px 0;
    max-width: 85%; float: left; clear: both;
}
.bubble-wrapper { overflow: hidden; margin-bottom: 8px; }

/* Curriculum card */
.curriculum-card {
    border: 2px solid #18181b; border-radius: 12px;
    padding: 20px 24px; margin-top: 16px;
    background: #fafafa;
}
.module-row {
    border-left: 3px solid #18181b;
    padding: 8px 12px; margin: 6px 0;
    background: #ffffff; border-radius: 0 8px 8px 0;
}

/* Validation badge */
.badge-pass {
    background: #dcfce7; color: #166534;
    border-radius: 20px; padding: 4px 12px;
    font-size: 13px; font-weight: 600;
    display: inline-block;
}
.badge-fail {
    background: #fee2e2; color: #991b1b;
    border-radius: 20px; padding: 4px 12px;
    font-size: 13px; font-weight: 600;
    display: inline-block;
}
.badge-warn {
    background: #fef9c3; color: #854d0e;
    border-radius: 20px; padding: 4px 12px;
    font-size: 13px; font-weight: 600;
    display: inline-block;
}

/* Tool pill */
.tool-pill {
    display: inline-block;
    background: #e4e4e7; color: #3f3f46;
    border-radius: 12px; padding: 2px 10px;
    font-size: 12px; margin: 2px;
}

/* Score bar */
.score-bar-bg {
    background: #e4e4e7; border-radius: 8px;
    height: 8px; width: 100%; margin-top: 4px;
}
.score-bar-fill {
    background: #18181b; border-radius: 8px; height: 8px;
}

/* Login card */
.login-card {
    max-width: 400px; margin: 80px auto;
    padding: 36px; border: 1px solid #e4e4e7;
    border-radius: 16px; background: #fff;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────

def api_post(path: str, payload: dict, token: str = "") -> dict | None:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.post(f"{BACKEND}{path}", json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"백엔드 서버에 연결할 수 없습니다: {BACKEND}")
    except requests.exceptions.HTTPError as e:
        st.error(f"API 오류 ({e.response.status_code}): {e.response.text}")
    except Exception as e:
        st.error(f"요청 오류: {e}")
    return None


def api_get(path: str, token: str = "") -> dict | None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(f"{BACKEND}{path}", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Session state init ────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "token": "",
        "username": "",
        "messages": [],       # list of {role, content}
        "curriculum": None,
        "validation": None,
        "tool_history": [],   # list of tool call lists per turn
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Login page ────────────────────────────────────────────────

def page_login() -> None:
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown("## 🧭 AX Compass")
    st.markdown("**Curriculum Design Agent** — 로그인하여 시작하세요")
    st.markdown("---")

    default_user, default_pw = _default_creds()
    username = st.text_input("사용자명", value=default_user, key="login_user")
    password = st.text_input("비밀번호", type="password", value=default_pw, key="login_pw")

    if st.button("로그인", use_container_width=True, type="primary"):
        with st.spinner("인증 중..."):
            res = api_post("/auth/login", {"username": username, "password": password})
        if res and "access_token" in res:
            st.session_state.token    = res["access_token"]
            st.session_state.username = username
            st.rerun()
        else:
            st.error("로그인 실패 — 사용자명/비밀번호를 확인하세요")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Curriculum renderer ───────────────────────────────────────

def render_curriculum(cur: dict) -> None:
    st.markdown("<div class='curriculum-card'>", unsafe_allow_html=True)
    st.markdown(f"### 📋 {cur.get('title', '커리큘럼')}")

    c1, c2, c3 = st.columns(3)
    c1.metric("총 시간", f"{cur.get('total_duration_minutes', 0)}분")
    c2.metric("대상", cur.get("target_audience", "-"))
    c3.metric("그룹 규모", cur.get("group_size", "-"))

    st.markdown("**학습 목표**")
    for obj in cur.get("learning_objectives", []):
        st.markdown(f"- {obj}")

    st.markdown("**모듈 구성**")
    modules = cur.get("modules", [])
    for i, m in enumerate(modules, 1):
        with st.expander(
            f"**{i}. {m.get('title', '모듈')}** — {m.get('duration_minutes', 0)}분",
            expanded=(i <= 2),
        ):
            if m.get("objectives"):
                st.markdown("**목표**")
                for o in m["objectives"]:
                    st.markdown(f"  - {o}")
            if m.get("activities"):
                st.markdown("**활동**")
                for a in m["activities"]:
                    st.markdown(f"  - {a}")
            if m.get("materials"):
                st.markdown("**자료**")
                for mat in m["materials"]:
                    st.markdown(f"  - {mat}")

    if cur.get("assessment"):
        st.markdown(f"**평가 방법**: {cur['assessment']}")
    if cur.get("notes"):
        st.info(f"📝 {cur['notes']}")

    # Module duration bar chart
    if modules:
        import json as _json
        labels  = [f"{m.get('title', '?')} ({m.get('duration_minutes', 0)}분)" for m in modules]
        durations = [m.get("duration_minutes", 0) for m in modules]
        chart_data = {"labels": labels, "data": durations}
        total = sum(durations)
        bar_html = ""
        for label, dur in zip(labels, durations):
            pct = (dur / total * 100) if total else 0
            bar_html += (
                f"<div style='display:flex;align-items:center;margin:3px 0;'>"
                f"<div style='width:180px;font-size:12px;color:#3f3f46;overflow:hidden;"
                f"white-space:nowrap;text-overflow:ellipsis;' title='{label}'>{label}</div>"
                f"<div style='flex:1;background:#e4e4e7;border-radius:4px;height:16px;margin:0 8px;'>"
                f"<div style='width:{pct:.1f}%;background:#18181b;height:16px;border-radius:4px;'></div>"
                f"</div><div style='font-size:12px;color:#71717a;'>{pct:.0f}%</div></div>"
            )
        st.markdown(bar_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_validation(vr: dict) -> None:
    valid  = vr.get("valid", False)
    errors = vr.get("errors", [])
    warns  = vr.get("warnings", [])
    score  = vr.get("score", 0.0)

    badge_cls  = "badge-pass" if valid else "badge-fail"
    badge_text = "✅ 검증 통과" if valid else "❌ 검증 실패"

    st.markdown(
        f"<span class='{badge_cls}'>{badge_text}</span>"
        f"&nbsp;<span style='font-size:13px;color:#71717a;'>품질 점수 {score*100:.0f}점</span>",
        unsafe_allow_html=True,
    )
    pct = score * 100
    st.markdown(
        f"<div class='score-bar-bg'>"
        f"<div class='score-bar-fill' style='width:{pct:.0f}%;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if errors:
        with st.expander("❌ 오류", expanded=True):
            for e in errors:
                st.markdown(f"- {e}")
    if warns:
        with st.expander("⚠️ 경고 (개선 권장)", expanded=False):
            for w in warns:
                st.markdown(f"- {w}")


# ── Chat page ─────────────────────────────────────────────────

def page_chat() -> None:
    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧭 AX Compass")
        st.markdown(f"**{st.session_state.username}** 님 환영합니다")

        # Health check
        health = api_get("/health", st.session_state.token)
        if health:
            rag_ok = health.get("rag_indexed", False)
            st.markdown(
                f"**서버**: 🟢 정상  \n**RAG 인덱스**: {'🟢 로드됨' if rag_ok else '🟡 미로드'}"
            )
        else:
            st.markdown("**서버**: 🔴 연결 안됨")

        st.markdown("---")
        st.markdown("**사용 가능한 도구**")
        st.markdown(
            "🔍 `rag_search` — 내부 지식베이스  \n"
            "🌐 `web_search` — 웹 검색 (Tavily)  \n"
            "📋 `generate_curriculum` — 커리큘럼 생성  \n"
            "✅ `validate_curriculum` — 규칙 검증"
        )
        st.markdown("---")

        if st.button("💬 새 대화 시작", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.curriculum  = None
            st.session_state.validation  = None
            st.session_state.tool_history = []
            st.rerun()

        if st.button("🚪 로그아웃", use_container_width=True):
            for k in ["token", "username", "messages", "curriculum", "validation", "tool_history"]:
                st.session_state[k] = "" if k in ("token", "username") else (None if k in ("curriculum", "validation") else [])
            st.rerun()

        # Show current curriculum in sidebar if exists
        if st.session_state.curriculum:
            st.markdown("---")
            st.markdown("**현재 생성된 커리큘럼**")
            cur = st.session_state.curriculum
            st.markdown(f"📋 **{cur.get('title', '제목 없음')}**")
            st.markdown(
                f"⏱ {cur.get('total_duration_minutes', 0)}분 | "
                f"👥 {cur.get('group_size', '-')}"
            )
            if st.button("📥 JSON 다운로드", use_container_width=True):
                st.download_button(
                    "다운로드",
                    data=json.dumps(cur, ensure_ascii=False, indent=2),
                    file_name="curriculum.json",
                    mime="application/json",
                )

    # ── Main chat area ────────────────────────────────────────
    st.markdown("## 🧭 AX Compass — Curriculum Design Agent")

    # Render conversation history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            role    = msg["role"]
            content = msg["content"]

            if role == "user":
                st.markdown(
                    f"<div class='bubble-wrapper'>"
                    f"<div class='user-bubble'>{content}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='bubble-wrapper'>"
                    f"<div class='agent-bubble'>{content}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Show tool calls for this assistant turn
                if i < len(st.session_state.tool_history):
                    tools_used = st.session_state.tool_history[i // 2]
                    if tools_used:
                        pills = "".join(
                            f"<span class='tool-pill'>⚙️ {t}</span>"
                            for t in dict.fromkeys(tools_used)  # deduplicate, preserve order
                        )
                        st.markdown(pills, unsafe_allow_html=True)

        # Show latest curriculum and validation if present
        if st.session_state.curriculum:
            render_curriculum(st.session_state.curriculum)
        if st.session_state.validation:
            render_validation(st.session_state.validation)

    st.markdown("---")

    # Input area
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_area(
                "메시지 입력",
                placeholder=(
                    "예: 신입 개발자 대상 코드 리뷰 문화 교육 커리큘럼을 만들어줘 (3시간, 15명)"
                ),
                height=80,
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("전송", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        user_msg = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": user_msg})

        # Build history for API (last 20 turns to stay within token limits)
        history = st.session_state.messages[:-1][-20:]

        with st.spinner("Agent 처리 중..."):
            res = api_post(
                "/chat",
                {
                    "message": user_msg,
                    "history": history,
                },
                token=st.session_state.token,
            )

        if res:
            reply    = res.get("reply", "")
            cur      = res.get("curriculum")
            val      = res.get("validation_result")
            tools    = res.get("tool_calls_made", [])
            complete = res.get("complete", False)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.tool_history.append(tools)

            if cur:
                st.session_state.curriculum = cur
            if val:
                st.session_state.validation = val

            if complete:
                st.balloons()

        st.rerun()


# ── Entry point ───────────────────────────────────────────────

def main() -> None:
    if not st.session_state.token:
        page_login()
    else:
        page_chat()


main()
