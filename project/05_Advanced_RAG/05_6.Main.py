"""
05_6.Main.py — Streamlit 웹 애플리케이션 (메인 진입점)

실행:
  streamlit run 05_6.Main.py

Docker:
  docker-compose up
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── 내부 모듈 임포트 ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from schemas import (  # type: ignore[import]
    DIFF, CAT, COMBO2_CARD, MAX_FRUITS,
    EMOTION_PURPOSE, EMOTION_EMOJI,
)
from auth import get_client, load_data, save_data  # type: ignore[import]
from indexing import build_index  # type: ignore[import]
from retrieval import (  # type: ignore[import]
    classify_emotion, get_mission, parse_mission,
    get_insight, get_motivational_nudge, summarize_mission,
    analyze_coverage,
)

# ── 모듈명 단축 alias (파일명 05_N. 접두어 제거용) ──────────
import importlib
for _alias, _mod in [
    ("schemas",   "05_2.Schemas"),
    ("auth",      "05_3.Auth"),
    ("indexing",  "05_4.Indexing"),
    ("retrieval", "05_5.Retrieval"),
]:
    if _alias not in sys.modules:
        try:
            sys.modules[_alias] = importlib.import_module(_mod)
        except ModuleNotFoundError:
            pass  # 이미 직접 임포트 성공한 경우


# ════════════════════════════════════════════════════════════
#  스타일
# ════════════════════════════════════════════════════════════

def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
    .main { background: #ffffff; }
    .block-container { padding: 2rem 3rem; max-width: 900px; }

    /* 사이드바 */
    section[data-testid="stSidebar"] { background: #0a0a0a; }
    section[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 6px 0; }

    /* 카드 */
    .mission-card {
        background: #fff; border: 2px solid #111; border-radius: 12px;
        padding: 1.4rem 1.6rem; margin: 1rem 0; box-shadow: 4px 4px 0 #111;
    }
    .wildcard-card {
        background: #fffbea; border: 2px solid #f39c12; border-radius: 12px;
        padding: 1.4rem 1.6rem; margin: 1rem 0; box-shadow: 4px 4px 0 #f39c12;
    }
    .insight-box {
        background: #f8f8f8; border-left: 4px solid #111;
        padding: 0.9rem 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
        font-size: 0.92rem; color: #333;
    }
    .combo-banner {
        background: #111; color: #f1c40f; text-align: center;
        padding: 0.5rem; border-radius: 8px; font-weight: 700;
        letter-spacing: 1px; font-size: 0.9rem; margin: 0.5rem 0;
    }
    .stat-box {
        background: #f9f9f9; border: 1px solid #ddd; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .stat-num { font-size: 2rem; font-weight: 700; color: #111; }
    .stat-lbl { font-size: 0.78rem; color: #777; margin-top: 2px; }
    .tag {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; margin-right: 4px;
    }
    .divider { border-top: 1px solid #e0e0e0; margin: 1.2rem 0; }
    h1, h2, h3 { color: #111; font-weight: 700; }
    .stButton > button {
        border-radius: 8px; border: 2px solid #111; background: #fff;
        color: #111; font-weight: 600; transition: all .15s;
    }
    .stButton > button:hover { background: #111; color: #fff; }
    .stButton > button[kind="primary"] { background: #111; color: #fff; }
    .stButton > button[kind="primary"]:hover { background: #333; }
    </style>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  RAG 인덱스 캐시 (앱 재시작 시 1회만 빌드)
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def build_index_cached(api_key: str):
    client = get_client()
    progress = st.progress(0, text="논문 인덱스 로딩 중...")

    def cb(cur, total, msg):
        pct = int(cur / max(total, 1) * 100)
        progress.progress(pct, text=msg)

    chunks, embeddings, bm25 = build_index(client, progress_callback=cb)
    progress.empty()
    return chunks, embeddings, bm25


# ════════════════════════════════════════════════════════════
#  시각화 헬퍼
# ════════════════════════════════════════════════════════════

CAT_COLOR = {
    "건강": "#e74c3c", "생산성": "#3498db",
    "재미": "#9b59b6", "성장":   "#27ae60", "돌발": "#f39c12",
}


def render_tree_chart(fruits: list):
    if not fruits:
        st.info("아직 열매가 없습니다. 첫 번째 미션을 완료해보세요!")
        return

    n = len(fruits)
    cap = 30 if n > 20 else (20 if n > 10 else 10)
    label = "🌲 완전한 나무" if cap == 30 else ("🌿 성장 나무" if cap == 20 else "🌱 새싹 나무")

    # 격자 좌표 계산 (피라미드 형태)
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
                sym = CAT.get(cat, {}).get("sym", "●")
                short = fruits[idx].get("mission", "")
                texts.append(sym)
                hover.append(f"[{cat}] {short}")
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
    # 나무 줄기
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.3, y1=0.5,
                  line=dict(color="#8B4513", width=8))
    fig.update_layout(
        title=dict(text=f"{label}  [{n}/{cap} 열매]", x=0.5, font=dict(size=15)),
        xaxis=dict(visible=False, range=[-3.5, 3.5]),
        yaxis=dict(visible=False, range=[-0.8, 9]),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(t=40, b=10, l=10, r=10),
        height=340, showlegend=False,
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
        title="카테고리 분포", height=280,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def render_success_gauge(total: int, success: int):
    rate = int(success / total * 100) if total else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        number={"suffix": "%", "font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#111"},
            "steps": [
                {"range": [0, 40],  "color": "#f5f5f5"},
                {"range": [40, 70], "color": "#e0e0e0"},
                {"range": [70, 100],"color": "#c0c0c0"},
            ],
        },
        title={"text": "성공률"},
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20),
                      paper_bgcolor="#fff")
    st.plotly_chart(fig, use_container_width=True)


def render_mission_freq_bar(fruits: list):
    if not fruits:
        return
    cnt = Counter(f.get("full_mission") or f.get("mission", "") for f in fruits)
    top = cnt.most_common(8)
    labels = [t[0][:18] + ("…" if len(t[0]) > 18 else "") for t in top]
    vals   = [t[1] for t in top]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color="#111", text=vals, textposition="outside",
    ))
    fig.update_layout(
        title="자주 수행한 미션 TOP 8", height=300,
        xaxis_title="횟수", yaxis=dict(autorange="reversed"),
        margin=dict(t=40, b=10, l=10, r=40),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_coverage_bar(coverage: dict):
    if not coverage or "sorted" not in coverage:
        return
    pairs = coverage["sorted"]
    labels = [p[0] for p in pairs]
    scores = [round(p[1] * 100, 1) for p in pairs]
    weak   = coverage.get("weak", [])
    colors = ["#e74c3c" if lbl in weak else "#111" for lbl in labels]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{s}%" for s in scores], textposition="outside",
    ))
    fig.update_layout(
        title="논문 활용도 분석 (낮을수록 다음 미션에서 우선 반영)",
        height=340,
        xaxis_title="활용도 (%)",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=10, l=10, r=50),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
    )
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  미션 페이지
# ════════════════════════════════════════════════════════════

def page_mission(client, chunks, embeddings, bm25):
    st.header("🌱 미션 시작")
    data = st.session_state["data"]
    step = st.session_state.get("mission_step", "input")

    # ── step: input ──────────────────────────────────────────
    if step == "input":
        if len(data["fruits"]) >= MAX_FRUITS:
            st.error("🌲 나무가 가득 찼습니다! [열매 & 카드] 메뉴에서 열매를 쪼개주세요.")
            return

        with st.form("mission_form"):
            mood = st.text_area("지금 기분을 자유롭게 적어주세요", height=80,
                                placeholder="예: 오늘 너무 지쳐서 아무것도 하기 싫어…")
            minutes = st.slider("사용 가능한 시간 (분)", 1, 60, 15, step=5)
            submitted = st.form_submit_button("미션 추천 받기", type="primary", use_container_width=True)

        if submitted:
            if not mood.strip():
                st.warning("기분을 입력해주세요.")
                return
            st.session_state["mission_input"] = {"mood": mood.strip(), "minutes": minutes}
            st.session_state["mission_step"] = "generating"
            st.rerun()

    # ── step: generating ─────────────────────────────────────
    elif step == "generating":
        inp = st.session_state["mission_input"]
        with st.spinner("감정 분석 중..."):
            emotion_type = classify_emotion(client, inp["mood"])
        st.session_state["emotion_type"] = emotion_type

        emoji   = EMOTION_EMOJI.get(emotion_type, "")
        purpose = EMOTION_PURPOSE.get(emotion_type, "")
        st.markdown(f"**{emoji} 감정 분류: {emotion_type}** → 목적: {purpose}")

        with st.spinner("미션 생성 중..."):
            raw, is_wildcard, sources = get_mission(
                client, inp["mood"], f"{inp['minutes']}분", inp["minutes"],
                chunks, embeddings, emotion_type, data, bm25=bm25,
            )
            mission = parse_mission(raw, is_wildcard, sources)

        if not mission["mission"]:
            st.error("미션 생성에 실패했습니다. 다시 시도해주세요.")
            st.session_state["mission_step"] = "input"
            st.rerun()

        st.session_state["current_mission"] = mission
        st.session_state["mission_step"] = "confirm"
        st.rerun()

    # ── step: confirm ────────────────────────────────────────
    elif step == "confirm":
        mission = st.session_state["current_mission"]
        inp     = st.session_state["mission_input"]
        emotion_type = st.session_state.get("emotion_type", "중립")

        card_css = "wildcard-card" if mission["is_wildcard"] else "mission-card"
        cat_color = CAT.get(mission["category"], {}).get("color", "#333")
        diff_color = DIFF.get(mission["difficulty"], {}).get("color", "#333")
        diff_sym   = DIFF.get(mission["difficulty"], {}).get("sym", "")
        cat_sym    = CAT.get(mission["category"], {}).get("sym", "")
        cat_lbl    = CAT.get(mission["category"], {}).get("label", mission["category"])

        if mission["is_wildcard"]:
            st.markdown("### ⚡ 돌발 미션 발동!")
        else:
            emoji   = EMOTION_EMOJI.get(emotion_type, "")
            purpose = EMOTION_PURPOSE.get(emotion_type, "")
            st.markdown(f"**{emoji} {emotion_type}** → {purpose}")

        st.markdown(f"""
        <div class="{card_css}">
          <div style="font-size:1.25rem;font-weight:700;margin-bottom:.8rem;">📋 {mission['mission']}</div>
          <span class="tag" style="background:{cat_color}20;color:{cat_color};border:1px solid {cat_color};">
            {cat_sym} {cat_lbl}
          </span>
          <span class="tag" style="background:{diff_color}20;color:{diff_color};border:1px solid {diff_color};">
            {diff_sym} {mission['difficulty']}
          </span>
          <div class="divider"></div>
          <div style="font-size:.9rem;color:#555;"><strong>📖 근거</strong><br>{mission.get('basis','')}</div>
          <div style="margin-top:.8rem;font-size:.9rem;color:#555;"><strong>✨ 효과</strong><br>{mission.get('effect','')}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ 수락하고 시작!", type="primary", use_container_width=True):
                st.session_state["mission_step"] = "doing"
                st.rerun()
        with c2:
            if st.button("🔄 다른 미션 보기", use_container_width=True):
                with st.spinner("격려 메시지..."):
                    nudge = get_motivational_nudge(client, inp["mood"], chunks, embeddings, bm25)
                st.info(f"💬 {nudge}")
                st.session_state["mission_step"] = "generating"
                st.rerun()
        with c3:
            if st.button("↩ 처음으로", use_container_width=True):
                st.session_state["mission_step"] = "input"
                st.rerun()

    # ── step: doing ──────────────────────────────────────────
    elif step == "doing":
        mission = st.session_state["current_mission"]
        inp     = st.session_state["mission_input"]
        emotion_type = st.session_state.get("emotion_type", "중립")

        st.markdown(f"### 🎯 {mission['mission']}")
        st.info("미션을 수행하고 결과를 선택해주세요.")

        result = st.radio("미션 결과", ["성공", "실패"], horizontal=True)
        memo   = st.text_input("메모 (선택)", placeholder="오늘 어땠나요?")

        if st.button("완료 기록하기", type="primary", use_container_width=True):
            success = (result == "성공")

            if not success:
                data["cards"].append({"card": "도전 카드", "difficulty": "도전"})
                save_data(data)
                st.success("😊 시도한 용기에 도전 카드를 드립니다! 🌱")
                st.session_state["mission_step"] = "input"
                st.rerun()
                return

            with st.spinner("인사이트 생성 중..."):
                insight = get_insight(client, mission["mission"], chunks, embeddings, bm25)
            st.markdown(f'<div class="insight-box">🔬 {insight}</div>', unsafe_allow_html=True)

            with st.spinner("미션 요약 중..."):
                short_name = summarize_mission(client, mission["mission"])

            # 콤보 계산
            combo_before = data.get("combo_count", 0)
            if mission["category"] != "돌발":
                last = data.get("last_category")
                if last == mission["category"]:
                    data["combo_count"] = combo_before + 1
                else:
                    data["combo_count"] = 1
                data["last_category"] = mission["category"]
            combo_after = data.get("combo_count", 0)

            # 콤보 보너스 카드
            if mission["category"] != "돌발" and combo_after >= 2:
                if combo_after >= 3:
                    data["cards"].append({"card": "골드 카드", "difficulty": "최상"})
                    st.markdown('<div class="combo-banner">🔥 3연속 콤보! 골드 카드 획득!</div>',
                                unsafe_allow_html=True)
                else:
                    card_name, diff = COMBO2_CARD.get(mission["category"], ("씨앗 카드", "하"))
                    data["cards"].append({"card": card_name, "difficulty": diff})
                    st.markdown(f'<div class="combo-banner">🔥 2연속 콤보! {card_name} 획득!</div>',
                                unsafe_allow_html=True)

            data["fruits"].append({
                "difficulty":   mission["difficulty"],
                "category":     mission["category"],
                "mission":      short_name,
                "full_mission": mission["mission"],
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                "mood":         inp["mood"],
                "time":         f"{inp['minutes']}분",
                "emotion_type": emotion_type,
                "effect":       mission.get("effect", ""),
                "success":      True,
                "memo":         memo if memo else None,
            })
            data.setdefault("mission_history", []).append(mission["mission"])
            if len(data["mission_history"]) > 20:
                data["mission_history"] = data["mission_history"][-20:]

            save_data(data)
            st.session_state["data"] = data
            st.success(f"🎉 열매 획득! [{mission['category']}] {short_name}")
            st.session_state["mission_step"] = "input"


# ════════════════════════════════════════════════════════════
#  미션 히스토리 페이지
# ════════════════════════════════════════════════════════════

def page_history():
    st.header("📋 미션 히스토리")
    data   = st.session_state["data"]
    fruits = [f for f in data.get("fruits", []) if "timestamp" in f]

    if not fruits:
        st.info("아직 기록된 미션이 없습니다.")
        return

    render_mission_freq_bar(fruits)

    st.divider()
    mission_counts = Counter(f.get("full_mission") or f.get("mission", "") for f in fruits)
    seen: set[str] = set()

    for f in reversed(fruits):
        full = f.get("full_mission") or f.get("mission", "")
        if full in seen:
            continue
        seen.add(full)

        cat      = f.get("category", "")
        cat_info = CAT.get(cat, {"color": "#999", "label": cat, "sym": ""})
        short    = f.get("mission", "")
        count    = mission_counts[full]
        ts       = f.get("timestamp", "")
        mood     = f.get("mood", "")

        badge = f'<span style="color:#f1c40f;font-weight:700;">×{count}회</span>' if count > 1 else ""
        st.markdown(f"""
        <div style="border-left:4px solid {cat_info['color']};padding:.6rem 1rem;margin:.5rem 0;">
          <span class="tag" style="background:{cat_info['color']}20;color:{cat_info['color']};
                border:1px solid {cat_info['color']};">{cat_info['sym']} {cat_info['label']}</span>
          {badge}
          <strong style="margin-left:6px;">{short}</strong>
          <div style="font-size:.82rem;color:#888;margin-top:4px;">{ts} · 기분: {mood[:20]}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  열매 & 카드 페이지
# ════════════════════════════════════════════════════════════

def page_cards():
    st.header("🌿 열매 & 카드")
    data   = st.session_state["data"]
    fruits = data.get("fruits", [])
    cards  = data.get("cards", [])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"열매 {len(fruits)}/{MAX_FRUITS}개")
        render_tree_chart(fruits)

        st.markdown("#### 열매 쪼개기 → 카드 변환")
        if fruits:
            options = [f"[{f.get('category','')}] {f.get('mission','')[:20]}" for f in fruits]
            sel_idx = st.selectbox("쪼갤 열매 선택", range(len(fruits)),
                                   format_func=lambda i: options[i])
            if st.button("🔪 쪼개기", use_container_width=True):
                fruit = fruits.pop(sel_idx)
                d     = fruit["difficulty"]
                card  = DIFF.get(d, DIFF["하"])["card"]
                data["cards"].append({"card": card, "difficulty": d})
                save_data(data)
                st.session_state["data"] = data
                st.success(f"✨ {card} 획득!")
                st.rerun()
        else:
            st.info("쪼갤 열매가 없습니다.")

    with c2:
        st.subheader(f"카드 {len(cards)}장")
        if cards:
            cnt = Counter(c["card"] for c in cards)
            for card, n in cnt.most_common():
                diff = next((c["difficulty"] for c in cards if c["card"] == card), "하")
                color = DIFF.get(diff, {}).get("color", "#333")
                st.markdown(f"""
                <div style="border:2px solid {color};border-radius:10px;
                            padding:.7rem 1.1rem;margin:.4rem 0;
                            box-shadow:3px 3px 0 {color}20;">
                  <span style="font-size:1.1rem;font-weight:700;">{card}</span>
                  <span style="color:{color};margin-left:8px;">×{n}장</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("보유 카드가 없습니다.")


# ════════════════════════════════════════════════════════════
#  내가 걸어온 길 페이지
# ════════════════════════════════════════════════════════════

def page_journey(client, chunks, embeddings):
    st.header("🌿 내가 걸어온 길")
    data      = st.session_state["data"]
    fruits    = data.get("fruits", [])
    successes = [f for f in fruits if f.get("success")]
    total     = len(fruits)
    s_count   = len(successes)

    # 통계 행
    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, total,   "전체 미션"),
        (c2, s_count, "완료"),
        (c3, total - s_count, "미완료"),
        (c4, len(data.get("cards", [])), "보유 카드"),
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
        cat      = f.get("category", "")
        cat_info = CAT.get(cat, {"color": "#999", "label": cat, "sym": ""})
        ts       = f.get("timestamp", "")
        mood     = f.get("mood", "")
        short    = f.get("mission", "")
        effect   = f.get("effect", "")
        memo     = f.get("memo")

        st.markdown(f"""
        <div style="border-left:4px solid {cat_info['color']};padding:.7rem 1rem;margin:.5rem 0;">
          <div style="font-size:.8rem;color:#888;">{ts}</div>
          <span class="tag" style="background:{cat_info['color']}20;color:{cat_info['color']};
                border:1px solid {cat_info['color']};margin-top:4px;">
            {cat_info['sym']} {cat_info['label']}
          </span>
          <strong style="margin-left:6px;">{short}</strong>
          <div style="font-size:.85rem;color:#555;margin-top:4px;">기분: {mood[:30]}</div>
          {f'<div style="font-size:.82rem;color:#888;">효과: {effect[:50]}</div>' if effect else ''}
          {f'<div style="font-size:.82rem;color:#aaa;font-style:italic;">📝 {memo}</div>' if memo else ''}
        </div>
        """, unsafe_allow_html=True)

    # 논문 커버리지 분석 (성공 미션 3개 이상)
    if len(successes) >= 3:
        st.divider()
        st.subheader("📊 논문 활용도 분석")
        if st.button("분석 실행", use_container_width=False):
            with st.spinner("분석 중..."):
                coverage = analyze_coverage(client, data, chunks, embeddings)
            st.session_state["data"] = data
            save_data(data)
            if coverage:
                weak = coverage.get("weak", [])
                if weak:
                    st.info(f"🔍 다음 미션에서 이 논문들이 우선 반영됩니다: **{', '.join(weak)}**")
                render_coverage_bar(coverage)


# ════════════════════════════════════════════════════════════
#  사이드바
# ════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌱 피어나기")
        st.markdown("작은 행동 하나가 나를 피워냅니다.")
        st.divider()

        data   = st.session_state.get("data", {})
        fruits = data.get("fruits", [])
        combo  = data.get("combo_count", 0)
        last   = data.get("last_category")

        # 진행 바
        progress_val = min(len(fruits) / MAX_FRUITS, 1.0)
        st.progress(progress_val, text=f"나무 {len(fruits)}/{MAX_FRUITS}")

        # 콤보 배너
        if combo >= 2 and last:
            cat_info = CAT.get(last, {"label": last})
            st.markdown(f'<div class="combo-banner">🔥 {cat_info["label"]} {combo}연속 콤보!</div>',
                        unsafe_allow_html=True)

        st.divider()
        page = st.radio(
            "메뉴",
            ["🌱 미션 시작", "📋 미션 히스토리", "🌿 열매 & 카드", "✨ 내가 걸어온 길"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(f"<small style='color:#888;'>열매: {len(fruits)} · 카드: {len(data.get('cards',[]))}</small>",
                    unsafe_allow_html=True)

    return page


# ════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="🌱 피어나기",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    # ── API 키 확인 ──────────────────────────────────────────
    try:
        client = get_client()
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()

    # ── 세션 초기화 ──────────────────────────────────────────
    if "data" not in st.session_state:
        st.session_state["data"] = load_data()
    if "mission_step" not in st.session_state:
        st.session_state["mission_step"] = "input"

    # ── 인덱스 로드 ──────────────────────────────────────────
    import os
    chunks, embeddings, bm25 = build_index_cached(os.environ.get("OPENAI_API_KEY", ""))

    # ── 네비게이션 ───────────────────────────────────────────
    page = render_sidebar()

    if page == "🌱 미션 시작":
        page_mission(client, chunks, embeddings, bm25)
    elif page == "📋 미션 히스토리":
        page_history()
    elif page == "🌿 열매 & 카드":
        page_cards()
    elif page == "✨ 내가 걸어온 길":
        page_journey(client, chunks, embeddings)


if __name__ == "__main__":
    main()
