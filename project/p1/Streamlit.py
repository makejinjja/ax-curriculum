#!/usr/bin/env python3
import os
import re
import sys
import numpy as np
import streamlit as st
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env")

PDF_PATH = Path(__file__).parent.parent / "data" / "inf.pdf.pdf"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Removal AI",
    page_icon="◼",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* reset & font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* page background */
  .stApp { background: #ffffff; }

  /* hide default header / hamburger */
  #MainMenu, header, footer { visibility: hidden; }

  /* hero section */
  .hero {
    border-bottom: 2px solid #000;
    padding: 2.5rem 0 1.5rem 0;
    margin-bottom: 2rem;
  }
  .hero-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #000;
    margin: 0;
  }
  .hero-sub {
    font-size: 0.85rem;
    color: #555;
    margin-top: 0.4rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  /* section label */
  .section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.6rem;
  }

  /* input card */
  .input-card {
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
  }

  /* radio / select overrides */
  .stRadio > div { gap: 0.4rem; }
  .stRadio label { font-size: 0.9rem !important; color: #222 !important; }

  /* submit button */
  .stButton > button {
    background: #000 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 2rem !important;
    width: 100%;
    transition: opacity 0.15s;
  }
  .stButton > button:hover { opacity: 0.75 !important; }

  /* result: primary action card */
  .action-card {
    background: #000;
    color: #fff;
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
  }
  .action-card .action-label {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.6rem;
  }
  .action-card .action-text {
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1.35;
    letter-spacing: -0.02em;
  }

  /* result: detail card */
  .detail-card {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    background: #fff;
  }
  .detail-card .card-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #000;
    border-bottom: 1.5px solid #000;
    padding-bottom: 0.45rem;
    margin-bottom: 0.85rem;
  }
  .detail-card .card-body {
    font-size: 0.9rem;
    color: #222;
    line-height: 1.7;
  }
  .detail-card .card-body p { margin: 0.25rem 0; }

  /* tag chips */
  .tag {
    display: inline-block;
    background: #f0f0f0;
    color: #333;
    font-size: 0.72rem;
    font-weight: 500;
    border-radius: 2px;
    padding: 0.2rem 0.55rem;
    margin: 0.15rem 0.15rem 0.15rem 0;
    letter-spacing: 0.04em;
  }

  /* time badge */
  .time-badge {
    display: inline-block;
    background: #000;
    color: #fff;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 3px;
    padding: 0.3rem 0.8rem;
  }

  /* divider */
  .divider { border-top: 1px solid #e8e8e8; margin: 1.5rem 0; }

  /* spinner override */
  .stSpinner > div { border-top-color: #000 !important; }
</style>
""", unsafe_allow_html=True)


# ── RAG helpers (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None, [], []
    client = OpenAI(api_key=api_key)
    reader = PdfReader(str(PDF_PATH))
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    chunks = _chunk_by_sections(raw_text)
    embeddings = _get_embeddings(client, [c["embed_text"] for c in chunks])
    return client, chunks, embeddings


def _chunk_by_sections(text: str) -> list[dict]:
    tab_labels = {
        "1": "각성-동기 상태 모델",
        "2": "개인 최적화 전략 유형",
        "3": "에너지 상태 분류",
        "4": "Circadian Rhythm 기반 시간대별 최적화",
    }
    tab_pattern = re.compile(
        r"탭\s*(?P<num>[1-4])\s*\n(?P<content>.*?)(?=탭\s*[1-4]|\Z)", re.DOTALL
    )
    sec_pattern = re.compile(
        r"(?m)^(?P<title>[1-4]\.\s+.+?(?:\(.+?\))?)\s*\n(?P<body>.*?)(?=^[1-4]\.|^탭\s*[1-4]|\Z)",
        re.DOTALL,
    )
    chunks = []
    for tm in tab_pattern.finditer(text):
        tab_label = tab_labels.get(tm.group("num"), f"탭{tm.group('num')}")
        for sm in sec_pattern.finditer(tm.group("content")):
            body = sm.group("body").strip()
            if len(body) < 30:
                continue
            chunks.append({
                "tab": tab_label,
                "title": sm.group("title").strip(),
                "body": body,
                "embed_text": f"[{tab_label}] {sm.group('title').strip()}\n{body}",
            })
    if not chunks:
        for i, p in enumerate(re.split(r"\n{2,}", text)):
            if len(p.strip()) > 80:
                chunks.append({"tab": "전체", "title": f"단락{i+1}", "body": p.strip(), "embed_text": p.strip()})
    return chunks


def _get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]


def _cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def _retrieve(query_emb, chunk_embs, chunks, k=4) -> list[dict]:
    scored = sorted(
        zip([_cosine_sim(query_emb, e) for e in chunk_embs], chunks),
        key=lambda x: x[0], reverse=True,
    )
    return [c for _, c in scored[:k]]


def _build_context(items: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[{i['tab']}] {i['title']}\n{i['body']}" for i in items)


def _build_query(state: dict) -> str:
    mood_map = {
        "무기력": "무기력형 Low Arousal Low Motivation 행동 개시 어려움 도파민 저활성 에너지 저하",
        "스트레스": "스트레스형 High Arousal Negative Emotion 불안 편도체 과활성 전전두엽 저하",
        "안정": "안정형 Moderate Arousal Positive Emotion 세로토닌 균형 회복 부교감신경",
        "집중됨": "집중형 Optimal Arousal High Engagement Flow 도파민 노르에피네프린 전전두엽",
    }
    goal_map = {
        "건강": "건강 중심형 Energy Maximizer 수면 운동 식단 회복 Self-Regulation",
        "생산성": "생산성 중심형 Time Optimizer Deep Work 집중 시간블록 전두엽",
        "돈": "돈 중심형 Capital Maximizer ROI 레버리지 도파민 보상",
        "균형": "균형 중심형 Well-being Optimizer 삶의 만족 자율성 관계",
    }
    energy_map = {
        "낮음": "에너지 저하 Low Energy State 피로 회복 자율신경 불균형",
        "보통": "에너지 균형 Balanced State 항상성 Homeostasis 교감 부교감",
        "높음": "에너지 활성 High Energy State 도파민 보상 집중력 선순환",
    }
    hour = int(state["현재 시간"].split(":")[0]) if ":" in state["현재 시간"] else 12
    if 5 <= hour < 10:
        time_ctx = "아침 코르티솔 각성 반응 의사결정 집중 작업 개시"
    elif 10 <= hour < 13:
        time_ctx = "오전 고집중 작업 구간 인지 속도 논리 학습"
    elif 13 <= hour < 15:
        time_ctx = "점심 후 졸음 Post-lunch Dip 낮잠 회복"
    elif 15 <= hour < 18:
        time_ctx = "오후 사회적 상호작용 감정 인식 의사소통"
    elif 18 <= hour < 21:
        time_ctx = "저녁 운동 최적화 체온 최고점 근력 유연성"
    else:
        time_ctx = "밤 창의적 사고 뇌 억제 약화 수면 준비"
    return (
        f"{mood_map.get(state['기분'], state['기분'])} "
        f"{goal_map.get(state['목표'], state['목표'])} "
        f"{energy_map.get(state['에너지 상태'], state['에너지 상태'])} "
        f"{time_ctx} 상황:{state['현재 상황']} 가용시간:{state['사용 가능 시간']}"
    )


SYSTEM_PROMPT = """너는 "Decision Removal AI"다.
아래 [심리·신경과학 지식 베이스]에서 검색된 이론적 근거를 참고하여
사용자의 현재 상태에 가장 적합한 "지금 당장 해야 할 행동 1개"를 결정해주는 시스템이다.

[핵심 원칙]
1. 항상 단 하나의 행동만 제시한다.
2. 행동은 5~30분 내 실행 가능한 것이어야 한다.
3. 모호한 조언 금지 (ex. "운동해라" → "15분 걷기")
4. 실행 가능성이 가장 높은 행동을 선택한다.
5. 사용자의 현재 상태를 최우선으로 반영한다.

[출력 형식] 반드시 아래 형식으로만 출력한다:

[행동]
- (지금 당장 해야 할 단 하나의 행동)

[심리·생리 상태 분석]
- 각성-동기 상태: (무기력형/스트레스형/안정형/집중형 중 해당 유형과 특징 1~2줄)
- 에너지 상태: (에너지 저하/균형/활성 상태 판단 및 생리적 근거 1줄)
- 시간대 최적화: (현재 시간대의 Circadian Rhythm 특성과 적합 활동 1줄)

[근거]
- 상태 이론: (각성-동기 모델 또는 에너지 상태 분류에서 이 행동이 도출되는 신경과학적 이유 2~3줄)
- 목표 연계: (사용자 목표 유형의 전략 특성과 행동의 연결고리 1~2줄)
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


def get_decision(client: OpenAI, state: dict, chunks, chunk_embs) -> str:
    query_emb = _get_embeddings(client, [_build_query(state)])[0]
    top = _retrieve(query_emb, chunk_embs, chunks, k=4)
    context = _build_context(top)
    user_msg = (
        "[사용자 현재 상태]\n"
        + "\n".join(f"- {k}: {v}" for k, v in state.items())
        + f"\n\n[심리·신경과학 지식 베이스 - 관련 이론]\n{context}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content


# ── Response parser ─────────────────────────────────────────────────────────────
def parse_response(text: str) -> dict:
    sections = {}
    pattern = re.compile(r"\[([^\]]+)\]\s*\n(.*?)(?=\n\[|\Z)", re.DOTALL)
    for m in pattern.finditer(text):
        key = m.group(1).strip()
        body = m.group(2).strip()
        sections[key] = body
    return sections


def bullet_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = line.strip().lstrip("-•·").strip()
        if line:
            lines.append(line)
    return lines


# ── UI rendering ────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero">
      <p class="hero-title">◼ Decision Removal AI</p>
      <p class="hero-sub">Neuroscience-backed · Single Action · Zero Decision Fatigue</p>
    </div>
    """, unsafe_allow_html=True)


def render_state_tags(state: dict):
    tags_html = "".join(f'<span class="tag">{v}</span>' for v in state.values())
    st.markdown(f'<div style="margin-bottom:1.2rem">{tags_html}</div>', unsafe_allow_html=True)


def render_result(raw: str, state: dict):
    sections = parse_response(raw)

    # ── 행동 ──
    action_text = ""
    if "행동" in sections:
        lines = bullet_lines(sections["행동"])
        action_text = lines[0] if lines else sections["행동"]

    st.markdown(f"""
    <div class="action-card">
      <div class="action-label">◼ 지금 당장 할 행동</div>
      <div class="action-text">{action_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 소요 시간 ──
    if "소요 시간" in sections:
        t = bullet_lines(sections["소요 시간"])
        t_text = t[0] if t else sections["소요 시간"]
        st.markdown(f'<div style="margin-bottom:1.2rem"><span class="time-badge">⏱ {t_text}</span></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 입력 상태 태그 ──
    st.markdown('<div class="section-label">입력 상태</div>', unsafe_allow_html=True)
    render_state_tags(state)

    # ── 심리·생리 상태 분석 ──
    analysis_key = next((k for k in sections if "심리" in k or "분석" in k), None)
    if analysis_key:
        body = sections[analysis_key]
        rows = []
        for line in body.splitlines():
            line = line.strip().lstrip("-").strip()
            if ":" in line:
                label, content = line.split(":", 1)
                rows.append(f"<p><strong>{label.strip()}</strong>: {content.strip()}</p>")
            elif line:
                rows.append(f"<p>{line}</p>")
        st.markdown(f"""
        <div class="detail-card">
          <div class="card-title">심리·생리 상태 분석</div>
          <div class="card-body">{''.join(rows)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 근거 ──
    rationale_key = next((k for k in sections if "근거" in k), None)
    if rationale_key:
        body = sections[rationale_key]
        rows = []
        for line in body.splitlines():
            line = line.strip().lstrip("-").strip()
            if ":" in line:
                label, content = line.split(":", 1)
                rows.append(f"<p><strong>{label.strip()}</strong>: {content.strip()}</p>")
            elif line:
                rows.append(f"<p>{line}</p>")
        st.markdown(f"""
        <div class="detail-card">
          <div class="card-title">근거</div>
          <div class="card-body">{''.join(rows)}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 실행 방법 ──
    how_key = next((k for k in sections if "실행" in k), None)
    if how_key:
        lines = bullet_lines(sections[how_key])
        rows = "".join(f"<p>→ {l}</p>" for l in lines)
        st.markdown(f"""
        <div class="detail-card">
          <div class="card-title">실행 방법</div>
          <div class="card-body">{rows}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    render_hero()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.stop()

    with st.spinner("지식 베이스 로딩 중..."):
        client, chunks, chunk_embs = load_knowledge_base()

    if not chunks:
        st.error("지식 베이스를 불러올 수 없습니다. PDF 경로를 확인하세요.")
        st.stop()

    # ── Input form ──
    st.markdown('<div class="section-label">현재 상태 입력</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        time_now = st.text_input("현재 시간", placeholder="예: 14:30", label_visibility="visible")
        energy = st.radio("에너지 상태", ["낮음", "보통", "높음"], horizontal=True)
        mood = st.radio("기분", ["무기력", "스트레스", "안정", "집중됨"], horizontal=True)
    with col2:
        location = st.radio("현재 상황", ["집", "회사", "이동중"], horizontal=True)
        goal = st.radio("목표", ["건강", "생산성", "돈", "균형"], horizontal=True)
        available_time = st.number_input("사용 가능 시간 (분)", min_value=5, max_value=120, value=20, step=5)

    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.button("◼ 지금 당장 할 행동 결정")

    if submitted:
        if not time_now.strip():
            st.warning("현재 시간을 입력해주세요.")
            st.stop()

        state = {
            "현재 시간": time_now.strip(),
            "에너지 상태": energy,
            "기분": mood,
            "현재 상황": location,
            "목표": goal,
            "사용 가능 시간": f"{available_time}분",
        }

        with st.spinner("분석 중..."):
            raw = get_decision(client, state, chunks, chunk_embs)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        render_result(raw, state)


if __name__ == "__main__":
    main()
