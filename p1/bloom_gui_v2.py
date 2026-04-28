#!/usr/bin/env python3
"""
bloom_gui_v2.py
기분전환 미션 AI + RAG — Streamlit GUI v2 (UI 리디자인)
"""
import os, json, re, random, hashlib
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env")

DATA_FILE  = Path(__file__).parent / ".mission_data.json"
PDF_DIR    = Path(__file__).parent.parent / "data"
CACHE_FILE = Path(__file__).parent / ".index_cache" / "mission_rag_v7_index.json"
MAX_FRUITS = 30

# ── Color Palette ─────────────────────────────────────────────────────────────
PRIMARY   = "#4CAF50"
SURFACE   = "#1E1E2E"
BORDER    = "#2E2E42"
TEXT_MAIN = "#E8E8F0"
TEXT_SUB  = "#888899"
DANGER    = "#E05C5C"

PDF_FILES = [
    ("01.CBT.pdf",     "CBT(인지행동치료)"),
    ("02.behav.pdf",   "행동 활성화 1"),
    ("03.behav.pdf",   "행동 활성화 2"),
    ("04.mind.pdf",    "마음챙김"),
    ("05.emotion.pdf", "감정 조절 전략"),
]

CAT = {
    "건강":   {"label": "건강",   "color": "#FF6B6B"},
    "생산성": {"label": "생산성", "color": "#4DABF7"},
    "재미":   {"label": "재미",   "color": "#CC5DE8"},
    "성장":   {"label": "성장",   "color": "#51CF66"},
    "돌발":   {"label": "돌발",   "color": "#FFD43B"},
}

DIFF = {
    "하":   {"card": "씨앗 카드",      "color": "#FF6B6B", "emoji": "🌱"},
    "중":   {"card": "새싹 카드",      "color": "#E8823A", "emoji": "🌿"},
    "상":   {"card": "햇살 카드",      "color": "#A8A8A8", "emoji": "☀️"},
    "최상": {"card": "황금 열매 카드", "color": "#FFD700", "emoji": "🌟"},
    "돌발": {"card": "번개 카드",      "color": "#FFD43B", "emoji": "⚡"},
    "도전": {"card": "도전 카드",      "color": "#4DABF7", "emoji": "💪"},
}

COMBO2_CARD = {
    "건강":   ("회복 카드",  "중"),
    "성장":   ("지식 카드",  "중"),
    "재미":   ("행운 카드",  "중"),
    "생산성": ("집중 카드",  "중"),
}

EMOTION_PURPOSE = {
    "부정적": "기분전환·회복",
    "중립":   "생산성·성장",
    "긍정적": "도전·재미·확장",
    "집중됨": "딥워크·몰입",
    "지루함": "자극·탐험",
}

EMOTION_EMOJI = {
    "부정적": "😔", "중립": "😐", "긍정적": "😊",
    "집중됨": "🎯", "지루함": "😑",
}

EMOTION_SOURCE_WEIGHT = {
    "부정적": {"CBT": 2.0, "행동 활성화": 1.5},
    "중립":   {"행동 활성화": 1.5, "마음챙김": 1.2},
    "긍정적": {"감정 조절": 1.5},
    "집중됨": {"마음챙김": 2.0},
    "지루함": {"행동 활성화": 2.0},
}

EMOTION_CAT_QUERY = {
    ("부정적", "건강"):   "physical activity exercise mood recovery depression fatigue",
    ("부정적", "재미"):   "pleasant activity scheduling reward mood lift behavioral",
    ("부정적", "성장"):   "cognitive reappraisal meaning making post-traumatic growth",
    ("부정적", "생산성"): "behavioral activation low energy task initiation depression",
    ("중립",   "생산성"): "motivation task initiation self-efficacy goal setting",
    ("중립",   "성장"):   "deliberate practice habit formation incremental improvement",
    ("중립",   "재미"):   "engagement novelty positive activity scheduling",
    ("긍정적", "재미"):   "savoring positive experience novelty exploration reward",
    ("긍정적", "성장"):   "strengths challenge goal pursuit self-efficacy mastery",
    ("긍정적", "생산성"): "flow state deep work peak performance engagement",
    ("집중됨", "생산성"): "flow state deep work cognitive engagement sustained attention",
    ("집중됨", "성장"):   "deliberate practice skill building mastery concentration",
    ("지루함", "재미"):   "boredom engagement activation stimulation arousal novelty",
    ("지루함", "생산성"): "boredom activation pleasant activity behavioral engagement",
    ("지루함", "성장"):   "curiosity exploration new experience boredom relief",
}

MOOD_QUERY_MAP = {
    "무기력": "low motivation lethargy behavioral activation energy low mood depression",
    "우울":   "depression low mood behavioral activation CBT cognitive distortion",
    "스트레스": "stress anxiety rumination cognitive reappraisal mindfulness relaxation",
    "불안":   "anxiety worry cognitive restructuring exposure mindfulness breathing",
    "짜증":   "irritability anger emotion regulation distraction reappraisal",
    "피곤":   "fatigue mental exhaustion restorative activity rest recovery",
    "지루":   "boredom engagement activation pleasant activity scheduling",
    "행복":   "positive emotion well-being savoring gratitude strengths",
    "좋":     "positive mood well-being engagement flow activity",
    "집중":   "focus flow state deep work cognitive engagement productivity",
    "설렘":   "positive arousal excitement approach motivation novelty seeking",
}

# ── Prompts ───────────────────────────────────────────────────────────────────
CLASSIFY_PROMPT = """사용자의 기분 입력을 보고 감정 유형을 분류해라.

분류 기준:
- 부정적: 우울, 무기력, 스트레스, 불안, 짜증, 피곤, 슬픔, 힘듦
- 중립: 보통, 평범, 모르겠음, 그냥, 별로 특별한 감정 없음
- 긍정적: 행복, 설렘, 에너지 넘침, 기분 좋음, 성취감, 뿌듯함
- 집중됨: 몰입, 집중, 의욕, 하고 싶은 것이 있음
- 지루함: 지루, 심심, 무료, 할 게 없음

반드시 아래 형식으로만 출력 (다른 말 금지):
부정적 또는 중립 또는 긍정적 또는 집중됨 또는 지루함"""

WILDCARD_PROMPT = """너는 돌발 미션 생성 AI다.
감정과 상관없이 기묘하거나 유쾌하고 즉흥적인 초단기 미션 1개를 제안한다.

규칙:
- 예상치 못한 행동, 사소하지만 재미있는 것
- 반드시 5분 이내 완료 가능
- 난이도는 항상 "하" 고정
- [카테고리]는 항상 "돌발" 고정

반드시 아래 형식으로만 출력:

[미션]
- (구체적 행동 1개, 10~25자 이내)

[카테고리]
- 돌발

[난이도]
- 하

[근거]
- 돌발 자극을 통한 주의 전환 및 각성 효과

[효과]
- (유쾌하거나 의외의 긍정 효과 1줄)"""


def make_mission_prompt(emotion_type: str, purpose: str,
                        recent_missions: str = "없음",
                        forced_difficulty: str = "하") -> str:
    return f"""너는 기분전환 미션 AI다.
아래 [심리학 논문 근거]에 제시된 과학적 개입법을 바탕으로 미션 1개를 제안한다.

[감정 분류 결과]
- 감정 유형: {emotion_type}
- 미션 목적: {purpose}

[최근 수행 미션 — 아래와 겹치지 않는 미션을 제안해라]
{recent_missions}

규칙:
- 반드시 위 미션 목적({purpose})에 맞는 미션을 제안해라
- 매번 다른 미션 (최대한 랜덤하게, 최근 미션과 겹치지 않게)
- 미션 소요 시간은 반드시 가용 시간과 일치해야 함
- 난이도는 반드시 "{forced_difficulty}"로 고정
- [심리학 논문 근거]의 내용을 실제로 반영해 미션을 생성할 것
- [카테고리]는 미션 내용을 보고 건강/생산성/재미/성장 중 하나를 선택

반드시 아래 형식으로만 출력:

[미션]
- (구체적 행동 1개, 10~30자 이내)

[카테고리]
- 건강 또는 생산성 또는 재미 또는 성장

[난이도]
- {forced_difficulty}

[근거]
- (적용된 심리학 기법/이론 1줄)

[효과]
- (성공 시 심리·신체 효과 1~2줄)"""


# ── 데이터 I/O ────────────────────────────────────────────────────────────────
def load_data() -> dict:
    if DATA_FILE.exists():
        d = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        d.setdefault("last_category", None)
        d.setdefault("combo_count", 0)
        d.setdefault("mission_history", [])
        return d
    return {"fruits": [], "cards": [], "last_category": None,
            "combo_count": 0, "mission_history": []}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_data() -> dict:
    if "app_data" not in st.session_state:
        st.session_state.app_data = load_data()
    return st.session_state.app_data


def commit_data(data: dict):
    save_data(data)
    st.session_state.app_data = data


# ── RAG ───────────────────────────────────────────────────────────────────────
def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


MAX_CHUNK_CHARS = 6000


def _chunk(text: str, source: str) -> list[dict]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks, idx = [], 0
    for para in paragraphs:
        para = para.strip()
        if len(para) < 80:
            continue
        for start in range(0, len(para), MAX_CHUNK_CHARS):
            chunks.append({"text": para[start:start + MAX_CHUNK_CHARS],
                           "source": source, "chunk_index": idx})
            idx += 1
    return chunks


@st.cache_resource
def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.markdown('<div class="msg msg-warn">OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.</div>',
                    unsafe_allow_html=True)
        st.stop()
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner="📚 논문 인덱싱 중... (최초 1회)")
def get_index() -> tuple[list[dict], list[list[float]]]:
    client = get_client()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    hashes = {fname: _file_hash(PDF_DIR / fname)
              for fname, _ in PDF_FILES if (PDF_DIR / fname).exists()}

    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cache.get("hashes") == hashes:
            return cache["chunks"], cache["embeddings"]

    all_chunks: list[dict] = []
    for fname, label in PDF_FILES:
        p = PDF_DIR / fname
        if p.exists():
            all_chunks.extend(_chunk(_extract_text(p), f"{fname} ({label})"))

    texts = [c["text"] for c in all_chunks]
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), 512):
        resp = client.embeddings.create(model="text-embedding-3-small",
                                        input=texts[i:i + 512])
        embeddings.extend([item.embedding for item in resp.data])

    CACHE_FILE.write_text(
        json.dumps({"hashes": hashes, "chunks": all_chunks, "embeddings": embeddings},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return all_chunks, embeddings


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def _time_query(minutes: int) -> str:
    if minutes <= 10:
        return "brief intervention micro-habit short activity immediate"
    elif minutes <= 40:
        return "moderate duration activity engagement exercise"
    return "extended activity deep work immersive exercise"


def _expand_query(mood: str, minutes: int,
                  emotion_type: str | None = None,
                  target_cat: str | None = None) -> str:
    if emotion_type and target_cat:
        kw = EMOTION_CAT_QUERY.get((emotion_type, target_cat))
        if kw:
            return f"{kw} {_time_query(minutes)} psychological intervention evidence-based"
    mood_kw = next((v for k, v in MOOD_QUERY_MAP.items() if k in mood),
                   "emotion regulation mood improvement well-being intervention")
    return f"{mood_kw} {_time_query(minutes)} psychological intervention evidence-based"


def retrieve(query_emb, chunks, embeddings, k=4, emotion_type=None) -> list[dict]:
    weights = EMOTION_SOURCE_WEIGHT.get(emotion_type, {}) if emotion_type else {}
    scored = []
    for emb, chunk in zip(embeddings, chunks):
        score = _cosine(query_emb, emb)
        for kw, mult in weights.items():
            if kw in chunk.get("source", ""):
                score *= mult
                break
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def build_context(top_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks
    )


# ── AI Functions ──────────────────────────────────────────────────────────────
def classify_emotion(mood: str) -> str:
    resp = get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": CLASSIFY_PROMPT},
                  {"role": "user",   "content": mood}],
        temperature=0.0, max_tokens=10,
    )
    raw = resp.choices[0].message.content.strip()
    return next((k for k in EMOTION_PURPOSE if k in raw), "중립")


def summarize_mission(mission_text: str) -> str:
    resp = get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "미션 텍스트를 10자 이내 짧은 동사형 한국어로 요약해라. 요약문만 출력."},
                  {"role": "user",   "content": mission_text}],
        temperature=0.0, max_tokens=20,
    )
    return resp.choices[0].message.content.strip()


def get_mission(mood: str, minutes: int, emotion_type: str, data: dict) -> tuple[str, bool, list[str]]:
    client = get_client()
    chunks, embeddings = get_index()
    is_wildcard = random.random() < 0.15

    if is_wildcard:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": WILDCARD_PROMPT},
                      {"role": "user",   "content": f"가용 시간: {minutes}분"}],
            temperature=1.0, max_tokens=200,
        )
        return resp.choices[0].message.content, True, []

    forced_difficulty = random.choices(["하", "중", "상", "최상"], weights=[50, 30, 15, 5])[0]
    emotion_to_cat = {"부정적": "건강", "중립": "생산성", "긍정적": "재미",
                      "집중됨": "생산성", "지루함": "재미"}
    target_cat = emotion_to_cat.get(emotion_type)

    query     = _expand_query(mood, minutes, emotion_type=emotion_type, target_cat=target_cat)
    query_emb = client.embeddings.create(model="text-embedding-3-small",
                                         input=[query]).data[0].embedding
    top       = retrieve(query_emb, chunks, embeddings, k=4, emotion_type=emotion_type)
    context   = build_context(top)
    sources   = list(dict.fromkeys(c["source"] for c in top))

    recent     = data.get("mission_history", [])[-5:]
    recent_str = "\n".join(f"- {m}" for m in recent) if recent else "없음"
    purpose    = EMOTION_PURPOSE.get(emotion_type, "기분전환·회복")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": make_mission_prompt(emotion_type, purpose,
                                                               recent_str, forced_difficulty)},
            {"role": "user",   "content": f"현재 기분: {mood}\n가용 시간: {minutes}분\n\n[심리학 논문 근거]\n{context}"},
        ],
        temperature=0.9, max_tokens=350,
    )
    return resp.choices[0].message.content, False, sources


def normalize_difficulty(text: str) -> str:
    return next((d for d in ["최상", "상", "중", "하", "돌발"] if d in text), "하")


def normalize_category(text: str) -> str:
    return next((c for c in ["생산성", "건강", "재미", "성장", "돌발"] if c in text), "건강")


def parse_mission(text: str, is_wildcard: bool, sources: list[str] | None = None) -> dict:
    blocks: dict[str, str] = {}
    current_tag: str | None = None
    current_lines: list[str] = []
    for line in text.split('\n'):
        m = re.match(r'^\[(.+?)\]', line.strip())
        if m:
            if current_tag:
                blocks[current_tag] = '\n'.join(current_lines).strip()
            current_tag, current_lines = m.group(1), []
        elif current_tag:
            content = re.sub(r'^-\s*', '', line.strip())
            if content:
                current_lines.append(content)
    if current_tag:
        blocks[current_tag] = '\n'.join(current_lines).strip()
    return {
        "mission":     blocks.get("미션", ""),
        "category":    "돌발" if is_wildcard else normalize_category(blocks.get("카테고리", "")),
        "difficulty":  "하"   if is_wildcard else normalize_difficulty(blocks.get("난이도", "")),
        "basis":       blocks.get("근거", ""),
        "effect":      blocks.get("효과", ""),
        "is_wildcard": is_wildcard,
        "sources":     sources or [],
    }


def format_effect(effect: str) -> str:
    if not effect:
        return ""
    return effect


def check_combo(data: dict, new_category: str) -> int:
    if new_category == "돌발":
        return 0
    last = data.get("last_category")
    data["combo_count"] = (data.get("combo_count", 0) + 1) if last == new_category else 1
    data["last_category"] = new_category
    return data["combo_count"]


def apply_combo_bonus(data: dict, combo: int, category: str) -> str | None:
    if combo < 2:
        return None
    if combo >= 3:
        data["cards"].append({"card": "골드 카드", "difficulty": "최상"})
        return "골드 카드"
    card_name, diff = COMBO2_CARD.get(category, ("씨앗 카드", "하"))
    data["cards"].append({"card": card_name, "difficulty": diff})
    return card_name


# ── Tree Chart ────────────────────────────────────────────────────────────────
def _tree_positions() -> list[tuple[float, float]]:
    positions = []
    rows = [1, 2, 3, 4, 5, 5, 5, 5]
    y = 8.0
    for count in rows:
        width = (count - 1) * 2.2
        for i in range(count):
            positions.append((-width / 2 + i * 2.2, y))
        y -= 1.0
    return positions


TREE_POS = _tree_positions()


def render_tree_chart(fruits: list) -> go.Figure:
    n   = len(fruits)
    fig = go.Figure()

    for path, opacity in [("M 0 9.3 L -7.2 0.8 L 7.2 0.8 Z", 0.07),
                           ("M 0 8.7 L -5.8 1.8 L 5.8 1.8 Z", 0.07)]:
        fig.add_shape(type="path", path=path,
                      fillcolor=f"rgba(76,175,80,{opacity})",
                      line=dict(color="rgba(76,175,80,0.18)", width=1))

    fig.add_trace(go.Scatter(x=[0, 0], y=[-0.7, 1.0], mode="lines",
                             line=dict(color="#6D4C41", width=12),
                             showlegend=False, hoverinfo="none"))
    for dx in [-1.2, 1.2]:
        fig.add_trace(go.Scatter(x=[0, dx], y=[-0.2, -0.7], mode="lines",
                                 line=dict(color="#6D4C41", width=6),
                                 showlegend=False, hoverinfo="none"))

    if n < MAX_FRUITS:
        ex = [TREE_POS[i][0] for i in range(n, MAX_FRUITS)]
        ey = [TREE_POS[i][1] for i in range(n, MAX_FRUITS)]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers",
                                 marker=dict(size=18, color="rgba(180,180,180,0.12)",
                                             line=dict(width=1, color="rgba(180,180,180,0.2)")),
                                 hoverinfo="none", showlegend=False))

    groups: dict[str, dict] = {c: {"x": [], "y": [], "hover": []} for c in CAT}
    for i, fruit in enumerate(fruits[:MAX_FRUITS]):
        x, y = TREE_POS[i]
        cat  = fruit.get("category", "건강")
        if cat not in groups:
            groups[cat] = {"x": [], "y": [], "hover": []}
        groups[cat]["x"].append(x)
        groups[cat]["y"].append(y)
        groups[cat]["hover"].append(
            f"<b>[{CAT.get(cat,{}).get('label',cat)}]</b> "
            f"{fruit.get('mission','')}<br><i>{fruit.get('timestamp','')}</i>"
        )

    for cat, g in groups.items():
        if not g["x"]:
            continue
        info = CAT.get(cat, {"color": "#999", "label": cat})
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["y"], mode="markers",
            name=info["label"],
            marker=dict(size=24, color=info["color"],
                        line=dict(width=2, color="white"), opacity=0.93),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=g["hover"],
        ))

    if n <= 10:
        stage = f"새싹 나무 · {n} / 10"
    elif n <= 20:
        stage = f"성장 나무 · {n} / 20"
    else:
        stage = f"완전한 나무 · {n} / 30"

    fig.update_layout(
        title=dict(text=stage, x=0.5, xanchor="center",
                   font=dict(size=15, color="#66BB6A")),
        height=470,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.04, font=dict(size=12)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-8.5, 8.5]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1.3, 10.2]),
        margin=dict(l=10, r=10, t=55, b=40),
        hovermode="closest",
    )
    return fig


# ── Analytics Charts ──────────────────────────────────────────────────────────
def emotion_pie(fruits: list) -> go.Figure | None:
    data = [f.get("emotion_type", "중립") for f in fruits if f.get("success")]
    if not data:
        return None
    counts = Counter(data)
    colors = {"부정적": "#FF6B6B", "중립": "#ADB5BD", "긍정적": "#FFD43B",
              "집중됨": "#4DABF7", "지루함": "#51CF66"}
    labels = [f"{EMOTION_EMOJI.get(k,'')} {k}" for k in counts]
    clrs   = [colors.get(k, "#999") for k in counts]
    fig = go.Figure(go.Pie(
        labels=labels, values=list(counts.values()), hole=0.44,
        marker=dict(colors=clrs, line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}회<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="감정별 분포", x=0.5, font=dict(size=13)),
        height=270, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def category_bar(fruits: list) -> go.Figure | None:
    data = [f.get("category", "건강") for f in fruits if f.get("success")]
    if not data:
        return None
    counts = Counter(data)
    cats   = sorted(counts, key=lambda c: counts[c], reverse=True)
    fig = go.Figure(go.Bar(
        x=[CAT.get(c, {"label": c})["label"] for c in cats],
        y=[counts[c] for c in cats],
        marker_color=[CAT.get(c, {"color": "#999"})["color"] for c in cats],
        text=[counts[c] for c in cats], textposition="outside",
        hovertemplate="%{x}: %{y}회<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="카테고리별 성공", x=0.5, font=dict(size=13)),
        height=270,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.15)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
.msg          { border-radius:8px; padding:12px 16px; margin:10px 0; font-size:14px; line-height:1.5; }
.msg-info     { background:#4DABF726; border-left:3px solid #4DABF7; color:#C8E6FA; }
.msg-success  { background:#4CAF5026; border-left:3px solid #4CAF50; color:#C8F5C8; }
.msg-warn     { background:#FFD43B26; border-left:3px solid #FFD43B; color:#FFF3C4; }

.metric-box   { background:#1E1E2E; border:1px solid #2E2E42; border-radius:10px; padding:16px; text-align:center; }
.metric-label { font-size:12px; color:#888899; margin-bottom:4px; }
.metric-value { font-size:22px; font-weight:700; color:#E8E8F0; }

hr.divider    { border:none; border-top:1px solid #2E2E42; margin:16px 0; }
p.caption     { font-size:12px; color:#888899; margin:4px 0; }

.badge        { display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; }

.mission-card  { border-radius:12px; padding:22px 26px; margin:14px 0; line-height:1.6; }
.mission-label { font-size:13px; font-weight:600; margin-bottom:8px; }
.mission-text  { font-size:21px; font-weight:700; line-height:1.5; color:#E8E8F0; }

.history-item { padding:12px 0; border-bottom:1px solid #2E2E42; }

.journey-item { border-radius:0 10px 10px 0; padding:14px 18px; margin:10px 0; }
.journey-ts   { font-size:12px; color:#888899; margin-bottom:5px; }
.journey-body { font-weight:600; font-size:15px; margin-left:8px; color:#E8E8F0; }
.journey-sub  { font-size:13px; color:#888899; margin-top:3px; }

.card-tile  { border-radius:10px; padding:20px 14px; text-align:center; margin:6px 0;
              border:1px solid #2E2E42; background:#1E1E2E; }
.card-emoji { font-size:32px; }
.card-name  { font-size:13px; font-weight:700; margin:8px 0 4px; }
.card-count { font-size:20px; font-weight:800; color:#E8E8F0; }

.split-item   { padding:12px 0; border-bottom:1px solid #2E2E42; }
.sidebar-stat { font-size:12px; color:#888899; text-align:center; margin-top:8px; }
.combo-badge  { text-align:center; font-weight:700; margin-top:4px; }

div[data-testid="stForm"] { border:none !important; padding:0 !important; }
</style>
"""


# ── UI Helpers ────────────────────────────────────────────────────────────────
def _badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}26; color:{color};">{label}</span>'


def _divider() -> None:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def _metric(label: str, value: str) -> str:
    return (f'<div class="metric-box">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'</div>')


def _msg(cls: str, text: str) -> None:
    st.markdown(f'<div class="msg {cls}">{text}</div>', unsafe_allow_html=True)


# ── Page: Home ────────────────────────────────────────────────────────────────
def render_home():
    data    = get_data()
    fruits  = data["fruits"]
    total   = len(fruits)
    success = len([f for f in fruits if f.get("success")])
    rate    = int(success / total * 100) if total > 0 else 0
    combo   = data.get("combo_count", 0)
    last    = data.get("last_category")

    st.markdown("## 🌱 피어나기")
    st.markdown('<p class="caption">작은 행동 하나가 나를 피워냅니다.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_metric("열매", f"{total} / {MAX_FRUITS}"), unsafe_allow_html=True)
    c2.markdown(_metric("성공", f"{success}회"), unsafe_allow_html=True)
    c3.markdown(_metric("성공률", f"{rate}%"), unsafe_allow_html=True)
    combo_val = f"{last} {combo}연속" if combo >= 2 and last else "-"
    c4.markdown(_metric("콤보", combo_val), unsafe_allow_html=True)

    _divider()
    st.plotly_chart(render_tree_chart(fruits), use_container_width=True)
    _divider()

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("🎯 미션 시작하기", use_container_width=True, type="primary"):
            st.session_state.page = "mission"
            st.session_state.mission_step = "input"
            st.rerun()

    history = [f for f in fruits if "timestamp" in f]
    if history:
        st.markdown("#### 최근 미션")
        for f in reversed(history[-5:]):
            cat  = f.get("category", "")
            info = CAT.get(cat, {"color": "#999", "label": cat})
            ok   = "✅" if f.get("success") else "❌"
            st.markdown(
                f'<div class="history-item">{ok} {_badge(info["label"], info["color"])} '
                f'<strong>{f.get("mission","")}</strong> '
                f'<span style="font-size:12px;color:#888899;">· {f.get("timestamp","")}</span></div>',
                unsafe_allow_html=True,
            )


# ── Page: Mission ─────────────────────────────────────────────────────────────
def render_mission():
    step = st.session_state.get("mission_step", "input")
    if step == "input":
        _mission_input()
    elif step == "show_mission":
        _mission_show()
    elif step == "result":
        _mission_result()
    elif step == "complete":
        _mission_complete()


def _mission_input():
    data   = get_data()
    fruits = data["fruits"]
    st.markdown("## 🎯 미션 시작")

    if len(fruits) >= MAX_FRUITS:
        _msg("msg-warn", "나무가 가득 찼습니다! 열매를 먼저 쪼개주세요.")
        if st.button("✂️ 열매 쪼개기"):
            st.session_state.page = "split"
            st.rerun()
        return

    with st.form("mission_form"):
        mood = st.text_area(
            "지금 기분을 자유롭게 적어주세요",
            placeholder="예: 오늘 너무 피곤하고 아무것도 하기 싫어요...",
            height=110,
        )
        minutes  = st.slider("가용 시간 (분)", min_value=5, max_value=60, value=15, step=5)
        submitted = st.form_submit_button("✨ 미션 생성", use_container_width=True, type="primary")

    if submitted:
        if not mood.strip():
            _msg("msg-warn", "기분을 입력해주세요.")
            return
        st.session_state.mood_input    = mood.strip()
        st.session_state.minutes_input = minutes

        with st.spinner("🔍 감정 분석 중..."):
            emotion_type = classify_emotion(mood.strip())
        st.session_state.emotion_type = emotion_type

        with st.spinner("📋 미션 생성 중..."):
            raw, is_wildcard, sources = get_mission(mood.strip(), minutes, emotion_type, data)
            m = parse_mission(raw, is_wildcard, sources)

        st.session_state.current_mission = m
        st.session_state.mission_step    = "show_mission"
        st.rerun()


def _mission_show():
    m            = st.session_state.get("current_mission", {})
    emotion_type = st.session_state.get("emotion_type", "중립")
    mood         = st.session_state.get("mood_input", "")
    minutes      = st.session_state.get("minutes_input", 15)

    cat      = m.get("category", "건강")
    diff     = m.get("difficulty", "하")
    cat_info = CAT.get(cat,  {"color": "#999", "label": cat})
    is_wc    = m.get("is_wildcard", False)

    st.markdown("## 📋 추천 미션")

    if is_wc:
        _msg("msg-warn", "돌발 미션 발동!")
    else:
        emoji   = EMOTION_EMOJI.get(emotion_type, "")
        purpose = EMOTION_PURPOSE.get(emotion_type, "")
        _msg("msg-info", f"{emoji} <strong>{emotion_type}</strong> → 목적: <strong>{purpose}</strong>")

    c = cat_info["color"]
    st.markdown(
        f'<div class="mission-card" style="background:{c}26; border-left:4px solid {c};">'
        f'<div class="mission-label" style="color:{c};">{cat_info["label"]} · {diff} · {minutes}분</div>'
        f'<div class="mission-text">{m.get("mission","")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if m.get("basis"):
        with st.expander("📖 심리학적 근거"):
            st.write(m["basis"])

    if m.get("effect"):
        _msg("msg-success", f"💡 <strong>기대 효과</strong>: {m['effect']}")

    if m.get("sources"):
        with st.expander("📚 참고 논문"):
            for s in m["sources"]:
                st.markdown(f'<p class="caption">• {s}</p>', unsafe_allow_html=True)

    _divider()
    col1, col2, col3 = st.columns([3, 3, 1])
    with col1:
        if st.button("✅ 이 미션 수락!", use_container_width=True, type="primary"):
            st.session_state.mission_step = "result"
            st.rerun()
    with col2:
        if st.button("🔄 다른 미션 보기", use_container_width=True):
            data = get_data()
            with st.spinner("📋 미션 생성 중..."):
                raw, is_wc2, sources = get_mission(mood, minutes, emotion_type, data)
                m2 = parse_mission(raw, is_wc2, sources)
            st.session_state.current_mission = m2
            st.rerun()
    with col3:
        if st.button("취소"):
            st.session_state.mission_step = "input"
            st.rerun()


def _mission_result():
    m    = st.session_state.get("current_mission", {})
    cat  = m.get("category", "건강")
    info = CAT.get(cat, {"label": cat, "color": "#999"})
    c    = info["color"]

    st.markdown("## 🏁 미션 결과")
    st.markdown(
        f'<div class="mission-card" style="background:{c}26; border-left:4px solid {c};">'
        f'<div class="mission-label" style="color:{c};">{info["label"]}</div>'
        f'<div class="mission-text">{m.get("mission","")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _divider()
    memo = st.text_input("📝 메모 (선택)", placeholder="오늘 미션 어떠셨나요?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎉 성공!", use_container_width=True, type="primary"):
            _record_mission(success=True, memo=memo)
    with c2:
        if st.button("😅 실패 (시도했어요)", use_container_width=True):
            _record_mission(success=False, memo=memo)


def _record_mission(success: bool, memo: str):
    m            = st.session_state.get("current_mission", {})
    emotion_type = st.session_state.get("emotion_type", "중립")
    mood         = st.session_state.get("mood_input", "")
    minutes      = st.session_state.get("minutes_input", 15)
    data         = get_data()

    if not success:
        data["cards"].append({"card": "도전 카드", "difficulty": "도전"})
        data["fruits"].append({
            "difficulty": m.get("difficulty", "하"),
            "category":   m.get("category", "건강"),
            "mission":    m.get("mission", ""),
            "full_mission": m.get("mission", ""),
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mood": mood, "time": f"{minutes}분",
            "emotion_type": emotion_type,
            "effect":     m.get("effect", ""),
            "success":    False,
            "memo":       memo if memo else None,
            "photo_path": None,
        })
        commit_data(data)
        st.session_state.update({"acquired_card": "도전 카드", "combo_card": None,
                                  "combo_count": 0, "success_result": False,
                                  "mission_step": "complete"})
        st.rerun()
        return

    with st.spinner("미션 요약 중..."):
        short_name = summarize_mission(m["mission"])

    data["fruits"].append({
        "difficulty":   m["difficulty"],
        "category":     m["category"],
        "mission":      short_name,
        "full_mission": m["mission"],
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mood": mood, "time": f"{minutes}분",
        "emotion_type": emotion_type,
        "effect":       m.get("effect", ""),
        "success":      True,
        "memo":         memo if memo else None,
        "photo_path":   None,
    })
    data.setdefault("mission_history", []).append(m["mission"])
    if len(data["mission_history"]) > 20:
        data["mission_history"] = data["mission_history"][-20:]

    combo      = check_combo(data, m["category"])
    combo_card = apply_combo_bonus(data, combo, m["category"])
    commit_data(data)
    st.session_state.update({"acquired_card": None, "short_name": short_name,
                              "combo_card": combo_card, "combo_count": combo,
                              "success_result": True, "mission_step": "complete"})
    st.rerun()


def _mission_complete():
    success    = st.session_state.get("success_result", True)
    combo_card = st.session_state.get("combo_card")
    combo      = st.session_state.get("combo_count", 0)
    m          = st.session_state.get("current_mission", {})
    short      = st.session_state.get("short_name", m.get("mission", ""))

    if success:
        st.balloons()
        st.markdown("## 🎉 미션 완료!")

        cat      = m.get("category", "건강")
        diff     = m.get("difficulty", "하")
        cat_info = CAT.get(cat, {"label": cat, "color": "#999"})
        c        = cat_info["color"]

        st.markdown(
            f'<div style="background:{c}26; border-radius:14px; padding:28px; text-align:center;'
            f' margin:16px 0; border:1px solid {c}4D;">'
            f'<div style="font-size:48px; margin-bottom:10px;">🍎</div>'
            f'<div style="font-size:17px; font-weight:700; margin-bottom:6px; color:#E8E8F0;">열매 획득!</div>'
            f'<div style="font-size:14px; color:{c}; margin-bottom:6px;">{cat_info["label"]} · {diff}</div>'
            f'<div style="font-size:13px; color:#888899;">{short}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if combo_card:
            st.markdown(
                f'<div style="background:#FFD70026; border:2px solid #FFD700; border-radius:12px;'
                f' padding:18px; text-align:center; margin:10px 0;">'
                f'<div style="font-weight:700; color:#FFD700; font-size:15px;">'
                f'🔥 {combo}연속 콤보! {combo_card} 획득!</div></div>',
                unsafe_allow_html=True,
            )

        _msg("msg-info", "열매 쪼개기에서 카드로 변환할 수 있어요.")
    else:
        st.markdown("## 🌱 괜찮아요!")
        _msg("msg-info", "시도했다는 것만으로도 충분합니다. 도전 카드를 드립니다.")
        st.markdown(
            '<div style="border:2px solid #4DABF7; border-radius:12px; padding:22px;'
            ' text-align:center; margin:10px 0; background:#4DABF726;">'
            '<div style="font-size:32px;">💪</div>'
            '<div style="font-weight:700; color:#4DABF7; font-size:16px; margin-top:8px;">도전 카드</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    _divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🏠 홈으로", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.mission_step = "input"
            st.rerun()
    with c2:
        if st.button("🎯 미션 하나 더!", use_container_width=True, type="primary"):
            st.session_state.mission_step = "input"
            st.rerun()


# ── Page: History ─────────────────────────────────────────────────────────────
def render_history():
    data    = get_data()
    history = [f for f in data["fruits"] if "timestamp" in f]

    if not history:
        st.markdown("## 📋 미션 리스트")
        _msg("msg-info", "아직 기록이 없습니다. 첫 미션을 시작해보세요!")
        return

    sel_cat  = st.selectbox("카테고리", ["전체"] + list(CAT.keys()),
                            label_visibility="collapsed")
    filtered = list(reversed(history))
    if sel_cat != "전체":
        filtered = [f for f in filtered if f.get("category") == sel_cat]

    st.markdown(f"## 📋 미션 리스트 ({len(filtered)}건)")
    _divider()

    for f in filtered:
        cat   = f.get("category", "")
        info  = CAT.get(cat, {"color": "#999", "label": cat})
        ok    = "✅" if f.get("success") else "❌"
        short = f.get("mission", "")
        full  = f.get("full_mission", "")
        sub   = (f'<br><span style="font-size:12px;color:#888899;padding-left:16px;">{full}</span>'
                 if full and full != short else "")
        st.markdown(
            f'<div class="history-item">{ok} {_badge(info["label"], info["color"])} '
            f'<strong>{short}</strong> '
            f'<span style="font-size:12px;color:#888899;">· {f.get("timestamp","")}</span>'
            f'{sub}</div>',
            unsafe_allow_html=True,
        )


# ── Page: Journey ─────────────────────────────────────────────────────────────
def render_journey():
    data      = get_data()
    fruits    = data["fruits"]
    successes = [f for f in fruits if f.get("success")]
    total     = len(fruits)
    scount    = len(successes)
    rate      = int(scount / total * 100) if total > 0 else 0

    st.markdown("## 🌿 내가 걸어온 길")

    if not fruits:
        _msg("msg-info", "아직 기록이 없습니다. 첫 미션을 시작해보세요!")
        return

    c1, c2, c3 = st.columns(3)
    c1.markdown(_metric("전체 시도", f"{total}회"), unsafe_allow_html=True)
    c2.markdown(_metric("성공", f"{scount}회"), unsafe_allow_html=True)
    c3.markdown(_metric("성공률", f"{rate}%"), unsafe_allow_html=True)
    st.progress(rate / 100)
    _divider()

    col1, col2 = st.columns(2)
    with col1:
        fig1 = emotion_pie(fruits)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = category_bar(fruits)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    _divider()
    st.markdown(f"#### 완료한 미션 ({scount}개)")

    for f in reversed(successes):
        cat    = f.get("category", "")
        info   = CAT.get(cat, {"color": "#999", "label": cat})
        c      = info["color"]
        effect = format_effect(f.get("effect", ""))
        memo   = f.get("memo")

        effect_html = f'<div class="journey-sub">💡 {effect}</div>' if effect else ""
        memo_html   = f'<div class="journey-sub">📝 {memo}</div>' if memo else ""

        st.markdown(
            f'<div class="journey-item" style="background:{c}26; border-left:3px solid {c};">'
            f'<div class="journey-ts">{f.get("timestamp","")}</div>'
            f'<div>{_badge(info["label"], c)}'
            f'<span class="journey-body">{f.get("mission","")}</span></div>'
            f'<div class="journey-sub">기분: {f.get("mood","")}</div>'
            f'{effect_html}{memo_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Page: Cards ───────────────────────────────────────────────────────────────
def render_cards():
    data  = get_data()
    cards = data.get("cards", [])

    st.markdown("## 🃏 카드 보기")

    if not cards:
        _msg("msg-info", "보유한 카드가 없습니다. 미션을 완료하면 카드를 획득할 수 있어요!")
        return

    st.markdown(f'<p class="caption">총 {len(cards)}장</p>', unsafe_allow_html=True)
    _divider()

    card_meta: dict[str, dict] = {}
    for c in cards:
        name = c["card"]
        if name not in card_meta:
            card_meta[name] = {"count": 0, "diff": c.get("difficulty", "하")}
        card_meta[name]["count"] += 1

    cols = st.columns(min(len(card_meta), 3))
    for idx, (name, meta) in enumerate(card_meta.items()):
        diff  = meta["diff"]
        color = DIFF.get(diff, {"color": "#999"})["color"]
        emoji = DIFF.get(diff, {"emoji": "🃏"})["emoji"]
        with cols[idx % 3]:
            st.markdown(
                f'<div class="card-tile" style="border-color:{color};">'
                f'<div class="card-emoji">{emoji}</div>'
                f'<div class="card-name" style="color:{color};">{name}</div>'
                f'<div class="card-count">× {meta["count"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _divider()
    counts  = Counter(c["card"] for c in cards)
    diff_of = {c["card"]: c.get("difficulty", "하") for c in cards}
    labels  = list(counts.keys())
    fig = go.Figure(go.Bar(
        x=labels, y=[counts[l] for l in labels],
        text=[counts[l] for l in labels], textposition="outside",
        marker_color=[DIFF.get(diff_of[l], {"color": "#999"})["color"] for l in labels],
        hovertemplate="%{x}: %{y}장<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="카드 보유 현황", x=0.5, font=dict(size=13)),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.15)"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Split ───────────────────────────────────────────────────────────────
def render_split():
    data   = get_data()
    fruits = data["fruits"]

    st.markdown("## ✂️ 열매 쪼개기")
    _msg("msg-warn", "열매를 쪼개면 카드로 변환되며 되돌릴 수 없습니다.")

    if not fruits:
        _msg("msg-info", "쪼갤 열매가 없습니다.")
        return

    _divider()
    for i, f in enumerate(fruits):
        cat      = f.get("category", "건강")
        diff     = f.get("difficulty", "하")
        cat_info = CAT.get(cat,  {"label": cat, "color": "#999"})
        diff_info= DIFF.get(diff, {"card": "", "color": "#999"})
        c        = cat_info["color"]

        col1, col2 = st.columns([7, 1])
        with col1:
            st.markdown(
                f'<div class="split-item">'
                f'{_badge(cat_info["label"], c)} '
                f'<strong>{f.get("mission","")}</strong> '
                f'<span style="font-size:12px;color:#888899;">· {diff} · {f.get("timestamp","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("✂️", key=f"split_{i}", help=f"→ {diff_info['card']}"):
                card = diff_info["card"]
                fruits.pop(i)
                data["cards"].append({"card": card, "difficulty": diff})
                commit_data(data)
                _msg("msg-success", f"🃏 {card} 획득!")
                st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.markdown("### 🌱 피어나기")
    _divider()

    nav = [
        ("home",    "🏠", "홈"),
        ("mission", "🎯", "미션 시작"),
        ("history", "📋", "미션 리스트"),
        ("journey", "🌿", "내가 걸어온 길"),
        ("cards",   "🃏", "카드 보기"),
        ("split",   "✂️", "열매 쪼개기"),
    ]

    current = st.session_state.get("page", "home")
    for key, emoji, label in nav:
        if st.button(f"{emoji}  {label}", use_container_width=True,
                     type="primary" if key == current else "secondary",
                     key=f"nav_{key}"):
            st.session_state.page = key
            if key == "mission":
                st.session_state.mission_step = "input"
            st.rerun()

    _divider()
    data    = get_data()
    fruits  = data["fruits"]
    total   = len(fruits)
    success = len([f for f in fruits if f.get("success")])
    rate    = int(success / total * 100) if total > 0 else 0
    st.markdown(
        f'<p class="sidebar-stat">🍎 {total}/{MAX_FRUITS} · ✅ {success}회 · 📊 {rate}%</p>',
        unsafe_allow_html=True,
    )

    combo = data.get("combo_count", 0)
    last  = data.get("last_category")
    if combo >= 2 and last:
        color = CAT.get(last, {"color": "#FFD700"})["color"]
        st.markdown(
            f'<p class="combo-badge" style="color:{color};">🔥 {last} {combo}연속!</p>',
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="🌱 피어나기",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("mission_step", "input")

    with st.sidebar:
        render_sidebar()

    page = st.session_state.get("page", "home")
    dispatch = {
        "home":    render_home,
        "mission": render_mission,
        "history": render_history,
        "journey": render_journey,
        "cards":   render_cards,
        "split":   render_split,
    }
    dispatch.get(page, render_home)()


if __name__ == "__main__":
    main()
