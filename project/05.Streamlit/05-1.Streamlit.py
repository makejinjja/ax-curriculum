#!/usr/bin/env python3
"""
bloom Streamlit — 대화형 기분전환 미션 AI
블랙 & 화이트 클린 디자인 + 시각화
"""
import os
import sys
import json
import re
import random
import hashlib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent / "p1"
load_dotenv(BASE_DIR / ".env")

DATA_FILE  = BASE_DIR / ".mission_data.json"
PDF_DIR    = BASE_DIR.parent / "data"
CACHE_FILE = BASE_DIR / ".index_cache" / "mission_rag_v7_index.json"

MAX_FRUITS = 30

PDF_FILES = [
    ("Brief_Cognitive_Behavioral_Therapy_Guide.pdf",                    "CBT(인지행동치료)"),
    ("Adolescent_Behavioral_Activation_Program.pdf",                    "행동 활성화 1"),
    ("Behavioral_Activation_Guided_Practice.pdf",                       "행동 활성화 2"),
    ("Brief_Mindfulness_Intervention_Stress_Response.pdf",              "마음챙김"),
    ("Emotion_Regulation_Gross_Process_Model.pdf",                      "감정 조절 전략"),
    ("Positive_Psychology_Progress_Seligman.pdf",                       "긍정 심리학"),
    ("Self_Determination_Theory_Ryan_Deci.pdf",                         "자기결정이론"),
    ("Positive_Emotions_Broaden_and_Build_Fredrickson.pdf",             "긍정 정서 확장"),
    ("The_Role_of_Self_Compassion_in_Psychotherapy.pdf",                "자기연민"),
    ("Social_Isolation_Depression_Anxiety_Older_Adults.pdf",            "사회적 고립·우울"),
    ("Emotion_Regulation_and_Sleep_Vandekerckhove.pdf",                 "수면·감정조절"),
    ("Benefits_of_Exercise_for_Clinically_Depressed_Craft_Perna.pdf",  "운동·정신건강"),
    ("Investigating_the_Flow_Experience_Abuhamdeh.pdf",                 "Flow 이론"),
    ("Motivational_Interviewing_Helping_People_Change_Miller_Rollnick.pdf", "동기면담"),
]

# ── 카테고리 / 난이도 메타 ────────────────────────────────────
CAT_COLOR = {
    "건강":   "#e74c3c",
    "생산성": "#3498db",
    "재미":   "#9b59b6",
    "성장":   "#27ae60",
    "돌발":   "#f39c12",
}

DIFF_COLOR = {
    "하":   "#e74c3c",
    "중":   "#e67e22",
    "상":   "#95a5a6",
    "최상": "#f1c40f",
    "돌발": "#f39c12",
    "도전": "#3498db",
}

DIFF_CARD = {
    "하":   "씨앗 카드",
    "중":   "새싹 카드",
    "상":   "햇살 카드",
    "최상": "황금 열매 카드",
    "돌발": "번개 카드",
    "도전": "도전 카드",
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
    "부정적": "😔",
    "중립":   "😐",
    "긍정적": "😊",
    "집중됨": "🎯",
    "지루함": "😑",
}

EMOTION_SOURCE_WEIGHT = {
    "부정적": {"CBT": 2.0, "행동 활성화": 1.5, "자기연민": 1.8, "동기면담": 1.3, "수면·감정조절": 1.2},
    "중립":   {"행동 활성화": 1.5, "마음챙김": 1.2, "자기결정이론": 1.5, "긍정 심리학": 1.2},
    "긍정적": {"긍정 정서 확장": 2.0, "긍정 심리학": 1.8, "Flow 이론": 1.5, "자기결정이론": 1.3},
    "집중됨": {"Flow 이론": 2.5, "자기결정이론": 1.5, "마음챙김": 1.2},
    "지루함": {"행동 활성화": 2.0, "긍정 심리학": 1.5, "Flow 이론": 1.3, "사회적 고립·우울": 1.2},
}

CAT_PRIMARY_SOURCE = {
    "건강":   "운동·정신건강",
    "생산성": "Flow 이론",
    "재미":   "긍정 정서 확장",
    "성장":   "자기결정이론",
    "돌발":   None,
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


# ── 글로벌 CSS ────────────────────────────────────────────────
def apply_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
        color: #111111;
        background-color: #ffffff;
    }

    /* 사이드바 */
    [data-testid="stSidebar"] {
        background-color: #f8f8f8;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * { color: #111111 !important; }

    /* 채팅 메시지 */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid #e8e8e8;
        background: #fafafa;
        margin-bottom: 8px;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #f0f0f0;
    }

    /* 버튼 */
    .stButton > button {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.4rem 1.2rem !important;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
    }

    /* 입력창 */
    .stTextInput > div > input,
    .stTextArea > div > textarea,
    .stNumberInput input,
    .stSelectbox > div {
        border: 1px solid #cccccc !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        color: #111111 !important;
    }

    /* 카드 박스 */
    .mission-card {
        background: #ffffff;
        border: 2px solid #111111;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .wildcard-card {
        background: #fffbf0;
        border: 2px solid #f39c12;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .insight-box {
        background: #f5f5f5;
        border-left: 4px solid #111111;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .stat-box {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .combo-banner {
        background: #111111;
        color: #ffffff;
        border-radius: 10px;
        padding: 10px 18px;
        margin: 8px 0;
        font-weight: 600;
    }

    /* 타이틀 */
    h1, h2, h3 { color: #111111; font-weight: 700; }
    hr { border-color: #e0e0e0; }

    /* 숨기기 */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ── 데이터 저장/로드 ──────────────────────────────────────────

def load_data() -> dict:
    if DATA_FILE.exists():
        d = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        d.setdefault("last_category", None)
        d.setdefault("combo_count", 0)
        d.setdefault("mission_history", [])
        d.setdefault("weak_paper_boost", [])
        return d
    return {"fruits": [], "cards": [], "last_category": None,
            "combo_count": 0, "mission_history": [], "weak_paper_boost": []}


def save_data(data: dict):
    DATA_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── RAG 인덱싱 ───────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


MAX_CHUNK_CHARS = 600
CHUNK_OVERLAP   = 100


def _chunk(text: str, source: str) -> list[dict]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    idx = 0
    for para in paragraphs:
        para = para.strip()
        if len(para) < 80:
            continue
        start = 0
        while start < len(para):
            segment = para[start:start + MAX_CHUNK_CHARS]
            if len(segment) >= 80:
                chunks.append({"text": segment, "source": source, "chunk_index": idx})
                idx += 1
            start += MAX_CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


@st.cache_resource(show_spinner="📚 논문 인덱싱 중...")
def build_index_cached(api_key: str):
    client = OpenAI(api_key=api_key)
    hashes = {}
    for fname, _ in PDF_FILES:
        p = PDF_DIR / fname
        if p.exists():
            hashes[fname] = _file_hash(p)

    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cache.get("hashes") == hashes:
            tokenized = [c["text"].split() for c in cache["chunks"]]
            bm25 = BM25Okapi(tokenized) if tokenized else None
            return cache["chunks"], cache["embeddings"], bm25

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    for fname, label in PDF_FILES:
        p = PDF_DIR / fname
        if not p.exists():
            continue
        text = _extract_text(p)
        all_chunks.extend(_chunk(text, f"{fname} ({label})"))

    texts = [c["text"] for c in all_chunks]
    embeddings: list[list[float]] = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[i:i + batch_size],
        )
        embeddings.extend([item.embedding for item in resp.data])

    CACHE_FILE.write_text(
        json.dumps({"hashes": hashes, "chunks": all_chunks, "embeddings": embeddings},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    tokenized = [c["text"].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return all_chunks, embeddings, bm25


# ── RAG 검색 ─────────────────────────────────────────────────

def _time_query(minutes: int) -> str:
    if minutes <= 10:
        return "brief intervention micro-habit short activity immediate"
    elif minutes <= 40:
        return "moderate duration activity engagement exercise"
    else:
        return "extended activity deep work immersive exercise"


def _expand_query(mood: str, minutes: int,
                  emotion_type: str | None = None,
                  target_cat: str | None = None) -> str:
    if emotion_type and target_cat:
        combo_kw = EMOTION_CAT_QUERY.get((emotion_type, target_cat))
        if combo_kw:
            return f"{combo_kw} {_time_query(minutes)} psychological intervention evidence-based"
    mood_kw = ""
    for k, v in MOOD_QUERY_MAP.items():
        if k in mood:
            mood_kw = v
            break
    if not mood_kw:
        mood_kw = "emotion regulation mood improvement well-being intervention"
    return f"{mood_kw} {_time_query(minutes)} psychological intervention evidence-based"


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


@st.cache_resource
def _get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _hyde_query(client: OpenAI, query: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "심리학 논문 스타일로 아래 주제에 맞는 개입법 설명을 영어로 100자 이내로 써라. "
                "논문 본문처럼 써야 하며 다른 말은 하지 마라."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.3, max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def _generate_multi_queries(client: OpenAI, query: str) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "아래 검색 쿼리를 다른 표현으로 3가지 변형해라. "
                "각 변형은 줄바꿈으로 구분하고 번호 없이 쿼리 텍스트만 출력해라."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.7, max_tokens=150,
    )
    lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
    return lines[:3]


def retrieve(query_emb, chunks, embeddings, k=4,
             emotion_type=None, query_text="",
             bm25=None, extra_boost=None) -> list[dict]:
    weights = dict(EMOTION_SOURCE_WEIGHT.get(emotion_type, {}) if emotion_type else {})
    if extra_boost:
        for lbl, mult in extra_boost.items():
            weights[lbl] = weights.get(lbl, 1.0) * mult

    cosine_scores = []
    for emb, chunk in zip(embeddings, chunks):
        score = _cosine(query_emb, emb)
        source = chunk.get("source", "")
        for keyword, multiplier in weights.items():
            if keyword in source:
                score *= multiplier
                break
        cosine_scores.append(score)

    if bm25 and query_text:
        bm25_raw = bm25.get_scores(query_text.split())
        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        cos_max  = max(cosine_scores) if max(cosine_scores) > 0 else 1.0
        hybrid_scores = [
            0.5 * (c / cos_max) + 0.5 * (b / bm25_max)
            for c, b in zip(cosine_scores, bm25_raw)
        ]
    else:
        hybrid_scores = cosine_scores

    top20_idx = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)[:20]
    candidates = [chunks[i] for i in top20_idx]

    if query_text and candidates:
        ce = _get_cross_encoder()
        pairs = [(query_text, c["text"]) for c in candidates]
        ce_scores = ce.predict(pairs)
        reranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in reranked[:k]]

    return candidates[:k]


def _multi_query_retrieve(client, query, chunks, embeddings, k,
                          emotion_type, bm25=None, extra_boost=None) -> list[dict]:
    queries = [query] + _generate_multi_queries(client, query)
    seen: set[tuple] = set()
    merged: list[dict] = []
    for q in queries:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small", input=[q]
        ).data[0].embedding
        results = retrieve(q_emb, chunks, embeddings, k=k,
                           emotion_type=emotion_type, query_text=q,
                           bm25=bm25, extra_boost=extra_boost)
        for chunk in results:
            uid = (chunk["source"], chunk["chunk_index"])
            if uid not in seen:
                seen.add(uid)
                merged.append(chunk)
    return merged[:k]


def build_context(top_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks
    )


# ── AI 함수 ──────────────────────────────────────────────────

def classify_emotion(client: OpenAI, mood: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user",   "content": mood},
        ],
        temperature=0.0, max_tokens=10,
    )
    raw = resp.choices[0].message.content.strip()
    for key in EMOTION_PURPOSE:
        if key in raw:
            return key
    return "중립"


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
- 난이도는 반드시 "{forced_difficulty}"로 고정
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


def get_mission(client: OpenAI, mood: str, time_str: str, minutes: int,
                chunks, embeddings, emotion_type: str, data: dict,
                bm25=None) -> tuple[str, bool, list[str]]:
    is_wildcard = random.random() < 0.15

    if is_wildcard:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": WILDCARD_PROMPT},
                {"role": "user",   "content": f"가용 시간: {time_str}"},
            ],
            temperature=1.0, max_tokens=200,
        )
        return resp.choices[0].message.content, True, []

    forced_difficulty = random.choices(
        ["하", "중", "상", "최상"], weights=[50, 30, 15, 5]
    )[0]

    emotion_to_default_cat = {
        "부정적": "건강", "중립": "생산성", "긍정적": "재미",
        "집중됨": "생산성", "지루함": "재미",
    }
    target_cat = emotion_to_default_cat.get(emotion_type)

    extra_boost: dict[str, float] = {}
    primary_src = CAT_PRIMARY_SOURCE.get(target_cat or "")
    if primary_src:
        extra_boost[primary_src] = 2.0
    for weak_src in data.get("weak_paper_boost", []):
        extra_boost[weak_src] = extra_boost.get(weak_src, 1.0) * 2.0

    query = _expand_query(mood, minutes, emotion_type=emotion_type, target_cat=target_cat)
    hyde_text = _hyde_query(client, query)

    top     = _multi_query_retrieve(client, hyde_text, chunks, embeddings,
                                    k=4, emotion_type=emotion_type, bm25=bm25,
                                    extra_boost=extra_boost or None)
    context = build_context(top)
    sources = list(dict.fromkeys(c["source"] for c in top))

    recent     = data.get("mission_history", [])[-5:]
    recent_str = "\n".join(f"- {m}" for m in recent) if recent else "없음"
    purpose    = EMOTION_PURPOSE.get(emotion_type, "기분전환·회복")

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": make_mission_prompt(
                emotion_type, purpose, recent_str, forced_difficulty)},
            {"role": "user", "content": (
                f"현재 기분: {mood}\n가용 시간: {time_str}\n\n"
                f"[심리학 논문 근거]\n{context}"
            )},
        ],
        temperature=0.9, max_tokens=350,
    )
    return resp.choices[0].message.content, False, sources


def normalize_difficulty(text: str) -> str:
    for d in ["최상", "상", "중", "하", "돌발"]:
        if d in text:
            return d
    return "하"


def normalize_category(text: str) -> str:
    for c in ["생산성", "건강", "재미", "성장", "돌발"]:
        if c in text:
            return c
    return "건강"


def parse_mission(text: str, is_wildcard: bool, sources=None) -> dict:
    blocks: dict[str, str] = {}
    current_tag = None
    current_lines: list[str] = []
    for line in text.split('\n'):
        header_match = re.match(r'^\[(.+?)\]', line.strip())
        if header_match:
            if current_tag:
                blocks[current_tag] = '\n'.join(current_lines).strip()
            current_tag = header_match.group(1)
            current_lines = []
        else:
            if current_tag:
                content = re.sub(r'^-\s*', '', line.strip())
                if content:
                    current_lines.append(content)
    if current_tag:
        blocks[current_tag] = '\n'.join(current_lines).strip()

    return {
        "mission":     blocks.get("미션", ""),
        "category":    "돌발" if is_wildcard else normalize_category(blocks.get("카테고리", "")),
        "difficulty":  "하" if is_wildcard else normalize_difficulty(blocks.get("난이도", "")),
        "basis":       blocks.get("근거", ""),
        "effect":      blocks.get("효과", ""),
        "is_wildcard": is_wildcard,
        "sources":     sources or [],
    }


def get_insight(client: OpenAI, mission_text: str, chunks, embeddings, bm25=None) -> str:
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[mission_text]
    ).data[0].embedding
    top = retrieve(q_emb, chunks, embeddings, k=2, query_text=mission_text, bm25=bm25)
    context = build_context(top)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "심리학자처럼 방금 완료한 미션이 어떤 심리학 이론/기법과 연결되는지 1~2줄로 설명해라. "
                "형식: '방금 하신 행동은 [이론명]의 [기법명] 기법입니다. [효과]'"
            )},
            {"role": "user", "content": f"완료한 미션: {mission_text}\n\n[논문 근거]\n{context}"},
        ],
        temperature=0.5, max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


def get_motivational_nudge(client: OpenAI, mood: str, chunks, embeddings, bm25=None) -> str:
    mi_chunks, mi_embs = [], []
    for c, e in zip(chunks, embeddings):
        if "동기면담" in c.get("source", ""):
            mi_chunks.append(c)
            mi_embs.append(e)
    if not mi_chunks:
        return "괜찮아요, 언제든 준비되면 다시 도전해볼 수 있어요."
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[mood]
    ).data[0].embedding
    top = retrieve(q_emb, mi_chunks, mi_embs, k=2, query_text=mood, bm25=None)
    context = build_context(top)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "동기면담(Motivational Interviewing) 전문가처럼 "
                "미션을 거절한 사람에게 공감하며 부드럽게 동기를 북돋는 말 1~2줄을 한국어로 해라."
            )},
            {"role": "user", "content": f"현재 기분: {mood}\n\n[동기면담 근거]\n{context}"},
        ],
        temperature=0.7, max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def summarize_mission(client: OpenAI, mission_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "미션 텍스트를 10자 이내 짧은 동사형 한국어로 요약해라. 다른 말 없이 요약문만 출력."},
            {"role": "user",   "content": mission_text},
        ],
        temperature=0.0, max_tokens=20,
    )
    return resp.choices[0].message.content.strip()


def analyze_coverage(client: OpenAI, data: dict, chunks, embeddings) -> dict | None:
    fruits = [f for f in data.get("fruits", []) if f.get("success")][-30:]
    if not fruits or not chunks:
        return None
    combined = " ".join(f.get("full_mission") or f.get("mission", "") for f in fruits)
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[combined[:2000]]
    ).data[0].embedding

    label_scores: dict[str, list[float]] = {}
    for c, e in zip(chunks, embeddings):
        src = c.get("source", "")
        m = re.search(r'\(([^)]+)\)$', src)
        label = m.group(1) if m else src
        label_scores.setdefault(label, []).append(_cosine(q_emb, e))

    if not label_scores:
        return None

    avg_scores = {lbl: sum(sc) / len(sc) for lbl, sc in label_scores.items()}
    sorted_pairs = sorted(avg_scores.items(), key=lambda x: x[1])
    weak = [lbl for lbl, _ in sorted_pairs[:2]]
    data["weak_paper_boost"] = weak
    return {"scores": dict(avg_scores), "weak": weak}


# ── 콤보 시스템 ──────────────────────────────────────────────

def check_combo(data: dict, new_category: str) -> int:
    if new_category == "돌발":
        return 0
    last = data.get("last_category")
    if last == new_category:
        data["combo_count"] = data.get("combo_count", 0) + 1
    else:
        data["combo_count"] = 1
    data["last_category"] = new_category
    return data["combo_count"]


def apply_combo_bonus(data: dict, combo: int, category: str) -> str | None:
    if combo < 2:
        return None
    if combo >= 3:
        bonus_card = "골드 카드"
        data["cards"].append({"card": bonus_card, "difficulty": "최상"})
        save_data(data)
        return f"🔥 {category} 미션 {combo}연속! **골드 카드** 자동 획득!"
    card_name, diff = COMBO2_CARD.get(category, ("씨앗 카드", "하"))
    data["cards"].append({"card": card_name, "difficulty": diff})
    save_data(data)
    return f"🔥 {category} 미션 2연속! **{card_name}** 획득!"


# ── 시각화 ───────────────────────────────────────────────────

def render_tree_chart(fruits: list):
    n = len(fruits)
    if n == 0:
        st.info("아직 열매가 없습니다. 미션을 완료하면 나무가 자랍니다!")
        return

    if n <= 10:
        label, cap = "🌱 새싹 나무", 10
    elif n <= 20:
        label, cap = "🌿 성장 나무", 20
    else:
        label, cap = "🌲 완전한 나무", 30

    st.markdown(f"### {label} &nbsp; `{n}/{cap}`")

    cols_per_row = [1, 2, 3, 4, 5, 5]
    if n > 10:
        cols_per_row = [1, 2, 3, 4, 5, 5, 5, 5]
    if n > 20:
        cols_per_row = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5]

    idx = 0
    for row_count in cols_per_row:
        if idx >= cap:
            break
        row_fruits = fruits[idx:idx + row_count]
        cols = st.columns(row_count)
        for ci in range(row_count):
            with cols[ci]:
                if ci < len(row_fruits):
                    f = row_fruits[ci]
                    cat  = f.get("category", "건강")
                    diff = f.get("difficulty", "하")
                    col  = CAT_COLOR.get(cat, "#888888")
                    sym  = {"건강": "H", "생산성": "P", "재미": "F",
                             "성장": "G", "돌발": "J"}.get(cat, "?")
                    st.markdown(
                        f'<div style="width:36px;height:36px;border-radius:50%;'
                        f'background:{col};color:#fff;display:flex;align-items:center;'
                        f'justify-content:center;font-weight:700;font-size:12px;'
                        f'margin:auto;" title="{cat} / {diff}">{sym}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="width:36px;height:36px;border-radius:50%;'
                        'background:#f0f0f0;margin:auto;"></div>',
                        unsafe_allow_html=True,
                    )
        idx += row_count

    st.markdown("""
    <div style="text-align:center;margin-top:4px;font-size:12px;color:#888;">
        <span style="color:#e74c3c">●</span> 건강 &nbsp;
        <span style="color:#3498db">●</span> 생산성 &nbsp;
        <span style="color:#9b59b6">●</span> 재미 &nbsp;
        <span style="color:#27ae60">●</span> 성장 &nbsp;
        <span style="color:#f39c12">●</span> 돌발
    </div>
    """, unsafe_allow_html=True)


def render_category_pie(fruits: list):
    cats = [f.get("category", "기타") for f in fruits if f.get("success")]
    if not cats:
        return
    cnt = Counter(cats)
    fig = go.Figure(go.Pie(
        labels=list(cnt.keys()),
        values=list(cnt.values()),
        marker_colors=[CAT_COLOR.get(c, "#888") for c in cnt.keys()],
        textinfo="label+percent",
        hole=0.4,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="white",
        font=dict(family="Noto Sans KR", color="#111"),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_mission_freq_bar(fruits: list):
    history = [f for f in fruits if "timestamp" in f]
    if not history:
        return
    cnt = Counter(f.get("full_mission") or f.get("mission", "") for f in history)
    top10 = cnt.most_common(10)
    labels = [t[0][:20] + "…" if len(t[0]) > 20 else t[0] for t in top10]
    values = [t[1] for t in top10]
    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color="#111111",
    ))
    fig.update_layout(
        xaxis_title="횟수",
        margin=dict(t=10, b=10, l=10, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Noto Sans KR", color="#111"),
        height=max(200, len(top10) * 36),
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_coverage_bar(scores: dict, weak: list):
    if not scores:
        return
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    labels = [i[0] for i in sorted_items]
    values = [i[1] for i in sorted_items]
    colors = ["#e74c3c" if lbl in weak else "#111111" for lbl in labels]
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        xaxis_title="활용도 점수",
        margin=dict(t=10, b=10, l=10, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Noto Sans KR", color="#111"),
        height=max(300, len(labels) * 30),
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"🔴 적게 활용된 논문: {', '.join(weak)} — 다음 미션에서 우선 반영됩니다.")


def render_success_gauge(total: int, success: int):
    rate = int(success / total * 100) if total > 0 else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        number={"suffix": "%", "font": {"size": 36, "color": "#111"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#111"},
            "bar": {"color": "#111111"},
            "bgcolor": "#f0f0f0",
            "steps": [
                {"range": [0, 50],  "color": "#f8f8f8"},
                {"range": [50, 80], "color": "#f0f0f0"},
                {"range": [80, 100],"color": "#e8e8e8"},
            ],
        },
    ))
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="white",
        height=200,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 미션 카드 렌더링 ─────────────────────────────────────────

def render_mission_card(m: dict, emotion_type: str | None = None):
    cat   = m.get("category", "")
    diff  = m.get("difficulty", "하")
    col   = CAT_COLOR.get(cat, "#111")
    dcol  = DIFF_COLOR.get(diff, "#111")

    if m.get("is_wildcard"):
        st.markdown(f"""
        <div class="wildcard-card">
            <div style="font-size:22px;font-weight:700;color:#f39c12;">⚡ 돌발 미션!</div>
            <div style="font-size:20px;font-weight:600;margin:12px 0;">{m['mission']}</div>
            <div>
                <span style="background:{col};color:#fff;padding:3px 10px;border-radius:12px;font-size:13px;margin-right:6px;">{cat}</span>
                <span style="background:{dcol};color:#fff;padding:3px 10px;border-radius:12px;font-size:13px;">{diff}</span>
            </div>
            <div style="margin-top:12px;font-size:14px;color:#555;">{m.get('effect','')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        emotion_line = ""
        if emotion_type:
            emoji   = EMOTION_EMOJI.get(emotion_type, "")
            purpose = EMOTION_PURPOSE.get(emotion_type, "")
            emotion_line = f'<div style="font-size:13px;color:#777;margin-bottom:8px;">{emoji} {emotion_type} → {purpose}</div>'
        st.markdown(f"""
        <div class="mission-card">
            {emotion_line}
            <div style="font-size:13px;color:#999;margin-bottom:4px;">📋 미션</div>
            <div style="font-size:20px;font-weight:600;margin-bottom:12px;">{m['mission']}</div>
            <div>
                <span style="background:{col};color:#fff;padding:3px 10px;border-radius:12px;font-size:13px;margin-right:6px;">{cat}</span>
                <span style="background:{dcol};color:#fff;padding:3px 10px;border-radius:12px;font-size:13px;">{diff}</span>
            </div>
            <div style="margin-top:12px;font-size:14px;color:#444;border-top:1px solid #eee;padding-top:10px;">
                <strong>📖 근거</strong> {m.get('basis','')}<br>
                <strong>효과</strong> {m.get('effect','')}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── 페이지: 미션 시작 ─────────────────────────────────────────

def page_mission(client, data, chunks, embeddings, bm25):
    st.header("🌱 미션 시작")

    if len(data["fruits"]) >= MAX_FRUITS:
        st.warning("🌲 나무가 가득 찼습니다! 열매를 먼저 쪼개야 미션을 시작할 수 있습니다.")
        return

    step = st.session_state.get("mission_step", "input")

    # ── 단계 1: 기분 & 시간 입력 ──────────────────────────────
    if step == "input":
        st.markdown("지금 기분과 사용 가능 시간을 알려주세요.")
        with st.form("mood_form"):
            mood = st.text_input("지금 기분을 자유롭게 입력하세요", placeholder="예: 피곤하고 무기력해요")
            minutes = st.number_input("사용 가능 시간 (분)", min_value=1, max_value=60, value=15)
            submitted = st.form_submit_button("미션 생성하기")

        if submitted and mood.strip():
            with st.spinner("감정 분석 중..."):
                emotion_type = classify_emotion(client, mood.strip())
            st.session_state["emotion_type"] = emotion_type
            st.session_state["mood"]    = mood.strip()
            st.session_state["minutes"] = int(minutes)
            st.session_state["time_str"] = f"{int(minutes)}분"

            emoji   = EMOTION_EMOJI.get(emotion_type, "")
            purpose = EMOTION_PURPOSE.get(emotion_type, "")
            st.success(f"{emoji} 감정 분류: **{emotion_type}** → 목적: {purpose}")

            with st.spinner("미션 생성 중..."):
                raw, is_wildcard, sources = get_mission(
                    client,
                    st.session_state["mood"],
                    st.session_state["time_str"],
                    st.session_state["minutes"],
                    chunks, embeddings, emotion_type, data, bm25=bm25
                )
                m = parse_mission(raw, is_wildcard, sources)

            st.session_state["current_mission"] = m
            st.session_state["mission_step"] = "confirm"
            st.rerun()

    # ── 단계 2: 미션 확인 & 수락/거절 ─────────────────────────
    elif step == "confirm":
        m = st.session_state.get("current_mission", {})
        emotion_type = st.session_state.get("emotion_type")

        render_mission_card(m, emotion_type if not m.get("is_wildcard") else None)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ 수락", use_container_width=True):
                st.session_state["mission_step"] = "doing"
                st.rerun()
        with col2:
            if st.button("🔄 다른 미션", use_container_width=True):
                mood = st.session_state.get("mood", "")
                with st.spinner("격려 메시지 생성 중..."):
                    nudge = get_motivational_nudge(client, mood, chunks, embeddings, bm25)
                st.info(f"💬 {nudge}")
                with st.spinner("새 미션 생성 중..."):
                    raw, is_wildcard, sources = get_mission(
                        client, mood,
                        st.session_state.get("time_str", "15분"),
                        st.session_state.get("minutes", 15),
                        chunks, embeddings,
                        st.session_state.get("emotion_type", "중립"),
                        data, bm25=bm25
                    )
                    m = parse_mission(raw, is_wildcard, sources)
                st.session_state["current_mission"] = m
                st.rerun()
        with col3:
            if st.button("❌ 종료", use_container_width=True):
                st.session_state["mission_step"] = "input"
                st.rerun()

    # ── 단계 3: 미션 수행 중 ──────────────────────────────────
    elif step == "doing":
        m = st.session_state.get("current_mission", {})
        st.markdown("### 미션을 시작하세요!")
        render_mission_card(m)

        st.markdown("---")
        st.markdown("#### 미션 결과")
        result = st.radio("어떻게 됐나요?", ["성공 ✅", "실패 ❌"], horizontal=True)

        memo = st.text_area("메모 (선택)", placeholder="미션 후 느낌이나 경험을 남겨보세요...")

        if st.button("결과 저장", type="primary"):
            success = result.startswith("성공")
            if not success:
                data["cards"].append({"card": "도전 카드", "difficulty": "도전"})
                save_data(data)
                st.info("괜찮아요, 시도했다는 것만으로도 충분합니다. 🌱 **도전 카드**를 드립니다!")
                st.session_state["mission_step"] = "input"
                st.rerun()
            else:
                with st.spinner("인사이트 생성 중..."):
                    insight = get_insight(client, m["mission"], chunks, embeddings, bm25)
                st.markdown(f"""
                <div class="insight-box">
                    🔬 {insight}
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("미션 요약 중..."):
                    short_name = summarize_mission(client, m["mission"])

                data["fruits"].append({
                    "difficulty":   m["difficulty"],
                    "category":     m["category"],
                    "mission":      short_name,
                    "full_mission": m["mission"],
                    "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "mood":         st.session_state.get("mood", ""),
                    "time":         st.session_state.get("time_str", ""),
                    "emotion_type": st.session_state.get("emotion_type", "중립"),
                    "effect":       m["effect"],
                    "success":      True,
                    "memo":         memo if memo else None,
                    "photo_path":   None,
                })

                data.setdefault("mission_history", []).append(m["mission"])
                if len(data["mission_history"]) > 20:
                    data["mission_history"] = data["mission_history"][-20:]

                combo = check_combo(data, m["category"])
                save_data(data)

                diff_col = DIFF_COLOR.get(m["difficulty"], "#111")
                cat_col  = CAT_COLOR.get(m["category"], "#111")
                st.markdown(f"""
                <div style="background:#f8f8f8;border-radius:10px;padding:16px;margin:8px 0;">
                    🍎 열매 획득!&nbsp;
                    <span style="background:{diff_col};color:#fff;padding:2px 8px;border-radius:8px;">{m['difficulty']}</span>&nbsp;
                    <span style="background:{cat_col};color:#fff;padding:2px 8px;border-radius:8px;">{m['category']}</span><br>
                    <span style="color:#555;font-size:14px;">{short_name}</span>
                </div>
                """, unsafe_allow_html=True)

                combo_msg = apply_combo_bonus(data, combo, m["category"])
                if combo_msg:
                    st.markdown(f'<div class="combo-banner">{combo_msg}</div>', unsafe_allow_html=True)

                st.session_state["mission_step"] = "input"
                st.session_state["data_dirty"] = True


# ── 페이지: 미션 리스트 ───────────────────────────────────────

def page_history(data):
    st.header("📋 미션 리스트")
    history = [f for f in data.get("fruits", []) if "timestamp" in f]

    if not history:
        st.info("기록된 미션이 없습니다.")
        return

    st.caption(f"총 {len(history)}건")

    mission_counts = Counter(
        f.get("full_mission") or f.get("mission", "") for f in history
    )

    seen: set[str] = set()
    display_items = []
    for f in reversed(history):
        full = f.get("full_mission") or f.get("mission", "")
        if full in seen:
            continue
        seen.add(full)
        display_items.append(f)

    for i, f in enumerate(display_items, 1):
        cat      = f.get("category", "")
        cat_col  = CAT_COLOR.get(cat, "#888")
        short    = f.get("mission", "")
        full     = f.get("full_mission") or short
        count    = mission_counts[full]

        count_badge = ""
        if count > 1:
            count_badge = f'<span style="background:#111;color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;margin-left:8px;">×{count}회</span>'

        st.markdown(f"""
        <div style="border:1px solid #e8e8e8;border-radius:10px;padding:14px 18px;margin-bottom:8px;background:#fafafa;">
            <span style="background:{cat_col};color:#fff;padding:2px 8px;border-radius:8px;font-size:12px;">{cat}</span>
            {count_badge}
            <div style="font-weight:600;margin-top:8px;">{short}</div>
            {"<div style='font-size:13px;color:#666;margin-top:4px;'>"+full+"</div>" if full != short else ""}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 미션 빈도 차트")
    render_mission_freq_bar(history)


# ── 페이지: 카드 보기 ─────────────────────────────────────────

def page_cards(data):
    st.header("🃏 카드 보기")
    cards = data.get("cards", [])

    if not cards:
        st.info("보유 카드가 없습니다.")
        return

    cnt = Counter(c["card"] for c in cards)
    cols = st.columns(min(len(cnt), 4))
    for i, (card, n) in enumerate(cnt.items()):
        diff  = next((c["difficulty"] for c in cards if c["card"] == card), "하")
        dcol  = DIFF_COLOR.get(diff, "#111")
        with cols[i % 4]:
            st.markdown(f"""
            <div style="border:2px solid {dcol};border-radius:12px;padding:20px 16px;
                        text-align:center;background:#fff;margin-bottom:12px;">
                <div style="font-size:24px;">✦</div>
                <div style="font-weight:600;margin:8px 0;">{card}</div>
                <div style="color:{dcol};font-size:14px;">x{n}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("교환하기 기능 준비 중")


# ── 페이지: 내가 걸어온 길 ────────────────────────────────────

def page_journey(client, data, chunks, embeddings):
    st.header("🌿 내가 걸어온 길")
    fruits    = data.get("fruits", [])
    successes = [f for f in fruits if f.get("success")]
    total     = len(fruits)
    success_count = len(successes)

    if not successes:
        st.info("완료한 미션이 없습니다.")
        return

    # 통계
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="stat-box"><div style="font-size:28px;font-weight:700;">{total}</div><div style="color:#777;font-size:13px;">전체 미션</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-box"><div style="font-size:28px;font-weight:700;">{success_count}</div><div style="color:#777;font-size:13px;">완료 미션</div></div>""", unsafe_allow_html=True)
    with c3:
        rate = int(success_count / total * 100) if total > 0 else 0
        st.markdown(f"""<div class="stat-box"><div style="font-size:28px;font-weight:700;">{rate}%</div><div style="color:#777;font-size:13px;">성공률</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("카테고리 분포")
        render_category_pie(fruits)
    with col_right:
        st.subheader("성공률")
        render_success_gauge(total, success_count)

    # 미션 타임라인
    st.markdown("---")
    st.subheader("미션 타임라인")
    for i, f in enumerate(reversed(successes), 1):
        cat      = f.get("category", "")
        cat_col  = CAT_COLOR.get(cat, "#888")
        ts       = f.get("timestamp", "")
        mood     = f.get("mood", "")
        short    = f.get("mission", "")
        effect   = f.get("effect", "")
        memo     = f.get("memo")

        memo_html = f'<div style="font-size:13px;color:#555;margin-top:4px;">📝 {memo}</div>' if memo else ""
        st.markdown(f"""
        <div style="border-left:3px solid {cat_col};padding:10px 16px;margin-bottom:10px;background:#fafafa;border-radius:0 8px 8px 0;">
            <div style="font-size:12px;color:#999;">{ts}</div>
            <div style="font-weight:600;margin:4px 0;">
                <span style="background:{cat_col};color:#fff;padding:2px 7px;border-radius:6px;font-size:12px;margin-right:6px;">{cat}</span>
                {short}
            </div>
            <div style="font-size:13px;color:#666;">기분: {mood}</div>
            {"<div style='font-size:13px;color:#555;margin-top:2px;'>"+effect+"</div>" if effect else ""}
            {memo_html}
        </div>
        """, unsafe_allow_html=True)

    # 논문 커버리지
    if len(successes) >= 3 and chunks:
        st.markdown("---")
        st.subheader("📊 논문 활용 분석")
        with st.spinner("논문 커버리지 분석 중..."):
            coverage = analyze_coverage(client, data, chunks, embeddings)
        if coverage:
            save_data(data)
            render_coverage_bar(coverage["scores"], coverage["weak"])
        else:
            st.info("PDF 논문 파일이 없어 분석을 수행할 수 없습니다. data/ 폴더에 논문 PDF를 추가하면 활용도 분석이 가능합니다.")
    elif not chunks:
        st.markdown("---")
        st.info("📄 논문 PDF 파일이 없습니다. `data/` 폴더에 PDF를 추가하면 논문 활용 분석이 가능합니다.")


# ── 사이드바 ─────────────────────────────────────────────────

def render_sidebar(data):
    with st.sidebar:
        st.markdown("## 🌱 피어나기")
        st.caption("작은 행동 하나가 나를 피워냅니다.")
        st.markdown("---")

        fruits = data.get("fruits", [])
        n = len(fruits)
        combo = data.get("combo_count", 0)
        last  = data.get("last_category")

        st.markdown(f"**열매** {n} / {MAX_FRUITS}")
        st.progress(n / MAX_FRUITS)

        if combo >= 2 and last:
            col = CAT_COLOR.get(last, "#111")
            st.markdown(
                f'<div style="background:{col};color:#fff;border-radius:8px;'
                f'padding:6px 12px;font-size:13px;font-weight:600;">🔥 {last} {combo}연속 콤보!</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        page = st.radio(
            "메뉴",
            ["🌱 미션 시작", "📋 미션 리스트", "🃏 카드 보기", "🌿 내가 걸어온 길"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(f"오늘: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return page


# ── 메인 ─────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="피어나기 — 기분전환 미션 AI",
        page_icon="🌱",
        layout="wide",
    )
    apply_styles()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. `.env` 파일을 확인하세요.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # 세션 상태 초기화
    if "mission_step" not in st.session_state:
        st.session_state["mission_step"] = "input"

    # 데이터 로드 (세션 캐시)
    if "data" not in st.session_state or st.session_state.get("data_dirty"):
        st.session_state["data"] = load_data()
        st.session_state["data_dirty"] = False
    data = st.session_state["data"]

    # 인덱스 로드
    chunks, embeddings, bm25 = build_index_cached(api_key)

    page = render_sidebar(data)

    # 나무 렌더링 (미션 시작 탭에서만)
    if page == "🌱 미션 시작":
        render_tree_chart(data["fruits"])
        st.markdown("---")
        page_mission(client, data, chunks, embeddings, bm25)

    elif page == "📋 미션 리스트":
        page_history(data)

    elif page == "🃏 카드 보기":
        page_cards(data)

    elif page == "🌿 내가 걸어온 길":
        page_journey(client, data, chunks, embeddings)


if __name__ == "__main__":
    main()
