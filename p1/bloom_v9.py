#!/usr/bin/env python3
"""
bloom.v8.py
기분전환 미션 AI + RAG + 감정분류 + 돌발미션 + 카테고리 열매 + 콤보 시스템
v8 변경사항 (14개 논문 전면 활용):
  1. EMOTION_SOURCE_WEIGHT — 14개 논문 전체로 확장
  2. CAT_PRIMARY_SOURCE — 카테고리별 주 논문 ×2.0 부스트
  3. get_insight() — 미션 완료 후 심리학 이론 인사이트 생성
  4. get_motivational_nudge() — 미션 거절 시 동기면담 기반 격려 메시지
  5. analyze_coverage() — 내가 걸어온 길에서 논문 활용 분석 및 약점 논문 자동 부스트
"""
import os
import sys
import json
import re
import random
import hashlib
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv(Path(__file__).parent / ".env")

DATA_FILE  = Path(__file__).parent / ".mission_data.json"
PDF_DIR    = Path(__file__).parent.parent / "data"
CACHE_FILE = Path(__file__).parent / ".index_cache" / "mission_rag_v7_index.json"

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

# ── ANSI 색상 ────────────────────────────────────────────────
R   = "\033[91m"
BR  = "\033[38;5;130m"
SV  = "\033[97m"
GD  = "\033[93m"
BL  = "\033[94m"
PU  = "\033[95m"
GN  = "\033[32m"
YL  = "\033[93m"
BLD = "\033[1m"
RST = "\033[0m"

# ── 난이도 테이블 ────────────────────────────────────────────
DIFF = {
    "하":   {"sym": "(R)", "col": R,  "card": "씨앗 카드"},
    "중":   {"sym": "(B)", "col": BR, "card": "새싹 카드"},
    "상":   {"sym": "(S)", "col": SV, "card": "햇살 카드"},
    "최상": {"sym": "(G)", "col": GD, "card": "황금 열매 카드"},
    "돌발": {"sym": "(J)", "col": YL, "card": "번개 카드"},
    "도전": {"sym": "(C)", "col": BL, "card": "도전 카드"},
}

# ── 카테고리 테이블 ──────────────────────────────────────────
CAT = {
    "건강":  {"sym": "(H)", "col": R,  "label": "건강"},
    "생산성":{"sym": "(P)", "col": BL, "label": "생산성"},
    "재미":  {"sym": "(F)", "col": PU, "label": "재미"},
    "성장":  {"sym": "(G)", "col": GN, "label": "성장"},
    "돌발":  {"sym": "(J)", "col": YL, "label": "돌발"},
}

# ── 2연속 콤보 카테고리별 고정 카드 ─────────────────────────
COMBO2_CARD = {
    "건강":   ("회복 카드",  "중"),
    "성장":   ("지식 카드",  "중"),
    "재미":   ("행운 카드",  "중"),
    "생산성": ("집중 카드",  "중"),
}

# ── 감정 분류 → 목적 매핑 ────────────────────────────────────
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

# ── 감정 → 논문 소스 가중치 (14개 논문 전체 반영) ──────────
EMOTION_SOURCE_WEIGHT = {
    "부정적": {"CBT": 2.0, "행동 활성화": 1.5, "자기연민": 1.8, "동기면담": 1.3, "수면·감정조절": 1.2},
    "중립":   {"행동 활성화": 1.5, "마음챙김": 1.2, "자기결정이론": 1.5, "긍정 심리학": 1.2},
    "긍정적": {"긍정 정서 확장": 2.0, "긍정 심리학": 1.8, "Flow 이론": 1.5, "자기결정이론": 1.3},
    "집중됨": {"Flow 이론": 2.5, "자기결정이론": 1.5, "마음챙김": 1.2},
    "지루함": {"행동 활성화": 2.0, "긍정 심리학": 1.5, "Flow 이론": 1.3, "사회적 고립·우울": 1.2},
}

# ── 카테고리별 주 논문 소스 (×2.0 부스트) ───────────────────
CAT_PRIMARY_SOURCE = {
    "건강":   "운동·정신건강",
    "생산성": "Flow 이론",
    "재미":   "긍정 정서 확장",
    "성장":   "자기결정이론",
    "돌발":   None,
}

# ── 감정+카테고리 조합 쿼리 맵 ──────────────────────────────
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


# ── 미션 생성 SYSTEM_PROMPT ──────────────────────────────────
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
- 난이도는 반드시 "{forced_difficulty}"로 고정 (다른 난이도 선택 금지)
- 난이도 기준:
  - 하: 매우 쉬운 즉각적 행동
  - 중: 약간의 노력 필요
  - 상: 집중과 노력 필요
  - 최상: 강한 의지력 필요
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


# ── 돌발 미션 전용 SYSTEM_PROMPT ────────────────────────────
WILDCARD_PROMPT = """너는 돌발 미션 생성 AI다.
감정과 상관없이 기묘하거나 유쾌하고 즉흥적인 초단기 미션 1개를 제안한다.

규칙:
- 예상치 못한 행동, 사소하지만 재미있는 것
- 반드시 5분 이내 완료 가능
- 난이도는 항상 "하" 고정
- [카테고리]는 항상 "돌발" 고정

예시 유형:
- 주변 물건으로 뭔가 만들기
- 핸드폰 배경화면 바꾸기
- 창문 밖 1분 멍하니 보기
- 모르는 단어 하나 사전에서 찾기
- 냉장고 안 확인하기
- 오늘 처음 본 색깔 기억하기
- 아무 책 펼쳐서 첫 문장 읽기

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


# ── 데이터 저장 ─────────────────────────────────────────────

def load_data() -> dict:
    if DATA_FILE.exists():
        d = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        d.setdefault("last_category", None)
        d.setdefault("combo_count", 0)
        d.setdefault("mission_history", [])
        return d
    return {"fruits": [], "cards": [], "last_category": None, "combo_count": 0, "mission_history": []}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── RAG: 인덱싱 ─────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


MAX_CHUNK_CHARS = 600   # 1. 청크 크기 최적화: 6000 → 600자
CHUNK_OVERLAP   = 100   # 1. 청크 간 overlap


def _chunk(text: str, source: str) -> list[dict]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    idx = 0
    for para in paragraphs:
        para = para.strip()
        if len(para) < 80:
            continue
        # 1. overlap 적용: 이전 청크 끝 100자를 다음 청크 앞에 붙임
        start = 0
        while start < len(para):
            segment = para[start:start + MAX_CHUNK_CHARS]
            if len(segment) >= 80:
                chunks.append({"text": segment, "source": source, "chunk_index": idx})
                idx += 1
            start += MAX_CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def build_index(client: OpenAI, force_rebuild: bool = False) -> tuple[list[dict], list[list[float]], BM25Okapi]:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for fname, _ in PDF_FILES:
        p = PDF_DIR / fname
        if p.exists():
            hashes[fname] = _file_hash(p)

    if not force_rebuild and CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cache.get("hashes") == hashes:
            print(f"📚 캐시 로드 완료 — 논문 {len(PDF_FILES)}개, 청크 {len(cache['chunks'])}개")
            # 2. BM25 인덱스는 캐시에서 복원 불가 → 청크 텍스트로 재생성
            tokenized = [c["text"].split() for c in cache["chunks"]]
            bm25 = BM25Okapi(tokenized) if tokenized else None
            return cache["chunks"], cache["embeddings"], bm25

    print("📚 논문 인덱싱 중...", end="", flush=True)
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
    print(f"\r📚 논문 {len(PDF_FILES)}개, 청크 {len(all_chunks)}개 인덱싱 완료")
    # 2. BM25 인덱스 생성
    tokenized = [c["text"].split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return all_chunks, embeddings, bm25


# ── RAG: 검색 ───────────────────────────────────────────────

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
            time_kw = _time_query(minutes)
            return f"{combo_kw} {time_kw} psychological intervention evidence-based"

    mood_kw = ""
    for k, v in MOOD_QUERY_MAP.items():
        if k in mood:
            mood_kw = v
            break
    if not mood_kw:
        mood_kw = "emotion regulation mood improvement well-being intervention"
    time_kw = _time_query(minutes)
    return f"{mood_kw} {time_kw} psychological intervention evidence-based"


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


# 3. 리랭킹용 Cross-Encoder (첫 호출 시 모델 로드)
_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# 4. HyDE: 쿼리를 가상 논문 문장으로 확장
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
        temperature=0.3,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


# 5. 멀티쿼리: 쿼리 3개 변형 생성
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
        temperature=0.7,
        max_tokens=150,
    )
    lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
    return lines[:3]


def retrieve(query_emb, chunks, embeddings, k=4,
             emotion_type: str | None = None,
             query_text: str = "",
             bm25: BM25Okapi | None = None,
             extra_boost: dict | None = None) -> list[dict]:
    weights = dict(EMOTION_SOURCE_WEIGHT.get(emotion_type, {}) if emotion_type else {})
    if extra_boost:
        for lbl, mult in extra_boost.items():
            weights[lbl] = weights.get(lbl, 1.0) * mult

    # 코사인 유사도 점수
    cosine_scores = []
    for emb, chunk in zip(embeddings, chunks):
        score = _cosine(query_emb, emb)
        source = chunk.get("source", "")
        for keyword, multiplier in weights.items():
            if keyword in source:
                score *= multiplier
                break
        cosine_scores.append(score)

    # 2. BM25 점수 계산 및 정규화 후 코사인과 0.5:0.5 합산
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

    # 상위 20개 후보 추출 (3. 리랭킹 준비)
    top20_idx = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)[:20]
    candidates = [chunks[i] for i in top20_idx]

    # 3. Cross-Encoder 리랭킹
    if query_text:
        ce = _get_cross_encoder()
        pairs = [(query_text, c["text"]) for c in candidates]
        ce_scores = ce.predict(pairs)
        reranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in reranked[:k]]

    return candidates[:k]


# 5. 멀티쿼리 검색: 쿼리 3개 → 각각 검색 → 합산 후 중복 제거
def _multi_query_retrieve(client: OpenAI, query: str,
                          chunks: list[dict], embeddings: list[list[float]],
                          k: int, emotion_type: str | None,
                          bm25: BM25Okapi | None = None,
                          extra_boost: dict | None = None) -> list[dict]:
    queries = [query] + _generate_multi_queries(client, query)
    seen: set[tuple] = set()
    merged: list[dict] = []

    for q in queries:
        q_emb = client.embeddings.create(
            model="text-embedding-3-small", input=[q]
        ).data[0].embedding
        results = retrieve(q_emb, chunks, embeddings,
                           k=k, emotion_type=emotion_type,
                           query_text=q, bm25=bm25, extra_boost=extra_boost)
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


def get_insight(client: OpenAI, mission_text: str,
                chunks: list[dict], embeddings: list[list[float]],
                bm25: BM25Okapi | None = None) -> str:
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
        temperature=0.5,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip()


def get_motivational_nudge(client: OpenAI, mood: str,
                           chunks: list[dict], embeddings: list[list[float]],
                           bm25: BM25Okapi | None = None) -> str:
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
                "미션을 거절한 사람에게 공감하며 부드럽게 동기를 북돋는 말 1~2줄을 한국어로 해라. "
                "강요하거나 설득하지 말고, 자율성을 존중하는 말투로."
            )},
            {"role": "user", "content": f"현재 기분: {mood}\n\n[동기면담 근거]\n{context}"},
        ],
        temperature=0.7,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def analyze_coverage(client: OpenAI, data: dict,
                     chunks: list[dict], embeddings: list[list[float]]) -> str:
    fruits = [f for f in data.get("fruits", []) if f.get("success")][-30:]
    if not fruits:
        return ""
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

    avg_scores = {lbl: sum(sc) / len(sc) for lbl, sc in label_scores.items()}
    sorted_pairs = sorted(avg_scores.items(), key=lambda x: x[1])
    weak = [lbl for lbl, _ in sorted_pairs[:2]]
    data["weak_paper_boost"] = weak

    lines = ["  📊 논문 활용 분석:"]
    for lbl, score in sorted_pairs:
        filled = int(score * 200)
        bar = "▓" * min(filled, 10) + "░" * max(0, 10 - filled)
        lines.append(f"    {lbl[:12]:12s} {bar} {score:.3f}")
    lines.append(f"\n  🔍 적게 활용된 논문: {', '.join(weak)}")
    lines.append("  → 다음 미션에서 이 논문들이 우선 반영됩니다.")
    return "\n".join(lines)


# ── 감정 분류 ────────────────────────────────────────────────

CLASSIFY_PROMPT = """사용자의 기분 입력을 보고 감정 유형을 분류해라.

분류 기준:
- 부정적: 우울, 무기력, 스트레스, 불안, 짜증, 피곤, 슬픔, 힘듦
- 중립: 보통, 평범, 모르겠음, 그냥, 별로 특별한 감정 없음
- 긍정적: 행복, 설렘, 에너지 넘침, 기분 좋음, 성취감, 뿌듯함
- 집중됨: 몰입, 집중, 의욕, 하고 싶은 것이 있음
- 지루함: 지루, 심심, 무료, 할 게 없음

반드시 아래 형식으로만 출력 (다른 말 금지):
부정적 또는 중립 또는 긍정적 또는 집중됨 또는 지루함"""


def classify_emotion(client: OpenAI, mood: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user",   "content": mood},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    raw = resp.choices[0].message.content.strip()
    for key in EMOTION_PURPOSE:
        if key in raw:
            return key
    return "중립"


# ── 미션 요약 (10자 동사형) ──────────────────────────────────

def summarize_mission(client: OpenAI, mission_text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "미션 텍스트를 10자 이내 짧은 동사형 한국어로 요약해라. 다른 말 없이 요약문만 출력."},
            {"role": "user",   "content": mission_text},
        ],
        temperature=0.0,
        max_tokens=20,
    )
    return resp.choices[0].message.content.strip()


# ── 미션 생성 (일반 + 돌발) ─────────────────────────────────

def get_mission(client: OpenAI, mood: str, time_str: str, minutes: int,
                chunks: list[dict], embeddings: list[list[float]],
                emotion_type: str, data: dict,
                bm25: BM25Okapi | None = None) -> tuple[str, bool, list[str]]:
    """반환: (raw_text, is_wildcard, sources)"""
    is_wildcard = random.random() < 0.15

    if is_wildcard:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": WILDCARD_PROMPT},
                {"role": "user",   "content": f"가용 시간: {time_str}"},
            ],
            temperature=1.0,
            max_tokens=200,
        )
        return resp.choices[0].message.content, True, []

    forced_difficulty = random.choices(
        ["하", "중", "상", "최상"],
        weights=[50, 30, 15, 5],
    )[0]

    emotion_to_default_cat = {
        "부정적": "건강",
        "중립":   "생산성",
        "긍정적": "재미",
        "집중됨": "생산성",
        "지루함": "재미",
    }
    target_cat = emotion_to_default_cat.get(emotion_type)

    # 카테고리 주 논문 + 적게 활용된 논문 부스트
    extra_boost: dict[str, float] = {}
    primary_src = CAT_PRIMARY_SOURCE.get(target_cat or "")
    if primary_src:
        extra_boost[primary_src] = 2.0
    for weak_src in data.get("weak_paper_boost", []):
        extra_boost[weak_src] = extra_boost.get(weak_src, 1.0) * 2.0

    query = _expand_query(mood, minutes, emotion_type=emotion_type, target_cat=target_cat)

    # 4. HyDE: 쿼리를 가상 논문 문장으로 확장
    hyde_text = _hyde_query(client, query)

    # 5. 멀티쿼리 + 하이브리드(BM25+코사인) + 리랭킹 + 카테고리·약점 논문 부스트
    top     = _multi_query_retrieve(client, hyde_text, chunks, embeddings,
                                    k=4, emotion_type=emotion_type, bm25=bm25,
                                    extra_boost=extra_boost or None)
    context = build_context(top)
    sources = list(dict.fromkeys(c["source"] for c in top))

    recent     = data.get("mission_history", [])[-5:]
    recent_str = "\n".join(f"- {m}" for m in recent) if recent else "없음"

    purpose    = EMOTION_PURPOSE.get(emotion_type, "기분전환·회복")
    system_msg = make_mission_prompt(emotion_type, purpose, recent_str, forced_difficulty)
    user_msg   = (
        f"현재 기분: {mood}\n"
        f"가용 시간: {time_str}\n\n"
        f"[심리학 논문 근거]\n{context}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.9,
        max_tokens=350,
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


def parse_mission(text: str, is_wildcard: bool, sources: list[str] | None = None) -> dict:
    """블록 파싱 — [태그] 헤더 기준으로 분리"""
    blocks: dict[str, str] = {}
    current_tag: str | None = None
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

    raw_diff = blocks.get("난이도", "")
    raw_cat  = blocks.get("카테고리", "")
    return {
        "mission":     blocks.get("미션", ""),
        "category":    "돌발" if is_wildcard else normalize_category(raw_cat),
        "difficulty":  "하" if is_wildcard else normalize_difficulty(raw_diff),
        "basis":       blocks.get("근거", ""),
        "effect":      blocks.get("효과", ""),
        "is_wildcard": is_wildcard,
        "sources":     sources or [],
    }


def print_mission(m: dict, emotion_type: str | None = None):
    diff_info = DIFF.get(m["difficulty"], {"col": "", "sym": ""})
    cat_info  = CAT.get(m["category"],   {"col": "", "sym": "", "label": m["category"]})

    if m["is_wildcard"]:
        print(f"\n{YL}{'⚡'*19}{RST}")
        print(f"{YL}{BLD}  ⚡ 돌발 미션 발동! ⚡{RST}")
        print(f"{YL}{'⚡'*19}{RST}")
    else:
        print(f"\n{'─'*38}")
        if emotion_type:
            emoji   = EMOTION_EMOJI.get(emotion_type, "")
            purpose = EMOTION_PURPOSE.get(emotion_type, "")
            print(f"  {emoji} 감정: {emotion_type} → 목적: {purpose}")
        print(f"{'─'*38}")

    print(f"  {BLD}📋 미션{RST}")
    print(f"  {m['mission']}")
    print(f"\n  {BLD}카테고리{RST}: {cat_info['col']}{cat_info['label']} {cat_info['sym']}{RST}")
    print(f"  {BLD}난이도{RST}:   {diff_info['col']}{m['difficulty']} {diff_info['sym']}{RST}")

    if m.get("basis"):
        print(f"\n  {BLD}📖 근거{RST}")
        print(f"  {m['basis']}")

    print(f"\n  {BLD}효과{RST}")
    print(f"  {m['effect']}")
    print(f"{'─'*38}")


# ── 나무 UI ──────────────────────────────────────────────────

def fruit_sym(fruit: dict) -> str:
    cat  = fruit.get("category", "")
    diff = fruit.get("difficulty", "하")
    if cat == "돌발":
        info = DIFF["돌발"]
    elif cat in CAT:
        info = CAT[cat]
    else:
        info = DIFF.get(diff, {"col": "", "sym": "(?)"})
    return f"{info['col']}{info['sym']}{RST}"


def render_tree(fruits: list):
    n = len(fruits)

    def s(i):
        return fruit_sym(fruits[i]) if i < n else "   "

    if n <= 10:
        label, cap = "🌱 새싹 나무", 10
        print(f"\n{GN}{'━'*38}{RST}")
        print(f"  {BLD}{label}   [{n}/{cap} 열매]{RST}")
        print(f"{GN}{'━'*38}{RST}")
        print(f"                {s(0)}")
        print(f"             {s(1)}   {s(2)}")
        print(f"          {s(3)}   {s(4)}   {s(5)}")
        print(f"       {s(6)}  {s(7)}  {s(8)}  {s(9)}")
    elif n <= 20:
        label, cap = "🌿 성장 나무", 20
        print(f"\n{GN}{'━'*38}{RST}")
        print(f"  {BLD}{label}   [{n}/{cap} 열매]{RST}")
        print(f"{GN}{'━'*38}{RST}")
        print(f"                {s(0)}")
        print(f"             {s(1)}   {s(2)}")
        print(f"          {s(3)}   {s(4)}   {s(5)}")
        print(f"       {s(6)}  {s(7)}  {s(8)}  {s(9)}")
        print(f"    {s(10)} {s(11)} {s(12)} {s(13)} {s(14)}")
        print(f"    {s(15)} {s(16)} {s(17)} {s(18)} {s(19)}")
    else:
        label, cap = "🌲 완전한 나무", 30
        print(f"\n{GN}{'━'*38}{RST}")
        print(f"  {BLD}{label}  [{n}/{cap} 열매]{RST}")
        print(f"{GN}{'━'*38}{RST}")
        print(f"                {s(0)}")
        print(f"             {s(1)}   {s(2)}")
        print(f"          {s(3)}   {s(4)}   {s(5)}")
        print(f"       {s(6)}  {s(7)}  {s(8)}  {s(9)}")
        print(f"    {s(10)} {s(11)} {s(12)} {s(13)} {s(14)}")
        print(f"    {s(15)} {s(16)} {s(17)} {s(18)} {s(19)}")
        print(f"    {s(20)} {s(21)} {s(22)} {s(23)} {s(24)}")
        print(f"    {s(25)} {s(26)} {s(27)} {s(28)} {s(29)}")

    print(f"          ━━━━━━━━━━━━━")
    print(f"               ┃┃┃")
    print(f"           ~~~~~~~~~~~")
    print(f"{GN}{'━'*38}{RST}")
    print(f"  {R}(H){RST}=건강  {BL}(P){RST}=생산성  {PU}(F){RST}=재미")
    print(f"  {GN}(G){RST}=성장  {YL}(J){RST}=돌발")


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
    if not cards:
        print("  보유 카드 없음")
        return
    for card, cnt in Counter(c["card"] for c in cards).items():
        print(f"  • {card}  x{cnt}")


def show_cards_menu(cards: list):
    print(f"\n{'='*38}")
    print(f"  {BLD}🃏 카드 보기{RST}")
    print(f"{'='*38}")
    show_cards(cards)
    print(f"\n  {'─'*34}")
    print(f"  교환하기 (준비 중)")
    input("\n  Enter를 눌러 돌아가기... ")


# ── 미션 리스트 ──────────────────────────────────────────────

def show_history(data: dict):
    history = [f for f in data.get("fruits", []) if "timestamp" in f]
    if not history:
        print(f"\n  {'─'*34}")
        print("  기록된 미션이 없습니다.")
        print(f"  {'─'*34}")
        input("\n  Enter를 눌러 돌아가기... ")
        return

    print(f"\n{'='*38}")
    print(f"  {BLD}📋 미션 리스트  ({len(history)}건){RST}")
    print(f"{'='*38}")

    for i, f in enumerate(reversed(history), 1):
        cat      = f.get("category", "")
        cat_info = CAT.get(cat, {"col": "", "label": cat})
        short    = f.get("mission", "")
        full     = f.get("full_mission", "")

        print(f"\n  [{i}] {cat_info['col']}[{cat_info['label']}]{RST}  {short}")
        if full and full != short:
            print(f"      내용: {full}")

    input("\n  Enter를 눌러 돌아가기... ")


# ── 효과 과거형 변환 ─────────────────────────────────────────

def format_effect(effect: str) -> str:
    if not effect:
        return ""
    if effect.endswith(("요", "다", "음")):
        return effect
    return effect


# ── 내가 걸어온 길 ───────────────────────────────────────────

def show_journey(client: OpenAI, data: dict,
                 chunks: list[dict], embeddings: list[list[float]]):
    fruits    = data.get("fruits", [])
    successes = [f for f in fruits if f.get("success")]
    total     = len(fruits)
    success_count = len(successes)
    rate      = int(success_count / total * 100) if total > 0 else 0

    print(f"\n{'='*38}")
    print(f"  {BLD}🌿 내가 걸어온 길{RST}")
    print(f"{'='*38}")

    if not successes:
        print(f"\n  완료한 미션이 없습니다.")
        print(f"{'='*38}")
        input("\n  Enter를 눌러 돌아가기... ")
        return

    print(f"\n  지금까지 총 {total}번의 미션 중 {success_count}개를 완료했습니다. (성공률 {rate}%)\n")

    for i, f in enumerate(successes, 1):
        cat       = f.get("category", "")
        cat_info  = CAT.get(cat, {"col": "", "label": cat})
        timestamp = f.get("timestamp", "")
        mood      = f.get("mood", "")
        short     = f.get("mission", "")
        effect    = format_effect(f.get("effect", ""))
        memo      = f.get("memo")

        print(f"  [{i}] {timestamp}")
        print(f"      기분: {mood}")
        print(f"      미션: {cat_info['col']}[{cat_info['label']}]{RST} {short}")
        if effect:
            print(f"      효과: {effect}")
        if memo:
            print(f"      메모: {memo}")
        print()

    print(f"{'='*38}")

    if len(successes) >= 3:
        print("\n  논문 커버리지 분석 중...", end="", flush=True)
        coverage = analyze_coverage(client, data, chunks, embeddings)
        print(f"\r{' '*30}\r", end="")
        if coverage:
            print(f"\n{coverage}")

    input("\n  Enter를 눌러 돌아가기... ")


# ── 열매 쪼개기 ──────────────────────────────────────────────

def split_fruit_menu(data: dict):
    fruits = data["fruits"]
    if not fruits:
        print("\n  쪼갤 열매가 없습니다.")
        return

    print("\n  쪼갤 열매 번호를 선택하세요 (0 = 취소)")
    for i, f in enumerate(fruits, 1):
        d        = f["difficulty"]
        cat      = f.get("category", "건강")
        info     = DIFF.get(d, {"col": "", "sym": "(?)"})
        cat_info = CAT.get(cat, {"col": "", "label": cat})
        print(f"  {i}. {info['col']}{info['sym']}{RST} "
              f"[{cat_info['col']}{cat}{RST}] {f.get('mission','')[:20]}")

    while True:
        sel = input("  > ").strip()
        if sel == "0":
            return
        if sel.isdigit() and 1 <= int(sel) <= len(fruits):
            fruit = fruits.pop(int(sel) - 1)
            d     = fruit["difficulty"]
            card  = DIFF.get(d, DIFF["하"])["card"]
            data["cards"].append({"card": card, "difficulty": d})
            save_data(data)
            show_card(card, d)
            print(f"\n  {card}를 획득했습니다!")
            return
        print(f"  1~{len(fruits)} 또는 0을 입력하세요.")


# ── 콤보 시스템 ──────────────────────────────────────────────

def check_combo(data: dict, new_category: str) -> int:
    if new_category == "돌발":
        return 0  # 돌발은 콤보 발동 방지

    last = data.get("last_category")
    if last == new_category:
        data["combo_count"] = data.get("combo_count", 0) + 1
    else:
        data["combo_count"] = 1
    data["last_category"] = new_category
    return data["combo_count"]


def apply_combo_bonus(data: dict, combo: int, category: str):
    if combo < 2:
        return
    cat_label = CAT.get(category, {"label": category})["label"]
    cat_col   = CAT.get(category, {"col": ""})["col"]

    if combo >= 3:
        bonus_card = "골드 카드"
        data["cards"].append({"card": bonus_card, "difficulty": "최상"})
        save_data(data)
        print(f"\n{GD}{'🔥'*2} {cat_col}{cat_label}{RST}{GD} 미션 {combo}연속! "
              f"골드 카드 자동 획득!{RST}")
        show_card(bonus_card, "최상")
    elif combo == 2:
        card_name, diff = COMBO2_CARD.get(category, ("씨앗 카드", "하"))
        data["cards"].append({"card": card_name, "difficulty": diff})
        save_data(data)
        print(f"\n{GD}🔥 {cat_col}{cat_label}{RST}{GD} 미션 2연속! "
              f"보너스 카드 획득!{RST}")
        show_card(card_name, diff)
        print(f"  {card_name} 획득!")


# ── 입력 수집 ────────────────────────────────────────────────

def collect_input() -> dict:
    now = datetime.now()
    print(f"\n  현재 시간: {now.strftime('%H:%M')}")

    print("\n지금 기분을 자유롭게 입력하세요")
    while True:
        mood = input("  > ").strip()
        if mood:
            break
        print("  입력이 필요합니다.")

    print("\n사용 가능 시간을 입력하세요 (1~60분, 예: 15)")
    while True:
        raw = input("  > ").strip()
        if raw.isdigit():
            minutes = int(raw)
            if 1 <= minutes <= 60:
                break
        print("  1~60 사이의 숫자를 입력하세요.")

    time_str = f"{minutes}분"
    return {"mood": mood, "time": time_str, "minutes": minutes}


# ── 미션 루프 ────────────────────────────────────────────────

def mission_loop(client: OpenAI, mood: str, time_str: str, minutes: int,
                 chunks: list[dict], embeddings: list[list[float]],
                 emotion_type: str, data: dict,
                 bm25: BM25Okapi | None = None) -> dict | None:
    while True:
        print("\n  미션 생성 중...", end="", flush=True)
        raw, is_wildcard, sources = get_mission(
            client, mood, time_str, minutes, chunks, embeddings, emotion_type, data, bm25=bm25
        )
        m = parse_mission(raw, is_wildcard, sources)
        print(f"\r{' '*20}\r", end="")

        if not m["mission"]:
            print("  미션 생성에 실패했습니다. 재시도합니다.")
            continue

        print_mission(m, emotion_type if not is_wildcard else None)

        print("\n  1. 수락   2. 다른 미션   3. 종료")
        sel = input("  선택: ").strip()
        if sel == "1":
            return m
        if sel == "2":
            print("\n  💬 격려 메시지 생성 중...", end="", flush=True)
            nudge = get_motivational_nudge(client, mood, chunks, embeddings, bm25)
            print(f"\r{' '*30}\r", end="")
            print(f"\n  💬 {nudge}")
            continue
        if sel == "3":
            return None


# ── 시작 메뉴 ────────────────────────────────────────────────

def start_menu(client: OpenAI, data: dict,
               chunks: list[dict], embeddings: list[list[float]],
               bm25: BM25Okapi | None = None) -> str:
    print(f"\n{'='*38}")
    print(f"  {BLD}🌱 피어나기{RST}")
    print(f"  작은 행동 하나가 나를 피워냅니다.")
    print(f"  힐링, 성장, 즐거움 — 지금 기분에 딱 맞는 미션으로 시작해보세요.")
    print(f"{'='*38}")
    print(f"\n  현재 시간: {datetime.now().strftime('%H:%M')}")

    combo = data.get("combo_count", 0)
    last  = data.get("last_category")
    if combo >= 2 and last:
        cat_col = CAT.get(last, {"col": ""})["col"]
        print(f"  {GD}🔥 현재 콤보: {cat_col}{last}{RST}{GD} {combo}연속{RST}")

    render_tree(data["fruits"])

    print(f"\n  1. 미션 시작")
    print(f"  2. 열매 쪼개기")
    print(f"  3. 카드 보기")
    print(f"  4. 미션 리스트")
    print(f"  5. 내가 걸어온 길")
    print(f"  6. 종료")

    while True:
        sel = input("  선택 (1~6): ").strip()
        if sel == "1":
            return "mission"
        if sel == "2":
            split_fruit_menu(data)
            render_tree(data["fruits"])
            return start_menu(client, data, chunks, embeddings, bm25)
        if sel == "3":
            show_cards_menu(data["cards"])
            return start_menu(client, data, chunks, embeddings, bm25)
        if sel == "4":
            show_history(data)
            return start_menu(client, data, chunks, embeddings, bm25)
        if sel == "5":
            show_journey(client, data, chunks, embeddings)
            return start_menu(client, data, chunks, embeddings, bm25)
        if sel == "6":
            return "exit"
        print("  1~6 중에 선택하세요.")


# ── main ─────────────────────────────────────────────────────

def main():
    force_rebuild = "--rebuild" in sys.argv

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    data   = load_data()

    chunks, embeddings, bm25 = build_index(client, force_rebuild=force_rebuild)

    action = start_menu(client, data, chunks, embeddings, bm25)
    if action == "exit":
        print("\n  종료합니다.\n")
        sys.exit(0)

    if len(data["fruits"]) >= MAX_FRUITS:
        print(f"\n  {BLD}🌲 나무가 가득 찼습니다! 열매를 먼저 쪼개야 미션을 시작할 수 있습니다.{RST}")
        while len(data["fruits"]) >= MAX_FRUITS:
            split_fruit_menu(data)
            render_tree(data["fruits"])
            if len(data["fruits"]) < MAX_FRUITS:
                print("\n  자리가 생겼습니다. 미션을 시작합니다!")
                break
            print("\n  아직 나무가 가득 찼습니다. 열매를 더 쪼개주세요.")

    state = collect_input()

    print("\n  감정 분석 중...", end="", flush=True)
    emotion_type = classify_emotion(client, state["mood"])
    print(f"\r{' '*20}\r", end="")
    emoji   = EMOTION_EMOJI.get(emotion_type, "")
    purpose = EMOTION_PURPOSE.get(emotion_type, "")
    print(f"\n  {emoji} 감정 분류: {BLD}{emotion_type}{RST} → 목적: {purpose}")

    mission = mission_loop(client, state["mood"], state["time"], state["minutes"],
                           chunks, embeddings, emotion_type, data, bm25=bm25)
    if not mission:
        print("\n  종료합니다.\n")
        sys.exit(0)

    print(f"\n  미션을 시작하세요!")
    input(f"\n  ▶ 준비 완료 → Enter ")

    print("\n  미션 결과를 선택하세요")
    print("  1. 성공   2. 실패")
    while True:
        result = input("  선택: ").strip()
        if result in ("1", "2"):
            break
        print("  1 또는 2를 입력하세요.")

    success = (result == "1")

    if not success:
        print(f"\n  괜찮아요. 시도했다는 것만으로도 충분합니다. 🌱")
        print(f"  다음에 또 도전해봐요!")
        data["cards"].append({"card": "도전 카드", "difficulty": "도전"})
        save_data(data)
        show_card("도전 카드", "도전")
        print("  시도한 용기에 도전 카드를 드립니다. 🌱")
        print("\n  오늘도 수고하셨습니다! 👍\n")
        sys.exit(0)

    print("\n  인사이트 생성 중...", end="", flush=True)
    insight = get_insight(client, mission["mission"], chunks, embeddings, bm25)
    print(f"\r{' '*25}\r", end="")
    print(f"\n  🔬 {insight}")

    print("\n  미션 메모를 남기시겠습니까? (Enter로 건너뛰기)")
    memo = input("  > ").strip()

    print("\n  미션 요약 중...", end="", flush=True)
    short_name = summarize_mission(client, mission["mission"])
    print(f"\r{' '*20}\r", end="")

    data["fruits"].append({
        "difficulty":   mission["difficulty"],
        "category":     mission["category"],
        "mission":      short_name,
        "full_mission": mission["mission"],
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mood":         state["mood"],
        "time":         state["time"],
        "emotion_type": emotion_type,
        "effect":       mission["effect"],
        "success":      True,
        "memo":         memo if memo else None,
        "photo_path":   None,
    })

    data.setdefault("mission_history", []).append(mission["mission"])
    if len(data["mission_history"]) > 20:
        data["mission_history"] = data["mission_history"][-20:]

    combo = check_combo(data, mission["category"])
    save_data(data)

    diff_info = DIFF.get(mission["difficulty"], {"col": "", "sym": ""})
    cat_info  = CAT.get(mission["category"],   {"col": "", "label": mission["category"]})
    print(f"\n  열매 획득! "
          f"{diff_info['col']}{diff_info['sym']}{RST} [{mission['difficulty']}] "
          f"{cat_info['col']}[{cat_info['label']}]{RST}")
    print(f"  ({short_name})")

    apply_combo_bonus(data, combo, mission["category"])

    render_tree(data["fruits"])

    print("\n  오늘도 잘 하셨습니다! 👍\n")


if __name__ == "__main__":
    main()
