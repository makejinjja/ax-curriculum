#!/usr/bin/env python3
"""
decision_mood_mission_rag_v4.py
기분전환 미션 AI + RAG + 감정분류 + 돌발미션 + 카테고리 열매 + 콤보 시스템
v4 변경사항:
  1. 나무 3단계 확장 (최대 30개: 새싹/성장/완전한 나무)
  2. [근거] 출처 표시 — 파일명 제거, 라벨만 표시
  3. 미션 저장 시 GPT로 10자 동사형 요약 → short name
  4. 미션 완료 후 성공/실패 선택 (실패 시 열매 없음 + 위로 메시지)
  5. 성공 후 메모 입력 (선택), photo_path 플레이스홀더
  6. parse_mission() — 블록 파싱으로 [효과] 잘림 버그 수정
  7. 기록 보기 메뉴 추가 (timestamp/기분/시간/감정/미션/효과/결과/메모)
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

load_dotenv(Path(__file__).parent / ".env")

DATA_FILE  = Path(__file__).parent / ".mission_data.json"
PDF_DIR    = Path(__file__).parent.parent / "data"
CACHE_FILE = Path(__file__).parent / ".index_cache" / "mission_rag_v4_index.json"

MAX_FRUITS = 30

PDF_FILES = [
    ("01.CBT.pdf",     "CBT(인지행동치료)"),
    ("02.behav.pdf",   "행동 활성화 1"),
    ("03.behav.pdf",   "행동 활성화 2"),
    ("04.mind.pdf",    "마음챙김"),
    ("05.emotion.pdf", "감정 조절 전략"),
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
}

# ── 카테고리 테이블 ──────────────────────────────────────────
CAT = {
    "건강":  {"sym": "(H)", "col": R,  "label": "건강"},
    "생산성":{"sym": "(P)", "col": BL, "label": "생산성"},
    "재미":  {"sym": "(F)", "col": PU, "label": "재미"},
    "성장":  {"sym": "(G)", "col": GN, "label": "성장"},
    "돌발":  {"sym": "(J)", "col": YL, "label": "돌발"},
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

# ── 감정 → 논문 소스 가중치 ─────────────────────────────────
EMOTION_SOURCE_WEIGHT = {
    "부정적": {"CBT":      2.0, "행동 활성화": 1.5},
    "중립":   {"행동 활성화": 1.5, "마음챙김":  1.2},
    "긍정적": {"감정 조절": 1.5},
    "집중됨": {"마음챙김":  2.0},
    "지루함": {"행동 활성화": 2.0},
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
def make_mission_prompt(emotion_type: str, purpose: str, recent_missions: str = "없음") -> str:
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
- 미션 소요 시간은 반드시 가용 시간과 일치해야 함:
  - 가용 시간 10분 → 실제 소요 5~15분인 미션만 제안
  - 가용 시간 30분 → 실제 소요 20~40분인 미션만 제안
  - 가용 시간 1시간 → 실제 소요 45~70분인 미션만 제안
- 난이도는 다음 확률로 결정해라: 하 50%, 중 30%, 상 15%, 최상 5%
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
- 하 또는 중 또는 상 또는 최상

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


MAX_CHUNK_CHARS = 6000


def _chunk(text: str, source: str) -> list[dict]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    idx = 0
    for para in paragraphs:
        para = para.strip()
        if len(para) < 80:
            continue
        for start in range(0, len(para), MAX_CHUNK_CHARS):
            segment = para[start:start + MAX_CHUNK_CHARS]
            chunks.append({"text": segment, "source": source, "chunk_index": idx})
            idx += 1
    return chunks


def build_index(client: OpenAI, force_rebuild: bool = False) -> tuple[list[dict], list[list[float]]]:
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
            return cache["chunks"], cache["embeddings"]

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
    return all_chunks, embeddings


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

TIME_QUERY_MAP = {
    "10분":  "brief intervention micro-habit short activity immediate",
    "30분":  "moderate duration activity engagement exercise",
    "1시간": "extended activity deep work immersive exercise",
}


def _expand_query(mood: str, time_str: str,
                  emotion_type: str | None = None,
                  target_cat: str | None = None) -> str:
    if emotion_type and target_cat:
        combo_kw = EMOTION_CAT_QUERY.get((emotion_type, target_cat))
        if combo_kw:
            time_kw = TIME_QUERY_MAP.get(time_str, "")
            return f"{combo_kw} {time_kw} psychological intervention evidence-based"

    mood_kw = ""
    for k, v in MOOD_QUERY_MAP.items():
        if k in mood:
            mood_kw = v
            break
    if not mood_kw:
        mood_kw = "emotion regulation mood improvement well-being intervention"
    time_kw = TIME_QUERY_MAP.get(time_str, "")
    return f"{mood_kw} {time_kw} psychological intervention evidence-based"


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def retrieve(query_emb, chunks, embeddings, k=4,
             emotion_type: str | None = None) -> list[dict]:
    weights = EMOTION_SOURCE_WEIGHT.get(emotion_type, {}) if emotion_type else {}

    scored = []
    for emb, chunk in zip(embeddings, chunks):
        score = _cosine(query_emb, emb)
        source = chunk.get("source", "")
        for keyword, multiplier in weights.items():
            if keyword in source:
                score *= multiplier
                break
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def build_context(top_chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[출처: {c['source']}]\n{c['text']}" for c in top_chunks
    )


def _source_label(s: str) -> str:
    """"01.CBT.pdf (CBT(인지행동치료))" → "CBT(인지행동치료)" """
    m = re.search(r'\((.+)\)$', s.strip())
    return m.group(1) if m else s


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

def get_mission(client: OpenAI, mood: str, time_str: str,
                chunks: list[dict], embeddings: list[list[float]],
                emotion_type: str, data: dict) -> tuple[str, bool, list[str]]:
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

    emotion_to_default_cat = {
        "부정적": "건강",
        "중립":   "생산성",
        "긍정적": "재미",
        "집중됨": "생산성",
        "지루함": "재미",
    }
    target_cat = emotion_to_default_cat.get(emotion_type)

    query     = _expand_query(mood, time_str, emotion_type=emotion_type, target_cat=target_cat)
    query_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding

    top     = retrieve(query_emb, chunks, embeddings, k=4, emotion_type=emotion_type)
    context = build_context(top)
    sources = list(dict.fromkeys(c["source"] for c in top))

    recent = data.get("mission_history", [])[-5:]
    recent_str = "\n".join(f"- {m}" for m in recent) if recent else "없음"

    purpose    = EMOTION_PURPOSE.get(emotion_type, "기분전환·회복")
    system_msg = make_mission_prompt(emotion_type, purpose, recent_missions=recent_str)
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
    """블록 파싱 — [태그] 헤더 기준으로 분리해 [효과] 잘림 버그 수정"""
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
        if m.get("sources"):
            for src in m["sources"]:
                label = _source_label(src)
                print(f"  {SV}└ {label}{RST}")

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
        label = "🌱 새싹 나무"
        cap   = 10
        print(f"\n{GN}{'━'*38}{RST}")
        print(f"  {BLD}{label}   [{n}/{cap} 열매]{RST}")
        print(f"{GN}{'━'*38}{RST}")
        print(f"                {s(0)}")
        print(f"             {s(1)}   {s(2)}")
        print(f"          {s(3)}   {s(4)}   {s(5)}")
        print(f"       {s(6)}  {s(7)}  {s(8)}  {s(9)}")
    elif n <= 20:
        label = "🌿 성장 나무"
        cap   = 20
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
        label = "🌲 완전한 나무"
        cap   = 30
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
    print(f"\n  {BLD}보유 카드{RST}")
    if not cards:
        print("  없음")
        return
    for card, cnt in Counter(c["card"] for c in cards).items():
        print(f"  • {card}  x{cnt}")


# ── 기록 보기 ────────────────────────────────────────────────

def show_history(data: dict):
    history = [f for f in data.get("fruits", []) if "timestamp" in f]
    if not history:
        print(f"\n  {'─'*34}")
        print("  기록된 미션이 없습니다.")
        print(f"  {'─'*34}")
        input("\n  Enter를 눌러 돌아가기... ")
        return

    print(f"\n{'='*38}")
    print(f"  {BLD}📋 미션 기록  ({len(history)}건){RST}")
    print(f"{'='*38}")

    for i, f in enumerate(reversed(history), 1):
        success_str = "✅ 성공" if f.get("success") else "❌ 실패"
        diff_info   = DIFF.get(f.get("difficulty", "하"), {"col": "", "sym": ""})
        cat_info    = CAT.get(f.get("category", "건강"), {"col": "", "label": "?"})
        print(f"\n  [{i}] {f.get('timestamp', '')}")
        print(f"  기분: {f.get('mood', '')}  |  시간: {f.get('time', '')}  |  감정: {f.get('emotion_type', '')}")
        print(f"  미션: {f.get('full_mission', f.get('mission', ''))}")
        if f.get("effect"):
            print(f"  효과: {f['effect']}")
        print(f"  결과: {success_str}   "
              f"{diff_info['col']}{f.get('difficulty','')}{RST} / "
              f"{cat_info['col']}{cat_info['label']}{RST}")
        if f.get("memo"):
            print(f"  메모: {f['memo']}")
        print(f"  {'─'*34}")

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
            card  = DIFF[d]["card"]
            data["cards"].append({"card": card, "difficulty": d})
            save_data(data)
            show_card(card, d)
            print(f"\n  {card}를 획득했습니다!")
            return
        print(f"  1~{len(fruits)} 또는 0을 입력하세요.")


# ── 콤보 시스템 ──────────────────────────────────────────────

def check_combo(data: dict, new_category: str) -> int:
    if new_category == "돌발":
        return data.get("combo_count", 0)

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
        if data["cards"]:
            bonus = data["cards"][-1]
            data["cards"].append({"card": bonus["card"], "difficulty": bonus["difficulty"]})
        else:
            data["cards"].append({"card": "목재 카드", "difficulty": "중"})
        save_data(data)
        bonus_name = data["cards"][-1]["card"]
        print(f"\n{GD}🔥 {cat_col}{cat_label}{RST}{GD} 미션 2연속! "
              f"보너스 카드 획득! ({bonus_name}){RST}")


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

    print("\n사용 가능 시간")
    opts = ["10분", "30분", "1시간"]
    for i, t in enumerate(opts, 1):
        print(f"  {i}. {t}")
    while True:
        sel = input("  선택 (1/2/3): ").strip()
        if sel in ("1", "2", "3"):
            time_str = opts[int(sel) - 1]
            break
        print("  1, 2, 3 중에 선택하세요.")

    return {"mood": mood, "time": time_str}


# ── 미션 루프 ────────────────────────────────────────────────

def mission_loop(client: OpenAI, mood: str, time_str: str,
                 chunks: list[dict], embeddings: list[list[float]],
                 emotion_type: str, data: dict) -> dict | None:
    while True:
        print("\n  미션 생성 중...", end="", flush=True)
        raw, is_wildcard, sources = get_mission(
            client, mood, time_str, chunks, embeddings, emotion_type, data
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
        if sel == "3":
            return None


# ── 시작 메뉴 ────────────────────────────────────────────────

def start_menu(data: dict) -> str:
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
    show_cards(data["cards"])

    print(f"\n  1. 미션 시작")
    print(f"  2. 열매 쪼개기")
    print(f"  3. 기록 보기")
    print(f"  4. 종료")

    while True:
        sel = input("  선택 (1/2/3/4): ").strip()
        if sel == "1":
            return "mission"
        if sel == "2":
            split_fruit_menu(data)
            render_tree(data["fruits"])
            show_cards(data["cards"])
            return start_menu(data)
        if sel == "3":
            show_history(data)
            return start_menu(data)
        if sel == "4":
            return "exit"
        print("  1, 2, 3, 4 중에 선택하세요.")


# ── main ─────────────────────────────────────────────────────

def main():
    force_rebuild = "--rebuild" in sys.argv

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    data   = load_data()

    chunks, embeddings = build_index(client, force_rebuild=force_rebuild)

    action = start_menu(data)
    if action == "exit":
        print("\n  종료합니다.\n")
        sys.exit(0)

    if len(data["fruits"]) >= MAX_FRUITS:
        print(f"\n  {BLD}🌲 나무가 가득 찼습니다! 열매를 먼저 쪼개야 미션을 시작할 수 있습니다.{RST}")
        while len(data["fruits"]) >= MAX_FRUITS:
            split_fruit_menu(data)
            render_tree(data["fruits"])
            show_cards(data["cards"])
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

    mission = mission_loop(client, state["mood"], state["time"],
                           chunks, embeddings, emotion_type, data)
    if not mission:
        print("\n  종료합니다.\n")
        sys.exit(0)

    print(f"\n  미션을 시작하세요!")
    print(f"  완료 후 아래에서 결과를 선택해 주세요.")
    input(f"\n  ▶ 준비 완료 → Enter ")

    # 성공/실패 선택
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
        # 미션 히스토리에는 추가하지 않음 (재도전 허용)
        print("\n  오늘도 수고하셨습니다! 👍\n")
        sys.exit(0)

    # 메모 입력 (선택)
    print("\n  미션 메모를 남기시겠습니까? (Enter로 건너뛰기)")
    memo = input("  > ").strip()

    # 미션 요약 (10자 동사형 short name)
    print("\n  미션 요약 중...", end="", flush=True)
    short_name = summarize_mission(client, mission["mission"])
    print(f"\r{' '*20}\r", end="")

    # 열매 저장
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

    # 미션 히스토리 저장 (성공한 미션만, 최대 20개)
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
    show_cards(data["cards"])

    if data["fruits"]:
        print("\n  열매를 쪼개시겠습니까? (y/n)")
        if input("  > ").strip().lower() == "y":
            split_fruit_menu(data)
            render_tree(data["fruits"])
            show_cards(data["cards"])

    print("\n  오늘도 잘 하셨습니다! 👍\n")


if __name__ == "__main__":
    main()
