#!/usr/bin/env python3
"""
decision_ai_v2.py — Decision Removal AI (고도화 RAG)
  - 14.indexing.py의 고도화 파이프라인 적용
  - DocType별 파서 / 메타데이터 사전필터 / 다양성 보장 / 증분 인덱싱
"""

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env")

PDF_PATH   = Path(__file__).parent.parent / "data" / "inf.pdf.pdf"
CACHE_DIR  = Path(__file__).parent / ".index_cache"
CACHE_FILE = CACHE_DIR / "inf_index.json"

CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """너는 "Decision Removal AI"다.
아래 [심리·신경과학 지식 베이스]에서 검색된 이론적 근거를 참고하여
사용자의 현재 상태에 가장 적합한 "지금 당장 해야 할 행동 1개"를 결정해주는 시스템이다.

너의 목적은 사용자의 생산성, 건강, 삶의 균형을 개선하는 것이지만
절대 여러 선택지를 주지 않는다.

너는 코치가 아니라 "결정 대행 엔진"이다.
사용자의 생각을 줄이고 즉시 행동하게 만드는 것이 최우선 목표다.

[핵심 원칙]
1. 항상 단 하나의 행동만 제시한다.
2. 행동은 5~30분 내 실행 가능한 것이어야 한다.
3. 모호한 조언 금지 (ex. "운동해라" → "15분 걷기")
4. 실행 가능성이 가장 높은 행동을 선택한다.
5. 사용자의 현재 상태를 최우선으로 반영한다.
6. 장기 계획보다 "지금 즉시 행동"에 집중한다.

[판단 로직]
1. 지식 베이스의 각성-동기 상태 모델로 사용자 심리 상태를 분석
2. 에너지 상태 분류와 Circadian Rhythm 정보로 생리적 최적 행동 필터링
3. 개인 최적화 전략 유형(목표)에 맞는 행동 우선 선택
4. 실행 난이도가 낮고 즉시 가능한 행동 우선 선택

[출력 형식] 반드시 아래 형식으로만 출력한다:

[행동]
- (지금 당장 해야 할 단 하나의 행동)

[심리·생리 상태 분석]
- 각성-동기 상태: (무기력형/스트레스형/안정형/집중형 중 해당 유형과 그 특징 1~2줄)
- 에너지 상태: (에너지 저하/균형/활성 상태 판단 및 생리적 근거 1줄)
- 시간대 최적화: (현재 시간대의 Circadian Rhythm 특성과 적합 활동 1줄)

[근거]
- 상태 이론: (각성-동기 모델 또는 에너지 상태 분류에서 이 행동이 도출되는 신경과학적 이유 2~3줄)
- 목표 연계: (사용자 목표 유형(건강/생산성/돈/균형)의 전략 특성과 행동의 연결고리 1~2줄)
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


# ══════════════════════════════════════════════════════════════════════════════
# 타입 정의
# ══════════════════════════════════════════════════════════════════════════════

class DocType(str, Enum):
    STATE_MODEL   = "state_model"
    STRATEGY_TYPE = "strategy_type"
    ENERGY_STATE  = "energy_state"
    CIRCADIAN     = "circadian"


@dataclass
class Chunk:
    chunk_id:      str
    doc_type:      DocType
    tab_id:        int
    tab_label:     str
    section_title: str
    body:          str
    subsections:   dict         = field(default_factory=dict)
    mood_tags:     list[str]    = field(default_factory=list)
    energy_tags:   list[str]    = field(default_factory=list)
    goal_tags:     list[str]    = field(default_factory=list)
    time_start:    Optional[int] = None
    time_end:      Optional[int] = None
    arousal_level: Optional[str] = None
    keywords:      list[str]    = field(default_factory=list)
    embed_text:    str          = ""
    char_count:    int          = 0
    embedding:     list[float]  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# PDF 추출 + 구조 보존 전처리
# ══════════════════════════════════════════════════════════════════════════════

def extract_structured_text(pdf_path: Path) -> dict[str, str]:
    reader = PdfReader(str(pdf_path))
    raw = "\n".join(p.extract_text() or "" for p in reader.pages)
    raw = re.sub(r"\r\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n[ \t]+", "\n", raw)

    tab_labels = {
        "1": "각성-동기 상태 모델",
        "2": "개인 최적화 전략 유형",
        "3": "에너지 상태 분류",
        "4": "Circadian Rhythm 기반 시간대별 최적화",
    }
    tab_pattern = re.compile(
        r"탭\s*(?P<num>[1-4])\s*\n(?P<content>.*?)(?=탭\s*[1-4]|\Z)", re.DOTALL
    )
    tabs: dict[str, str] = {}
    for m in tab_pattern.finditer(raw):
        num = m.group("num")
        tabs[tab_labels[num]] = m.group("content").strip()
    return tabs if tabs else {"전체": raw}


def extract_subsections(body: str) -> dict[str, str]:
    known = ["특징", "신경학적 상태", "이론적 근거", "요약", "개요", "행동 특성",
             "핵심 근거", "함의", "정의", "주요 특징", "생리적 배경", "신경계 특성"]
    pattern = re.compile(
        r"(?P<key>" + "|".join(known) + r")\s*\n(?P<val>.*?)(?=" + "|".join(known) + r"|\Z)",
        re.DOTALL,
    )
    result = {}
    for m in pattern.finditer(body):
        key = m.group("key").strip()
        val = re.sub(r"\n{2,}", "\n", m.group("val")).strip()
        result[key] = val
    return result


def extract_keywords(text: str) -> list[str]:
    kws = set()
    kws.update(re.findall(r"\b[A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z]+)*\b", text))
    for line in text.splitlines():
        line = line.strip().lstrip("●•-").strip()
        if line and len(line) <= 20:
            kws.add(line.split("(")[0].strip())
    return [k for k in kws if len(k) >= 2][:20]


# ══════════════════════════════════════════════════════════════════════════════
# DocType별 파서
# ══════════════════════════════════════════════════════════════════════════════

_SEC_PATTERN = re.compile(
    r"(?m)^(?P<title>[1-9]\.\s+.+?)(?:\s*\(.+?\))?\s*\n(?P<body>.*?)(?=^[1-9]\.|^탭\s*[1-4]|\Z)",
    re.DOTALL,
)

_AROUSAL_MAP = {"무기력형": "low", "스트레스형": "high", "안정형": "moderate", "집중형": "optimal"}
_MOOD_MAP    = {"무기력형": ["무기력"], "스트레스형": ["스트레스"], "안정형": ["안정"], "집중형": ["집중됨"]}
_GOAL_MAP    = {"건강 중심형": ["건강"], "생산성 중심형": ["생산성"], "돈 중심형": ["돈"], "균형 중심형": ["균형"]}
_ENERGY_LEVEL_MAP = {"에너지 저하 상태": "낮음", "에너지 균형 상태": "보통", "에너지 활성 상태": "높음"}
_CIRCADIAN_SLOTS  = [
    (5,  10, "아침"), (10, 13, "오전"), (13, 15, "점심후"),
    (15, 18, "오후"), (18, 21, "저녁"), (21, 24, "밤"), (0, 5, "새벽"),
]


def parse_state_model(tab_text: str) -> list[Chunk]:
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title, body = m.group("title").strip(), m.group("body").strip()
        if len(body) < 40:
            continue
        key         = next((k for k in _MOOD_MAP if k in title), None)
        mood_tags   = _MOOD_MAP.get(key, [])
        arousal     = _AROUSAL_MAP.get(key)
        energy_tags = ["낮음"] if arousal == "low" else (["높음"] if arousal == "optimal" else ["보통"])
        subsec      = extract_subsections(body)
        kws         = extract_keywords(body)
        embed_text  = (
            f"[상태모델] [기분:{','.join(mood_tags)}] [에너지:{','.join(energy_tags)}]\n"
            f"섹션: {title}\n"
            + (f"특징: {subsec.get('특징','')}\n" if subsec.get('특징') else "")
            + (f"신경학적 상태: {subsec.get('신경학적 상태','')}\n" if subsec.get('신경학적 상태') else "")
            + (f"요약: {subsec.get('요약','')}\n" if subsec.get('요약') else "")
            + f"키워드: {', '.join(kws[:8])}"
        )
        chunks.append(Chunk(
            chunk_id=f"tab1_{re.sub(r'\\W','_',title)}", doc_type=DocType.STATE_MODEL,
            tab_id=1, tab_label="각성-동기 상태 모델", section_title=title, body=body,
            subsections=subsec, mood_tags=mood_tags, energy_tags=energy_tags,
            goal_tags=["건강","생산성","돈","균형"], arousal_level=arousal,
            keywords=kws, embed_text=embed_text, char_count=len(body),
        ))
    return chunks


def parse_strategy_types(tab_text: str) -> list[Chunk]:
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title, body = m.group("title").strip(), m.group("body").strip()
        if len(body) < 40:
            continue
        key       = next((k for k in _GOAL_MAP if k in title), None)
        goal_tags = _GOAL_MAP.get(key, [])
        subsec    = extract_subsections(body)
        kws       = extract_keywords(body)
        embed_text = (
            f"[전략유형] [목표:{','.join(goal_tags)}]\n섹션: {title}\n"
            + (f"개요: {subsec.get('개요','')}\n" if subsec.get('개요') else "")
            + (f"행동특성: {subsec.get('행동 특성','')}\n" if subsec.get('행동 특성') else "")
            + (f"함의: {subsec.get('함의','')}\n" if subsec.get('함의') else "")
            + f"키워드: {', '.join(kws[:8])}"
        )
        chunks.append(Chunk(
            chunk_id=f"tab2_{re.sub(r'\\W','_',title)}", doc_type=DocType.STRATEGY_TYPE,
            tab_id=2, tab_label="개인 최적화 전략 유형", section_title=title, body=body,
            subsections=subsec, mood_tags=["무기력","스트레스","안정","집중됨"],
            energy_tags=["낮음","보통","높음"], goal_tags=goal_tags,
            keywords=kws, embed_text=embed_text, char_count=len(body),
        ))
    return chunks


def parse_energy_states(tab_text: str) -> list[Chunk]:
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title, body = m.group("title").strip(), m.group("body").strip()
        if len(body) < 40:
            continue
        energy_level = next((v for k, v in _ENERGY_LEVEL_MAP.items() if k in title), "보통")
        subsec = extract_subsections(body)
        kws    = extract_keywords(body)
        embed_text = (
            f"[에너지상태] [에너지:{energy_level}]\n섹션: {title}\n"
            + (f"정의: {subsec.get('정의','')}\n" if subsec.get('정의') else "")
            + (f"주요특징: {subsec.get('주요 특징','')}\n" if subsec.get('주요 특징') else "")
            + (f"신경계특성: {subsec.get('신경계 특성','')}\n" if subsec.get('신경계 특성') else "")
            + (f"요약: {subsec.get('요약','')}\n" if subsec.get('요약') else "")
            + f"키워드: {', '.join(kws[:8])}"
        )
        chunks.append(Chunk(
            chunk_id=f"tab3_{re.sub(r'\\W','_',title)}", doc_type=DocType.ENERGY_STATE,
            tab_id=3, tab_label="에너지 상태 분류", section_title=title, body=body,
            subsections=subsec, mood_tags=["무기력","스트레스","안정","집중됨"],
            energy_tags=[energy_level], goal_tags=["건강","생산성","돈","균형"],
            keywords=kws, embed_text=embed_text, char_count=len(body),
        ))
    return chunks


def parse_circadian(tab_text: str) -> list[Chunk]:
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title, body = m.group("title").strip(), m.group("body").strip()
        if len(body) < 40:
            continue
        t_start, t_end, slot_name = None, None, ""
        for ts, te, sn in _CIRCADIAN_SLOTS:
            if sn in title:
                t_start, t_end, slot_name = ts, te, sn
                break
        kws = extract_keywords(body)
        embed_text = (
            f"[Circadian] [시간:{slot_name or title}]"
            + (f"[{t_start}시-{t_end}시]" if t_start is not None else "")
            + f"\n섹션: {title}\n{body[:300]}\n키워드: {', '.join(kws[:8])}"
        )
        chunks.append(Chunk(
            chunk_id=f"tab4_{re.sub(r'\\W','_',title)}", doc_type=DocType.CIRCADIAN,
            tab_id=4, tab_label="Circadian Rhythm 기반 시간대별 최적화", section_title=title,
            body=body, subsections={}, mood_tags=["무기력","스트레스","안정","집중됨"],
            energy_tags=["낮음","보통","높음"], goal_tags=["건강","생산성","돈","균형"],
            time_start=t_start, time_end=t_end, keywords=kws,
            embed_text=embed_text, char_count=len(body),
        ))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 증분 인덱싱
# ══════════════════════════════════════════════════════════════════════════════

def compute_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_cache() -> Optional[dict]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return None


def save_cache(source_hash: str, chunks: list[Chunk]):
    payload = {
        "source_hash": source_hash,
        "indexed_at":  datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "chunks":      [asdict(c) for c in chunks],
    }
    CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chunks_from_cache(data: dict) -> list[Chunk]:
    result = []
    for d in data["chunks"]:
        d["doc_type"] = DocType(d["doc_type"])
        result.append(Chunk(**d))
    return result


def build_index(client: OpenAI, force: bool = False) -> tuple[list[Chunk], bool]:
    source_hash = compute_hash(PDF_PATH)
    cache = load_cache()
    if not force and cache and cache.get("source_hash") == source_hash:
        return chunks_from_cache(cache), True

    tabs = extract_structured_text(PDF_PATH)
    parsers = {
        "각성-동기 상태 모델":                    parse_state_model,
        "개인 최적화 전략 유형":                   parse_strategy_types,
        "에너지 상태 분류":                       parse_energy_states,
        "Circadian Rhythm 기반 시간대별 최적화":   parse_circadian,
    }
    chunks: list[Chunk] = []
    for label, text in tabs.items():
        parser = parsers.get(label)
        if parser:
            chunks.extend(parser(text))

    texts = [c.embed_text for c in chunks]
    resp  = client.embeddings.create(model="text-embedding-3-small", input=texts)
    for chunk, item in zip(chunks, resp.data):
        chunk.embedding = item.embedding

    save_cache(source_hash, chunks)
    return chunks, False


# ══════════════════════════════════════════════════════════════════════════════
# 향상된 검색
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def metadata_filter(chunks: list[Chunk], state: dict) -> list[Chunk]:
    mood   = state.get("기분", "")
    energy = state.get("에너지 상태", "")
    goal   = state.get("목표", "")
    hour   = int(state.get("현재 시간", "12:00").split(":")[0])

    def passes(c: Chunk) -> bool:
        mood_ok   = not c.mood_tags   or mood   in c.mood_tags
        energy_ok = not c.energy_tags or energy in c.energy_tags
        goal_ok   = not c.goal_tags   or goal   in c.goal_tags
        time_ok   = (c.time_start is None) or (c.time_start <= hour < c.time_end)
        if c.doc_type == DocType.CIRCADIAN:
            return time_ok
        return mood_ok or energy_ok or goal_ok

    filtered = [c for c in chunks if passes(c)]
    return filtered if filtered else chunks


def hybrid_retrieve(query_emb: list[float], chunks: list[Chunk], state: dict, k: int = 4) -> list[Chunk]:
    pool = metadata_filter(chunks, state)
    scored = sorted(
        [(cosine_sim(query_emb, c.embedding), c) for c in pool],
        key=lambda x: x[0], reverse=True,
    )
    seen_types: set[DocType] = set()
    result, remainder = [], []
    for score, chunk in scored:
        if chunk.doc_type not in seen_types:
            result.append((score, chunk))
            seen_types.add(chunk.doc_type)
        else:
            remainder.append((score, chunk))
        if len(result) >= min(k, len(DocType)):
            break
    for score, chunk in remainder:
        if len(result) >= k:
            break
        result.append((score, chunk))
    return [c for _, c in sorted(result, key=lambda x: x[0], reverse=True)]


def build_query(state: dict) -> str:
    mood_map = {
        "무기력": "무기력형 Low Arousal Low Motivation 행동 개시 어려움 도파민 저활성 에너지 저하",
        "스트레스": "스트레스형 High Arousal Negative Emotion 불안 편도체 과활성 전전두엽 저하",
        "안정": "안정형 Moderate Arousal Positive Emotion 세로토닌 균형 회복 부교감신경",
        "집중됨": "집중형 Optimal Arousal High Engagement Flow 도파민 노르에피네프린 전전두엽 활성",
    }
    goal_map = {
        "건강": "건강 중심형 Energy Maximizer 수면 운동 식단 회복 Self-Regulation",
        "생산성": "생산성 중심형 Time Optimizer Deep Work 집중 시간블록 전두엽",
        "돈": "돈 중심형 Capital Maximizer ROI 레버리지 도파민 보상 시스템",
        "균형": "균형 중심형 Well-being Optimizer 삶의 만족 자율성 관계 Hedonic Adaptation",
    }
    energy_map = {
        "낮음": "에너지 저하 Low Energy State 피로 회복 우선 자율신경 불균형",
        "보통": "에너지 균형 Balanced State 항상성 Homeostasis 교감 부교감 균형",
        "높음": "에너지 활성 High Energy State 도파민 보상 높은 집중력 행동 선순환",
    }
    hour = int(state.get("현재 시간", "12:00").split(":")[0])
    if 5 <= hour < 10:
        time_ctx = "아침 코르티솔 각성 반응 의사결정 집중 작업 개시"
    elif 10 <= hour < 13:
        time_ctx = "오전 고집중 작업 구간 인지 속도 논리 학습"
    elif 13 <= hour < 15:
        time_ctx = "점심 후 졸음 Post-lunch Dip 낮잠 회복 생산성"
    elif 15 <= hour < 18:
        time_ctx = "오후 사회적 상호작용 감정 인식 의사소통"
    elif 18 <= hour < 21:
        time_ctx = "저녁 운동 최적화 체온 최고점 근력 유연성"
    else:
        time_ctx = "밤 창의적 사고 뇌 억제 약화 Glymphatic 수면 준비"
    return (
        f"{mood_map.get(state.get('기분',''), state.get('기분',''))} "
        f"{goal_map.get(state.get('목표',''), state.get('목표',''))} "
        f"{energy_map.get(state.get('에너지 상태',''), state.get('에너지 상태',''))} "
        f"{time_ctx} 상황:{state.get('현재 상황','')} 가용시간:{state.get('사용 가능 시간','')}"
    )


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        header = f"[{c.tab_label}] {c.section_title}"
        meta   = " | ".join(filter(None, [
            f"기분:{c.mood_tags}" if c.mood_tags and len(c.mood_tags) < 4 else "",
            f"에너지:{c.energy_tags}" if c.energy_tags and len(c.energy_tags) < 3 else "",
            f"목표:{c.goal_tags}" if c.goal_tags and len(c.goal_tags) < 4 else "",
        ]))
        parts.append(f"{header}\n{meta}\n{c.body}" if meta else f"{header}\n{c.body}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_input(prompt, options=None):
    while True:
        value = input(prompt).strip()
        if not value:
            print("  입력이 필요합니다.")
            continue
        if options and value not in options:
            print(f"  선택지: {' / '.join(options)}")
            continue
        return value


def collect_state() -> dict:
    print("\n=== Decision Removal AI v2 (고도화 RAG) ===\n")
    time_now = get_input("현재 시간 (예: 14:30): ")
    print("\n에너지 상태")
    energy = get_input("  낮음 / 보통 / 높음: ", ["낮음", "보통", "높음"])
    print("\n기분")
    mood = get_input("  무기력 / 스트레스 / 안정 / 집중됨: ", ["무기력", "스트레스", "안정", "집중됨"])
    print("\n현재 상황")
    location = get_input("  집 / 회사 / 이동중: ", ["집", "회사", "이동중"])
    print("\n목표")
    goal = get_input("  건강 / 생산성 / 돈 / 균형: ", ["건강", "생산성", "돈", "균형"])
    available_time = get_input("\n사용 가능 시간 (분, 예: 20): ")
    return {
        "현재 시간": time_now,
        "에너지 상태": energy,
        "기분": mood,
        "현재 상황": location,
        "목표": goal,
        "사용 가능 시간": f"{available_time}분",
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    force  = "--rebuild" in sys.argv

    print("지식 베이스 로딩 중...", end="", flush=True)
    t0 = time.time()
    chunks, was_cached = build_index(client, force=force)
    elapsed = time.time() - t0
    status  = "캐시" if was_cached else "신규 빌드"
    print(f" [{status}] {len(chunks)}개 청크 완료 ({elapsed:.1f}s)")

    try:
        state = collect_state()

        query     = build_query(state)
        query_emb = client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
        top_chunks = hybrid_retrieve(query_emb, chunks, state, k=4)
        context    = build_context(top_chunks)

        user_msg = (
            "[사용자 현재 상태]\n"
            + "\n".join(f"- {k}: {v}" for k, v in state.items())
            + f"\n\n[심리·신경과학 지식 베이스 - 관련 이론]\n{context}"
        )

        print("\n결정 중...\n")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=600,
        )

        print(response.choices[0].message.content)
        print()

    except KeyboardInterrupt:
        print("\n\n종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
