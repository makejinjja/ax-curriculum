#!/usr/bin/env python3
"""
14.indexing.py — 고도화된 RAG 인덱싱 파이프라인
────────────────────────────────────────────────
개선 포인트
  1. 문서 종류(doc_type)별 인덱싱 전략 분리
  2. 청킹 전 문서 구조 보존 전처리
  3. 검색에 유리한 메타데이터 확장
  4. 증분 인덱싱(hash 기반 캐시) 지원

실행:  python 14.indexing.py
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


# ══════════════════════════════════════════════════════════════════════════════
# 1. 타입 정의
# ══════════════════════════════════════════════════════════════════════════════

class DocType(str, Enum):
    STATE_MODEL   = "state_model"     # 탭1 · 각성-동기 상태 모델
    STRATEGY_TYPE = "strategy_type"   # 탭2 · 개인 최적화 전략 유형
    ENERGY_STATE  = "energy_state"    # 탭3 · 에너지 상태 분류
    CIRCADIAN     = "circadian"       # 탭4 · Circadian Rhythm


@dataclass
class Chunk:
    # 식별
    chunk_id:      str
    doc_type:      DocType

    # 원문
    tab_id:        int
    tab_label:     str
    section_title: str
    body:          str

    # 구조 보존 필드 (전처리 결과)
    subsections:   dict = field(default_factory=dict)   # {"특징": "...", "신경학적 상태": "..."}

    # 메타데이터 (검색 필터링용)
    mood_tags:     list[str] = field(default_factory=list)   # ["무기력"] / ["스트레스"] 등
    energy_tags:   list[str] = field(default_factory=list)   # ["낮음"] / ["보통"] / ["높음"]
    goal_tags:     list[str] = field(default_factory=list)   # ["건강"] / ["생산성"] 등
    time_start:    Optional[int] = None   # 시(hour) 시작 — circadian 전용
    time_end:      Optional[int] = None   # 시(hour) 끝   — circadian 전용
    arousal_level: Optional[str] = None  # "low" / "moderate" / "high" / "optimal"
    keywords:      list[str] = field(default_factory=list)

    # 임베딩용 텍스트 (메타데이터 태그 포함 · 풍부한 표현)
    embed_text:    str = ""

    # 통계
    char_count:    int = 0

    # 임베딩 (증분 저장)
    embedding:     list[float] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PDF 추출 + 구조 보존 전처리
# ══════════════════════════════════════════════════════════════════════════════

def extract_structured_text(pdf_path: Path) -> dict[str, str]:
    """탭 단위로 분리된 원문 반환 + 불필요한 공백·헤더 정리."""
    reader = PdfReader(str(pdf_path))
    raw = "\n".join(p.extract_text() or "" for p in reader.pages)

    # 전처리: 과도한 개행 정규화, 불필요 공백 제거
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
        r"탭\s*(?P<num>[1-4])\s*\n(?P<content>.*?)(?=탭\s*[1-4]|\Z)",
        re.DOTALL,
    )

    tabs: dict[str, str] = {}
    for m in tab_pattern.finditer(raw):
        num = m.group("num")
        tabs[tab_labels[num]] = m.group("content").strip()

    if not tabs:
        tabs["전체"] = raw

    return tabs


def extract_subsections(body: str) -> dict[str, str]:
    """본문에서 소제목(특징 / 신경학적 상태 / 이론적 근거 / 요약 등)을 키-값으로 파싱."""
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
    """불릿 항목, 괄호 속 영어 개념, 핵심 명사 추출."""
    kws = set()
    # 영어 전문 용어
    kws.update(re.findall(r"\b[A-Z][a-zA-Z\-]+(?:\s[A-Z][a-zA-Z]+)*\b", text))
    # 불릿 첫 명사구 (2~6자)
    for line in text.splitlines():
        line = line.strip().lstrip("●•-").strip()
        if line and len(line) <= 20:
            kws.add(line.split("(")[0].strip())
    return [k for k in kws if len(k) >= 2][:20]


# ══════════════════════════════════════════════════════════════════════════════
# 3. DocType별 파서
# ══════════════════════════════════════════════════════════════════════════════

_SEC_PATTERN = re.compile(
    r"(?m)^(?P<title>[1-9]\.\s+.+?)(?:\s*\(.+?\))?\s*\n(?P<body>.*?)(?=^[1-9]\.|^탭\s*[1-4]|\Z)",
    re.DOTALL,
)

_AROUSAL_MAP = {
    "무기력형": "low",
    "스트레스형": "high",
    "안정형": "moderate",
    "집중형": "optimal",
}

_MOOD_MAP = {
    "무기력형": ["무기력"],
    "스트레스형": ["스트레스"],
    "안정형": ["안정"],
    "집중형": ["집중됨"],
}

_GOAL_MAP = {
    "건강 중심형": ["건강"],
    "생산성 중심형": ["생산성"],
    "돈 중심형": ["돈"],
    "균형 중심형": ["균형"],
}

_ENERGY_LEVEL_MAP = {
    "에너지 저하 상태": "낮음",
    "에너지 균형 상태": "보통",
    "에너지 활성 상태": "높음",
}

_CIRCADIAN_SLOTS = [
    (5,  10, "아침", "아침 코르티솔 각성"),
    (10, 13, "오전", "오전 고집중 작업"),
    (13, 15, "점심후", "점심 후 졸음"),
    (15, 18, "오후", "오후 사회적 상호작용"),
    (18, 21, "저녁", "저녁 운동"),
    (21, 24, "밤", "밤 창의적 사고"),
    (0,   5, "새벽", "수면 뇌 정화"),
]


def parse_state_model(tab_text: str) -> list[Chunk]:
    """탭1: 각성-동기 상태 모델 — 4개 유형 파싱."""
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title = m.group("title").strip()
        body  = m.group("body").strip()
        if len(body) < 40:
            continue

        key = next((k for k in _MOOD_MAP if k in title), None)
        mood_tags   = _MOOD_MAP.get(key, [])
        arousal     = _AROUSAL_MAP.get(key)
        energy_tags = ["낮음"] if arousal in ("low",) else (
                      ["높음"] if arousal == "optimal" else ["보통"])

        subsec = extract_subsections(body)
        kws    = extract_keywords(body)

        embed_text = (
            f"[상태모델] [기분:{','.join(mood_tags)}] [에너지:{','.join(energy_tags)}]\n"
            f"섹션: {title}\n"
            + (f"특징: {subsec.get('특징','')}\n" if subsec.get('특징') else "")
            + (f"신경학적 상태: {subsec.get('신경학적 상태','')}\n" if subsec.get('신경학적 상태') else "")
            + (f"요약: {subsec.get('요약','')}\n" if subsec.get('요약') else "")
            + f"키워드: {', '.join(kws[:8])}"
        )

        chunks.append(Chunk(
            chunk_id      = f"tab1_{re.sub(r'\\W','_',title)}",
            doc_type      = DocType.STATE_MODEL,
            tab_id        = 1,
            tab_label     = "각성-동기 상태 모델",
            section_title = title,
            body          = body,
            subsections   = subsec,
            mood_tags     = mood_tags,
            energy_tags   = energy_tags,
            goal_tags     = ["건강", "생산성", "돈", "균형"],
            arousal_level = arousal,
            keywords      = kws,
            embed_text    = embed_text,
            char_count    = len(body),
        ))
    return chunks


def parse_strategy_types(tab_text: str) -> list[Chunk]:
    """탭2: 개인 최적화 전략 유형 — 4개 목표 유형 파싱."""
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title = m.group("title").strip()
        body  = m.group("body").strip()
        if len(body) < 40:
            continue

        key       = next((k for k in _GOAL_MAP if k in title), None)
        goal_tags = _GOAL_MAP.get(key, [])
        subsec    = extract_subsections(body)
        kws       = extract_keywords(body)

        embed_text = (
            f"[전략유형] [목표:{','.join(goal_tags)}]\n"
            f"섹션: {title}\n"
            + (f"개요: {subsec.get('개요','')}\n" if subsec.get('개요') else "")
            + (f"행동특성: {subsec.get('행동 특성','')}\n" if subsec.get('행동 특성') else "")
            + (f"함의: {subsec.get('함의','')}\n" if subsec.get('함의') else "")
            + f"키워드: {', '.join(kws[:8])}"
        )

        chunks.append(Chunk(
            chunk_id      = f"tab2_{re.sub(r'\\W','_',title)}",
            doc_type      = DocType.STRATEGY_TYPE,
            tab_id        = 2,
            tab_label     = "개인 최적화 전략 유형",
            section_title = title,
            body          = body,
            subsections   = subsec,
            mood_tags     = ["무기력", "스트레스", "안정", "집중됨"],
            energy_tags   = ["낮음", "보통", "높음"],
            goal_tags     = goal_tags,
            keywords      = kws,
            embed_text    = embed_text,
            char_count    = len(body),
        ))
    return chunks


def parse_energy_states(tab_text: str) -> list[Chunk]:
    """탭3: 에너지 상태 분류 — 3단계 파싱."""
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title = m.group("title").strip()
        body  = m.group("body").strip()
        if len(body) < 40:
            continue

        energy_level = next(
            (v for k, v in _ENERGY_LEVEL_MAP.items() if k in title), "보통"
        )
        subsec = extract_subsections(body)
        kws    = extract_keywords(body)

        embed_text = (
            f"[에너지상태] [에너지:{energy_level}]\n"
            f"섹션: {title}\n"
            + (f"정의: {subsec.get('정의','')}\n" if subsec.get('정의') else "")
            + (f"주요특징: {subsec.get('주요 특징','')}\n" if subsec.get('주요 특징') else "")
            + (f"신경계특성: {subsec.get('신경계 특성','')}\n" if subsec.get('신경계 특성') else "")
            + (f"요약: {subsec.get('요약','')}\n" if subsec.get('요약') else "")
            + f"키워드: {', '.join(kws[:8])}"
        )

        chunks.append(Chunk(
            chunk_id      = f"tab3_{re.sub(r'\\W','_',title)}",
            doc_type      = DocType.ENERGY_STATE,
            tab_id        = 3,
            tab_label     = "에너지 상태 분류",
            section_title = title,
            body          = body,
            subsections   = subsec,
            mood_tags     = ["무기력", "스트레스", "안정", "집중됨"],
            energy_tags   = [energy_level],
            goal_tags     = ["건강", "생산성", "돈", "균형"],
            keywords      = kws,
            embed_text    = embed_text,
            char_count    = len(body),
        ))
    return chunks


def parse_circadian(tab_text: str) -> list[Chunk]:
    """탭4: Circadian Rhythm — 시간 슬롯별 파싱."""
    chunks = []
    for m in _SEC_PATTERN.finditer(tab_text):
        title = m.group("title").strip()
        body  = m.group("body").strip()
        if len(body) < 40:
            continue

        # 시간 범위 추출
        t_start, t_end, slot_name = None, None, ""
        for ts, te, sn, _ in _CIRCADIAN_SLOTS:
            if sn in title or any(w in title for w in ["아침", "오전", "점심", "오후", "저녁", "밤", "수면"]):
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
            chunk_id      = f"tab4_{re.sub(r'\\W','_',title)}",
            doc_type      = DocType.CIRCADIAN,
            tab_id        = 4,
            tab_label     = "Circadian Rhythm 기반 시간대별 최적화",
            section_title = title,
            body          = body,
            subsections   = {},
            mood_tags     = ["무기력", "스트레스", "안정", "집중됨"],
            energy_tags   = ["낮음", "보통", "높음"],
            goal_tags     = ["건강", "생산성", "돈", "균형"],
            time_start    = t_start,
            time_end      = t_end,
            keywords      = kws,
            embed_text    = embed_text,
            char_count    = len(body),
        ))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 4. 증분 인덱싱 — IndexManager
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
    """
    증분 인덱싱:
      - PDF 해시가 캐시와 같으면 임베딩 재사용 (API 호출 0)
      - 달라지면 청킹 재실행 + 임베딩 재생성
    Returns (chunks, was_cached)
    """
    source_hash = compute_hash(PDF_PATH)
    cache = load_cache()

    if not force and cache and cache.get("source_hash") == source_hash:
        chunks = chunks_from_cache(cache)
        return chunks, True   # 캐시 히트

    # ── 전처리 ──────────────────────────────────────────────
    tabs = extract_structured_text(PDF_PATH)

    # ── DocType별 파싱 ──────────────────────────────────────
    chunks: list[Chunk] = []
    parsers = {
        "각성-동기 상태 모델":           parse_state_model,
        "개인 최적화 전략 유형":          parse_strategy_types,
        "에너지 상태 분류":              parse_energy_states,
        "Circadian Rhythm 기반 시간대별 최적화": parse_circadian,
    }
    for label, text in tabs.items():
        parser = parsers.get(label)
        if parser:
            chunks.extend(parser(text))

    # ── 임베딩 ─────────────────────────────────────────────
    texts = [c.embed_text for c in chunks]
    resp  = client.embeddings.create(model="text-embedding-3-small", input=texts)
    for chunk, item in zip(chunks, resp.data):
        chunk.embedding = item.embedding

    save_cache(source_hash, chunks)
    return chunks, False   # 새로 빌드


# ══════════════════════════════════════════════════════════════════════════════
# 5. 향상된 검색 — 메타데이터 사전필터 + 벡터 검색
# ══════════════════════════════════════════════════════════════════════════════

def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def metadata_filter(chunks: list[Chunk], state: dict) -> list[Chunk]:
    """
    사용자 상태 기반 하드 필터 (벡터 검색 전 범위 축소).
    필터 조건:
      - mood_tags  ∋ 사용자 기분
      - energy_tags ∋ 사용자 에너지
      - goal_tags  ∋ 사용자 목표
      - circadian: time_start ~ time_end 안에 현재 시간 포함
    최소 1개 조건이라도 맞으면 통과 (OR 방식으로 recall 보장).
    """
    mood   = state.get("기분", "")
    energy = state.get("에너지 상태", "")
    goal   = state.get("목표", "")
    hour   = int(state.get("현재 시간", "12:00").split(":")[0])

    def passes(c: Chunk) -> bool:
        mood_ok   = not c.mood_tags   or mood   in c.mood_tags
        energy_ok = not c.energy_tags or energy in c.energy_tags
        goal_ok   = not c.goal_tags   or goal   in c.goal_tags
        time_ok   = (c.time_start is None) or (c.time_start <= hour < c.time_end)
        # Circadian 청크는 시간 조건 우선
        if c.doc_type == DocType.CIRCADIAN:
            return time_ok
        return mood_ok or energy_ok or goal_ok

    filtered = [c for c in chunks if passes(c)]
    return filtered if filtered else chunks   # 필터 결과 없으면 전체 반환


def hybrid_retrieve(
    query_emb: list[float],
    chunks: list[Chunk],
    state: dict,
    k: int = 4,
) -> list[tuple[float, Chunk]]:
    """
    메타데이터 필터 → 코사인 유사도 정렬 → 상위 k개.
    doc_type 다양성 보장: 각 type에서 최소 1개 포함.
    """
    pool = metadata_filter(chunks, state)

    scored: list[tuple[float, Chunk]] = sorted(
        [(cosine_sim(query_emb, c.embedding), c) for c in pool],
        key=lambda x: x[0],
        reverse=True,
    )

    # doc_type 다양성 보장
    seen_types: set[DocType] = set()
    result: list[tuple[float, Chunk]] = []
    remainder: list[tuple[float, Chunk]] = []

    for score, chunk in scored:
        if chunk.doc_type not in seen_types:
            result.append((score, chunk))
            seen_types.add(chunk.doc_type)
        else:
            remainder.append((score, chunk))
        if len(result) >= min(k, len(DocType)):
            break

    # 나머지 슬롯은 점수 순으로 채우기
    for score, chunk in remainder:
        if len(result) >= k:
            break
        result.append((score, chunk))

    return sorted(result, key=lambda x: x[0], reverse=True)


def build_query(state: dict) -> str:
    mood_map = {
        "무기력": "무기력형 Low Arousal Low Motivation 도파민 저활성 행동 개시 어려움",
        "스트레스": "스트레스형 High Arousal Negative Emotion 편도체 과활성 코르티솔",
        "안정": "안정형 Moderate Arousal 세로토닌 균형 부교감신경 회복",
        "집중됨": "집중형 Optimal Arousal Flow 도파민 노르에피네프린 전전두엽",
    }
    goal_map = {
        "건강": "건강 중심형 Energy Maximizer 수면 운동 Self-Regulation",
        "생산성": "생산성 중심형 Time Optimizer Deep Work 집중",
        "돈": "돈 중심형 Capital Maximizer ROI 레버리지 도파민",
        "균형": "균형 중심형 Well-being Optimizer 삶의 만족 자율성",
    }
    energy_map = {
        "낮음": "에너지 저하 Low Energy 피로 회복 자율신경 불균형",
        "보통": "에너지 균형 Balanced Homeostasis 항상성",
        "높음": "에너지 활성 High Energy 도파민 보상 집중 선순환",
    }
    hour = int(state.get("현재 시간", "12:00").split(":")[0])
    if 5 <= hour < 10:
        time_ctx = "아침 코르티솔 각성 의사결정 집중"
    elif 10 <= hour < 13:
        time_ctx = "오전 고집중 인지 속도 논리 학습"
    elif 13 <= hour < 15:
        time_ctx = "점심후 Post-lunch Dip 졸음 회복 낮잠"
    elif 15 <= hour < 18:
        time_ctx = "오후 사회적 상호작용 감정 인식"
    elif 18 <= hour < 21:
        time_ctx = "저녁 운동 체온 최고점 근력"
    else:
        time_ctx = "밤 창의적 사고 수면 Glymphatic"

    return (
        f"{mood_map.get(state.get('기분',''), state.get('기분',''))} "
        f"{goal_map.get(state.get('목표',''), state.get('목표',''))} "
        f"{energy_map.get(state.get('에너지 상태',''), state.get('에너지 상태',''))} "
        f"{time_ctx} 상황:{state.get('현재 상황','')} 가용:{state.get('사용 가능 시간','')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. 전후 비교 데모
# ══════════════════════════════════════════════════════════════════════════════

def run_old_pipeline(client: OpenAI, raw_text: str, query_emb: list[float], k=4):
    """기존 decision_ai.py 방식 (flat chunking, 최소 메타데이터)."""
    tab_pattern = re.compile(
        r"탭\s*(?P<num>[1-4])\s*\n(?P<content>.*?)(?=탭\s*[1-4]|\Z)", re.DOTALL
    )
    sec_pattern = re.compile(
        r"(?m)^(?P<title>[1-4]\.\s+.+?(?:\(.+?\))?)\s*\n(?P<body>.*?)(?=^[1-4]\.|^탭\s*[1-4]|\Z)",
        re.DOTALL,
    )
    tab_labels = {"1": "각성-동기 상태 모델", "2": "개인 최적화 전략 유형",
                  "3": "에너지 상태 분류", "4": "Circadian"}
    old_chunks = []
    for tm in tab_pattern.finditer(raw_text):
        lab = tab_labels.get(tm.group("num"), "")
        for sm in sec_pattern.finditer(tm.group("content")):
            body = sm.group("body").strip()
            if len(body) < 30:
                continue
            old_chunks.append({
                "tab": lab,
                "title": sm.group("title").strip(),
                "body": body,
                "embed_text": f"[{lab}] {sm.group('title').strip()}\n{body}",
            })

    texts  = [c["embed_text"] for c in old_chunks]
    resp   = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embs   = [i.embedding for i in resp.data]

    scored = sorted(
        zip([cosine_sim(query_emb, e) for e in embs], old_chunks),
        key=lambda x: x[0], reverse=True,
    )
    return old_chunks, [(s, c) for s, c in scored[:k]]


def print_separator(char="─", width=70):
    print(char * width)


def demo(client: OpenAI, chunks: list[Chunk]):
    sample_state = {
        "현재 시간": "09:30",
        "에너지 상태": "낮음",
        "기분": "무기력",
        "현재 상황": "집",
        "목표": "생산성",
        "사용 가능 시간": "20분",
    }

    print_separator("═")
    print("  전후 비교 데모")
    print(f"  쿼리 상태: {sample_state}")
    print_separator("═")

    query_str = build_query(sample_state)
    query_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[query_str]
    ).data[0].embedding

    # ── BEFORE ──────────────────────────────────────────────────────────────
    print("\n【 BEFORE — 기존 파이프라인 (flat chunking) 】")
    print_separator()
    raw_text = "\n".join(
        p.extract_text() or "" for p in PdfReader(str(PDF_PATH)).pages
    )
    old_all, old_top = run_old_pipeline(client, raw_text, query_emb)

    print(f"  총 청크 수: {len(old_all)}")
    print(f"  메타데이터: tab, title, body, embed_text (4개 필드)")
    print(f"  검색 방식: 전체 벡터 검색 (필터 없음)")
    print(f"\n  상위 {len(old_top)}개 검색 결과:")
    for i, (score, c) in enumerate(old_top, 1):
        print(f"  [{i}] {score:.4f}  [{c['tab']}] {c['title']}")

    # ── AFTER ───────────────────────────────────────────────────────────────
    print("\n【 AFTER — 고도화 파이프라인 (typed + metadata + diverse) 】")
    print_separator()

    type_counts = {}
    for c in chunks:
        type_counts[c.doc_type.value] = type_counts.get(c.doc_type.value, 0) + 1

    total_fields = sum(len(asdict(c)) for c in chunks[:1])
    print(f"  총 청크 수: {len(chunks)}")
    print(f"  doc_type 분포: {type_counts}")
    print(f"  메타데이터 필드 수: {total_fields}개 (mood_tags, energy_tags, goal_tags, time_range, arousal_level, keywords, subsections 등)")
    print(f"  검색 방식: 메타데이터 사전필터 → 벡터 검색 → doc_type 다양성 보장")

    filtered = metadata_filter(chunks, sample_state)
    top = hybrid_retrieve(query_emb, chunks, sample_state, k=4)

    print(f"\n  메타데이터 필터 후 후보: {len(filtered)}개 (전체 {len(chunks)}개 중)")
    print(f"\n  상위 {len(top)}개 검색 결과:")
    for i, (score, c) in enumerate(top, 1):
        mood  = f"기분:{c.mood_tags}"    if c.mood_tags   else ""
        eng   = f"에너지:{c.energy_tags}" if c.energy_tags else ""
        goal  = f"목표:{c.goal_tags}"    if c.goal_tags   else ""
        print(f"  [{i}] {score:.4f}  [{c.doc_type.value}] {c.section_title}")
        print(f"       {mood} {eng} {goal}")

    # ── 요약 ────────────────────────────────────────────────────────────────
    print()
    print_separator("═")
    print("  개선 효과 요약")
    print_separator("─")
    improvements = [
        ("문서 종류 분리",    "단일 파서",      "4개 DocType별 전용 파서"),
        ("메타데이터",        "4개 필드",       f"{total_fields}개 필드 (mood/energy/goal/time/arousal)"),
        ("검색 방식",         "전체 벡터 검색", "메타데이터 필터 → 벡터 → 다양성 보장"),
        ("증분 인덱싱",       "없음",           "SHA-256 해시 캐시 (.index_cache/inf_index.json)"),
        ("embed_text 품질",   "제목+본문",      "타입태그+소제목+요약+키워드 결합"),
        ("검색 후보 풀",      f"{len(old_all)}개", f"{len(filtered)}개 (사전필터로 {100-int(len(filtered)/len(chunks)*100)}% 축소)"),
    ]
    for label, before, after in improvements:
        print(f"  {label:<16} │ Before: {before:<18} │ After: {after}")
    print_separator("═")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    force = "--rebuild" in sys.argv

    print("인덱싱 파이프라인 시작...")
    t0 = time.time()
    chunks, was_cached = build_index(client, force=force)
    elapsed = time.time() - t0

    status = "캐시 로드" if was_cached else "신규 빌드"
    print(f"  [{status}] {len(chunks)}개 청크 완료 ({elapsed:.1f}s)")
    if not was_cached:
        print(f"  인덱스 저장: {CACHE_FILE}")

    print()
    demo(client, chunks)


if __name__ == "__main__":
    main()
