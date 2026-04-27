"""
schemas.py — 상수, 경로, 에이전트 설정, Pydantic API 모델
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel

# ── 경로 설정 ────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent  # 07.SingleAgent 루트

PDF_DIR    = Path(os.environ.get("BLOOM_PDF_DIR",    str(_BASE.parent / "data")))
CACHE_FILE = Path(os.environ.get("BLOOM_CACHE_FILE", str(_BASE / "data" / ".index_cache" / "sa_rag_index.json")))
DATA_FILE  = Path(os.environ.get("BLOOM_DATA_FILE",  str(_BASE / "data" / ".mission_data.json")))

# ── 에이전트 설정 ────────────────────────────────────────────
MAX_AGENT_ITERATIONS   = 10
MAX_GENERATION_RETRIES = 3
MAX_CHUNK_CHARS        = 600
CHUNK_OVERLAP          = 100

# ── 논문 목록 (05_Advanced_RAG와 동일) ──────────────────────
PDF_FILES: list[tuple[str, str]] = [
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

# ── 감정 → 논문 가중치 ───────────────────────────────────────
EMOTION_SOURCE_WEIGHT: dict[str, dict[str, float]] = {
    "부정적": {
        "CBT":        2.0,
        "행동 활성화": 1.5,
        "자기연민":   1.8,
        "동기면담":   1.3,
        "수면·감정조절": 1.2,
    },
    "중립": {
        "행동 활성화":  1.5,
        "마음챙김":    1.2,
        "자기결정이론": 1.5,
        "긍정 심리학":  1.2,
    },
    "긍정적": {
        "긍정 정서 확장": 2.0,
        "긍정 심리학":   1.8,
        "Flow 이론":    1.5,
        "자기결정이론":  1.3,
    },
    "집중됨": {
        "Flow 이론":    2.5,
        "자기결정이론":  1.5,
        "마음챙김":    1.2,
    },
    "지루함": {
        "행동 활성화":   2.0,
        "긍정 심리학":   1.5,
        "Flow 이론":    1.3,
        "사회적 고립·우울": 1.2,
    },
}

# ── 난이도 / 카테고리 허용값 ──────────────────────────────────
VALID_DIFFICULTIES = {"하", "중", "상", "최상"}
VALID_CATEGORIES   = {"건강", "생산성", "재미", "성장", "돌발"}

# 난이도별 최소 필요 시간 (분) — 이 값보다 available_minutes가 작으면 검증 실패
DIFFICULTY_MIN_MINUTES: dict[str, int] = {
    "하":   5,
    "중":   15,
    "상":   40,
    "최상": 90,
}

# ── API Pydantic 모델 ────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    available_minutes: Optional[int] = 30


class ChatResponse(BaseModel):
    response: str
    mission: Optional[dict] = None
    tool_calls_made: List[str] = []
    session_id: str


class HealthResponse(BaseModel):
    status: str
    rag_index_loaded: bool
    version: str = "1.0.0"
