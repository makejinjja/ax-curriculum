"""
schemas.py — 상수, 경로, 에이전트 설정, Pydantic API 모델
05_Advanced_RAG / bloom_v10 기반 전체 상수 이식
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel

# ── 경로 설정 ────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent  # 08.MultiAgent 루트

PDF_DIR    = Path(os.environ.get("BLOOM_PDF_DIR",    str(_BASE.parent / "data")))
CACHE_FILE = Path(os.environ.get("BLOOM_CACHE_FILE", str(_BASE / "data" / ".index_cache" / "ma_rag_index.json")))
DATA_DIR   = Path(os.environ.get("BLOOM_DATA_DIR",   str(_BASE / "data")))

# ── 에이전트 / RAG 설정 ──────────────────────────────────────
MAX_AGENT_ITERATIONS   = 10
MAX_GENERATION_RETRIES = 3
MAX_CHUNK_CHARS        = 600
CHUNK_OVERLAP          = 100
MAX_FRUITS             = 30

# ── 논문 목록 ────────────────────────────────────────────────
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

# ── 난이도 테이블 ────────────────────────────────────────────
DIFF: dict[str, dict] = {
    "하":   {"sym": "🌱", "color": "#e74c3c", "card": "씨앗 카드",     "label_en": "Easy"},
    "중":   {"sym": "🌿", "color": "#8B4513", "card": "새싹 카드",     "label_en": "Medium"},
    "상":   {"sym": "🌟", "color": "#bdc3c7", "card": "햇살 카드",     "label_en": "Hard"},
    "최상": {"sym": "🏆", "color": "#f1c40f", "card": "황금 열매 카드", "label_en": "Expert"},
    "돌발": {"sym": "⚡", "color": "#f39c12", "card": "번개 카드",     "label_en": "Wildcard"},
    "도전": {"sym": "🎯", "color": "#3498db", "card": "도전 카드",     "label_en": "Challenge"},
}

# ── 카테고리 테이블 ──────────────────────────────────────────
CAT: dict[str, dict] = {
    "건강":   {"sym": "💪", "color": "#e74c3c", "label": "건강"},
    "생산성": {"sym": "📋", "color": "#3498db", "label": "생산성"},
    "재미":   {"sym": "🎮", "color": "#9b59b6", "label": "재미"},
    "성장":   {"sym": "📚", "color": "#27ae60", "label": "성장"},
    "돌발":   {"sym": "⚡", "color": "#f39c12", "label": "돌발"},
}

# ── 2연속 콤보 카드 ───────────────────────────────────────────
COMBO2_CARD: dict[str, tuple[str, str]] = {
    "건강":   ("회복 카드",  "중"),
    "성장":   ("지식 카드",  "중"),
    "재미":   ("행운 카드",  "중"),
    "생산성": ("집중 카드",  "중"),
}

# ── 감정 분류 ────────────────────────────────────────────────
EMOTION_PURPOSE: dict[str, str] = {
    "부정적": "기분전환·회복",
    "중립":   "생산성·성장",
    "긍정적": "도전·재미·확장",
    "집중됨": "딥워크·몰입",
    "지루함": "자극·탐험",
}

EMOTION_EMOJI: dict[str, str] = {
    "부정적": "😔",
    "중립":   "😐",
    "긍정적": "😊",
    "집중됨": "🎯",
    "지루함": "😑",
}

# ── 감정 → 논문 가중치 ───────────────────────────────────────
EMOTION_SOURCE_WEIGHT: dict[str, dict[str, float]] = {
    "부정적": {"CBT": 2.0, "행동 활성화": 1.5, "자기연민": 1.8, "동기면담": 1.3, "수면·감정조절": 1.2},
    "중립":   {"행동 활성화": 1.5, "마음챙김": 1.2, "자기결정이론": 1.5, "긍정 심리학": 1.2},
    "긍정적": {"긍정 정서 확장": 2.0, "긍정 심리학": 1.8, "Flow 이론": 1.5, "자기결정이론": 1.3},
    "집중됨": {"Flow 이론": 2.5, "자기결정이론": 1.5, "마음챙김": 1.2},
    "지루함": {"행동 활성화": 2.0, "긍정 심리학": 1.5, "Flow 이론": 1.3, "사회적 고립·우울": 1.2},
}

# ── 카테고리별 주 논문 (×2.0 부스트) ────────────────────────
CAT_PRIMARY_SOURCE: dict[str, str | None] = {
    "건강":   "운동·정신건강",
    "생산성": "Flow 이론",
    "재미":   "긍정 정서 확장",
    "성장":   "자기결정이론",
    "돌발":   None,
}

# ── 감정+카테고리 조합 쿼리 ──────────────────────────────────
EMOTION_CAT_QUERY: dict[tuple[str, str], str] = {
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

# ── 기분 키워드 → 검색 쿼리 ──────────────────────────────────
MOOD_QUERY_MAP: dict[str, str] = {
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

# ── 난이도 / 카테고리 허용값 ──────────────────────────────────
VALID_DIFFICULTIES = {"하", "중", "상", "최상"}
VALID_CATEGORIES   = {"건강", "생산성", "재미", "성장", "돌발"}

DIFFICULTY_MIN_MINUTES: dict[str, int] = {
    "하":   5,
    "중":   15,
    "상":   40,
    "최상": 90,
}

# ── LLM 프롬프트 ─────────────────────────────────────────────
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


def make_mission_prompt(
    emotion_type: str,
    purpose: str,
    recent_missions: str = "없음",
    forced_difficulty: str = "하",
) -> str:
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
- [심리학 논문 근거]의 내용을 실제로 반영해 미션을 생성할 것
- [카테고리]는 미션 내용을 보고 건강/생산성/재미/성장 중 하나를 선택
- [근거]에는 반드시 논문 컨텍스트에서 확인한 구체적 심리학 이론명·기법명을 명시한다
- [효과]에는 논문에서 언급한 구체적 결과나 수치 표현을 포함한다

반드시 아래 형식으로만 출력:

[미션]
- (구체적 행동 1개, 10~30자 이내)

[카테고리]
- 건강 또는 생산성 또는 재미 또는 성장

[난이도]
- {forced_difficulty}

[근거]
- (위 [심리학 논문 근거]에서 확인한 이론명·기법명을 반드시 포함해 2줄 이내로 서술)

[효과]
- (논문 근거에 기반한 구체적 심리·신체 효과, 이론 연구에서 확인된 결과를 포함해 1~2줄)"""


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
    user_data: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    rag_index_loaded: bool
    version: str = "1.0.0"
