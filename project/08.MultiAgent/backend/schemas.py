from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Any

# ── 경로 설정 (절대경로 기반으로 CWD 독립) ─────────────────────
_ROOT = Path(__file__).parent.parent          # 08.MultiAgent/
_PROJECT = _ROOT.parent                        # project/

PDF_DIR    = Path(os.environ.get("BLOOM_PDF_DIR",    str(_PROJECT / "data")))
CACHE_FILE = Path(os.environ.get("BLOOM_CACHE_FILE", str(_ROOT / "data" / ".index_cache" / "ma_rag_index.json")))
DATA_FILE  = Path(os.environ.get("BLOOM_DATA_FILE",  str(_ROOT / "data" / ".mission_data.json")))
CURRICULA_DIR = Path(os.environ.get("BLOOM_CURRICULA_DIR", str(_ROOT / "data" / ".curricula")))
PROMPTS_DIR   = _ROOT / "prompts"

# ── 에이전트 설정 ──────────────────────────────────────────────
MAX_GENERATION_RETRIES = 3
CODE_VALIDATION_SCORE_THRESHOLD = 70

# ── 난이도 최소 소요 시간(분) ──────────────────────────────────
DIFFICULTY_MIN_MINUTES: dict[str, int] = {
    "하": 5, "중": 15, "상": 40, "최상": 90,
}

# ── 유효 카테고리 ──────────────────────────────────────────────
VALID_CATEGORIES = {"건강", "생산성", "재미", "성장", "돌발"}

# ── PDF 파일 목록 (project/data/ 실제 파일명 기준) ───────────────
PDF_FILES: dict[str, str] = {
    "behavioral_activation":  "Adolescent_Behavioral_Activation_Program.pdf",
    "behavioral_activation2": "Behavioral_Activation_Guided_Practice.pdf",
    "exercise_mental":        "Benefits_of_Exercise_for_Clinically_Depressed_Craft_Perna.pdf",
    "CBT":                    "Brief_Cognitive_Behavioral_Therapy_Guide.pdf",
    "mindfulness":            "Brief_Mindfulness_Intervention_Stress_Response.pdf",
    "emotion_regulation":     "Emotion_Regulation_Gross_Process_Model.pdf",
    "sleep_mental":           "Emotion_Regulation_and_Sleep_Vandekerckhove.pdf",
    "flow_theory":            "Investigating_the_Flow_Experience_Abuhamdeh.pdf",
    "motivational":           "Motivational_Interviewing_Helping_People_Change_Miller_Rollnick.pdf",
    "broaden_build":          "Positive_Emotions_Broaden_and_Build_Fredrickson.pdf",
    "positive_psychology":    "Positive_Psychology_Progress_Seligman.pdf",
    "self_determination":     "Self_Determination_Theory_Ryan_Deci.pdf",
    "social_support":         "Social_Isolation_Depression_Anxiety_Older_Adults.pdf",
    "self_compassion":        "The_Role_of_Self_Compassion_in_Psychotherapy.pdf",
}

# ── 감정-논문 가중치 부스트 ────────────────────────────────────
EMOTION_SOURCE_WEIGHT: dict[str, dict[str, float]] = {
    "스트레스": {"mindfulness": 1.5, "CBT": 1.3, "exercise_mental": 1.2},
    "우울":     {"behavioral_activation": 1.5, "positive_psychology": 1.3, "social_support": 1.2},
    "불안":     {"emotion_regulation": 1.5, "mindfulness": 1.3, "CBT": 1.2},
    "무기력":   {"behavioral_activation": 1.5, "broaden_build": 1.3, "flow_theory": 1.2},
    "피로":     {"sleep_mental": 1.5, "mindfulness": 1.3, "exercise_mental": 1.2},
    "분노":     {"self_compassion": 1.5, "mindfulness": 1.3, "CBT": 1.2},
    "긍정":     {"flow_theory": 1.5, "self_determination": 1.3, "broaden_build": 1.2},
    "중립":     {},
}

# ── API 모델 ───────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    available_minutes: int = 30

class ChatResponse(BaseModel):
    response: str
    session_id: str
    mission: dict | None = None
    curriculum_id: str | None = None
    agent_trace: list[dict] = []

# ── 내부 데이터 모델 ───────────────────────────────────────────
class EmotionAnalysis(BaseModel):
    emotion_type: str
    intensity: int
    triggers: list[str] = []
    needs: list[str] = []
    summary: str

class ResearchResult(BaseModel):
    key_findings: list[str] = []
    recommended_activities: list[str] = []
    psychological_basis: str
    expected_effect: str
    raw_chunks: list[dict] = []

class CodeValidationResult(BaseModel):
    has_required_tags: bool
    valid_category: bool
    valid_difficulty: bool
    time_feasible: bool
    passed: bool
    errors: list[str] = []

class LLMValidationResult(BaseModel):
    scores: dict[str, int]
    total_score: int
    is_valid: bool
    feedback: str = ""
    strengths: str = ""

class ValidationResult(BaseModel):
    code_result: CodeValidationResult
    llm_result: LLMValidationResult | None = None
    overall_valid: bool
    attempt: int

class CurriculumRecord(BaseModel):
    id: str
    username: str
    created_at: str
    emotion: EmotionAnalysis
    research: ResearchResult
    mission: dict
    validation: ValidationResult
    final_response: str
    available_minutes: int
