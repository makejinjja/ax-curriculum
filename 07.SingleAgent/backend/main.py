"""
main.py — FastAPI Single Agent 백엔드

엔드포인트:
  POST /auth/login     — JWT 토큰 발급
  GET  /auth/verify    — 토큰 검증 (Bearer)
  GET  /health         — 헬스체크 + RAG 상태
  POST /chat           — 에이전트 대화 (단일 엔드포인트)

- startup 시 RAG 인덱스를 빌드·캐시한다.
- 사용자별 JSON 파일로 fruits/cards/combo/mission_history를 영속화한다.
- 세션별 대화 이력을 인메모리로 관리한다 (최근 20턴 유지).
"""
from __future__ import annotations
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agent import SingleAgent
from auth import (
    authenticate_user,
    create_access_token,
    decode_token,
    get_openai_client,
    load_data,
    save_data,
)
from rag import build_index, summarize_mission
from schemas import (
    ChatRequest,
    ChatResponse,
    COMBO2_CARD,
    MAX_FRUITS,
    HealthResponse,
    LoginRequest,
    LoginResponse,
)

# ── 앱 수준 상태 ─────────────────────────────────────────────
_rag_state: dict = {"chunks": [], "embeddings": [], "bm25": None}
_agents:    dict[str, SingleAgent] = {}
_sessions:  dict[str, list[dict]] = {}   # session_id → 대화 이력
_openai_client = None


def _load_rag() -> None:
    """RAG 인덱스를 백그라운드 스레드에서 로드한다."""
    global _openai_client
    try:
        _openai_client = get_openai_client()
        chunks, embeddings, bm25 = build_index(_openai_client)
        _rag_state.update({"chunks": chunks, "embeddings": embeddings, "bm25": bm25})
        print(f"[RAG] 인덱스 로드 완료 — {len(chunks)}개 청크")
    except Exception as exc:
        print(f"[RAG] 인덱스 로드 실패 (빈 인덱스로 시작): {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 동기 build_index가 이벤트 루프를 블로킹하지 않도록 별도 스레드에서 실행
    threading.Thread(target=_load_rag, daemon=True).start()
    yield
    _agents.clear()
    _sessions.clear()


app = FastAPI(
    title="Bloom Single Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


# ── 인증 헬퍼 ────────────────────────────────────────────────

def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    try:
        return decode_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _get_agent(username: str) -> SingleAgent:
    if username not in _agents:
        _agents[username] = SingleAgent(
            client=get_openai_client(),
            rag_state=_rag_state,
        )
    return _agents[username]


# ── 미션 완료 시 데이터 업데이트 ────────────────────────────

def _update_on_completion(
    user_data: dict,
    last_mission: dict,
    client=None,
) -> str | None:
    """
    미션 성공 완료 시 fruits·cards·combo를 업데이트한다.
    콤보 카드 이름(있으면)을 반환한다.
    """
    cat  = last_mission.get("category", "")
    diff = last_mission.get("difficulty", "하")

    # 미션 요약명 (짧게)
    short_name = last_mission.get("mission", "")
    if client and short_name:
        try:
            short_name = summarize_mission(client, short_name)
        except Exception:
            pass

    # 콤보 계산
    combo_card_name: str | None = None
    if cat and cat != "돌발":
        if user_data.get("last_category") == cat:
            user_data["combo_count"] = user_data.get("combo_count", 0) + 1
        else:
            user_data["combo_count"] = 1
        user_data["last_category"] = cat

    combo = user_data.get("combo_count", 0)
    if cat and cat != "돌발" and combo >= 2:
        if combo >= 3:
            user_data["cards"].append({"card": "골드 카드", "difficulty": "최상"})
            combo_card_name = "골드 카드"
        else:
            card_name, card_diff = COMBO2_CARD.get(cat, ("씨앗 카드", "하"))
            user_data["cards"].append({"card": card_name, "difficulty": card_diff})
            combo_card_name = card_name

    # 열매 추가
    user_data["fruits"].append({
        "difficulty":   diff,
        "category":     cat,
        "mission":      short_name,
        "full_mission": last_mission.get("mission", ""),
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "success":      True,
    })

    # MAX_FRUITS 초과 시 오래된 것부터 제거
    if len(user_data["fruits"]) > MAX_FRUITS:
        user_data["fruits"] = user_data["fruits"][-MAX_FRUITS:]

    return combo_card_name


# ── 엔드포인트 ───────────────────────────────────────────────

@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    if not authenticate_user(req.username, req.password):
        raise HTTPException(status_code=401, detail="잘못된 사용자명 또는 비밀번호입니다.")
    return LoginResponse(access_token=create_access_token(req.username))


@app.get("/auth/verify")
def verify(username: str = Depends(get_current_user)):
    return {"valid": True, "username": username}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        rag_index_loaded=len(_rag_state.get("chunks", [])) > 0,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    username: str = Depends(get_current_user),
):
    session_id = req.session_id or str(uuid.uuid4())
    history    = _sessions.get(session_id, [])
    agent      = _get_agent(username)
    user_data  = load_data(username)

    result = agent.chat(
        user_message=req.message,
        available_minutes=req.available_minutes or 30,
        conversation_history=history,
        user_data=user_data,
    )

    user_data = result.get("user_data", user_data)

    # 새 미션이 생성된 경우 last_mission 보존
    if result.get("mission"):
        user_data["last_mission"] = result["mission"]

    # get_insight 호출 = 미션 완료 → fruits/cards/combo 업데이트
    combo_card: str | None = None
    if "get_insight" in result["tool_calls_made"] and user_data.get("last_mission"):
        combo_card = _update_on_completion(
            user_data,
            user_data.pop("last_mission"),
            client=_openai_client,
        )

    save_data(username, user_data)

    # 대화 이력 업데이트 (최근 20턴 유지)
    history.append({"role": "user",      "content": req.message})
    history.append({"role": "assistant", "content": result["response"]})
    _sessions[session_id] = history[-20:]

    # 프론트엔드에 내부 필드 제외한 user_data 반환
    public_data = {
        "fruits":        user_data.get("fruits", []),
        "cards":         user_data.get("cards", []),
        "combo_count":   user_data.get("combo_count", 0),
        "last_category": user_data.get("last_category"),
        "combo_card":    combo_card,
    }

    return ChatResponse(
        response=result["response"],
        mission=result.get("mission"),
        tool_calls_made=result.get("tool_calls_made", []),
        session_id=session_id,
        user_data=public_data,
    )
