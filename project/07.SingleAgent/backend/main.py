"""
main.py — FastAPI Single Agent 백엔드

엔드포인트:
  POST /auth/login     — JWT 토큰 발급
  GET  /auth/verify    — 토큰 검증 (Bearer)
  GET  /health         — 헬스체크 + RAG 상태
  POST /chat           — 에이전트 대화 (단일 엔드포인트)

startup 시 RAG 인덱스를 빌드·캐시하고, 사용자별 SingleAgent 인스턴스를 캐싱한다.
세션별 대화 이력을 인메모리로 관리한다 (최근 20턴 유지).
"""
from __future__ import annotations
import uuid
from contextlib import asynccontextmanager
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
)
from rag import build_index
from schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
)

# ── 앱 수준 상태 ─────────────────────────────────────────────
_rag_state: dict = {"chunks": [], "embeddings": [], "bm25": None}
_agents:    dict[str, SingleAgent] = {}
_sessions:  dict[str, list[dict]] = {}  # session_id → 대화 이력


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = get_openai_client()
        chunks, embeddings, bm25 = build_index(client)
        _rag_state.update({"chunks": chunks, "embeddings": embeddings, "bm25": bm25})
        print(f"[RAG] 인덱스 로드 완료 — {len(chunks)}개 청크")
    except Exception as exc:
        print(f"[RAG] 인덱스 로드 실패 (빈 인덱스로 시작): {exc}")
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

    result = agent.chat(
        user_message=req.message,
        available_minutes=req.available_minutes or 30,
        conversation_history=history,
    )

    # 대화 이력 업데이트 (최근 20턴 유지)
    history.append({"role": "user",      "content": req.message})
    history.append({"role": "assistant", "content": result["response"]})
    _sessions[session_id] = history[-20:]

    return ChatResponse(
        response=result["response"],
        mission=result.get("mission"),
        tool_calls_made=result.get("tool_calls_made", []),
        session_id=session_id,
    )
