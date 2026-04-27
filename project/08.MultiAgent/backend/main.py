from __future__ import annotations
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth import authenticate_user, create_access_token, decode_token, get_openai_client
from orchestrator import Orchestrator
from rag import build_index
from schemas import (
    ChatRequest, ChatResponse, LoginRequest,
    CURRICULA_DIR,
)

# ── RAG 전역 상태 ──────────────────────────────────────────────
_rag_state: dict[str, Any] = {
    "chunks":     [],
    "embeddings": None,
    "bm25":       None,
    "loaded":     False,
}

# ── 오케스트레이터 캐시 ────────────────────────────────────────
_orchestrators: dict[str, Orchestrator] = {}

# ── 세션 히스토리 (멀티에이전트는 history 불필요하지만 UX용 유지) ─
_sessions: dict[str, list[dict]] = {}
MAX_SESSION_TURNS = 20


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = get_openai_client()
        chunks, emb, bm25 = build_index(
            client,
            force_rebuild=False,
            progress_callback=lambda m: print(m),
        )
        _rag_state.update({"chunks": chunks, "embeddings": emb, "bm25": bm25, "loaded": True})
    except Exception as e:
        print(f"[WARN] RAG 인덱스 빌드 실패: {e}")
    yield


app = FastAPI(title="Bloom MultiAgent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


def _get_username(creds: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        return decode_token(creds.credentials)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def _get_orchestrator(username: str) -> Orchestrator:
    if username not in _orchestrators:
        _orchestrators[username] = Orchestrator(get_openai_client())
    return _orchestrators[username]


# ── 인증 엔드포인트 ────────────────────────────────────────────
@app.post("/auth/login")
def login(req: LoginRequest):
    if not authenticate_user(req.username, req.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증 실패")
    return {"access_token": create_access_token(req.username), "token_type": "bearer"}


@app.get("/auth/verify")
def verify(username: str = Depends(_get_username)):
    return {"username": username, "valid": True}


# ── 헬스체크 ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_index_loaded": _rag_state["loaded"],
        "chunks": len(_rag_state["chunks"]),
    }


# ── 채팅 엔드포인트 ────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, username: str = Depends(_get_username)):
    if not _rag_state["loaded"]:
        raise HTTPException(status_code=503, detail="RAG 인덱스 로딩 중입니다.")

    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = []

    orchestrator = _get_orchestrator(username)
    result = orchestrator.run(
        user_message=req.message,
        available_minutes=req.available_minutes,
        chunks=_rag_state["chunks"],
        embeddings=_rag_state["embeddings"],
        bm25=_rag_state["bm25"],
        username=username,
    )

    # 세션 히스토리 (최대 MAX_SESSION_TURNS 쌍)
    history = _sessions[session_id]
    history.append({"role": "user",  "content": req.message})
    history.append({"role": "agent", "content": result["response"]})
    if len(history) > MAX_SESSION_TURNS * 2:
        _sessions[session_id] = history[-(MAX_SESSION_TURNS * 2):]

    return ChatResponse(
        response=result["response"],
        session_id=session_id,
        mission=result.get("mission"),
        curriculum_id=result.get("curriculum_id"),
        agent_trace=result.get("agent_trace", []),
    )


# ── 커리큘럼 엔드포인트 ────────────────────────────────────────
@app.get("/curriculum")
def list_curricula(username: str = Depends(_get_username)):
    CURRICULA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CURRICULA_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    records = []
    for f in files[:50]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("username") == username:
                records.append({
                    "id":         data["id"],
                    "created_at": data["created_at"],
                    "emotion":    data["emotion"]["emotion_type"],
                    "mission":    data["mission"]["mission"],
                    "category":   data["mission"]["category"],
                    "difficulty": data["mission"]["difficulty"],
                    "score":      (data.get("validation") or {}).get("llm_result", {}) and
                                  data["validation"]["llm_result"].get("total_score"),
                })
        except Exception:
            continue
    return {"curricula": records}


@app.get("/curriculum/{curriculum_id}")
def get_curriculum(curriculum_id: str, username: str = Depends(_get_username)):
    path = CURRICULA_DIR / f"{curriculum_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="커리큘럼을 찾을 수 없습니다.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("username") != username:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")
    return data
