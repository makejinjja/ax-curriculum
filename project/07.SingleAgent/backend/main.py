"""
main.py — FastAPI application for the AX Compass Single Agent

Endpoints:
  GET  /health          — liveness probe
  POST /auth/login      — username/password → JWT
  GET  /auth/verify     — Bearer token → user info
  POST /chat            — single agent conversation turn (JWT required)

Run locally:
  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Ensure backend/ is on sys.path when launched from outside the directory
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

for _env_path in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from openai import OpenAI

from schemas import (
    LoginRequest, TokenResponse, VerifyResponse,
    ChatRequest, ChatResponse, HealthResponse,
)
from auth import authenticate_user, create_access_token, decode_token
from agent import run_agent
import rag as rag_module

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(
    title="AX Compass Single Agent",
    description="ReAct agent for curriculum design with RAG + web search",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bearer = HTTPBearer(auto_error=False)


# ── Dependency helpers ────────────────────────────────────────

def _get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY not configured on server",
        )
    return OpenAI(api_key=key)


def _require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(creds.credentials)


# ── Routes ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    indexed = rag_module._index is not None and rag_module._index.size > 0
    return HealthResponse(rag_indexed=indexed)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(req: LoginRequest):
    if not authenticate_user(req.username, req.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": req.username})
    return TokenResponse(access_token=token)


@app.get("/auth/verify", response_model=VerifyResponse, tags=["Auth"])
def verify(user: dict = Depends(_require_auth)):
    return VerifyResponse(username=user.get("sub", ""))


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(
    req: ChatRequest,
    user: dict = Depends(_require_auth),
    client: OpenAI = Depends(_get_client),
):
    return run_agent(client, req.message, req.history)
