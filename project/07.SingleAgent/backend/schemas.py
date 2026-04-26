"""
schemas.py — Pydantic request/response models for the AX Compass Single Agent API
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class VerifyResponse(BaseModel):
    username: str
    valid: bool = True


# ── Chat ──────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


# ── Curriculum domain ─────────────────────────────────────────

class CurriculumModule(BaseModel):
    title: str
    duration_minutes: int
    objectives: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)


class Curriculum(BaseModel):
    title: str
    target_audience: str
    total_duration_minutes: int
    group_size: str
    learning_objectives: list[str]
    modules: list[CurriculumModule]
    assessment: str = ""
    notes: str = ""


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score: float = 0.0  # 0.0 ~ 1.0


# ── Agent response ────────────────────────────────────────────

class ChatResponse(BaseModel):
    reply: str
    complete: bool = False
    curriculum: dict[str, Any] | None = None
    validation_result: ValidationResult | None = None
    tool_calls_made: list[str] = Field(default_factory=list)


# ── Utility ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    rag_indexed: bool = False
