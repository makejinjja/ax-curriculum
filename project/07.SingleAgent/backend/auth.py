"""
auth.py — JWT 인증 모듈

- JWT 발급 / 검증 (python-jose)
- bcrypt 비밀번호 해싱 (passlib)
- 데모용 인메모리 사용자 DB (환경변수로 비밀번호 오버라이드 가능)
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from jose import jwt, JWTError
from openai import OpenAI
from passlib.context import CryptContext

# ── .env 로드 ────────────────────────────────────────────────
_ENV_PATHS = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent / "p1" / ".env",
]
for _p in _ENV_PATHS:
    if _p.exists():
        load_dotenv(_p)
        break

# ── JWT 설정 ─────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "bloom-single-agent-secret-change-in-prod")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24  # 24시간

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── 사용자 DB (데모용) ───────────────────────────────────────
_USERS: dict[str, str] = {
    "admin": pwd_context.hash(os.environ.get("ADMIN_PASSWORD", "bloom1234")),
    "user":  pwd_context.hash(os.environ.get("USER_PASSWORD",  "user1234")),
}


def authenticate_user(username: str, password: str) -> bool:
    hashed = _USERS.get(username)
    if not hashed:
        return False
    return pwd_context.verify(password, hashed)


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """Returns username or raises ValueError."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise ValueError("토큰 페이로드가 유효하지 않습니다.")
        return username
    except JWTError as exc:
        raise ValueError(f"토큰 검증 실패: {exc}") from exc


def get_openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "  • 로컬: .env 파일에 OPENAI_API_KEY=sk-... 추가\n"
            "  • Docker: env_file 또는 -e OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=key)
