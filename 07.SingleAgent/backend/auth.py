"""
auth.py — JWT 인증 모듈

- JWT 발급 / 검증 (python-jose, HS256)
- bcrypt 비밀번호 해싱 (passlib)
- 사용자 계정: BLOOM_USERNAME / BLOOM_PASSWORD 환경변수로 오버라이드
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from jose import jwt, JWTError
from openai import OpenAI
from passlib.context import CryptContext

from schemas import DATA_DIR

# ── .env 로드 ────────────────────────────────────────────────
for _p in [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent / "p1" / ".env",
]:
    if _p.exists():
        load_dotenv(_p)
        break

# ── JWT 설정 ─────────────────────────────────────────────────
SECRET_KEY           = os.environ.get("JWT_SECRET_KEY", "bloom-single-agent-secret-change-in-prod")
ALGORITHM            = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24  # 24시간

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── 사용자 DB ────────────────────────────────────────────────
_BLOOM_USERNAME = os.environ.get("BLOOM_USERNAME", "bloom")
_BLOOM_PASSWORD = os.environ.get("BLOOM_PASSWORD", "bloom1234")

_USERS: dict[str, str] = {
    _BLOOM_USERNAME: pwd_context.hash(_BLOOM_PASSWORD),
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


# ── 사용자별 데이터 영속성 ────────────────────────────────────

def load_data(username: str) -> dict:
    data_file = DATA_DIR / f"{username}.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    if data_file.exists():
        d = json.loads(data_file.read_text(encoding="utf-8"))
        d.setdefault("last_category",    None)
        d.setdefault("combo_count",      0)
        d.setdefault("mission_history",  [])
        d.setdefault("weak_paper_boost", [])
        d.setdefault("last_mission",     None)
        return d
    return {
        "fruits":           [],
        "cards":            [],
        "last_category":    None,
        "combo_count":      0,
        "mission_history":  [],
        "weak_paper_boost": [],
        "last_mission":     None,
    }


def save_data(username: str, data: dict) -> None:
    data_file = DATA_DIR / f"{username}.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
