from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "bloom-multi-agent-secret-change-in-prod")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 24

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

_USERS: dict[str, str] = {
    "admin": os.environ.get("ADMIN_PASSWORD", "bloom1234"),
    "user":  os.environ.get("USER_PASSWORD",  "user1234"),
}


def authenticate_user(username: str, password: str) -> bool:
    return _USERS.get(username) == password


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise ValueError("Invalid token")
        return username
    except JWTError as e:
        raise ValueError(str(e)) from e


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)
