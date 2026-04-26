"""
auth.py — JWT authentication helpers
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "ax-compass-dev-secret-change-in-prod")
ALGORITHM = "HS256"
EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Demo user store — replace with a real database for production
_USERS: dict[str, str] = {
    os.environ.get("DEMO_USER", "admin"): _pwd.hash(
        os.environ.get("DEMO_PASSWORD", "password")
    ),
}


def authenticate_user(username: str, password: str) -> bool:
    hashed = _USERS.get(username)
    return bool(hashed and _pwd.verify(password, hashed))


def create_access_token(data: dict[str, Any]) -> str:
    payload = {
        **data,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
