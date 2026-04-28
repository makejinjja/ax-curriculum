"""
05_3.Auth.py — 인증 및 데이터 영속성

- OpenAI 클라이언트 팩토리 (API 키 검증 포함)
- 유저 데이터 JSON 로드 / 저장
- .env 자동 로드 (로컬 개발 시)
"""
from __future__ import annotations
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from schemas import DATA_FILE  # type: ignore[import]

# ── .env 로드 ────────────────────────────────────────────────
# Docker 환경에서는 환경변수가 직접 주입되므로 .env 파일이 없어도 된다.
_ENV_PATHS = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / "p1" / ".env",
]
for _p in _ENV_PATHS:
    if _p.exists():
        load_dotenv(_p)
        break


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "  • 로컬: .env 파일에 OPENAI_API_KEY=sk-... 추가\n"
            "  • Docker: -e OPENAI_API_KEY=sk-... 또는 docker-compose .env"
        )
    return key


def get_client() -> OpenAI:
    return OpenAI(api_key=get_api_key())


# ── 데이터 영속성 ────────────────────────────────────────────

def load_data() -> dict:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        d = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        d.setdefault("last_category",   None)
        d.setdefault("combo_count",     0)
        d.setdefault("mission_history", [])
        d.setdefault("weak_paper_boost", [])
        return d
    return {
        "fruits":           [],
        "cards":            [],
        "last_category":    None,
        "combo_count":      0,
        "mission_history":  [],
        "weak_paper_boost": [],
    }


def save_data(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
