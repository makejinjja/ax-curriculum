from __future__ import annotations
from pathlib import Path

from schemas import PROMPTS_DIR


class BaseAgent:
    """외부 프롬프트 파일을 로드하고 OpenAI chat completion을 호출하는 기반 클래스."""

    prompt_filename: str = ""
    model: str = "gpt-4o-mini"

    def __init__(self, client, prompt_override: str | None = None):
        self.client = client
        if prompt_override:
            self.system_prompt = prompt_override
        else:
            prompt_path = PROMPTS_DIR / self.prompt_filename
            if not prompt_path.exists():
                raise FileNotFoundError(f"프롬프트 파일 없음: {prompt_path}")
            self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    def _chat(
        self,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: dict | None = None,
    ) -> str:
        kwargs: dict = {
            "model":       self.model,
            "messages":    [
                {"role": "system",  "content": self.system_prompt},
                {"role": "user",    "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
