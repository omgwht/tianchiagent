from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-flash"


def load_env(path: Path = ENV_PATH) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_client() -> OpenAI:
    load_env()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(f"请先在 {ENV_PATH} 中设置 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
    )


def chat_once(prompt: str, model: str = DEFAULT_MODEL) -> str:
    client = build_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content or ""


def chat_json(messages: list[dict[str, str]], model: str = DEFAULT_MODEL) -> dict[str, Any]:
    client = build_client()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("模型未返回 JSON 对象")
    return parsed


if __name__ == "__main__":
    print(chat_once("你是谁？"))
