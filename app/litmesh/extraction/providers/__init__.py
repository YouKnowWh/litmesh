"""LLM Provider abstraction — unified interface for multiple backends.

All providers accept a simple text-completion interface:
    provider.complete(prompt, system, temperature) -> str

Internally they convert to the native API format.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Iterator

logger = logging.getLogger("litmesh.llm")


class BaseProvider(ABC):
    """Minimal interface: text in, text out."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "", temperature: float = 0.1,
                 max_tokens: int = 4096) -> str:
        ...

    def complete_json(self, prompt: str, system: str = "",
                       temperature: float = 0.1) -> dict:
        """Return parsed JSON from completion."""
        text = self.complete(prompt, system or "You output only valid JSON.", temperature)
        return _parse_json(text)


# ---- OpenAI-compatible (DeepSeek, Groq, vLLM, Ollama...) ----

class OpenAICompatibleProvider(BaseProvider):
    """Standard /chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1,
                 max_tokens: int = 4096) -> str:
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        t0 = time.monotonic()
        resp = httpx.post(url, json=body, headers=headers, timeout=180)
        dt = time.monotonic() - t0
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        logger.info("model=%s prompt_len=%d resp_len=%d time=%.1fs tokens_in=%s tokens_out=%s",
                    self.model, len(prompt), len(content), dt,
                    usage.get("prompt_tokens", "?"),
                    usage.get("completion_tokens", "?"))
        return content


# ---- Anthropic Messages API ----

class AnthropicProvider(BaseProvider):
    """Anthropic Claude Messages API."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1,
                 max_tokens: int = 4096) -> str:
        import anthropic as anthropic_sdk

        client = anthropic_sdk.Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a precise academic text analyst.",
            messages=[{"role": "user", "content": prompt}],
        )

        content = message.content
        if isinstance(content, list):
            return "".join(
                block.text if hasattr(block, "text") else str(block)
                for block in content
            )
        return str(content)


# ---- Gemini ----

class GeminiProvider(BaseProvider):
    """Google Gemini API."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1,
                 max_tokens: int = 4096) -> str:
        import httpx

        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        resp = httpx.post(url, json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        return ""


# ---- Factory ----

_PROVIDER_MAP = {
    "openai_compatible": OpenAICompatibleProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def create_provider(provider: str, base_url: str, api_key: str, model: str) -> BaseProvider:
    cls = _PROVIDER_MAP.get(provider)
    if not cls:
        raise ValueError(f"Unknown provider: {provider}. Supported: {list(_PROVIDER_MAP.keys())}")
    return cls(base_url=base_url, api_key=api_key, model=model)


# ---- JSON helpers ----

def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"_parse_error": True, "_raw": raw[:500]}
