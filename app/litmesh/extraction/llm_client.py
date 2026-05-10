"""
LLM client for LitMesh extraction tasks.

Delegates to the provider abstraction for multi-backend support.
Public interface unchanged: complete(prompt) -> str, complete_json(prompt) -> dict.
"""

import os
import json
from typing import Optional

from .providers import create_provider, BaseProvider


class LLMClient:
    """LLM client with configurable backend.

    Usage:
        client = LLMClient()                                          # env-vars
        client = LLMClient(provider="openai_compatible", base_url=..., api_key=..., model=...)
        client = LLMClient(provider="anthropic")                      # uses ANTHROPIC_BASE_URL etc.
        text = client.complete("Summarize this paper.")
    """

    def __init__(
        self,
        provider: str = "",
        model: str = "",
        base_url: str = "",
        api_key: str = "",
    ):
        def _getenv(*keys: str) -> str:
            for k in keys:
                v = os.getenv(k, "").strip()
                if v:
                    return v
            return ""

        provider = provider or _getenv("LITMESH_LLM_PROVIDER") or "openai_compatible"

        if provider == "anthropic":
            self.model = model or _getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest"
            self.base_url = base_url or _getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
            self.api_key = api_key or _getenv("ANTHROPIC_AUTH_TOKEN")
        elif provider in ("openai_compatible", "openai"):
            self.model = model or _getenv("OPENAI_MODEL") or "deepseek-chat"
            self.base_url = base_url or _getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
            self.api_key = api_key or _getenv("OPENAI_API_KEY", "DEEPSEEK_API_KEY")
        elif provider == "gemini":
            self.model = model or _getenv("GEMINI_MODEL") or "gemini-2.5-flash"
            self.base_url = base_url or _getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta"
            self.api_key = api_key or _getenv("GEMINI_API_KEY")
        else:
            self.model = model
            self.base_url = base_url
            self.api_key = api_key

        self._provider: BaseProvider = create_provider(
            provider=provider,
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
        )

    @property
    def provider_type(self) -> str:
        return self._provider.__class__.__name__

    def complete(
        self,
        prompt: str,
        system: str = "You are a precise academic text analyst.",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        return self._provider.complete(prompt, system, temperature, max_tokens)

    def complete_json(self, prompt: str, system: Optional[str] = None,
                       temperature: float = 0.1, max_tokens: int = 4096) -> dict:
        text = self.complete(prompt, system=system or "You output only valid JSON.",
                            temperature=temperature, max_tokens=max_tokens)
        return self._parse_json(text)

    @staticmethod
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
