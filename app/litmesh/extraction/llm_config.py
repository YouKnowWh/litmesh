"""
Multi-model LLM configuration for LitMesh.

Each subsystem can use a different LLM endpoint:
  - extraction: claim/evidence/limitation/concept extraction (cheap, fast)
  - segment:    PDF page-window segmentation (cheap, fast, JSON-only)
  - review:     bridge detection, series detection, inbox review (medium)
  - compilation: PromptPacket compilation, knowledge queries (expensive, good)
  - default:    fallback for everything else

Inspired by KokoroMemo's per-subsystem LLM config pattern.

Environment variables for each role (role: EXTRACTION, SEGMENT, REVIEW, COMPILATION, REPAIR, DEFAULT):
  LITMESH_{ROLE}_PROVIDER   — provider name (anthropic, openai_compatible, gemini)
  LITMESH_{ROLE}_MODEL      — model name
  LITMESH_{ROLE}_BASE_URL   — API base URL
  LITMESH_{ROLE}_API_KEY    — API key

Top-level env vars (LITMESH_LLM_PROVIDER, ANTHROPIC_AUTH_TOKEN, etc.) are used
as fallback defaults for all roles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .providers import create_provider, BaseProvider


@dataclass
class LLMEndpoint:
    """Configuration for a single LLM endpoint."""

    provider: str = "openai_compatible"
    model: str = ""
    base_url: str = ""
    api_key: str = ""

    def create_client(self) -> "LLMClient":
        """Create an LLMClient from this endpoint config."""
        return LLMClient(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )


# ---- Config loading from environment ----

def _env(key: str, default: str = "") -> str:
    val = os.getenv(key, "").strip()
    return val if val else default


def _resolve_provider(role: str) -> str:
    """Resolve provider for a role, falling back to global defaults."""
    role_key = f"LITMESH_{role}_PROVIDER"
    if _env(role_key):
        return _env(role_key)
    if _env("LITMESH_LLM_PROVIDER"):
        return _env("LITMESH_LLM_PROVIDER")
    # Infer from available keys
    if _env("DEEPSEEK_API_KEY") or _env("OPENAI_API_KEY"):
        return "openai_compatible"
    if _env("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("GEMINI_API_KEY"):
        return "gemini"
    return "openai_compatible"


def _resolve_model(role: str, provider: str) -> str:
    role_key = f"LITMESH_{role}_MODEL"
    if _env(role_key):
        return _env(role_key)
    if provider == "anthropic":
        return _env("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    elif provider in ("openai", "openai_compatible"):
        if role == "SEGMENT":
            return _env("OPENAI_MODEL", "deepseek-chat")
        return _env("OPENAI_MODEL", "deepseek-chat")
    elif provider == "gemini":
        return _env("GEMINI_MODEL", "gemini-2.5-flash")
    return ""


def _resolve_base_url(role: str, provider: str) -> str:
    role_key = f"LITMESH_{role}_BASE_URL"
    if _env(role_key):
        return _env(role_key)
    if provider == "anthropic":
        return _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    elif provider in ("openai", "openai_compatible"):
        return _env("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    elif provider == "gemini":
        return _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    return ""


def _resolve_api_key(role: str, provider: str) -> str:
    role_key = f"LITMESH_{role}_API_KEY"
    if _env(role_key):
        return _env(role_key)
    if provider == "anthropic":
        return _env("ANTHROPIC_AUTH_TOKEN", "")
    elif provider in ("openai", "openai_compatible"):
        return _env("OPENAI_API_KEY", _env("DEEPSEEK_API_KEY", ""))
    elif provider == "gemini":
        return _env("GEMINI_API_KEY", "")
    return ""


def load_endpoint(role: str) -> LLMEndpoint:
    """Load LLM endpoint config for a given role from environment variables.

    Args:
        role: One of 'EXTRACTION', 'SEGMENT', 'REVIEW', 'COMPILATION', 'DEFAULT' (case-insensitive).

    Returns:
        LLMEndpoint with resolved provider/model/base_url/api_key.
    """
    role = role.upper()
    provider = _resolve_provider(role)
    model = _resolve_model(role, provider)
    base_url = _resolve_base_url(role, provider)
    api_key = _resolve_api_key(role, provider)
    return LLMEndpoint(provider=provider, model=model, base_url=base_url, api_key=api_key)


def load_all_endpoints() -> dict[str, LLMEndpoint]:
    """Load all role-specific LLM endpoints.

    Returns:
        Dict mapping role name (lowercase) to LLMEndpoint.
    """
    roles = ["EXTRACTION", "SEGMENT", "REVIEW", "COMPILATION", "REPAIR", "DEFAULT"]
    return {role.lower(): load_endpoint(role) for role in roles}


# ---- Multi-client container ----

class MultiLLMClient:
    """Holds role-specific LLMClient instances, created lazily from endpoints.

    Usage:
        configs = load_all_endpoints()
        clients = MultiLLMClient(configs)
        claim_text = clients.extraction.complete(prompt)
        packet = clients.compilation.complete(prompt)
    """

    def __init__(self, endpoints: Optional[dict[str, LLMEndpoint]] = None):
        if endpoints is None:
            endpoints = load_all_endpoints()
        self._endpoints = endpoints
        self._clients: dict[str, "LLMClient"] = {}

    def _get(self, role: str) -> "LLMClient":
        if role not in self._clients:
            endpoint = self._endpoints.get(role, self._endpoints.get("default"))
            if endpoint is None:
                endpoint = load_endpoint("DEFAULT")
            self._clients[role] = endpoint.create_client()
        return self._clients[role]

    @property
    def extraction(self) -> "LLMClient":
        return self._get("extraction")

    @property
    def review(self) -> "LLMClient":
        return self._get("review")

    @property
    def segment(self) -> "LLMClient":
        return self._get("segment")

    @property
    def compilation(self) -> "LLMClient":
        return self._get("compilation")

    @property
    def repair(self) -> "LLMClient":
        return self._get("repair")

    @property
    def default(self) -> "LLMClient":
        return self._get("default")

    def get(self, role: str) -> "LLMClient":
        """Get client for an arbitrary role name."""
        return self._get(role.lower())


# Need to import at bottom to avoid circular import
from .llm_client import LLMClient
