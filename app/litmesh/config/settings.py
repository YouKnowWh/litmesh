"""
Settings manager: load from env vars, overlay from data/settings.json,
provide public view with masked API keys, save updates with deep-merge.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Optional

SETTINGS_PATH = Path("data/settings.json")

# Roles in the LLM subsystem
LLM_ROLES = ["extraction", "segment", "review", "compilation", "default"]


def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return key[:3] + "***" + key[-3:]


def _env_or(key: str, default: str = "") -> str:
    val = os.getenv(key, "").strip()
    return val if val else default


def _load_llm_defaults() -> dict:
    """Load LLM config from env vars as defaults."""
    provider = _env_or("LITMESH_LLM_PROVIDER") or "openai_compatible"
    deepseek_key = _env_or("DEEPSEEK_API_KEY")
    openai_key = _env_or("OPENAI_API_KEY")
    api_key = openai_key or deepseek_key or _env_or("ANTHROPIC_AUTH_TOKEN") or _env_or("GEMINI_API_KEY")
    model = _env_or("OPENAI_MODEL", "deepseek-chat")
    base_url = _env_or("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

    roles = {}
    for role in LLM_ROLES:
        r = role.upper()
        roles[role] = {
            "provider": _env_or(f"LITMESH_{r}_PROVIDER") or provider,
            "model": _env_or(f"LITMESH_{r}_MODEL") or model,
            "base_url": _env_or(f"LITMESH_{r}_BASE_URL") or base_url,
            "api_key": _env_or(f"LITMESH_{r}_API_KEY") or api_key,
        }
    return roles


def _load_embedding_defaults() -> dict:
    """Load embedding config from env vars."""
    return {
        "provider": _env_or("LITMESH_EMBED_PROVIDER", "openai_compatible"),
        "model": _env_or("LITMESH_EMBED_MODEL", "BAAI/bge-large-zh-v1.5"),
        "base_url": _env_or("LITMESH_EMBED_BASE_URL", "https://api.siliconflow.cn/v1"),
        "api_key": _env_or("LITMESH_EMBED_API_KEY", ""),
        "dimension": int(_env_or("LITMESH_EMBED_DIMENSION", "1024")),
    }


def _load_parser_defaults() -> dict:
    return {
        "mineru_api_url": _env_or("LITMESH_MINERU_API_URL", ""),
        "mineru_api_timeout": int(_env_or("LITMESH_MINERU_API_TIMEOUT", "1800")),
        "mineru_api_backend": _env_or("LITMESH_MINERU_API_BACKEND", "pipeline"),
        "mineru_api_return_md": _env_or("LITMESH_MINERU_API_RETURN_MD", "true"),
    }


def _load_segment_defaults() -> dict:
    return {
        "max_concurrency": int(_env_or("LITMESH_SEGMENT_CONCURRENCY", "4")),
        "segment_max_tokens": int(_env_or("LITMESH_SEGMENT_MAX_TOKENS", "2000")),
        "window_size": int(_env_or("LITMESH_SEGMENT_WINDOW_SIZE", "4")),
        "overlap_pages": int(_env_or("LITMESH_SEGMENT_OVERLAP", "2")),
    }


class SettingsManager:
    """Load, save, and reload LitMesh settings."""

    def __init__(self):
        self._settings: dict = {}
        self.load()

    def load(self) -> dict:
        """Load defaults from env vars, overlay from settings.json."""
        self._settings = {
            "llm": _load_llm_defaults(),
            "embedding": _load_embedding_defaults(),
            "parser": _load_parser_defaults(),
            "segment": _load_segment_defaults(),
        }
        if SETTINGS_PATH.exists():
            try:
                overlay = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                _deep_merge(self._settings, overlay)
            except Exception:
                pass
        return self._settings

    def get_public(self) -> dict:
        """Return settings with API keys masked."""
        public = copy.deepcopy(self._settings)
        for role in LLM_ROLES:
            key = public["llm"][role].get("api_key", "")
            public["llm"][role]["api_key_set"] = bool(key)
            public["llm"][role]["api_key_masked"] = _mask_key(key) if key else ""
            public["llm"][role]["api_key"] = ""  # never send raw key
        ek = public["embedding"].get("api_key", "")
        public["embedding"]["api_key_set"] = bool(ek)
        public["embedding"]["api_key_masked"] = _mask_key(ek) if ek else ""
        public["embedding"]["api_key"] = ""
        return public

    def get_raw(self, role: str) -> dict:
        """Get raw LLM endpoint for a role."""
        return self._settings.get("llm", {}).get(role, {})

    def get_raw_embedding(self) -> dict:
        return self._settings.get("embedding", {})

    def get_raw_parser(self) -> dict:
        return self._settings.get("parser", {})

    def get_raw_segment(self) -> dict:
        return self._settings.get("segment", {})

    def save(self, updates: dict) -> bool:
        """Deep-merge updates into settings, persist to settings.json."""
        _deep_merge(self._settings, updates)
        # Also write back to env vars that can be set
        self._apply_llm_to_env()
        self._apply_embedding_to_env()
        self._apply_parser_to_env()
        self._apply_segment_to_env()
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self._settings, indent=2, ensure_ascii=False), encoding="utf-8")
        return True

    def _apply_llm_to_env(self):
        """Write LLM settings back to environment so new clients pick them up."""
        llm = self._settings.get("llm", {})
        for role in LLM_ROLES:
            cfg = llm.get(role, {})
            r = role.upper()
            if cfg.get("provider"):
                os.environ[f"LITMESH_{r}_PROVIDER"] = cfg["provider"]
            if cfg.get("model"):
                os.environ[f"LITMESH_{r}_MODEL"] = cfg["model"]
            if cfg.get("base_url"):
                os.environ[f"LITMESH_{r}_BASE_URL"] = cfg["base_url"]
            if cfg.get("api_key"):
                os.environ[f"LITMESH_{r}_API_KEY"] = cfg["api_key"]

    def _apply_embedding_to_env(self):
        emb = self._settings.get("embedding", {})
        for k in ("provider", "model", "base_url", "api_key"):
            if emb.get(k):
                os.environ[f"LITMESH_EMBED_{k.upper()}"] = str(emb[k])
        if emb.get("dimension"):
            os.environ["LITMESH_EMBED_DIMENSION"] = str(emb["dimension"])

    def _apply_parser_to_env(self):
        p = self._settings.get("parser", {})
        if p.get("mineru_api_url"):
            os.environ["LITMESH_MINERU_API_URL"] = p["mineru_api_url"]
        if p.get("mineru_api_timeout"):
            os.environ["LITMESH_MINERU_API_TIMEOUT"] = str(p["mineru_api_timeout"])
        if p.get("mineru_api_backend"):
            os.environ["LITMESH_MINERU_API_BACKEND"] = p["mineru_api_backend"]
        if p.get("mineru_api_return_md"):
            os.environ["LITMESH_MINERU_API_RETURN_MD"] = p["mineru_api_return_md"]

    def _apply_segment_to_env(self):
        s = self._settings.get("segment", {})
        if s.get("max_concurrency"):
            os.environ["LITMESH_SEGMENT_CONCURRENCY"] = str(s["max_concurrency"])
        if s.get("segment_max_tokens"):
            os.environ["LITMESH_SEGMENT_MAX_TOKENS"] = str(s["segment_max_tokens"])
        if s.get("window_size"):
            os.environ["LITMESH_SEGMENT_WINDOW_SIZE"] = str(s["window_size"])
        if s.get("overlap_pages"):
            os.environ["LITMESH_SEGMENT_OVERLAP"] = str(s["overlap_pages"])

    def reload_llm_clients(self):
        """Rebuild MultiLLMClient and embedding provider from current settings."""
        from ..extraction.llm_config import load_all_endpoints
        from ..extraction.llm_config import MultiLLMClient
        endpoints = load_all_endpoints()
        return MultiLLMClient(endpoints)


def _deep_merge(base: dict, overlay: dict):
    """Merge overlay into base in-place, recursively."""
    for key, value in overlay.items():
        if key not in base:
            base[key] = value
        elif isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif value is not None and value != "":
            base[key] = value
