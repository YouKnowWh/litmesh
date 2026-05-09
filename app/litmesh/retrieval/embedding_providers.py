"""
Embedding provider abstraction for LitMesh vector retrieval.

Supported providers:
  - openai_compatible: OpenAI / DeepSeek / Voyage / ModelArk / local vLLM
  - sentence_transformers: local HuggingFace model (no API call needed)
  - dummy: deterministic pseudo-embeddings for testing

Environment variables (role: EXTRACTION, DEFAULT — same as LLM config):
  LITMESH_{ROLE}_EMBED_PROVIDER  — embedding provider name
  LITMESH_{ROLE}_EMBED_MODEL     — embedding model name
  LITMESH_{ROLE}_EMBED_BASE_URL  — API base URL (for openai_compatible)
  LITMESH_{ROLE}_EMBED_API_KEY   — API key (for openai_compatible)

Global fallback:
  LITMESH_EMBED_PROVIDER / LITMESH_EMBED_MODEL / LITMESH_EMBED_BASE_URL / LITMESH_EMBED_API_KEY
"""

from __future__ import annotations

import os
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    provider_name: str = ""
    model: str = ""
    dimension: int = 768

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text, returning a float vector."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Override for batch API support."""
        return [self.embed(t) for t in texts]


# ---- OpenAI-compatible ----

class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """OpenAI / DeepSeek / Voyage / ModelArk / local vLLM embeddings API."""

    def __init__(self, base_url: str, api_key: str, model: str, dimension: int = 1536, timeout: int = 30):
        self.provider_name = "openai_compatible"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx
        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model, "input": texts}

        resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        embeddings = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            vec = item["embedding"]
            if len(vec) != self.dimension:
                self.dimension = len(vec)  # adapt
            embeddings.append(vec)
        return embeddings


# ---- Sentence Transformers (local) ----

class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Local HuggingFace sentence-transformers model. No API call needed.

    Install: pip install sentence-transformers
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", dimension: int = 384, device: str = "cpu"):
        self.provider_name = "sentence_transformers"
        self.model = model
        self.dimension = dimension
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model, device=self.device)
            # Detect actual dimension from model
            test_vec = self._model.encode("test")
            self.dimension = len(test_vec)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        return model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return model.encode(texts).tolist()


# ---- Dummy (testing) ----

class DummyEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-embeddings for testing. No external dependencies."""

    def __init__(self, dimension: int = 768):
        self.provider_name = "dummy"
        self.model = "dummy"
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [((h[i % len(h)] / 255.0) * 2 - 1) for i in range(self.dimension)]


# ---- Config & factory ----

@dataclass
class EmbeddingEndpoint:
    """Configuration for an embedding provider."""
    provider: str = "openai_compatible"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    dimension: int = 1536

    def create_provider(self) -> EmbeddingProvider:
        if self.provider == "sentence_transformers":
            return SentenceTransformersEmbeddingProvider(
                model=self.model or "all-MiniLM-L6-v2",
                dimension=self.dimension,
            )
        elif self.provider == "dummy":
            return DummyEmbeddingProvider(dimension=self.dimension)
        else:
            # openai_compatible (default)
            return OpenAICompatibleEmbeddingProvider(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                dimension=self.dimension,
            )


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def load_embedding_endpoint(role: str = "DEFAULT") -> EmbeddingEndpoint:
    """Load embedding endpoint config from environment variables.

    Args:
        role: Role name (EXTRACTION, DEFAULT, etc.). Falls back to global env vars.

    Priority per setting:
      1. LITMESH_{ROLE}_EMBED_{KEY}
      2. LITMESH_EMBED_{KEY}
      3. Sensible defaults (openai_compatible + infer from LLM config)
    """
    role = role.upper()
    role_prefix = f"LITMESH_{role}_EMBED_"

    provider = (
        _env(f"{role_prefix}PROVIDER")
        or _env("LITMESH_EMBED_PROVIDER")
        or _infer_embed_provider()
    )

    model = (
        _env(f"{role_prefix}MODEL")
        or _env("LITMESH_EMBED_MODEL")
        or _default_embed_model(provider)
    )

    base_url = (
        _env(f"{role_prefix}BASE_URL")
        or _env("LITMESH_EMBED_BASE_URL")
        or _default_embed_base_url(provider)
    )

    api_key = (
        _env(f"{role_prefix}API_KEY")
        or _env("LITMESH_EMBED_API_KEY")
        or _default_embed_api_key(provider)
    )

    dimension = int(
        _env(f"{role_prefix}DIMENSION")
        or _env("LITMESH_EMBED_DIMENSION")
        or str(_default_embed_dimension(provider))
    )

    return EmbeddingEndpoint(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        dimension=dimension,
    )


def _infer_embed_provider() -> str:
    """Infer embedding provider from available credentials."""
    if _env("ANTHROPIC_AUTH_TOKEN") or _env("LITMESH_LLM_PROVIDER") == "anthropic":
        pass  # Anthropic doesn't have embeddings yet, fall through
    if _env("OPENAI_API_KEY"):
        return "openai_compatible"
    # Default: DeepSeek or compatible
    return "openai_compatible"


def _default_embed_model(provider: str) -> str:
    if provider == "sentence_transformers":
        return "all-MiniLM-L6-v2"
    if provider == "dummy":
        return "dummy"
    # openai_compatible
    return _env("LITMESH_EMBED_MODEL", "text-embedding-3-small")


def _default_embed_base_url(provider: str) -> str:
    if provider == "sentence_transformers" or provider == "dummy":
        return ""
    # openai_compatible: reuse LLM base URL if possible
    return _env("OPENAI_BASE_URL", _env("ANTHROPIC_BASE_URL", "https://api.openai.com/v1"))


def _default_embed_api_key(provider: str) -> str:
    if provider == "sentence_transformers" or provider == "dummy":
        return ""
    return _env("OPENAI_API_KEY", _env("ANTHROPIC_AUTH_TOKEN", ""))


def _default_embed_dimension(provider: str) -> int:
    if provider == "sentence_transformers":
        return 384  # all-MiniLM-L6-v2
    if provider == "dummy":
        return 768
    return 1536  # text-embedding-3-small
