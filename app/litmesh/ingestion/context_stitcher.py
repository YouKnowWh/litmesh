"""
LLM-based context stitcher for merging adjacent text blocks.

Uses a small/fast model (flash) to decide whether two blocks
are part of the same paragraph, with no thinking/reasoning overhead.
"""

import os
import re

from ..extraction.llm_client import LLMClient


_STITCH_PROMPT = """下面两个文本块是否属于同一段落？只回答是或否。

块A: {block_a}

块B: {block_b}"""


class ContextStitcher:
    """Uses a fast LLM to stitch broken paragraphs."""

    def __init__(self, model: str = "", base_url: str = "", api_key: str = ""):
        self.model = model or os.getenv("LITMESH_STITCH_MODEL", "deepseek-chat")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
        self.client = LLMClient(
            provider="openai_compatible",
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self._call_count = 0

    def should_merge(self, block_a: str, block_b: str) -> bool:
        """Ask flash model: are these two blocks part of the same paragraph?"""
        if not block_a or not block_b:
            return False

        # Only pre-filter obvious cases to save API calls
        a_end = block_a.strip()[-1] if block_a.strip() else ''
        b_start = block_b.strip()[:6] if block_b.strip() else ''

        # Definitely sentence boundary
        if a_end in ('。', '！', '？') and b_start not in '一二三四五六七八九':
            return False
        # Definitely a new chapter/section header
        if any(b_start.startswith(s) for s in ('第', '思考', '探究', '练习', '本章', '复习')):
            return False

        # Everything else: ask LLM
        try:
            return self._ask_llm(block_a, block_b)
        except Exception:
            return False

    def _ask_llm(self, block_a: str, block_b: str) -> bool:
        """Call flash model for merge decision via OpenAI-compatible API."""
        prompt = _STITCH_PROMPT.format(
            block_a=block_a[-200:],
            block_b=block_b[:200],
        )
        content = self.client.complete(
            prompt,
            system="你只回答“是”或“否”。",
            temperature=0,
        )
        self._call_count += 1
        return bool(re.search(r'是|yes|同一', content, re.IGNORECASE))
