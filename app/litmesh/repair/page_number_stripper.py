"""
LLM-based page number stripper.

Replaces the regex `_strip_page_number()` with a safer approach:
  1. Collect — scan all SectionBlock.raw_text for leading number patterns
  2. Classify — batch LLM call to determine which are actually page numbers
  3. Apply — only strip confirmed page numbers, preserve years/dates/chapters

One LLM call per document, regardless of section count.
Results are written to the repair audit log.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..models.section import SectionBlock

logger = logging.getLogger("litmesh.repair.page_number")

# Matches a leading digit (1-4 chars) followed by whitespace.
# Captures the full pattern so we know exactly what to strip.
_LEADING_NUM_RE = re.compile(r"^(\d{1,4})\s+")

# Never-strip words: if the number is immediately followed by one of these,
# skip collection entirely — it's never a page number.
_NEVER_STRIP_AFTER = re.compile(
    r"^(年|月|日|世纪|年代|万年|亿年|章|节|篇|部|条|款|项|\.\d|%|％|°|℃|℉)"
)


@dataclass
class PageNumberCandidate:
    """A single leading-number pattern found in section text."""
    candidate_id: str
    number_str: str           # e.g. "144", "2007"
    context: str              # e.g. "144 印度共产党在1957年..."
    section_id: str
    # Accumulation: same pattern appearing in multiple sections
    count: int = 1


class PageNumberStripper:
    """Collect leading-number patterns, classify via LLM, strip only page numbers."""

    # Rough estimate: one candidate JSON is ~100 chars ≈ 25 tokens.
    # 400 candidates ≈ 10k tokens, well within any modern LLM's context.
    # If we somehow exceed this, chunk automatically.
    _max_candidates_per_call: int = 400

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLMClient for classification. If None, falls back to regex.
        """
        self.llm = llm_client

    # ---- Public API ----

    def process(
        self,
        sections: List[SectionBlock],
        paper_id: str = "",
    ) -> tuple[List[SectionBlock], dict]:
        """Full pipeline: collect → classify → apply.

        Returns (stripped_sections, report).
        """
        t0 = time.monotonic()
        report = {
            "candidates_collected": 0,
            "unique_patterns": 0,
            "page_numbers_found": 0,
            "stripped_sections": 0,
            "llm_used": False,
            "elapsed_ms": 0,
        }

        # Step 1: Collect
        candidates = self.collect_candidates(sections)
        report["candidates_collected"] = sum(c.count for c in candidates)
        report["unique_patterns"] = len(candidates)

        if not candidates:
            report["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
            return sections, report

        # Deduplicate by number_str
        unique = _deduplicate_by_number(candidates)

        # Step 2: Classify
        page_numbers: set[str] = set()
        if self.llm is not None:
            try:
                page_numbers = self._classify_with_llm(unique, paper_id)
                report["llm_used"] = True
            except Exception as e:
                logger.warning("LLM page number classification failed: %s — "
                               "falling back to regex", e)
                page_numbers = self._classify_with_regex(unique)
        else:
            page_numbers = self._classify_with_regex(unique)

        report["page_numbers_found"] = len(page_numbers)

        # Step 3: Apply
        if page_numbers:
            sections = self._apply(sections, page_numbers)
            report["stripped_sections"] = sum(
                1 for s in sections
                if _LEADING_NUM_RE.match(s.raw_text) is None
                and s.raw_text  # text was modified
            )

        # Count actual stripped sections by comparing raw_text length
        # More accurate: just count sections that had a page number stripped
        report["stripped_sections"] = sum(
            1 for c in candidates if c.number_str in page_numbers
        )

        report["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Page number stripping: %d unique patterns, %d page numbers, "
            "%d sections stripped, LLM=%s, %dms",
            report["unique_patterns"], report["page_numbers_found"],
            report["stripped_sections"], report["llm_used"],
            report["elapsed_ms"],
        )
        return sections, report

    # ---- Step 1: Collect ----

    def collect_candidates(self, sections: List[SectionBlock]) -> List[PageNumberCandidate]:
        """Scan sections for leading-number patterns, skipping obvious non-page-numbers."""
        candidates: List[PageNumberCandidate] = []
        for s in sections:
            text = s.raw_text.strip()
            m = _LEADING_NUM_RE.match(text)
            if not m:
                continue
            number_str = m.group(1)
            after = text[m.end():]

            # Skip if followed by never-strip words (years, chapters, etc.)
            if _NEVER_STRIP_AFTER.match(after):
                continue

            # Build context snippet: number + up to 50 chars after
            context = text[:m.end() + min(50, len(after))]

            candidates.append(PageNumberCandidate(
                candidate_id=f"pn:{number_str}:{s.section_id}",
                number_str=number_str,
                context=context,
                section_id=s.section_id,
            ))
        return candidates

    # ---- Step 2a: LLM Classification ----

    def _classify_with_llm(
        self, candidates: List[PageNumberCandidate], paper_id: str
    ) -> set[str]:
        """Classify all candidates via LLM, auto-chunking if needed.

        Sends everything in one call for typical documents;
        splits into chunks only when candidate count exceeds context safety limit.
        """
        page_numbers: set[str] = set()
        chunk_size = self._max_candidates_per_call
        for i in range(0, len(candidates), chunk_size):
            batch = candidates[i:i + chunk_size]
            result = self._call_llm_batch(batch)
            for item in result:
                if item.get("is_page_number"):
                    page_numbers.add(item["number_str"])
        return page_numbers

    def _call_llm_batch(self, candidates: List[PageNumberCandidate]) -> list[dict]:
        """Send one batch to the LLM, parse response."""
        items = [
            {
                "id": c.candidate_id,
                "number_str": c.number_str,
                "context": c.context,
            }
            for c in candidates
        ]

        prompt = f"""你是一个文档清洗助手。以下是文本段落开头的数字模式。

对每一项判断：这个数字是"页码"还是"年份/日期/章节号/其他有意义数字"？

判断规则：
- 页码：数字后面紧跟着正文内容，没有明显的年份/日期/章节标记
  例："144 印度共产党在..." → 页码
  例："35 本章小结..." → 页码
- 非页码：数字是年份、日期、章节号、或其他有意义的编号
  例："2007 年 1 月..." → 年份，不是页码
  例："1949 年，中国..." → 年份，不是页码
  例："21 世纪以来..." → 世纪，不是页码
  例："1990 年代..." → 年代，不是页码

待判断项：
{json.dumps(items, ensure_ascii=False, indent=2)}

请用 JSON 数组回答，不要输出其他内容。每项包含 id、number_str、is_page_number (true/false)：
[{{"id": "...", "number_str": "...", "is_page_number": true/false}}, ...]"""

        response = self.llm.complete_json(
            prompt,
            system="You are a precise document cleaning assistant. Reply with JSON only.",
            temperature=0.0,
            max_tokens=2048,
        )

        # Normalize: if LLM returns a dict with a key containing the array,
        # or the array directly
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            for v in response.values():
                if isinstance(v, list):
                    return v
            # Maybe it's a flat dict — return as single-item list
            return [response]
        logger.warning("Unexpected LLM response format: %s", type(response))
        return []

    # ---- Step 2b: Regex Fallback ----

    def _classify_with_regex(self, candidates: List[PageNumberCandidate]) -> set[str]:
        """Regex-based classification as fallback.

        Uses the conservative negative lookahead from _strip_page_number.
        """
        page_numbers: set[str] = set()
        _safe_num_re = re.compile(r"^\d{1,4}\s+(?!年|月|日|世纪|年代|万年|亿年)")
        for c in candidates:
            if _safe_num_re.match(c.context):
                page_numbers.add(c.number_str)
        return page_numbers

    # ---- Step 3: Apply ----

    def _apply(
        self, sections: List[SectionBlock], page_numbers: set[str]
    ) -> List[SectionBlock]:
        """Strip confirmed page numbers from section raw_text."""
        for s in sections:
            text = s.raw_text.strip()
            for num_str in sorted(page_numbers, key=len, reverse=True):
                prefix = num_str + " "
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    break
                # Also try without trailing space
                if text.startswith(num_str) and len(text) > len(num_str) and text[len(num_str)] in (' ', '\t'):
                    text = text[len(num_str):].strip()
                    break
            s.raw_text = text
        return sections


def _deduplicate_by_number(candidates: List[PageNumberCandidate]) -> List[PageNumberCandidate]:
    """Merge candidates with the same number_str, keeping the first context."""
    seen: dict[str, PageNumberCandidate] = {}
    for c in candidates:
        if c.number_str in seen:
            seen[c.number_str].count += 1
        else:
            seen[c.number_str] = c
    return list(seen.values())
