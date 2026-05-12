"""
LLM fallback judge for grey-zone, high-impact repair candidates.

Only invoked when:
  - Reranker score is in the grey zone (0.5-0.85)
  - Repair type is high-impact (TOC boundary, chapter boundary, front matter)

Uses a lightweight model (deepseek-chat) via LLMClient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from .candidate_detector import RepairCandidate
from .reranker_client import RerankerScore
from ..models.section import SectionBlock


@dataclass
class RepairLLMDecision:
    candidate_id: str
    decision: str     # merge | split | keep | mark_heading | mark_not_heading | mark_toc | mark_front_matter
    confidence: float
    reasoning: str


# ---- Prompt templates ----

_REPAIR_PROMPT_TEMPLATES = {
    "adjacent_merge": """你是一个文档结构分析助手。判断以下两个文本块是否应该合并为同一段落。

文本块 A（标题: {heading_a}）:
{text_a}

文本块 B（标题: {heading_b}）:
{text_b}

请用 JSON 回答，不要输出其他内容：
{{"should_merge": true/false, "confidence": 0.0-1.0, "reasoning": "一句话解释"}}""",

    "toc_boundary": """你是一个文档结构分析助手。判断以下文本块是否是目录/目录残留，不应该被当作正文。

文本块:
{text}

请用 JSON 回答，不要输出其他内容：
{{"is_toc": true/false, "confidence": 0.0-1.0, "reasoning": "一句话解释"}}""",

    "front_matter_boundary": """你是一个文档结构分析助手。判断以下文本块是否是前言/序言/编写说明。

文本块（标题: {heading}）:
{text}

请用 JSON 回答，不要输出其他内容：
{{"is_front_matter": true/false, "confidence": 0.0-1.0, "reasoning": "一句话解释"}}""",

    "chapter_boundary": """你是一个文档结构分析助手。以下两个相邻文本块的标题层级出现异常跳跃，判断这是否合理。

块 A: 层级={level_a}, 标题="{heading_a}"
块 B: 层级={level_b}, 标题="{heading_b}"

可选操作:
- "fix_levels": 层级异常，需要修正
- "keep": 层级跳跃合理（例如特殊排版）

请用 JSON 回答，不要输出其他内容：
{{"action": "fix_levels"/"keep", "confidence": 0.0-1.0, "reasoning": "一句话解释"}}""",

    "heading_role": """你是一个文档结构分析助手。判断下面这段文本的标签是否正确。

当前标签: "{heading}"
正文开头:
{text}

可选操作:
- "mark_not_heading": 这个标签太像正文，应该降级
- "mark_heading": 正文第一行更像是标题
- "keep": 标签和正文匹配，保持不变

请用 JSON 回答，不要输出其他内容：
{{"action": "mark_not_heading"/"mark_heading"/"keep", "confidence": 0.0-1.0, "reasoning": "一句话解释"}}""",
}


class FallbackLLM:
    """LLM-based repair judge for ambiguous structural decisions."""

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient instance (typically using deepseek-chat).
        """
        self.llm = llm_client

    def judge(
        self,
        candidate: RepairCandidate,
        sections: List[SectionBlock],
        reranker_score: RerankerScore,
    ) -> RepairLLMDecision:
        """Get LLM judgment for a grey-zone candidate."""
        id_map = {s.section_id: s for s in sections}

        prompt = self._build_prompt(candidate, id_map)
        system = "You are a precise document structure analyst. Reply with JSON only."

        try:
            result = self.llm.complete_json(prompt, system=system, temperature=0.0, max_tokens=512)
        except Exception:
            # If LLM fails, return a conservative "keep" decision
            return RepairLLMDecision(
                candidate_id=candidate.candidate_id,
                decision="keep",
                confidence=0.3,
                reasoning="LLM call failed, defaulting to keep",
            )

        return self._parse_response(candidate, result, reranker_score)

    def _build_prompt(self, candidate: RepairCandidate, id_map: dict) -> str:
        template = _REPAIR_PROMPT_TEMPLATES.get(
            candidate.repair_type,
            _REPAIR_PROMPT_TEMPLATES["adjacent_merge"],
        )

        sids = candidate.section_ids
        s0 = id_map.get(sids[0])

        if candidate.repair_type == "adjacent_merge":
            s1 = id_map.get(sids[1]) if len(sids) > 1 else None
            return template.format(
                heading_a=s0.heading if s0 else "(无)",
                text_a=(s0.raw_text if s0 else "")[:1500],
                heading_b=s1.heading if s1 else "(无)",
                text_b=(s1.raw_text if s1 else "")[:1500],
            )

        elif candidate.repair_type in ("toc_boundary", "front_matter_boundary"):
            return template.format(
                text=(s0.raw_text if s0 else "")[:2000],
                heading=s0.heading if s0 else "(无)",
            )

        elif candidate.repair_type == "chapter_boundary":
            s1 = id_map.get(sids[1]) if len(sids) > 1 else None
            return template.format(
                level_a=s0.heading_level.value if s0 else "?",
                heading_a=s0.heading if s0 else "(无)",
                level_b=s1.heading_level.value if s1 else "?",
                heading_b=s1.heading if s1 else "(无)",
            )

        elif candidate.repair_type == "heading_role":
            return template.format(
                heading=s0.heading if s0 else "(无标题)",
                text=(s0.raw_text if s0 else "")[:1500],
            )

        # Generic fallback
        return template

    def _parse_response(
        self,
        candidate: RepairCandidate,
        result: dict,
        reranker_score: RerankerScore,
    ) -> RepairLLMDecision:
        """Parse LLM JSON response into a RepairLLMDecision."""
        confidence = float(result.get("confidence", 0.5))
        reasoning = str(result.get("reasoning", ""))

        repair_type = candidate.repair_type
        if repair_type == "adjacent_merge":
            should_merge = result.get("should_merge", False)
            decision = "merge" if should_merge else "keep"
        elif repair_type == "toc_boundary":
            is_toc = result.get("is_toc", False)
            decision = "mark_toc" if is_toc else "keep"
        elif repair_type == "front_matter_boundary":
            is_fm = result.get("is_front_matter", False)
            decision = "mark_front_matter" if is_fm else "keep"
        elif repair_type == "chapter_boundary":
            decision = str(result.get("action", "keep"))
        elif repair_type == "heading_role":
            decision = str(result.get("action", "keep"))
        else:
            decision = str(result.get("action", "keep"))

        return RepairLLMDecision(
            candidate_id=candidate.candidate_id,
            decision=decision,
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )
