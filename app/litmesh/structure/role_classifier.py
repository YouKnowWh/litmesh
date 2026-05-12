"""
Language-agnostic block role classifier.

Uses structural signals (numbering, position, density, adjacency) rather than
language-specific word lists. Word lists are weak hints only — they adjust
confidence but never make the final decision alone.
"""

from __future__ import annotations

import re
from typing import Optional

from .block_role import BlockRole


# ---- Language-agnostic patterns ----

# Chapter/section numbering (Chinese, Western, Roman)
_STRUCTURAL_NUM = re.compile(
    r"^(第\s*[一二三四五六七八九十\d]+\s*[章节篇部]|"
    r"Chapter\s+\d+|Part\s+\d+|Section\s+\d+|"
    r"^[IVX]+\.\s+|^\d+\.\d+\s+|"
    r"^[A-G]\.\s+)",
    re.IGNORECASE,
)

# Activity/exercise numbering
_CONTEXT_NUM = re.compile(
    r"^[\d①②③④⑤⑥⑦⑧⑨⑩]+[\.\、．)]\s*|"
    r"^[\(（]\d+[\)）]\s*|"
    r"^[•·●○]\s*",
)

# TOC dot-leader pattern (language-agnostic)
_TOC_DOTS = re.compile(r"[\.…·•]{3,}\s*\d{1,4}\s*$")

# Page header/footer patterns
_PAGE_NUM = re.compile(r"^\d{1,4}\s*$")
_HEADER_FOOTER = re.compile(r"^\d{1,4}\s*[-–—]\s*\d{1,4}\s*$")

# Weak word hints for front matter (NOT hard rules)
_FRONT_HINTS = {"preface", "foreword", "acknowledgment", "appendix",
                "前言", "序", "后记", "附录", "致谢", "目录", "contents"}

# Weak word hints for context blocks
_CONTEXT_HINTS = {"discussion", "example", "note", "exercise", "activity",
                  "讨论", "案例", "练习", "注意", "提示", "说明", "问题"}

# Characters with high semantic density
_HIGH_DENSITY = re.compile(r"[一-鿿a-zA-Z0-9]")


class RoleClassifier:
    """Classify a text block into one of five universal roles."""

    def classify(
        self,
        text: str,
        order_index: int = 0,
        total_blocks: int = 1,
        heading_level: int = 0,
        neighbour_roles: Optional[list[str]] = None,
    ) -> BlockRole:
        """Determine block role from structural signals.

        Args:
            text: Block text content (first 200 chars is sufficient).
            order_index: Position in document (0-based).
            total_blocks: Total number of blocks in document.
            heading_level: Parser-reported heading level (0 = not a heading).
            neighbour_roles: Roles of adjacent blocks (for context inference).
        """
        text_stripped = text.strip()
        if not text_stripped:
            return BlockRole.NOISE

        # ---- NOISE (skip for headings — short heading text is valid) ----
        if heading_level == 0 and _is_noise(text_stripped):
            return BlockRole.NOISE

        # ---- Position signals ----
        position_ratio = order_index / max(total_blocks, 1)

        # ---- FRONT ----
        if position_ratio < 0.05 or position_ratio > 0.95:
            if _TOC_DOTS.search(text_stripped):
                return BlockRole.FRONT
            if _has_hint(text_stripped, _FRONT_HINTS):
                return BlockRole.FRONT

        # ---- Heading-based signals ----
        if heading_level >= 1 or _STRUCTURAL_NUM.match(text_stripped):
            # Front matter can appear at any heading level
            if position_ratio < 0.05 or position_ratio > 0.95:
                if _has_hint(text_stripped, _FRONT_HINTS) or _TOC_DOTS.search(text_stripped):
                    return BlockRole.FRONT
            # Check context hints first (even at level 2, "问题探讨" is context)
            if heading_level >= 2 and _has_hint(text_stripped, _CONTEXT_HINTS):
                return BlockRole.CONTEXT
            if heading_level <= 2:
                return BlockRole.STRUCTURAL
            if heading_level <= 3:
                # Could be context or structural — check if it looks like a context hint
                if _has_hint(text_stripped, _CONTEXT_HINTS):
                    return BlockRole.CONTEXT
                return BlockRole.STRUCTURAL
            # Level 4+: likely context
            if _has_hint(text_stripped, _CONTEXT_HINTS):
                return BlockRole.CONTEXT
            return BlockRole.CONTEXT

        # ---- Context numbering ----
        if _CONTEXT_NUM.match(text_stripped):
            return BlockRole.CONTEXT

        # ---- Density-based ----
        if _is_low_density(text_stripped):
            if _has_hint(text_stripped, _CONTEXT_HINTS):
                return BlockRole.CONTEXT
            # Very short, not clearly structural → content
            pass

        # ---- Adjacency inference ----
        if neighbour_roles:
            prev = neighbour_roles[0] if neighbour_roles else ""
            if prev == BlockRole.STRUCTURAL.value and heading_level >= 2:
                return BlockRole.CONTEXT

        # ---- Default ----
        return BlockRole.CONTENT


def _is_noise(text: str) -> bool:
    """Check if block is noise (page number, header/footer, debris)."""
    if len(text) < 3:
        return True
    if _PAGE_NUM.match(text):
        return True
    if _HEADER_FOOTER.match(text):
        return True
    # >70% non-semantic characters
    density = len(_HIGH_DENSITY.findall(text)) / max(len(text), 1)
    if density < 0.2:
        return True
    return False


def _is_low_density(text: str) -> bool:
    """Check if text has low semantic density (short label-like block)."""
    density = len(_HIGH_DENSITY.findall(text)) / max(len(text), 1)
    return len(text) < 40 and density < 0.6


def _has_hint(text: str, hint_set: set[str]) -> bool:
    """Check if any hint word appears in the text."""
    text_lower = text.lower()
    return any(h in text_lower for h in hint_set)
