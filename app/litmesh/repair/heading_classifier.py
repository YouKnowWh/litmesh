"""
Heading role classifier — separates headings into 6 roles for downstream routing.

Classification priority (highest first):
  1. FRONT_MATTER  — preface, TOC header, epilogue, etc.
  2. NOISE         — garbage, too short, punctuation-only
  3. TOC_ENTRY     — real table-of-contents entry (chapter/section with page number)
  4. STRUCTURAL_HEADING — body chapter/section-level heading
  5. CONTEXT_HEADING — local activity/discussion/exercise label
  Fallback: DECORATIVE — column/special-topic/person name

Each role maps to different downstream handling:
  - TOC_ENTRY → outline_nodes, chapter graph skeleton
  - STRUCTURAL_HEADING → heading_path, section grouping
  - CONTEXT_HEADING → context_group, local paragraph cluster
  - FRONT_MATTER → reserved section
  - DECORATIVE → display label only
  - NOISE → skip
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class HeadingRole(str, Enum):
    TOC_ENTRY = "toc_entry"
    STRUCTURAL_HEADING = "structural_heading"
    CONTEXT_HEADING = "context_heading"
    FRONT_MATTER = "front_matter"
    DECORATIVE = "decorative"
    NOISE = "noise"


# ---- Word Lists ----

_FRONT_MATTER_EXACT: set[str] = {
    "目录", "目 录", "目次", "Contents", "Table of Contents",
    "前言", "序言", "序", "Foreword", "Preface",
    "编写说明", "编辑说明", "凡例", "使用说明",
    "后记", "跋", "结语", "後記",
    "致谢", "Acknowledgments",
    "参考文献", "References", "Bibliography",
    "附录", "Appendix", "附錄",
    "注释", "Notes",
    "译者简介", "作者简介", "编者简介",
}

_CONTEXT_HEADING_EXACT: set[str] = {
    "问题探讨", "讨论", "本节聚焦", "思考与讨论",
    "探究·实践", "探究与实践", "实验", "实践活动",
    "练习与应用", "练习", "复习与提高", "复习题",
    "自我检测", "本章小结", "内容提要", "本节要点",
    "材料用具", "方法步骤", "结果和结论", "目的要求",
    "实验原理", "实验步骤", "实验结果", "讨论与思考",
    "相关信息", "与社会的联系", "学科交叉", "科学方法",
}

_DECORATIVE_EXACT: set[str] = {
    "科学家访谈", "科学·技术·社会", "科学•技术•社会",
    "与生物学有关的职业", "生物科技进展",
    "知识链接", "拓展视野", "课外阅读", "课外实践",
    "STS", "STSE",
    "袁隆平", "育种工作者", "遗传咨询师",
}

# Numbered activity headings: "一、概念检测", "二、拓展应用", etc.
_NUMBERED_ACTIVITY_RE = re.compile(
    r"^[一二三四五六七八九十]+[、，．.]\s*.+"
)

# Chapter/section number patterns
_CHAPTER_RE = re.compile(
    r"^(第\s*[一二三四五六七八九十\d]+\s*[章节篇部]|"
    r"Chapter\s+\d+|Part\s+\d+|"
    r"第\s*[一二三四五六七八九十\d]+\s*节)"
)

# TOC dot-leader + page number
_TOC_LINE_RE = re.compile(
    r"[\.…·•]{2,}\s*\d{1,4}\s*$|"
    r"第\s*[一二三四五六七八九十\d]+\s*[章节].+\d{1,4}\s*$"
)

# Page header/footer patterns
_PAGE_NUM_RE = re.compile(r"^第?\d{1,4}[页頁]?\s*$")


class HeadingClassifier:
    """Rule-based heading role classifier.

    Usage:
        classifier = HeadingClassifier()
        role = classifier.classify("第1章 遗传因子的发现", context={"is_toc_region": True})
    """

    def classify(
        self,
        text: str,
        heading_level: int = 1,
        context: Optional[dict] = None,
    ) -> HeadingRole:
        """Classify a heading text into one of six roles.

        Args:
            text: The heading text.
            heading_level: Parser-reported heading level (1-6).
            context: Optional dict with keys:
                - is_toc_region: bool — element is in/near a TOC block
                - heading_count: int — total headings in document
        """
        ctx = context or {}
        text_clean = text.strip()
        if not text_clean:
            return HeadingRole.NOISE

        # Step 1: FRONT_MATTER
        role = self._classify_front_matter(text_clean)
        if role:
            return role

        # Step 2: NOISE
        role = self._classify_noise(text_clean)
        if role:
            return role

        # Step 3: TOC_ENTRY
        is_toc_region = ctx.get("is_toc_region", False)
        role = self._classify_toc_entry(text_clean, is_toc_region)
        if role:
            return role

        # Step 4: STRUCTURAL_HEADING
        role = self._classify_structural(text_clean, heading_level)
        if role:
            return role

        # Step 5: CONTEXT_HEADING
        role = self._classify_context(text_clean)
        if role:
            return role

        # Fallback: DECORATIVE
        role = self._classify_decorative(text_clean)
        if role:
            return role

        return HeadingRole.STRUCTURAL_HEADING  # default for unknown headings

    # ---- per-role classifiers ----

    def _classify_front_matter(self, text: str) -> Optional[HeadingRole]:
        if text in _FRONT_MATTER_EXACT:
            return HeadingRole.FRONT_MATTER
        return None

    def _classify_noise(self, text: str) -> Optional[HeadingRole]:
        if len(text) < 2:
            return HeadingRole.NOISE
        if _PAGE_NUM_RE.match(text):
            return HeadingRole.NOISE
        # Mostly digits/symbols
        alpha_chars = sum(1 for c in text if c.isalpha() or '一' <= c <= '鿿')
        if alpha_chars / len(text) < 0.3:
            return HeadingRole.NOISE
        return None

    def _classify_toc_entry(self, text: str, is_toc_region: bool) -> Optional[HeadingRole]:
        if not is_toc_region:
            return None
        # TOC entries have chapter numbering or dot-leader patterns
        if _CHAPTER_RE.search(text):
            return HeadingRole.TOC_ENTRY
        if _TOC_LINE_RE.search(text):
            return HeadingRole.TOC_ENTRY
        return None

    def _classify_structural(self, text: str, heading_level: int) -> Optional[HeadingRole]:
        # Chapter/section numbering → structural
        if _CHAPTER_RE.search(text):
            return HeadingRole.STRUCTURAL_HEADING
        # Top-level heading in a flat doc
        if heading_level <= 2 and len(text) > 5:
            # Exclude context/decorative
            if text in _CONTEXT_HEADING_EXACT:
                return None
            if text in _DECORATIVE_EXACT:
                return None
            if _NUMBERED_ACTIVITY_RE.match(text):
                return None
            return HeadingRole.STRUCTURAL_HEADING
        return None

    def _classify_context(self, text: str) -> Optional[HeadingRole]:
        if text in _CONTEXT_HEADING_EXACT:
            return HeadingRole.CONTEXT_HEADING
        if _NUMBERED_ACTIVITY_RE.match(text):
            return HeadingRole.CONTEXT_HEADING
        return None

    def _classify_decorative(self, text: str) -> Optional[HeadingRole]:
        if text in _DECORATIVE_EXACT:
            return HeadingRole.DECORATIVE
        return None


# Module-level convenience
_classifier: Optional[HeadingClassifier] = None


def classify_heading(text: str, heading_level: int = 1, **context) -> HeadingRole:
    """Module-level convenience wrapper."""
    global _classifier
    if _classifier is None:
        _classifier = HeadingClassifier()
    return _classifier.classify(text, heading_level, context if context else None)
