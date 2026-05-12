"""
Title augmenter — generates enhanced display_title in "heading｜keyword" format.

Uses HeadingRole to decide augmentation strategy:
  - CONTEXT_HEADING → forced augmentation
  - STRUCTURAL_HEADING → optional augmentation
  - TOC_ENTRY / FRONT_MATTER / DECORATIVE → no augmentation
"""

from __future__ import annotations

from .heading_classifier import HeadingRole
from .keyword_extractor import KeywordExtractor


class TitleAugmenter:
    """Generate enhanced display titles."""

    def __init__(self, extractor: KeywordExtractor | None = None):
        self.extractor = extractor or KeywordExtractor()

    def augment(self, heading: str, raw_text: str, role: str) -> str:
        """Return enhanced display_title.

        Args:
            heading: Current heading text.
            raw_text: Full section text for keyword extraction.
            role: HeadingRole value string.

        Returns:
            Enhanced display title.
        """
        if not heading and not raw_text:
            return ""

        if not heading:
            # No heading: use keyword summary directly
            kw = self.extractor.extract(raw_text, "", role)
            return kw or raw_text[:40]

        # Decide strategy by role
        if role == HeadingRole.CONTEXT_HEADING.value:
            # Forced augmentation
            kw = self.extractor.extract(raw_text, heading, role)
            if kw:
                return f"{heading}｜{kw}"
            return heading

        elif role == HeadingRole.STRUCTURAL_HEADING.value:
            # Optional: only add keyword if meaningful
            kw = self.extractor.extract(raw_text, heading, role)
            if kw and len(kw) >= 4:
                return f"{heading}｜{kw}"
            return heading

        elif role == HeadingRole.FRONT_MATTER.value:
            return heading

        elif role == HeadingRole.TOC_ENTRY.value:
            return heading

        elif role == HeadingRole.DECORATIVE.value:
            return heading

        # Default: heading as-is
        return heading or raw_text[:40]
