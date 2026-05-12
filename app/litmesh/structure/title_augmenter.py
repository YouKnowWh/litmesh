"""
Title augmenter — generates structure_title and display_title.

structure_title = the structural anchor label (from TOC / heading)
display_title  = the display label (structure_title + keyword if generic)
"""

from __future__ import annotations

from .block_role import BlockRole
from .keyword_summary import KeywordExtractor


class TitleAugmenter:
    """Generate two-tier titles for document blocks."""

    def __init__(self, extractor: KeywordExtractor | None = None):
        self.extractor = extractor or KeywordExtractor()

    def generate(
        self,
        heading: str,
        raw_text: str,
        role: str,
        toc_title: str = "",
    ) -> tuple[str, str]:
        """Generate (structure_title, display_title) for a block.

        Args:
            heading: Current block's heading text.
            raw_text: Block text for keyword extraction.
            role: BlockRole value.
            toc_title: TOC/anchor title if available (preferred for structure_title).

        Returns:
            (structure_title, display_title) tuple.
        """
        # Structure title: TOC > heading > empty
        structure_title = toc_title or heading or ""

        # Display title generation by role
        if role == BlockRole.FRONT.value:
            return structure_title, heading or structure_title

        if role == BlockRole.STRUCTURAL.value:
            title = toc_title or heading
            if not title:
                return "", ""
            # Optional keyword augmentation
            kw = self.extractor.extract(raw_text, title)
            if kw and len(kw) >= 4:
                display = f"{title}｜{kw}"
            else:
                display = title
            return title, display

        if role == BlockRole.CONTEXT.value:
            if not heading:
                return "", raw_text[:40]
            kw = self.extractor.extract(raw_text, heading)
            if kw:
                display = f"{heading}｜{kw}"
            else:
                display = heading
            return heading, display

        if role == BlockRole.CONTENT.value:
            # Content blocks: use heading if available, else keyword from text
            if heading:
                return heading, heading
            kw = self.extractor.extract(raw_text, "")
            return kw, kw or raw_text[:40]

        if role == BlockRole.NOISE.value:
            return "", ""

        # Fallback
        return heading or "", heading or raw_text[:40]

    def generate_display(
        self,
        heading: str,
        raw_text: str,
        role: str,
        structure_title: str = "",
    ) -> str:
        """Convenience: return only display_title."""
        _, display = self.generate(heading, raw_text, role, structure_title)
        return display
