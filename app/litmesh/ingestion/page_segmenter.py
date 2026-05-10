"""
PageTextSegmenter: secondary text processing for pdfplumber output.

Takes raw page-level text from pdfplumber and produces cleaned,
paragraph-level elements suitable for SectionBlock generation.

Key operations:
1. Clean: strip cover fragments, page headers/footers, page numbers
2. Detect: find front matter boundary (TOC → first chapter)
3. Segment: split each body page into paragraphs
4. Merge: join broken lines within paragraphs (common in CJK PDFs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parsed_document import ParsedElement, ElementType, QualityReport

# ---- TOC / Front matter detection ----

_TOC_LINE_RE = re.compile(r'\.{4,}\s*\d{1,4}\s*$')  # Trailing dots + page number
_LEADING_DOTS_RE = re.compile(r'^\.{4,}')             # Leading dots (TOC left-side filler)
_TOC_PAGE_NUM_RE = re.compile(r'\s+\d{1,4}\s*$')     # Line ending with isolated page number
_CHAPTER_IN_LINE = re.compile(r'第\s*[一二三四五六七八九十\d]+\s*[章节]')
_CHAPTER_ANYWHERE = re.compile(r'第\s*[一二三四五六七八九十\d]+\s*[章节]')
_PAGE_NUM_ONLY = re.compile(r'^\s*\d{1,4}\s*$')
_BLANK_LINE = re.compile(r'^\s*$')


@dataclass
class SegmenterConfig:
    min_paragraph_chars: int = 30       # Skip shorter fragments
    body_page_threshold: float = 0.3    # If >30% of lines on a page are TOC-like, it's TOC
    max_title_lines: int = 4           # Max consecutive heading-like lines before it's body text
    merge_short_lines: int = 6         # If a line has ≤N CJK chars, merge with next line


class PageTextSegmenter:
    """Produces ParsedElements from pdfplumber page text via secondary segmentation."""

    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()

    def segment(self, pages: list[dict]) -> tuple[list[ParsedElement], QualityReport, int]:
        """Process pages into elements.

        Phase 1: identify front matter vs body pages.
        Phase 2: concatenate all BODY text, split into paragraphs globally.
        Phase 3: track page boundaries for each paragraph.

        Returns:
            (elements, quality_report, body_start_page)
        """
        cfg = self.config

        # Phase 1: classify pages
        body_start_page = 1
        in_front_matter = True
        toc_page_count = 0
        front_matter_pages = 0
        body_texts = []  # (page_num, text) for body pages

        for page in pages:
            page_num = page.get("page_num", 0)
            text = page.get("text", "")

            if not text or len(text.strip()) < 20:
                continue

            is_toc = self._is_toc_page(text)
            is_cover = self._is_cover_page(text, page_num)

            if is_toc:
                toc_page_count += 1
            if is_cover or is_toc or (in_front_matter and not self._has_body_signal(text)):
                front_matter_pages += 1
                continue

            if in_front_matter and self._has_body_signal(text):
                in_front_matter = False
                body_start_page = page_num

            body_texts.append((page_num, text))

        # Phase 2 + 3: concatenate all body text, split globally, track pages
        # Join with space so text flows continuously across page boundaries
        combined = " ".join(t for _, t in body_texts)
        # Clean up: collapse multiple spaces
        combined = re.sub(r' {2,}', ' ', combined)
        # Restore paragraph breaks from original double-newlines
        combined = re.sub(r'\n{2,}', '\n\n', combined)
        all_elements = []
        global_order = 0

        # Build page offset map: find where each page's text starts in combined
        page_offsets = []
        for pg_num, txt in body_texts:
            pos = combined.find(txt)
            if pos >= 0:
                page_offsets.append((pos, pos + len(txt), pg_num))

        def _page_for_pos(pos):
            for start, end, pg in page_offsets:
                if start <= pos < end:
                    return pg
            return body_start_page

        # Split combined text into paragraphs
        raw_paragraphs = self._split_paragraphs(combined)

        # Rule segmenter must stay deterministic and cheap. Window-level LLM
        # segmentation lives in LLMPageSegmenter; this path is fallback only.
        paragraphs = raw_paragraphs

        for para_text in paragraphs:
            if len(para_text) < cfg.min_paragraph_chars:
                continue
            if len(para_text) < 100 and (
                _TOC_LINE_RE.search(para_text) or _LEADING_DOTS_RE.match(para_text)
            ):
                continue
            global_order += 1
            elem_type = ElementType.HEADING if self._is_heading(para_text) else ElementType.PARAGRAPH

            # Find which page this paragraph starts on
            para_pos = combined.find(para_text)
            page_num = _page_for_pos(para_pos) if para_pos >= 0 else body_start_page

            all_elements.append(ParsedElement(
                element_id=f"elem_{global_order}",
                type=elem_type,
                text=para_text,
                page_start=page_num, page_end=page_num,
                order_index=global_order,
            ))

        # Quality report
        total = max(len(all_elements), 1)
        body_pages = len(pages) - front_matter_pages
        quality = QualityReport(
            parser_name="pdfplumber+segmenter",
            parser_version="1.0",
            segmenter_name="rule_page_segmenter",
            total_elements=total,
            paragraph_count=sum(1 for e in all_elements if e.type == ElementType.PARAGRAPH),
            heading_count=sum(1 for e in all_elements if e.type == ElementType.HEADING),
            toc_detected=toc_page_count > 0,
            front_matter_block_count=front_matter_pages,
            empty_fragment_count=0,
            suspicious_long_heading_count=0,
            average_paragraph_length=round(sum(len(e.text) for e in all_elements) / total, 1),
            body_page_count=body_pages,
            body_start_page=body_start_page,
            too_short_paragraph_count=sum(1 for e in all_elements if len(e.text) < cfg.min_paragraph_chars),
            needs_structure_review=body_pages < 5 or total < max(10, body_pages),
        )
        return all_elements, quality, body_start_page

    # ---- Paragraph splitting ----

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split page text into paragraphs.

        Strategy:
        1. Split on blank lines first (natural paragraph breaks)
        2. Within each block, merge CJK lines that are broken mid-sentence
        3. Keep heading-like lines separate
        """
        # Split on blank lines
        raw_blocks = re.split(r'\n\s*\n', text)
        paragraphs = []

        for block in raw_blocks:
            block = block.strip()
            if not block or len(block) < self.config.min_paragraph_chars:
                continue

            lines = block.split('\n')

            # If single line, keep as-is
            if len(lines) == 1:
                paragraphs.append(lines[0])
                continue

            # Multi-line block: merge broken CJK lines
            merged = []
            current = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if current:
                        merged.append(current)
                        current = ""
                    continue

                # CJK line that ends mid-sentence: merge
                cjk_chars = sum(1 for c in line if '一' <= c <= '鿿')
                if current and cjk_chars > self.config.merge_short_lines:
                    current += line
                elif current:
                    merged.append(current)
                    current = line
                else:
                    current = line

            if current:
                merged.append(current)

            paragraphs.extend(merged)

        return paragraphs

    # ---- Detection helpers ----

    @staticmethod
    def _is_toc_page(text: str) -> bool:
        """Check if page looks like a TOC: many dot-leader lines or title+page-number patterns."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return False
        # Count various TOC line patterns
        toc_dot = sum(1 for l in lines if _TOC_LINE_RE.search(l))
        toc_lead = sum(1 for l in lines if _LEADING_DOTS_RE.match(l))
        toc_page = sum(1 for l in lines if _TOC_PAGE_NUM_RE.search(l) and len(l) < 60)
        has_chapter = any(_CHAPTER_ANYWHERE.search(l) for l in lines)
        toc_total = toc_dot + toc_lead + toc_page
        # Dominated by TOC patterns or many TOC lines with chapter indicators
        return (toc_total >= 8) or (toc_total >= 3 and has_chapter and toc_total >= len(lines) * 0.3)

    @staticmethod
    def _is_cover_page(text: str, page_num: int) -> bool:
        """Detect cover/title pages (first few pages with minimal body text)."""
        if page_num > 4:
            return False
        text_stripped = text.replace(' ', '').replace(' ', '').replace('\n', '')
        return len(text_stripped) < 100

    @staticmethod
    def _has_body_signal(text: str) -> bool:
        """Check if text contains actual body content (not TOC/front matter)."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return False

        # Count all TOC-like patterns
        toc_like = sum(1 for l in lines if (
            _TOC_LINE_RE.search(l) or _LEADING_DOTS_RE.match(l) or
            _PAGE_NUM_ONLY.match(l) or
            (_TOC_PAGE_NUM_RE.search(l) and _CHAPTER_IN_LINE.search(l))
        ))
        toc_ratio = toc_like / len(lines)
        avg_len = sum(len(l) for l in lines) / len(lines)

        # Reject obvious TOC/front matter
        if toc_ratio > 0.2 or avg_len < 25:
            return False

        long_lines = sum(1 for l in lines if len(l) > 40)
        return long_lines >= 3 and toc_ratio < 0.15

    @staticmethod
    def _is_heading(text: str) -> bool:
        """Heuristic: short text with chapter/section patterns is likely a heading."""
        if len(text) > 120:
            return False
        stripped = text.strip()
        lines = stripped.split('\n')
        if len(lines) > 2:
            return False
        return bool(
            _CHAPTER_ANYWHERE.search(stripped) or
            stripped.startswith('第') or
            (len(stripped) < 40 and ('节' in stripped[:6] or '章' in stripped[:6] or '探究' in stripped[:6]))
        )


_SENTENCE_END = set('。！？!?."\'」】）)')
_SENTENCE_START = set('第一二三四五六七八九「（(')
