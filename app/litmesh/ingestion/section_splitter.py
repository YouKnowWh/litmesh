"""
Section splitter: splits extracted PDF text into SectionBlocks.

Strategy:
1. Detect heading patterns via regex (numbered sections, Chinese chapter headings, etc.)
2. Build heading hierarchy (heading_path)
3. Split text into sections at each heading boundary
4. Assign page ranges based on text offset lookup
5. Link sections with prev/next/parent pointers

This is heuristic-based for v0.1. Future versions can use LLM for heading detection.
"""

import hashlib
import re
from typing import Optional

from ..models.section import SectionBlock, HeadingLevel


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Heading detection patterns (Chinese academic papers)
HEADING_PATTERNS = [
    # Numbered: "1 标题", "1.1 标题", "1.1.1 标题"
    (re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)"), HeadingLevel.SECTION),
    # Chinese: "一、标题", "（一）标题"
    (re.compile(r"^[（(]?([一二三四五六七八九十]+)[）)]\s*(.+)"), HeadingLevel.SECTION),
    # Keywords: "摘要", "关键词", "引言", "绪论", "结论", "参考文献", "致谢"
    (re.compile(r"^(摘要|关键词|引言|绪论|前言|结论|总结|参考文献|致谢|附录)\s*$"), HeadingLevel.SECTION),
    # English: "Abstract", "Introduction", "Conclusion", "References"
    (re.compile(r"^(Abstract|Introduction|Conclusion|References|Acknowledgments)\s*$", re.IGNORECASE), HeadingLevel.SECTION),
    # Chinese chapter: "第一章", "第1章"
    (re.compile(r"^第[一二三四五六七八九十\d]+章\s*(.*)"), HeadingLevel.CHAPTER),
    # Bold-style: lines that are short (< 60 chars) and followed by substantial text
    (re.compile(r"^([一-鿿\w][一-鿿\w\s]{2,50})$"), HeadingLevel.SUBSECTION),
]


def _detect_heading_level(text: str, depth: int) -> HeadingLevel:
    """Map numbered depth to heading level."""
    if depth == 0:
        return HeadingLevel.TITLE
    elif depth == 1:
        return HeadingLevel.CHAPTER
    elif depth == 2:
        return HeadingLevel.SECTION
    elif depth == 3:
        return HeadingLevel.SUBSECTION
    else:
        return HeadingLevel.SUBSUBSECTION


def _find_page_for_offset(offset: int, page_map: list[tuple[int, int]]) -> int:
    """Find the page number for a character offset."""
    for page_num, start, end in page_map:
        if start <= offset < end:
            return page_num
    return 1


def _build_page_map(pages: list[dict]) -> list[tuple[int, int, int]]:
    """Build (page_num, char_start, char_end) tuples."""
    offset = 0
    page_map = []
    for p in pages:
        text = p["text"]
        page_map.append((p["page_num"], offset, offset + len(text)))
        offset += len(text) + 2  # +2 for the "\n\n" separator
    return page_map


def split_sections(
    full_text: str,
    paper_id: str,
    graph_id: str,
    pages: list[dict],
    min_section_chars: int = 200,
) -> list[SectionBlock]:
    """Split extracted PDF text into SectionBlocks.

    Algorithm:
    1. Walk through lines; when a heading pattern matches, start a new section.
    2. Accumulate text until the next heading.
    3. Skip sections shorter than min_section_chars (merge into parent).
    4. Build heading_path and link prev/next/parent.
    """
    lines = full_text.split("\n")
    page_map = _build_page_map(pages)

    # Phase 1: detect headings with their positions
    headings = []  # [(line_idx, heading_text, level, depth)]
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        for pattern, default_level in HEADING_PATTERNS:
            m = pattern.match(line_stripped)
            if m:
                depth = len(m.group(1).split(".")) if "." in (m.group(1) or "") else 1
                level = _detect_heading_level(line_stripped, depth)
                headings.append((i, line_stripped, level, depth))
                break

    if not headings:
        # No headings found: treat entire text as one section
        return [_make_section(
            paper_id=paper_id, graph_id=graph_id,
            heading="正文", heading_path=["正文"],
            heading_level=HeadingLevel.SECTION,
            text=full_text, pages=pages, page_map=page_map,
            char_start=0, char_end=len(full_text),
        )]

    # Phase 2: build sections between headings
    sections = []
    heading_stack = []  # For parent tracking

    for idx, (line_idx, heading_text, level, depth) in enumerate(headings):
        # Determine text span
        char_start = sum(len(l) + 1 for l in lines[:line_idx])
        if idx + 1 < len(headings):
            next_line_idx = headings[idx + 1][0]
            char_end = sum(len(l) + 1 for l in lines[:next_line_idx])
        else:
            char_end = len(full_text)

        section_text = full_text[char_start:char_end].strip()

        # Build heading path
        while heading_stack and heading_stack[-1][1] >= depth:
            heading_stack.pop()
        heading_stack.append((heading_text, depth))
        heading_path = [h[0] for h in heading_stack]

        page_start = _find_page_for_offset(char_start, page_map)
        page_end = _find_page_for_offset(char_end, page_map)

        section = _make_section(
            paper_id=paper_id, graph_id=graph_id,
            heading=heading_text, heading_path=list(heading_path),
            heading_level=level,
            text=section_text, pages=pages, page_map=page_map,
            char_start=char_start, char_end=char_end,
            page_start=page_start, page_end=page_end,
            parent_section_id=None,  # Linked in phase 3
        )
        sections.append(section)

    # Phase 3: link prev/next/parent
    for i, section in enumerate(sections):
        if i > 0:
            section.prev_section_id = sections[i - 1].section_id
        if i < len(sections) - 1:
            section.next_section_id = sections[i + 1].section_id
        # Parent is the closest preceding section with shallower depth
        current_depth = len(section.heading_path)
        for j in range(i - 1, -1, -1):
            if len(sections[j].heading_path) < current_depth:
                section.parent_section_id = sections[j].section_id
                break

    return sections


def _make_section(
    paper_id: str, graph_id: str,
    heading: str, heading_path: list[str],
    heading_level: HeadingLevel,
    text: str, pages: list[dict], page_map: list[tuple[int, int, int]],
    char_start: int, char_end: int,
    page_start: Optional[int] = None, page_end: Optional[int] = None,
    parent_section_id: Optional[str] = None,
) -> SectionBlock:
    """Create a single SectionBlock."""
    return SectionBlock(
        paper_id=paper_id,
        graph_id=graph_id,
        heading=heading,
        heading_path=heading_path,
        heading_level=heading_level,
        raw_text=text,
        page_start=page_start or _find_page_for_offset(char_start, page_map),
        page_end=page_end or _find_page_for_offset(char_end, page_map),
        parent_section_id=parent_section_id,
        content_hash=sha256(text),
    )
