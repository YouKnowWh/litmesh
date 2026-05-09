"""
Section splitter: splits extracted PDF text into paragraph-level SectionBlocks.

Strategy:
1. Detect heading patterns via regex (numbered sections, Chinese chapter headings, etc.)
2. Use headings as structural context (heading_path), not as the primary chunk
3. Split the body into paragraph blocks
4. Assign page ranges based on text offset lookup
5. Link paragraph blocks with prev/next pointers

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
    # Common short academic headings without explicit numbering
    (re.compile(r"^[一-鿿A-Za-z0-9\-—（）()·:：\s]{2,40}(框架设计|风险分析|实验验证|分析|设计|方法|结果|讨论)$"), HeadingLevel.SECTION),
]

_PARAGRAPH_ENDINGS = tuple("。！？!?；;.")


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
    min_section_chars: int = 40,
) -> list[SectionBlock]:
    """Split extracted PDF text into paragraph-level SectionBlocks.

    Algorithm:
    1. Walk through lines; when a heading pattern matches, update heading context.
    2. Split the text under each heading into paragraph blocks.
    3. Keep heading_path on every paragraph so traversal can recover context.
    4. Link paragraph blocks with prev/next. The actual next pointer is written
       after DB insertion in pipeline.py to avoid FK violations.
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
                token = m.group(1) if m.groups() else ""
                depth = len(token.split(".")) if "." in token else 1
                level = _detect_heading_level(line_stripped, depth) if "." in token else default_level
                headings.append((i, line_stripped, level, depth))
                break

    # Phase 2: build paragraph blocks between headings.
    paragraph_blocks = []
    heading_stack = []  # For parent tracking

    segments = []
    if headings:
        for idx, (line_idx, heading_text, level, depth) in enumerate(headings):
            heading_start = sum(len(l) + 1 for l in lines[:line_idx])
            content_start = heading_start + len(lines[line_idx]) + 1
            if idx + 1 < len(headings):
                next_line_idx = headings[idx + 1][0]
                content_end = sum(len(l) + 1 for l in lines[:next_line_idx])
            else:
                content_end = len(full_text)

            while heading_stack and heading_stack[-1][1] >= depth:
                heading_stack.pop()
            heading_stack.append((heading_text, depth))
            heading_path = [h[0] for h in heading_stack]
            segments.append((heading_text, list(heading_path), content_start, content_end))
    else:
        segments.append(("正文", ["正文"], 0, len(full_text)))

    for heading_text, heading_path, content_start, content_end in segments:
        segment_text = full_text[content_start:content_end]
        paragraphs = _split_paragraphs(segment_text, content_start)
        para_index = 1
        for paragraph_text, char_start, char_end in paragraphs:
            if len(paragraph_text) < min_section_chars and paragraph_blocks:
                continue
            page_start = _find_page_for_offset(char_start, page_map)
            page_end = _find_page_for_offset(char_end, page_map)
            paragraph_label = f"P{para_index}"
            paragraph_blocks.append(_make_section(
                paper_id=paper_id,
                graph_id=graph_id,
                heading=f"{heading_text} {paragraph_label}",
                heading_path=[*heading_path, paragraph_label],
                heading_level=HeadingLevel.PARAGRAPH_GROUP,
                text=paragraph_text,
                pages=pages,
                page_map=page_map,
                char_start=char_start,
                char_end=char_end,
                page_start=page_start,
                page_end=page_end,
            ))
            para_index += 1

    # If all sections were filtered out, fall back to whole-text
    if not paragraph_blocks:
        return [_make_section(
            paper_id=paper_id, graph_id=graph_id,
            heading="正文", heading_path=["正文"],
            heading_level=HeadingLevel.PARAGRAPH_GROUP,
            text=full_text, pages=pages, page_map=page_map,
            char_start=0, char_end=len(full_text),
        )]

    # Phase 3: link prev/next paragraph context
    # Note: next_section_id is left NULL during initial insert to avoid FK violations.
    # It will be updated after all sections are inserted into the database.
    for i, section in enumerate(paragraph_blocks):
        if i > 0:
            section.prev_section_id = paragraph_blocks[i - 1].section_id
        # next_section_id is set after insertion (see pipeline.py)

    return paragraph_blocks


def _split_paragraphs(text: str, base_offset: int) -> list[tuple[str, int, int]]:
    """Split a heading body into paragraph spans.

    Blank lines are treated as hard paragraph boundaries. Single newlines are
    preserved as softer boundaries, with short wrapped lines merged when they
    appear to be PDF line-break artifacts.
    """
    paragraphs: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_start: Optional[int] = None
    cursor = base_offset

    def flush(end_offset: int):
        nonlocal current, current_start
        if not current or current_start is None:
            current = []
            current_start = None
            return
        paragraph = "".join(current).strip()
        if paragraph:
            paragraphs.append((paragraph, current_start, end_offset))
        current = []
        current_start = None

    for raw_line in text.splitlines(keepends=True):
        line_start = cursor
        cursor += len(raw_line)
        stripped = raw_line.strip()
        if not stripped:
            flush(line_start)
            continue
        if current_start is None:
            current_start = line_start
            current.append(stripped)
            continue
        previous = current[-1]
        if previous.endswith(_PARAGRAPH_ENDINGS):
            flush(line_start)
            current_start = line_start
            current.append(stripped)
        else:
            current.append(stripped)

    flush(base_offset + len(text))
    return paragraphs


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
