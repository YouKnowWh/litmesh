"""
Section splitter: splits cleaned PDF text into paragraph-level SectionBlocks.

Strategy (revised for poor-layout Chinese PDFs):
1. Split on natural paragraph breaks (double newlines).
2. Skip heading detection — garbled text makes pattern matching unreliable.
3. Every block is a paragraph_group with stable indices.
4. First sentence of each block becomes the display_title.
5. The LLM extraction phase (v0.2) handles semantic structure.
"""

import hashlib
import json
import logging
import re
import time
from typing import Optional

from ..models.section import SectionBlock, HeadingLevel, StructureStatus
from .parsed_document import ElementType, ParsedDocument, ParsedElement, OutlineItem
from .toc_extractor import normalize_title

logger = logging.getLogger("litmesh.outline")


def _generate_outline_id(paper_id: str, item) -> str:
    """Generate a stable outline_id from a paper_id and an OutlineItem.

    Uses element_id if available, otherwise builds a deterministic id.
    """
    if getattr(item, "element_id", ""):
        return item.element_id
    # Fallback: deterministic id from paper + title + level
    import hashlib as _hl
    key = f"{paper_id}:{item.title}:{item.level}:{item.page}"
    return f"ol_{_hl.sha256(key.encode()).hexdigest()[:12]}"


def _outline_items_to_nodes(
    outline: list, paper_id: str, graph_id: str
) -> list:
    """Convert OutlineItem list to DocumentOutlineNode list for DB persistence."""
    from .parsed_document import DocumentOutlineNode
    nodes = []
    for i, item in enumerate(outline):
        ol_id = _generate_outline_id(paper_id, item)
        # Find parent: nearest prior item with lower level
        parent_id = ""
        for j in range(i - 1, -1, -1):
            if outline[j].level < item.level:
                parent_id = _generate_outline_id(paper_id, outline[j])
                break
        nodes.append(DocumentOutlineNode(
            outline_id=ol_id,
            paper_id=paper_id,
            graph_id=graph_id,
            title=item.title,
            normalized_title=item.normalized_title,
            level=item.level,
            toc_page=item.toc_page,
            printed_page=item.printed_page,
            body_page=item.body_page or item.page,
            parent_outline_id=parent_id,
            order_index=i + 1,
            confidence=item.confidence,
            source=item.source or "parser_outline",
        ))
    return nodes


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_page_for_offset(offset: int, page_map: list[tuple[int, int, int]]) -> int:
    for page_num, start, end in page_map:
        if start <= offset < end:
            return page_num
    return 1


def _build_page_map(pages: list[dict]) -> list[tuple[int, int, int]]:
    offset = 0
    page_map = []
    for p in pages:
        text = p["text"]
        page_map.append((p["page_num"], offset, offset + len(text)))
        offset += len(text) + 1
    return page_map


def _strip_page_number(text: str) -> str:
    """Remove leading page numbers like '144', 'P144', '第144页'.

    Does NOT strip years and dates ('2007 年', '1949 年', '21 世纪', etc.).
    """
    import re
    text = text.strip()
    # Strip leading page number patterns
    text = re.sub(r'^[pP]\s*\d{1,4}\s*', '', text)
    text = re.sub(r'^第?\d{1,4}[页頁]\s*', '', text)
    text = re.sub(r'^\d{1,4}\s*/\s*\d{1,4}\s*', '', text)
    # Strip bare leading numbers only when NOT followed by a date/time word
    text = re.sub(r'^\d{1,4}\s+(?!年|月|日|世纪|年代|万年|亿年)', '', text)
    return text.strip()


# ---- Special block classification ----

# Patterns for blocks that should retain their structural identity
_SPECIAL_BLOCK_PATTERNS = [
    (re.compile(r"^(前言|序言|序|Foreword|Preface|编者序|作者序|译者序|推荐序|代序)$"), "前言"),
    (re.compile(r"^(目录|目次|Contents|Table of Contents)$"), "目录"),
    (re.compile(r"^(编写说明|编辑说明|凡例|使用说明|导读)$"), "编写说明"),
    (re.compile(r"^(参考文献|References|Bibliography|引用文献)$"), "参考文献"),
    (re.compile(r"^(附录|Appendix|附錄)$"), "附录"),
    (re.compile(r"^(后记|跋|结语|後記|あとがき)$"), "后记"),
    (re.compile(r"^(注释|Notes|注釈|注解)$"), "注释"),
    (re.compile(r"^(译者简介|作者简介|编者简介)$"), "简介"),
]


def _classify_special_block(text: str, heading: str = "") -> tuple[str, str]:
    """Check if a block is a structural marker (TOC, preface, etc.).

    Returns (normalized_heading, display_title) if special, else ("", "").
    """
    candidate = (heading or text.split("\n")[0])[:40].strip()
    for pattern, label in _SPECIAL_BLOCK_PATTERNS:
        if pattern.match(candidate):
            return label, label
    return "", ""


def _is_quality_suspect(text: str) -> list[str]:
    """Return a list of quality concerns for a text block."""
    concerns = []
    if len(text) < 50:
        concerns.append("too_short")
    if text and not _FIRST_CHAR_RE.match(text[0]):
        concerns.append("suspicious_first_char")
    if text and text[0] in '，。！？、；：」』）】》':
        concerns.append("starts_with_punctuation")
    return concerns


_FIRST_CHAR_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿa-zA-Z0-9\"'“‘（\(《〈【\[‘“]")
_LOOKS_LIKE_PUB_INFO = re.compile(r"(出版社|出版|印刷|版次|印次|ISBN|定价| CIP |中国图书馆)")


def _first_sentence(text: str, max_len: int = 50) -> str:
    """Extract a meaningful first sentence for display_title fallback.

    Avoids producing fragments like '年 1 月，...' by:
    - Stripping page numbers first
    - Extending short first sentences with a second sentence
    - Falling back to a longer prefix if sentences are too short
    """
    text = _strip_page_number(text).strip()
    if not text:
        return ""

    # Find sentence boundaries
    boundaries = []
    for i, ch in enumerate(text):
        if ch in '。！？!?.\n':
            boundaries.append(i + 1)
        if i >= max_len * 3:
            break

    if not boundaries:
        return text[:max_len]

    # Take first sentence
    first = text[:boundaries[0]]
    if len(first) >= 10:
        return first

    # First sentence too short — try first two sentences
    if len(boundaries) >= 2:
        combined = text[:boundaries[1]]
        if len(combined) >= 15:
            return combined

    # Still too short — take first N chars
    return text[:max_len]


def split_sections(
    full_text: str,
    paper_id: str,
    graph_id: str,
    pages: list[dict],
    min_section_chars: int = 200,
    layout_lines: list[dict] | None = None,
    blocks: list[dict] | None = None,
) -> list[SectionBlock]:
    """Split extracted PDF text into paragraph-level SectionBlocks.

    If ``blocks`` (PyMuPDF text blocks) are provided, each becomes one
    SectionBlock — this preserves the PDF's natural paragraph divisions.
    Otherwise falls back to page-based splitting.
    """
    if blocks and len(blocks) >= 10:
        return _split_by_blocks(paper_id, graph_id, blocks, pages)

    page_map = _build_page_map(pages)

    # Split on paragraph breaks (double newlines, or single newline after CJK period)
    # Normalize: collapse 3+ newlines to 2, then split on double newlines
    text = re.sub(r'\n{3,}', '\n\n', full_text)
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    blocks = []
    global_order = 0
    char_offset = 0

    for para_text in raw_paragraphs:
        if len(para_text) < min_section_chars:
            char_offset += len(para_text) + 2
            continue

        global_order += 1
        page_start = _find_page_for_offset(char_offset, page_map)
        page_end = _find_page_for_offset(char_offset + len(para_text), page_map)
        display_title = _first_sentence(para_text)

        block = SectionBlock(
            paper_id=paper_id,
            graph_id=graph_id,
            heading=display_title[:60],
            heading_path=[display_title[:30]],
            heading_level=HeadingLevel.PARAGRAPH_GROUP,
            heading_confidence=0.3,  # Low: we don't trust auto-detected headings
            display_title=display_title,
            structure_status=StructureStatus.CLEAN,
            chapter_index=0,
            section_index=global_order,
            block_index=1,
            global_order_index=global_order,
            raw_text=para_text,
            page_start=page_start,
            page_end=page_end,
            content_hash=sha256(para_text),
        )
        blocks.append(block)
        char_offset += len(para_text) + 2

    # Link prev only (next is set by pipeline after DB insertion to avoid FK violations)
    for i, block in enumerate(blocks):
        if i > 0:
            block.prev_section_id = blocks[i-1].section_id

    # Fallback: if natural paragraphs produce too few blocks, split by pages
    if len(blocks) < max(10, len(pages) // 5):
        blocks = _split_by_pages(full_text, paper_id, graph_id, pages, page_map, min_section_chars)

    # Apply TOC-based chapter/section labels using second-occurrence detection
    toc = _parse_toc(full_text, pages)
    if toc:
        _apply_toc_labels(blocks, toc, full_text, pages)

    return blocks


def split_parsed_document(
    parsed: ParsedDocument,
    paper_id: str,
    graph_id: str,
    min_section_chars: int = 20,
    include_front_matter: bool = False,
    include_toc: bool = False,
) -> list[SectionBlock]:
    """Build paragraph SectionBlocks from a ParsedDocument.

    The external parser owns layout/element detection. LitMesh only turns
    content-bearing elements into stable paragraph blocks and keeps headings as
    context, not as primary chunks.
    """
    sections: list[SectionBlock] = []
    heading_path: list[str] = []
    outline = _usable_outline(parsed.outline)
    outline_first = bool(outline) and not _outline_is_order_only(parsed, outline)
    has_headings = (not outline_first) and any(e.type in (ElementType.TITLE, ElementType.HEADING) for e in parsed.elements)
    chapter_index = 0
    section_index = 0
    block_index = 0
    global_order = 0
    _context_heading = ""  # non-structural heading for the next section
    body_started = include_front_matter or outline_first or not has_headings

    content_types = {
        ElementType.PARAGRAPH,
        ElementType.LIST_ITEM,
        ElementType.TABLE,
        ElementType.CAPTION,
        ElementType.UNKNOWN,
    }
    skip_types = {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_NUMBER, ElementType.FIGURE}
    if not include_toc:
        skip_types.add(ElementType.TOC)

    # Roles that should NOT enter the body SectionBlock chain
    non_body_roles = {"sidebar", "activity", "exercise"}
    _toc_blocks: list[ParsedElement] = []

    for element in sorted(parsed.elements, key=lambda e: e.order_index):
        text = _normalize_element_text(element.text)
        if not text:
            continue

        reserved_label = _reserved_label(element)
        if reserved_label:
            if len(text) >= 6:
                global_order += 1
                sections.append(_make_reserved_section(
                    paper_id=paper_id,
                    graph_id=graph_id,
                    element=element,
                    text=text,
                    label=reserved_label,
                    global_order=global_order,
                    parser_name=parsed.parser_name,
                ))
            continue
        if element.type == ElementType.TOC or (getattr(element, "role", "") or "").lower() in {"toc", "front_matter"}:
            # Collect TOC blocks for recovery as a single "目录" section.
            # Do not let them enter the body extraction chain.
            _toc_blocks.append(element)
            continue

        if element.type in skip_types:
            continue

        # Skip non-body roles from the main section chain
        if getattr(element, "role", "") in non_body_roles:
            continue

        if element.type in (ElementType.TITLE, ElementType.HEADING):
            if outline_first:
                # TOC-derived outline owns heading_path. Parser/LLM headings are
                # not authoritative when a usable TOC exists.
                continue
            # Classify heading role — context/decorative/noise headings
            # should not enter the structural heading_path
            from ..repair.heading_classifier import HeadingClassifier, HeadingRole
            _hclassifier = HeadingClassifier()
            _hrole = _hclassifier.classify(text, heading_level=_normalize_heading_level(element))
            if _hrole in (HeadingRole.CONTEXT_HEADING, HeadingRole.DECORATIVE,
                          HeadingRole.FRONT_MATTER, HeadingRole.NOISE):
                # Store as context heading for the next section but don't
                # modify the structural heading_path
                _context_heading = text
                continue
            level = _normalize_heading_level(element)
            if level <= 1:
                chapter_index += 1
                section_index = 0
                block_index = 0
                heading_path = [text]
                body_started = True
            else:
                section_index += 1
                block_index = 0
                if not body_started:
                    body_started = True
                if heading_path:
                    heading_path = [heading_path[0], text]
                else:
                    heading_path = [text]
            continue

        if element.type not in content_types:
            continue
        if len(text) < min_section_chars:
            continue
        if not body_started and not include_front_matter:
            # Keep the parser's TOC/front matter out of extraction blocks.
            continue

        global_order += 1
        block_index += 1

        # Check for special structural blocks (TOC, preface, etc.)
        spec_heading, spec_title = _classify_special_block(text, heading_path[-1] if heading_path else "")
        if spec_heading:
            heading_path = [spec_heading]
            display_title = spec_title
        elif heading_path:
            display_title = heading_path[-1][:80]
        else:
            display_title = _first_sentence(text)

        title = display_title
        section = SectionBlock(
            paper_id=paper_id,
            graph_id=graph_id,
            heading=(_context_heading or (heading_path[-1] if heading_path else title))[:80],
            heading_path=[*heading_path] if heading_path else [],
            heading_level=HeadingLevel.PARAGRAPH_GROUP,
            heading_confidence=0.8 if heading_path else 0.3,
            display_title=title,
            structure_status=StructureStatus.CLEAN if heading_path else StructureStatus.RECONSTRUCTED,
            chapter_index=chapter_index,
            section_index=section_index,
            block_index=block_index,
            global_order_index=global_order,
            raw_text=text,
            page_start=element.page_start,
            page_end=element.page_end or element.page_start,
            parser_name=parsed.parser_name,
            parser_element_id=element.element_id,
            parser_confidence=element.confidence,
            content_hash=sha256(text),
        )
        _context_heading = ""  # consumed

        sections.append(section)

    if outline_first:
        _apply_outline_to_sections(sections, outline, parsed)

    # ---- TOC recovery ----
    toc_recovered = 0
    toc_recovery_mode = ""
    if _toc_blocks and not include_toc:
        toc_text = "\n\n".join(e.text.strip() for e in _toc_blocks if len(e.text.strip()) >= 6)
        if toc_text and _looks_like_real_toc(toc_text):
            toc_recovery_mode = "collected_toc_blocks"
            toc_recovered = len(_toc_blocks)
        elif _toc_blocks:
            # Check each block individually for partial TOC patterns
            partial = [e for e in _toc_blocks if _looks_like_toc_block(e.text)]
            if partial:
                toc_text = "\n\n".join(e.text.strip() for e in partial)
                toc_recovery_mode = "partial_toc_pattern"
                toc_recovered = len(partial)
        if toc_text and toc_recovered:
            global_order += 1
            sections.insert(0, SectionBlock(
                paper_id=paper_id, graph_id=graph_id,
                heading="目录", heading_path=["目录"],
                heading_level=HeadingLevel.SECTION,
                heading_confidence=1.0,
                display_title="目录",
                structure_status=StructureStatus.CLEAN,
                chapter_index=0, section_index=0, block_index=0,
                global_order_index=global_order,
                raw_text=toc_text,
                page_start=min(e.page_start for e in (_toc_blocks or [parsed.elements[0]]) if e.page_start),
                page_end=max(e.page_end or e.page_start for e in (_toc_blocks or [parsed.elements[0]])),
                parser_name=parsed.parser_name,
                content_hash=sha256(toc_text),
            ))
            import logging
            logger = logging.getLogger("litmesh.section")
            logger.info("toc_recovered mode=%s blocks=%d text_len=%d",
                        toc_recovery_mode, toc_recovered, len(toc_text))

    # ---- Paragraph stitching ----
    stitched = _stitch_paragraphs(sections)

    for i, section in enumerate(sections):
        if i > 0:
            section.prev_section_id = sections[i - 1].section_id

    _augment_display_titles(sections)

    _log_quality_concerns(sections, paper_id)

    return sections


def _augment_display_titles(sections: list[SectionBlock]):
    """Enhance display_title for context/structural headings with keywords.

    Uses structure/ module (language-agnostic) with repair/ fallback.
    """
    try:
        from ..structure.role_classifier import RoleClassifier
        from ..structure.title_augmenter import TitleAugmenter
        from ..structure.block_role import BlockRole

        classifier = RoleClassifier()
        augmenter = TitleAugmenter()
        n = len(sections)

        augmented = 0
        for i, section in enumerate(sections):
            heading = section.heading or ""
            heading_level = 1 if section.heading_path else (3 if heading else 0)
            classify_text = heading if heading else section.raw_text[:200]
            role = classifier.classify(
                text=classify_text,
                order_index=i,
                total_blocks=n,
                heading_level=heading_level,
            )
            section.block_role = role.value

            if role in (BlockRole.CONTEXT, BlockRole.STRUCTURAL):
                if heading:
                    kw = augmenter.extractor.extract(section.raw_text, heading)
                    if kw:
                        section.keyword_summary = kw
                    _, display = augmenter.generate(heading, section.raw_text, role.value)
                    section.display_title = display
                    augmented += 1
            elif role == BlockRole.FRONT:
                if heading:
                    section.display_title = heading

        if augmented:
            logger.info("display_title_augmented sections=%d augmented=%d",
                        len(sections), augmented)
    except ImportError:
        # Fallback to repair/
        try:
            from ..repair.keyword_extractor import KeywordExtractor as K
            from ..repair.title_augmenter import TitleAugmenter as T
            from ..repair.heading_classifier import HeadingClassifier as H, HeadingRole as HR
            extractor = K()
            augmenter = T(extractor)
            classifier = H()
            augmented = 0
            for section in sections:
                heading = section.heading or ""
                if not heading:
                    continue
                role = classifier.classify(heading, heading_level=1)
                if role in (HR.CONTEXT_HEADING, HR.STRUCTURAL_HEADING):
                    kw = extractor.extract(section.raw_text, heading, role.value)
                    if kw:
                        section.keyword_summary = kw
                        section.display_title = augmenter.augment(heading, section.raw_text, role.value)
                        augmented += 1
                elif role == HR.FRONT_MATTER:
                    section.display_title = heading
            if augmented:
                logger.info("display_title_augmented (repair fallback) sections=%d augmented=%d",
                            len(sections), augmented)
        except ImportError:
            pass


def _log_quality_concerns(sections: list[SectionBlock], paper_id: str):
    """Log suspicious blocks for downstream repair analysis."""
    qlog = logging.getLogger("litmesh.section")
    suspicious = 0
    for section in sections:
        concerns = _is_quality_suspect(section.raw_text)
        if concerns:
            suspicious += 1
            qlog.debug(
                "quality_concern section=%s heading=%s concerns=%s preview=%s",
                section.section_id, section.heading[:40], ",".join(concerns),
                section.raw_text[:60],
            )
    if suspicious:
        qlog.info("split_quality paper=%s sections=%d suspicious=%d",
                  paper_id, len(sections), suspicious)


def _reserved_label(element: ParsedElement) -> str:
    role = (getattr(element, "role", "") or "").lower()
    text = element.text.strip()
    if (element.type == ElementType.TOC or role == "toc") and _looks_like_real_toc(text):
        return "目录"
    if role == "front_matter":
        if "前言" in text[:80]:
            return "前言"
        if "编写" in text[:80] and "说明" in text[:120]:
            return "编写说明"
    return ""


def _looks_like_real_toc(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(line in {"目录", "目 录", "目　录"} for line in lines):
        return True
    toc_like = 0
    for line in lines:
        if re.search(r"[\.\u2026·•]{2,}\s*\d{1,4}\s*$", line):
            toc_like += 1
        elif re.search(r"第\s*[一二三四五六七八九十\d]+\s*[章节].+\s+\d{1,4}\s*$", line):
            toc_like += 1
    return toc_like >= 1 and len(text) < 4000


def _looks_like_toc_block(text: str) -> bool:
    """Check a single block for TOC dot-leader patterns."""
    text = text.strip()
    if not text or len(text) < 10:
        return False
    if "目录" in text[:10] or "目 录" in text[:10]:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    toc_lines = 0
    for ln in lines:
        if re.search(r"[\.…·•]{2,}\s*\d{1,4}\s*$", ln):
            toc_lines += 1
        elif re.search(r"第\s*[一二三四五六七八九十\d]+\s*[章节].+\s+\d{1,4}\s*$", ln):
            toc_lines += 1
    return toc_lines >= 1 and len(text) < 4000


_SENTENCE_END_RE = re.compile(r'[。！？.!?」】)）]$')


def _stitch_paragraphs(sections: list) -> int:
    """Merge adjacent body paragraphs that look like continuations.

    Only merges when:
    - Both sections have no heading_path (pure body paragraphs)
    - Previous section does not end with sentence-ending punctuation
    - Next section does not start with a heading/activity/exercise pattern
    - Both are reasonably sized (not too short, not too long combined)
    """
    import logging
    logger = logging.getLogger("litmesh.section")
    if len(sections) < 2:
        return 0

    merged = 0
    candidates = 0
    _HEADING_START = re.compile(
        r'^(第\s*[一二三四五六七八九十\d]+\s*[章节]|思考|讨论|探究|活动|练习|复习|本章|小结|实验|第[一二三四五六七八九十\d]+节|第[一二三四五六七八九十\d]+单元)'
    )
    _EXERCISE_START = re.compile(r'^[\d①②③④⑤⑥⑦⑧⑨⑩]+\s*[\.\、．)\s]')
    # Structural blocks that should never be merged away
    _RESERVED_HEADINGS = {"目录", "前言", "编写说明", "参考文献", "附录", "后记", "注释", "简介"}

    def _is_reserved(section) -> bool:
        return any(
            section.heading == h or (section.heading_path and section.heading_path[0] == h)
            for h in _RESERVED_HEADINGS
        )

    i = 1
    while i < len(sections):
        prev = sections[i - 1]
        curr = sections[i]

        if _is_reserved(prev) or _is_reserved(curr):
            i += 1
            continue
        if prev.structure_status != StructureStatus.CLEAN and prev.structure_status != StructureStatus.RECONSTRUCTED:
            i += 1
            continue
        if curr.structure_status != StructureStatus.CLEAN and curr.structure_status != StructureStatus.RECONSTRUCTED:
            i += 1
            continue

        prev_text = prev.raw_text.strip()
        curr_text = curr.raw_text.strip()

        # Short block merge: very short blocks with same heading_path
        same_heading = prev.heading_path == curr.heading_path and prev.heading_path
        if same_heading and (len(prev_text) < 50 or len(curr_text) < 50) and (len(prev_text) + len(curr_text) < 2000):
            prev.raw_text = prev_text + "\n\n" + curr_text
            prev.page_end = max(prev.page_end or 0, curr.page_end or 0)
            prev.content_hash = sha256(prev.raw_text)
            sections.pop(i)
            merged += 1
            logger.debug("stitch_short section=%s heading=%s len_prev=%d len_curr=%d",
                         prev.section_id, prev.heading_path[0] if prev.heading_path else "?",
                         len(prev_text), len(curr_text))
            continue

        # Only merge pure body paragraphs for continuation stitching
        if (prev.heading_path or curr.heading_path):
            i += 1
            continue

        # Size checks
        if len(prev_text) < 20 or len(curr_text) < 20:
            i += 1
            continue
        if len(prev_text) + len(curr_text) > 800:
            i += 1
            continue

        # Previous must not end with terminal punctuation
        if _SENTENCE_END_RE.search(prev_text[-5:]):
            i += 1
            continue

        # Next must not start with heading/exercise pattern
        if _HEADING_START.match(curr_text):
            i += 1
            continue
        if _EXERCISE_START.match(curr_text) and len(curr_text) < 100:
            i += 1
            continue

        candidates += 1
        # Stitch: merge curr into prev
        prev.raw_text = prev_text + "\n" + curr_text
        prev.page_end = max(prev.page_end or 0, curr.page_end or 0)
        prev.content_hash = sha256(prev.raw_text)
        prev.display_title = prev.display_title or curr.display_title
        # Remove curr from list
        sections.pop(i)
        merged += 1
        # Don't increment i — re-check prev against the new next

    if candidates:
        logger.info("paragraph_stitch_summary merged=%d candidates=%d total=%d",
                    merged, candidates, len(sections))
    return merged


def _make_reserved_section(
    paper_id: str,
    graph_id: str,
    element: ParsedElement,
    text: str,
    label: str,
    global_order: int,
    parser_name: str,
) -> SectionBlock:
    return SectionBlock(
        paper_id=paper_id,
        graph_id=graph_id,
        heading=label,
        heading_path=[label],
        heading_level=HeadingLevel.SECTION,
        heading_confidence=1.0,
        display_title=label,
        structure_status=StructureStatus.CLEAN,
        chapter_index=0,
        section_index=0,
        block_index=global_order,
        global_order_index=global_order,
        raw_text=text,
        page_start=element.page_start,
        page_end=element.page_end or element.page_start,
        parser_name=parser_name,
        parser_element_id=element.element_id,
        parser_confidence=element.confidence,
        content_hash=sha256(text),
    )


def _usable_outline(outline: list[OutlineItem]) -> list[OutlineItem]:
    items = [
        item for item in outline
        if item.title and item.level > 0 and (item.body_page or item.page)
    ]
    return sorted(items, key=lambda i: (i.body_page or i.page, i.level, i.title))


def _outline_is_order_only(parsed: ParsedDocument, outline: list[OutlineItem]) -> bool:
    """Markdown-style parsers often have headings but no real page positions."""
    if not outline:
        return False
    parser_name = (parsed.parser_name or "").lower()
    if parser_name not in {
        "markdown",
        "external_markdown",
        "mineru_api",
        "mineru_markdown",
        "marker_markdown",
        "docling",
    }:
        return False
    positions = {(item.body_page or item.page or 0) for item in outline}
    return len(positions) <= 1


def _apply_outline_to_sections(sections: list[SectionBlock], outline: list[OutlineItem], parsed: ParsedDocument):
    if not sections or not outline:
        return

    chapter_entries = [item for item in outline if item.level <= 1]
    assigned = 0
    fallback = 0
    keyword_fallback = 0

    # Detect order-based anchoring: when all sections share page=1 (markdown-style),
    # use global_order_index instead of page number for outline matching
    pages = {s.page_start for s in sections if not _is_reserved_section(s)}
    use_order = len(pages) <= 1 and (1 in pages or None in pages)
    if use_order:
        outline = _refine_order_only_outline_from_sections(outline, sections)

    for section in sections:
        if _is_reserved_section(section):
            continue
        if use_order:
            # Order-based: match by section's global_order_index range
            section_order = section.global_order_index or 0
            active = _active_outline_path_by_order(outline, section_order)
        else:
            page = section.page_start or 0
            active = _active_outline_path(outline, page, section.raw_text)
        if not active:
            fallback += 1
            _write_outline_audit(section.paper_id, "heading_fallback_used",
                                 section_id=section.section_id,
                                 page_start=section.page_start,
                                 text_head=section.raw_text[:60])
            continue

        path = [item.title for item in active]
        chapter = next((item for item in reversed(active) if item.level <= 1), None)
        subsection = next((item for item in reversed(active) if item.level >= 2), None)

        section.heading_path = path
        section.heading = " > ".join(path)
        section.heading_confidence = min(0.95, max(item.confidence for item in active) if active else 0.7)
        section.structure_status = StructureStatus.CLEAN

        # Bind to the finest-matching TOC node
        anchor = active[-1] if active else None
        if anchor:
            section.toc_anchor_id = anchor.element_id or _generate_outline_id(
                section.paper_id, anchor
            )
            section.toc_anchor_title = anchor.title
        if chapter:
            try:
                section.chapter_index = chapter_entries.index(chapter) + 1
            except ValueError:
                section.chapter_index = 0
        if subsection:
            siblings = [
                item for item in outline
                if item.level >= 2 and _same_chapter(outline, item, chapter)
            ]
            try:
                section.section_index = siblings.index(subsection) + 1
            except ValueError:
                section.section_index = section.section_index or 0
        else:
            section.section_index = 0
        assigned += 1
        _write_outline_audit(section.paper_id, "section_outline_assigned",
                             section_id=section.section_id,
                             page_start=section.page_start,
                             heading_path=path)

    # Re-number paragraph blocks inside each TOC path.
    counters: dict[tuple[str, ...], int] = {}
    for section in sections:
        if _is_reserved_section(section):
            continue
        key = tuple(section.heading_path)
        counters[key] = counters.get(key, 0) + 1
        section.block_index = counters[key]

    if parsed.quality_report:
        parsed.quality_report.outline_assigned_section_count = assigned
        parsed.quality_report.fallback_heading_count = fallback
        parsed.quality_report.keyword_heading_fallback_count = keyword_fallback
        if fallback:
            parsed.quality_report.quality_gate_reasons.append("heading_fallback_used")
        parsed.quality_report.outline = [
            {
                "title": item.title,
                "level": item.level,
                "page": item.page,
                "toc_page": item.toc_page,
                "printed_page": item.printed_page,
                "body_page": item.body_page,
                "confidence": item.confidence,
                "source": item.source,
            }
            for item in outline
        ]

    logger.info(
        "outline_apply_done sections=%d chapter_assigned=%d section_assigned=%d fallback_heading=%d",
        len(sections),
        sum(1 for s in sections if s.chapter_index > 0),
        sum(1 for s in sections if s.section_index > 0),
        fallback,
    )


def _is_reserved_section(section: SectionBlock) -> bool:
    return bool(section.heading_path and _compact_label(section.heading_path[0]) in {"目录", "前言", "编写说明"})


def _compact_label(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _active_outline_path(outline: list[OutlineItem], page: int, text: str) -> list[OutlineItem]:
    candidates = [item for item in outline if (item.body_page or item.page) <= page]
    if not candidates:
        return []
    same_page = [item for item in candidates if (item.body_page or item.page) == page]
    text_norm = normalize_title(text)
    matching = [item for item in same_page if item.normalized_title and item.normalized_title in text_norm]
    latest = matching[-1] if matching else candidates[-1]
    path = []
    chapter = _nearest_prior(outline, latest, max_level=1)
    if chapter and chapter is not latest:
        path.append(chapter)
    path.append(latest)
    # Collapse duplicate chapter path.
    deduped = []
    seen = set()
    for item in path:
        key = (item.level, item.title)
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _active_outline_path_by_order(
    outline: list[OutlineItem], section_order: int
) -> list[OutlineItem]:
    """Find the active outline item for a section by order_index proximity.

    Used for markdown-style documents where page numbers are unreliable (all page=1).
    Matches the nearest prior outline item whose body_page (order_index) <= section_order.
    """
    if not outline:
        return []
    # Find the latest outline item whose body_page <= section_order
    prior = [item for item in outline if (item.body_page or 0) <= section_order]
    if not prior:
        return []
    latest = prior[-1]
    path = []
    chapter = _nearest_prior(outline, latest, max_level=1)
    if chapter and chapter is not latest:
        path.append(chapter)
    path.append(latest)
    return path


def _refine_order_only_outline_from_sections(
    outline: list[OutlineItem], sections: list[SectionBlock]
) -> list[OutlineItem]:
    """Refine markdown-style outline body orders using section text evidence.

    Some markdown/MinerU outputs preserve the TOC and chapter titles, but the
    heading element order can drift badly in the latter half of the document.
    When that happens, later TOC entries end up with body orders far beyond the
    last content block, causing entire chapters to be absorbed into the previous
    section. This pass uses the actual section text/display titles to pull TOC
    entries back toward the earliest plausible matching section order.
    """
    ordered = [
        item
        for _, item in sorted(
            enumerate(outline),
            key=lambda pair: (
                getattr(pair[1], "order_index", pair[0]) or pair[0],
                pair[1].level,
                pair[1].title,
            ),
        )
    ]
    content_sections = [
        s for s in sorted(sections, key=lambda s: s.global_order_index or 0)
        if not _is_reserved_section(s)
    ]
    if not content_sections:
        return ordered

    max_section_order = max((s.global_order_index or 0) for s in content_sections)
    # If all outline anchors are already comfortably inside section range,
    # keep the original ordering untouched.
    if max((item.body_page or 0) for item in ordered) <= max_section_order:
        return ordered

    searchable = []
    for s in content_sections:
        title_text = normalize_title(" ".join(filter(None, [s.heading, s.display_title])))
        body_text = normalize_title((s.raw_text or "")[:400])
        searchable.append((s.global_order_index or 0, title_text, body_text))
    available_orders = [order for order, _, _ in searchable]

    def _find_match(item: OutlineItem, min_order: int) -> int:
        keys = _outline_search_keys(item.title)
        if not keys:
            return 0
        best_order = 0
        best_score = 0
        for order, title_text, body_text in searchable:
            if order <= min_order:
                continue
            score = 0
            for key in keys:
                if len(key) < 2:
                    continue
                if key in title_text:
                    score = max(score, 100 + len(key))
                elif key in body_text:
                    score = max(score, len(key))
            if score > best_score:
                best_order = order
                best_score = score
                if score >= 104:
                    break
        return best_order

    last_order = 0
    for item in ordered:
        if item.level >= 2:
            matched_order = _find_match(item, last_order)
            if not matched_order and available_orders:
                matched_order = next((order for order in available_orders if order > last_order), 0)
            if matched_order and (not item.body_page or item.body_page > matched_order):
                item.body_page = matched_order
                item.page = matched_order
            if not item.body_page:
                item.body_page = max(1, last_order + 1)
                item.page = item.body_page
            if item.body_page <= last_order:
                item.body_page = last_order + 1
                item.page = item.body_page
            last_order = item.body_page

    # Pull chapter entries up to just before their first child section.
    for i, item in enumerate(ordered):
        if item.level > 1:
            continue
        child_orders = [
            child.body_page or 0
            for child in ordered[i + 1 :]
            if child.level > item.level
        ]
        next_chapter_idx = next((j for j in range(i + 1, len(ordered)) if ordered[j].level <= item.level), len(ordered))
        child_orders = [
            child.body_page or 0
            for child in ordered[i + 1 : next_chapter_idx]
            if child.level > item.level
        ]
        if child_orders:
            first_child = min(child_orders)
            candidate = max(0, first_child - 1)
            if not item.body_page or item.body_page > candidate:
                item.body_page = candidate
                item.page = candidate

    # Final monotonic cleanup
    prev = -1
    prev_level = 0
    for item in ordered:
        current = item.body_page or item.page or 0
        if current < prev or (current == prev and item.level <= prev_level):
            current = prev + 1
            item.body_page = current
            item.page = current
        prev = current
        prev_level = item.level
    return ordered


def _outline_search_keys(title: str) -> list[str]:
    """Generate normalized search keys for fuzzy section-to-outline matching."""
    core = normalize_title(title)
    core = re.sub(r"^第[一二三四五六七八九十\d]+[章节篇部]", "", core)
    core = re.sub(r"^(chapter|section|part)[\divx]+", "", core)
    core = core.strip()
    if not core:
        return []

    candidates = [core]
    if "的" in core:
        suffix = core.split("的", 1)[-1]
        if len(suffix) >= 3:
            candidates.append(suffix)
    if core.startswith("细胞") and len(core) > 4:
        candidates.append(core[2:])
    for part in re.split(r"[和与及、/（）()《》“”\"'·\-]+", core):
        part = part.strip()
        if len(part) >= 2:
            candidates.append(part)
    for ascii_term in re.findall(r"[a-z0-9]{2,}", core):
        candidates.append(ascii_term)

    seen = set()
    ordered = []
    for key in sorted(candidates, key=len, reverse=True):
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _nearest_prior(outline: list[OutlineItem], item: OutlineItem, max_level: int) -> OutlineItem | None:
    item_page = item.body_page or item.page
    prior = [
        candidate for candidate in outline
        if candidate.level <= max_level and (candidate.body_page or candidate.page) <= item_page
    ]
    return prior[-1] if prior else None


def _same_chapter(outline: list[OutlineItem], item: OutlineItem, chapter: OutlineItem | None) -> bool:
    if chapter is None:
        return True
    return _nearest_prior(outline, item, max_level=1) == chapter


def _write_outline_audit(paper_id: str, event: str, **kwargs):
    if not paper_id:
        return
    try:
        from pathlib import Path
        path = Path("logs/parse_audit") / f"{paper_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": event,
                "paper_id": paper_id,
                "timestamp": time.time(),
                **kwargs,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _normalize_element_text(text: str) -> str:
    text = _strip_page_number(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _normalize_heading_level(element: ParsedElement) -> int:
    if element.level > 0:
        return element.level
    text = element.text.strip()
    if re.match(r"^第\s*[一二三四五六七八九十\d]+\s*章", text):
        return 1
    if re.match(r"^第\s*[一二三四五六七八九十\d]+\s*节", text):
        return 2
    return 2


def _structure_label(chapter_index: int, section_index: int, block_index: int, global_order: int) -> str:
    if chapter_index > 0:
        return f"C{chapter_index:02d}-S{section_index:02d}-P{block_index:03d}"
    return f"P{global_order:03d}"


def _split_by_blocks(paper_id, graph_id, blocks, pages):
    """Create a SectionBlock per PyMuPDF text block.

    Each block is a natural paragraph/heading from the PDF layout.
    Text flows continuously within a block — no mid-paragraph cuts.
    """
    sections = []
    for i, b in enumerate(blocks):
        order = i + 1
        text = b["text"]
        display_title = _first_sentence(text)
        section = SectionBlock(
            paper_id=paper_id, graph_id=graph_id,
            heading=display_title[:60], heading_path=[display_title[:30]],
            heading_level=HeadingLevel.PARAGRAPH_GROUP,
            heading_confidence=0.3, display_title=display_title,
            structure_status=StructureStatus.CLEAN,
            chapter_index=0, section_index=order, block_index=1,
            global_order_index=order,
            raw_text=text,
            page_start=b.get("page_num", 1), page_end=b.get("page_num", 1),
            content_hash=sha256(text),
        )
        sections.append(section)

    for i, s in enumerate(sections):
        if i > 0:
            s.prev_section_id = sections[i-1].section_id
    return sections


def _make_block(paper_id, graph_id, text, page_map, pages, order, pg_start, pg_end, display_title):
    return SectionBlock(
        paper_id=paper_id, graph_id=graph_id,
        heading=display_title[:60], heading_path=[display_title[:30]],
        heading_level=HeadingLevel.PARAGRAPH_GROUP,
        heading_confidence=0.3, display_title=display_title,
        structure_status=StructureStatus.CLEAN,
        chapter_index=0, section_index=order, block_index=1, global_order_index=order,
        raw_text=text, page_start=pg_start, page_end=pg_end,
        content_hash=sha256(text),
    )


def _split_by_pages(full_text, paper_id, graph_id, pages, page_map, min_chars):
    """Fallback: split by page boundaries when paragraph splitting fails."""
    blocks = []
    order = 0
    for p in pages:
        text = p["text"].strip()
        if len(text) < min_chars:
            continue
        order += 1
        display_title = _first_sentence(text)
        blocks.append(_make_block(
            paper_id, graph_id, text, page_map, pages,
            order, p["page_num"], p["page_num"], display_title
        ))
    # Link prev
    for i, b in enumerate(blocks):
        if i > 0:
            b.prev_section_id = blocks[i-1].section_id
    return blocks


# ---- TOC parsing ----

# Unicode spaces: fullwidth (U+3000), em-space (U+2003), en-space (U+2002), thin-space (U+2009), hair-space (U+200A), ideographic-space (U+3000)
_UNI_SPACE = r'[\s　    ​　  ]'
_CHAPTER_RE = re.compile(r'^第\s*([一二三四五六七八九十\d]+)\s*章' + _UNI_SPACE + r'*(.+?)(?:\s*\.{2,}.*)?$')
_SECTION_RE = re.compile(r'^第\s*([一二三四五六七八九十\d]+)\s*节' + _UNI_SPACE + r'*(.+?)(?:\s*\.{2,}.*)?$')
_PAGE_NUM_RE = re.compile(r'\.{2,}\s*(\d+)\s*$|^第[章节].+?(\d+)\s*$')


def _parse_toc(full_text: str, pages: list[dict]) -> list[dict]:
    """Parse table of contents from text to get chapter/section page numbers.

    Returns list of {level: 1|2, title: str, page: int}
    """
    # Find the TOC section — look for "目录" or dense chapter listings
    lines = full_text.split('\n')
    toc_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ('目录', '目 录', '目 录') or (stripped.startswith('目') and '录' in stripped and len(stripped) < 10):
            toc_start = i
            break

    if toc_start < 0:
        # Try to find TOC by looking for "第1章" patterns early in the text
        for i, line in enumerate(lines[:200]):
            if _CHAPTER_RE.match(line.strip()):
                toc_start = i - 1
                break

    if toc_start < 0:
        return []

    # Scan TOC area + all lines after TOC for chapter/section entries
    entries = []
    chapter_titles = set()

    # First pass: from TOC start, scan ~200 lines
    for i in range(toc_start, min(toc_start + 200, len(lines))):
        _collect_toc_line(lines[i].strip(), entries, chapter_titles)

    # Second pass: if we're missing chapters (e.g., ch4-6), scan more text
    # looking specifically for chapter headers that weren't found
    found_chapters = {e['title'] for e in entries if e['level'] == 1}
    for i in range(toc_start + 200, min(len(lines), toc_start + 2000)):
        line = lines[i].strip()
        if not line:
            continue
        cm = _CHAPTER_RE.match(line)
        if cm:
            title = f"第{cm.group(1)}章 {cm.group(2).strip()}"
            if title not in found_chapters:
                pg = _extract_page_number(line)
                if pg and pg > 0:
                    entries.append({'level': 1, 'title': title, 'page': pg})
                    found_chapters.add(title)
                    chapter_titles.add(title)
        sm = _SECTION_RE.match(line)
        if sm:
            title = f"第{sm.group(1)}节 {sm.group(2).strip()}"
            pg = _extract_page_number(line)
            if pg and pg > 0 and pg < 300:
                entries.append({'level': 2, 'title': title, 'page': pg})

    return entries


def _collect_toc_line(line, entries, chapter_titles):
    """Parse a single line as a potential TOC entry."""
    if not line:
        return
    cm = _CHAPTER_RE.match(line)
    if cm:
        pg = _extract_page_number(line)
        if pg and pg > 0:
            title = f"第{cm.group(1)}章 {cm.group(2).strip()}"
            if title not in chapter_titles:
                entries.append({'level': 1, 'title': title, 'page': pg})
                chapter_titles.add(title)
        return

    sm = _SECTION_RE.match(line)
    if sm:
        pg = _extract_page_number(line)
        if pg and pg > 0:
            title = f"第{sm.group(1)}节 {sm.group(2).strip()}"
            entries.append({'level': 2, 'title': title, 'page': pg})


def _is_likely_toc(line: str) -> bool:
    """Check if a line looks like a TOC entry (has trailing dots/page numbers)."""
    return bool(re.search(r'\.{3,}\s*\d+', line))


def _extract_page_number(line: str) -> int | None:
    """Extract trailing page number from a TOC line."""
    # "........ 42" or "........42"
    m = re.search(r'\.{2,}\s*(\d+)\s*$', line)
    if m:
        return int(m.group(1))
    # "标题 42" at end
    m = re.search(r'\s+(\d{1,3})\s*$', line)
    if m and int(m.group(1)) < 300:
        return int(m.group(1))
    return None


def _find_body_page_for_title(full_text: str, title_key: str, pages: list[dict]) -> int:
    """Find the PDF page where a title appears for the SECOND time.

    The first occurrence is in the TOC. The second is in the body.
    Returns the page number, or 0 if not found.
    """
    occurrences = []
    key = title_key.replace(' ', '').replace(' ', '').replace('　', '')
    for p in pages:
        text = p['text'].replace(' ', '').replace(' ', '').replace('　', '').replace('\n', '')
        if key in text:
            occurrences.append(p['page_num'])
    return occurrences[1] if len(occurrences) >= 2 else (occurrences[0] if occurrences else 0)


def _apply_toc_labels(blocks: list, toc: list[dict], full_text: str = "", pages: list = None):
    """Apply chapter/section names from TOC to blocks.

    Uses 'second occurrence' heuristic: the first time a title appears is in
    the TOC, the second time is in the body. The body page becomes the boundary.
    Everything before the first body page is front matter.
    """
    if not toc or not blocks:
        return

    chapters = [e for e in toc if e['level'] == 1]
    sections = [e for e in toc if e['level'] == 2]
    if not chapters:
        return

    # Use second-occurrence to find body page for each chapter
    page_map = {}
    if pages and full_text:
        for e in toc:
            key = e['title'][:8]
            body_page = _find_body_page_for_title(full_text, key, pages)
            if body_page > 0:
                page_map[e['title']] = body_page

    # The first body chapter page is where front matter ends
    body_starts = sorted([p for p in page_map.values() if p > 0])
    first_content_page = body_starts[0] if body_starts else 8

    for block in blocks:
        page = block.page_start or 0

        if page < first_content_page:
            block.heading_path = ['前言/目录']
            block.heading = '前言/目录'
            block.heading_confidence = 0.7
            block.chapter_index = 0
            continue

        # Find chapter: find the latest chapter whose body page <= current page
        chapter_title = ''
        chapter_body_page = 0
        for c in chapters:
            cp = page_map.get(c['title'], 0)
            if cp > 0 and cp <= page:
                chapter_title = c['title']
                chapter_body_page = cp

        # Find section: within the current chapter, find the latest section
        section_title = ''
        for s in sections:
            sp = page_map.get(s['title'], 0)
            if sp > 0 and sp >= chapter_body_page and sp <= page:
                section_title = s['title']

        # If we have chapter but no section, inherit the latest section within the same chapter
        if chapter_title and not section_title:
            for s in reversed(sections):
                sp = page_map.get(s['title'], 0)
                if sp > 0 and sp >= chapter_body_page and sp <= page:
                    section_title = s['title']
                    break

        heading_path = []
        if chapter_title:
            heading_path.append(chapter_title)
        if section_title:
            heading_path.append(section_title)

        if heading_path:
            block.heading_path = heading_path
            block.heading = ' > '.join(heading_path)
            block.heading_confidence = 0.9
            block.chapter_index = chapters.index(next((c for c in chapters if c['title'] == chapter_title), chapters[0])) + 1
    return blocks
