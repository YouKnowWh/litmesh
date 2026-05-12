"""
Unified intermediate format for document parsing results.

LitMesh doesn't parse PDFs itself. It consumes ParsedDocument from
external parsers (Docling, MinerU, pdfplumber fallback) and builds
structured knowledge (SectionBlocks, source_spans, Claims, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ElementType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    SIDEBAR = "sidebar"
    ACTIVITY = "activity"
    EXERCISE = "exercise"
    FOOTER = "footer"
    HEADER = "header"
    PAGE_NUMBER = "page_number"
    TOC = "toc"
    UNKNOWN = "unknown"


@dataclass
class ParsedElement:
    """A single document element from an external parser."""
    element_id: str
    type: ElementType
    text: str
    page_start: int
    page_end: int = 0
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    level: int = 0           # Heading level (1=chapter, 2=section, etc.)
    order_index: int = 0     # Global sequential position
    confidence: float = 1.0  # Parser's confidence in this element
    role: str = ""           # LLM-assigned role: body/heading/sidebar/activity/exercise/etc.


@dataclass
class OutlineItem:
    """TOC / outline entry."""
    title: str
    level: int
    page: int
    element_id: str = ""
    toc_page: int = 0
    printed_page: int = 0
    body_page: int = 0
    normalized_title: str = ""
    confidence: float = 0.0
    source: str = ""


@dataclass
class DocumentOutlineNode:
    """Persistent TOC tree node — the structural backbone of a document.

    Unlike OutlineItem (parser intermediate), this is stored in the database
    and referenced by SectionBlock.toc_anchor_id.
    """
    outline_id: str           # "ol_{uuid4 hex[:12]}"
    paper_id: str
    graph_id: str
    title: str                # "第一章 社会主义革命：1949-1976"
    normalized_title: str = ""
    level: int = 1            # 1=章 2=节 3=小节
    toc_page: int = 0         # Page where this entry appears in the TOC
    printed_page: int = 0     # Page number printed in the TOC entry
    body_page: int = 0        # Actual body page in the PDF
    parent_outline_id: str = ""
    order_index: int = 0
    confidence: float = 0.0
    source: str = ""          # "parser_outline" | "text_toc" | "heading_fallback"


@dataclass
class QualityReport:
    """Parser quality assessment."""
    parser_name: str = ""
    parser_version: str = ""
    segmenter_name: str = ""
    total_elements: int = 0
    paragraph_count: int = 0
    heading_count: int = 0
    toc_detected: bool = False
    toc_entry_count: int = 0
    toc_page_count: int = 0
    toc_source: str = ""
    toc_alignment_confidence: float = 0.0
    toc_printed_page_offset: int = 0
    toc_unaligned_entries: int = 0
    outline_assigned_section_count: int = 0
    fallback_heading_count: int = 0
    keyword_heading_fallback_count: int = 0
    quality_gate_reasons: list[str] = field(default_factory=list)
    outline: list[dict] = field(default_factory=list)
    footer_header_removed: int = 0
    suspicious_long_heading_count: int = 0
    empty_fragment_count: int = 0
    average_paragraph_length: float = 0.0
    needs_structure_review: bool = False
    body_page_count: int = 0
    body_start_page: int = 0
    front_matter_block_count: int = 0
    too_short_paragraph_count: int = 0
    llm_window_count: int = 0
    llm_failed_windows: int = 0
    alignment_fail_count: int = 0
    rejected_segment_count: int = 0
    duplicate_overlap_count: int = 0
    containment_duplicate_count: int = 0
    sidebar_segment_count: int = 0
    activity_segment_count: int = 0
    exercise_segment_count: int = 0
    cross_window_merge_count: int = 0
    cross_page_paragraph_count: int = 0
    front_matter_segment_count: int = 0
    toc_segment_count: int = 0
    low_confidence_segment_count: int = 0
    rejected_segments: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Unified output from any document parser."""
    pages: list[dict] = field(default_factory=list)            # [{"page_num": 1, "text": "..."}]
    elements: list[ParsedElement] = field(default_factory=list)
    outline: list[OutlineItem] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parser_name: str = ""
    parser_version: str = ""
    quality_report: Optional[QualityReport] = None
    full_text: str = ""    # Concatenated full text for FTS/search
