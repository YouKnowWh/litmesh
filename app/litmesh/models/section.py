"""
SectionBlock: structured chapter/section/paragraph chunk.

Design rationale:
- SectionBlock is the bridge between raw PDF text and semantic claims.
- Stable structure indices (chapter_index, section_index, block_index, global_order_index)
  allow reliable document ordering even when heading text is garbled.
- The raw_text + summary pattern supports both full-text search and LLM extraction.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class HeadingLevel(str, Enum):
    TITLE = "title"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    SUBSUBSECTION = "subsubsection"
    PARAGRAPH_GROUP = "paragraph_group"


class StructureStatus(str, Enum):
    CLEAN = "clean"
    NEEDS_REVIEW = "needs_structure_review"
    RECONSTRUCTED = "reconstructed"


class SectionBlock(BaseModel):
    section_id: str = Field(default_factory=lambda: f"sec_{uuid4().hex[:12]}")
    graph_id: str
    paper_id: str

    # ---- Stable structure indices (v0.10) ----
    chapter_index: int = Field(default=0, description="1-based chapter number")
    section_index: int = Field(default=0, description="1-based section number within chapter")
    block_index: int = Field(default=0, description="1-based paragraph block number within section")
    global_order_index: int = Field(default=0, description="Global document order (1-based)")

    # ---- Display ----
    heading: str = Field(description="Section heading text")
    heading_path: list[str] = Field(default_factory=list)
    heading_level: HeadingLevel = HeadingLevel.PARAGRAPH_GROUP
    heading_confidence: float = Field(default=1.0, description="Confidence that heading is correct (0-1)")
    display_title: str = Field(default="", description="Fallback display title when heading is unreliable")
    structure_status: StructureStatus = StructureStatus.CLEAN

    # ---- Content ----
    raw_text: str = Field(description="Full text of this section")
    summary: str = Field(default="")

    # ---- Page reference ----
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # ---- TOC anchor ----
    toc_anchor_id: Optional[str] = Field(default=None, description="DocumentOutlineNode this block belongs to")
    toc_anchor_title: Optional[str] = Field(default=None, description="Title of the TOC node this block anchors to")

    # ---- Parser provenance ----
    parser_name: str = Field(default="", description="Document parser that produced this block")
    parser_element_id: str = Field(default="", description="Stable element id from the parser output")
    parser_confidence: float = Field(default=1.0, description="Parser confidence for this block")

    # ---- Concept keys ----
    concept_keys: list[str] = Field(default_factory=list)

    # ---- Linked-list navigation ----
    parent_section_id: Optional[str] = None
    prev_section_id: Optional[str] = None
    next_section_id: Optional[str] = None

    # ---- Hash ----
    content_hash: str = Field(default="")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
