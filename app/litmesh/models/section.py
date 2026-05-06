"""
SectionBlock: structured chapter/section chunk.

Design rationale:
- SectionBlock is the bridge between raw PDF text and semantic claims.
- It carries structural navigation (prev/next/parent) for trace_mode traversal.
- The raw_text + summary pattern supports both full-text search and LLM extraction.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class HeadingLevel(str, Enum):
    """Semantic heading level. Not always 1:1 with markdown heading depth."""
    TITLE = "title"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    SUBSUBSECTION = "subsubsection"
    PARAGRAPH_GROUP = "paragraph_group"


class SectionBlock(BaseModel):
    section_id: str = Field(default_factory=lambda: f"sec_{uuid4().hex[:12]}")
    graph_id: str
    paper_id: str

    # Structural position
    heading: str = Field(description="Section heading text")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Full heading hierarchy, e.g. ['Chapter 3', '3.1 Methods', '3.1.1 Sample']"
    )
    heading_level: HeadingLevel = HeadingLevel.SECTION

    # Content
    raw_text: str = Field(description="Full text of this section")
    summary: str = Field(default="", description="LLM-generated section summary (v0.2+)")

    # Page reference (1-based, from PDF)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # Concept keys surfaced in this section (populated after extraction)
    concept_keys: list[str] = Field(default_factory=list)

    # Linked-list navigation within the same paper
    parent_section_id: Optional[str] = None
    prev_section_id: Optional[str] = None
    next_section_id: Optional[str] = None

    # Content hash for change detection
    content_hash: str = Field(default="", description="SHA256 of raw_text")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
