"""
SourceSpan: the anchor that ties every claim/evidence/limitation back to source text.

Principle 5: No source_span, no active claim.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SpanType(str, Enum):
    """Type of span: what kind of text region this points to."""
    PAGE = "page"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SECTION = "section"
    CUSTOM = "custom"


class SpanPosition(BaseModel):
    """Precise position in source document.

    Design rationale:
    - Multiple position representations so the system can rebuild spans
      if the document format changes (e.g., PDF -> extracted text -> markdown).
    - char_offset is the canonical position within the normalized text.
    - page/line are human-readable references.
    """
    char_start: int = Field(description="Character offset start in normalized text")
    char_end: int = Field(description="Character offset end in normalized text")
    page_start: Optional[int] = Field(default=None, description="Starting page number (1-based)")
    page_end: Optional[int] = Field(default=None, description="Ending page number (1-based)")
    line_start: Optional[int] = Field(default=None, description="Starting line number in normalized text")
    line_end: Optional[int] = Field(default=None, description="Ending line number in normalized text")
    # For PDFs, store the bbox if available (future use)
    pdf_bbox: Optional[str] = Field(default=None, description="PDF bounding box string (future)")


class SourceSpan(BaseModel):
    """A verified anchor tying extracted content back to source text.

    Why SourceSpan is a first-class object, not just a field:
    1. It must survive across extraction runs, format migrations, and corpus updates.
    2. It's the only way to audit "did the LLM hallucinate this claim or is it real?"
    3. It enables trace_mode traversal (claim -> section -> paper).
    4. It makes LLM extraction failures visible (extraction that can't produce a span
       goes to orphan/candidate, not active).
    """
    span_id: str = Field(
        default_factory=lambda: f"span_{uuid4().hex[:12]}",
        description="Unique span identifier"
    )
    paper_id: str = Field(description="FK to PaperCard")
    section_id: Optional[str] = Field(default=None, description="FK to SectionBlock (nullable for paper-level spans)")
    span_type: SpanType = Field(default=SpanType.PARAGRAPH)

    # The actual source text this span points to.
    # Stored here so the span is self-contained: you can verify the extraction
    # without re-opening the PDF.
    source_text: str = Field(description="Verbatim source text at this span")

    # Position tracking
    position: SpanPosition

    # Hash of source_text for dedup and integrity checks
    content_hash: str = Field(default="", description="SHA256 of source_text for integrity")

    # Normalized form (optional, for better matching)
    normalized_text: Optional[str] = Field(
        default=None,
        description="Normalized version of source_text (lowercase, whitespace-collapsed)"
    )

    # Audit
    verified: bool = Field(default=False, description="Has a human verified this span?")
    verified_by: Optional[str] = Field(default=None, description="Who verified")
    verified_at: Optional[datetime] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
