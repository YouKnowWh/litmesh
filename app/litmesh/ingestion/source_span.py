"""
Source span constructor for ingestion pipeline.

Creates SourceSpan records from section text with precise character offsets.
"""

import hashlib
from ..models.source_span import SourceSpan, SpanPosition, SpanType


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_span(
    paper_id: str,
    section_id: str,
    text: str,
    char_start: int,
    char_end: int,
    page_start: int = 1,
    page_end: int = 1,
    span_type: SpanType = SpanType.PARAGRAPH,
) -> SourceSpan:
    """Create a SourceSpan anchored to a specific text region.

    This is the canonical entry point for creating verifiable source anchors.
    Every ClaimBlock, EvidenceBlock, and LimitationBlock must eventually link
    to a SourceSpan via source_span_id.
    """
    return SourceSpan(
        paper_id=paper_id,
        section_id=section_id,
        span_type=span_type,
        source_text=text,
        position=SpanPosition(
            char_start=char_start,
            char_end=char_end,
            page_start=page_start,
            page_end=page_end,
        ),
        content_hash=sha256(text),
        normalized_text=_normalize(text),
    )


def make_span_for_section(section) -> SourceSpan:
    """Create a span covering an entire section."""
    return make_span(
        paper_id=section.paper_id,
        section_id=section.section_id,
        text=section.raw_text[:500],  # First 500 chars as representative
        char_start=0,
        char_end=len(section.raw_text),
        page_start=section.page_start or 1,
        page_end=section.page_end or 1,
        span_type=SpanType.SECTION,
    )


def make_span_for_claim(
    paper_id: str,
    section_id: str,
    claim_text: str,
    full_section_text: str,
    page_start: int = 1,
) -> SourceSpan:
    """Create a span for a claim by finding its position in the section text."""
    idx = full_section_text.find(claim_text)
    if idx == -1:
        # Try normalized match
        idx = _normalize(full_section_text).find(_normalize(claim_text))
    char_start = max(0, idx) if idx >= 0 else 0
    char_end = char_start + len(claim_text) if idx >= 0 else 0
    return make_span(
        paper_id=paper_id,
        section_id=section_id,
        text=claim_text,
        char_start=char_start,
        char_end=char_end,
        page_start=page_start,
        span_type=SpanType.SENTENCE,
    )


def _normalize(text: str) -> str:
    """Normalize text for matching: lowercase, collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", text.lower().strip())
