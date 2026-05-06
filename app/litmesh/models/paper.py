"""
PaperCard: metadata card for each literature item.

One PaperCard = one paper, one book chapter cluster, or one document.
It belongs to a CorpusCard via graph_id.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ResearchType(str, Enum):
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"
    CASE_STUDY = "case_study"
    REVIEW = "review"
    PRACTICE_REPORT = "practice_report"
    POLICY = "policy"
    TEXTBOOK_CHAPTER = "textbook_chapter"
    OTHER = "other"


class PaperCard(BaseModel):
    paper_id: str = Field(default_factory=lambda: f"paper_{uuid4().hex[:12]}")
    graph_id: str = Field(description="FK to SeriesGraph")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    source_file: str = Field(description="Path to source PDF or document")
    # Metadata extracted or manually entered
    abstract: str = ""
    abstract_summary: str = Field(
        default="",
        description="LLM-generated one-paragraph summary of the abstract"
    )
    keywords: list[str] = Field(default_factory=list)
    research_type: ResearchType = ResearchType.OTHER
    main_framework: str = Field(
        default="",
        description="Primary theoretical framework mentioned (e.g., PACADI, CPE-3DF)"
    )
    # Internal bookkeeping
    raw_text_hash: str = Field(default="", description="SHA256 of the full extracted text")
    page_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
