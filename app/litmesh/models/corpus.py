"""
CorpusCard: top-level container for a literature collection.

Design: each corpus is a domain-scoped grouping (e.g., "AI Education Papers").
It owns one or more SeriesGraphs but is not itself a graph node.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class CorpusType(str, Enum):
    PAPER_COLLECTION = "paper_collection"
    TEXTBOOK = "textbook"
    PROJECT_DOCS = "project_docs"
    MANUAL = "manual"
    COURSE_MATERIAL = "course_material"
    CUSTOM = "custom"


class IntegrationPolicy(str, Enum):
    """How items in this corpus can be integrated with other corpora."""
    ISOLATED = "isolated"         # No cross-corpus integration
    BRIDGE_REVIEW = "bridge_review"  # Bridge only after review
    LOOSE = "loose"               # Allow bridge candidates automatically (still requires approval)


class CorpusCard(BaseModel):
    corpus_id: str = Field(default_factory=lambda: f"corpus_{uuid4().hex[:12]}")
    name: str
    corpus_type: CorpusType = CorpusType.PAPER_COLLECTION
    domain: str = Field(default="", description="e.g., AI_education, operating_systems, networking")
    description: str = ""
    source_items: list[str] = Field(
        default_factory=list,
        description="List of source file paths or identifiers in this corpus"
    )
    default_graph_id: Optional[str] = Field(default=None, description="Primary SeriesGraph for this corpus")
    integration_policy: IntegrationPolicy = IntegrationPolicy.BRIDGE_REVIEW
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
