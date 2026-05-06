"""
SeriesGraph: a self-contained knowledge graph for one literature series.

Principle 1: Series-first, bridge-later.
Each corpus/domain gets its own graph. Cross-graph integration requires
BridgeRelations with higher confidence bars.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class GraphType(str, Enum):
    PAPER_COLLECTION = "paper_collection"
    TEXTBOOK = "textbook"
    PROJECT_DOCS = "project_docs"
    DOMAIN = "domain"


class GraphStatus(str, Enum):
    ACTIVE = "active"
    BUILDING = "building"    # Still being populated
    FROZEN = "frozen"        # No new items without review
    ARCHIVED = "archived"


class CrossGraphPolicy(str, Enum):
    STRICT = "strict"         # All cross-graph bridges require review
    MODERATE = "moderate"     # same_as bridges require review, others generate candidates
    LOOSE = "loose"           # All bridge types generate candidates (still requires approval)


class SeriesGraph(BaseModel):
    graph_id: str = Field(default_factory=lambda: f"graph_{uuid4().hex[:12]}")
    corpus_id: str = Field(description="FK to CorpusCard")
    name: str
    graph_type: GraphType = GraphType.PAPER_COLLECTION
    domain: str = Field(default="", description="e.g., AI_education, operating_systems")
    description: str = ""

    # Namespace prefix for concepts in this graph
    concept_namespace: str = Field(default="", description="e.g., 'AI_edu', 'OS', 'net'")

    # Integration policies
    merge_policy: str = Field(default="review_required", description="For within-graph concept merging")
    cross_graph_policy: CrossGraphPolicy = CrossGraphPolicy.STRICT

    status: GraphStatus = GraphStatus.BUILDING

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
