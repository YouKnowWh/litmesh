"""
GraphRelation and BridgeRelation: typed pointers between nodes.

Principle: Relations are NOT just "related_to". Every edge has a type that
determines how TraversalExecutor routes through the mesh.

GraphRelation: within a single SeriesGraph.
BridgeRelation: across different SeriesGraphs (requires higher confidence bar).
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# --- Within-graph relation types ---

class GraphRelationType(str, Enum):
    """Typed pointers within a single SeriesGraph.

    First edition supports 10 types:

    Structural:
    - belongs_to: node X is part of node Y (e.g., claim belongs to section)
    - mentions: node X mentions concept Y

    Argumentative:
    - supports: evidence E supports claim C
    - contradicts: claim A contradicts claim B
    - constrains: limitation L constrains claim C

    Evolutionary:
    - derived_from: claim B is derived from claim A
    - refines: claim B refines/improves claim A
    - extends: claim B extends claim A to new domain
    - supersedes: claim B replaces claim A
    - same_as: concept X and concept Y are the same (merge candidate)
    """

    # Structural
    BELONGS_TO = "belongs_to"
    MENTIONS = "mentions"

    # Argumentative
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONSTRAINS = "constrains"

    # Evolutionary
    DERIVED_FROM = "derived_from"
    REFINES = "refines"
    EXTENDS = "extends"
    SUPERSEDES = "supersedes"
    SAME_AS = "same_as"

    # Navigation (section structure)
    PARENT = "section_parent"
    NEXT = "section_next"


class GraphRelation(BaseModel):
    """A typed edge between two nodes in the same SeriesGraph.

    source_id and target_id are string keys that can reference any node type:
    claim_id, evidence_id, limitation_id, concept_key, section_id, paper_id.
    """
    relation_id: str = Field(default_factory=lambda: f"rel_{uuid4().hex[:12]}")
    graph_id: str

    source_id: str = Field(description="Source node ID (e.g., claim_id, evidence_id)")
    target_id: str = Field(description="Target node ID")
    source_type: str = Field(description="Type of source node: claim, evidence, limitation, concept, section")
    target_type: str = Field(description="Type of target node")

    relation_type: GraphRelationType

    # Confidence that this relation is correct
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="How important this edge is for traversal")

    # Traversal cost (higher = more expensive to traverse, used by TraversalExecutor)
    traversal_cost: float = Field(default=1.0, ge=0.0, description="Higher cost = less likely to be traversed first")

    # Audit
    extraction_run_id: Optional[str] = None
    evidence_json: str = Field(default="", description="JSON snippet of source evidence for this relation")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# --- Cross-graph bridge types ---

class BridgeRelationType(str, Enum):
    """Cross-graph bridge relations. Higher bar for confidence.

    These are NOT intra-graph relations. They connect concepts/claims
    across different SeriesGraphs without merging them.

    Key distinctions:
    - same_as: concept X in graph A IS concept Y in graph B (strongest claim)
    - broader_than: concept X in graph A is broader than concept Y in graph B
    - narrower_than: the inverse
    - analogous_to: concept X in graph A is ANALOGOUS to concept Y in graph B (NOT same!)
    - applies_to: framework/method X in graph A applies to domain Y in graph B
    - conflicts_with: claim X in graph A conflicts with claim Y in graph B
    - transfers_to: finding X in graph A may transfer to context Y in graph B
    """
    SAME_AS = "same_as"
    BROADER_THAN = "broader_than"
    NARROWER_THAN = "narrower_than"
    ANALOGOUS_TO = "analogous_to"
    APPLIES_TO = "applies_to"
    CONFLICTS_WITH = "conflicts_with"
    TRANSFERS_TO = "transfers_to"


class BridgeStatus(str, Enum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    REJECTED = "rejected"


class BridgeRelation(BaseModel):
    """A typed bridge between nodes in different SeriesGraphs.

    BridgeRelations must go through BridgeInbox review before becoming active.
    Cross-graph integration standards are significantly higher than within-graph.
    """
    bridge_id: str = Field(default_factory=lambda: f"bridge_{uuid4().hex[:12]}")

    source_graph_id: str
    target_graph_id: str
    source_key: str = Field(description="Source node key (typically a concept_key or claim_id)")
    target_key: str = Field(description="Target node key in the other graph")

    bridge_type: BridgeRelationType

    bridge_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_json: str = Field(default="", description="Evidence supporting this bridge")

    # Warnings are critical for cross-graph bridges
    warning: str = Field(
        default="",
        description="Explicit warning, e.g. 'These concepts look similar but belong to different theoretical traditions'"
    )

    review_status: BridgeStatus = BridgeStatus.CANDIDATE
    extraction_run_id: Optional[str] = None

    # Higher traversal cost for cross-graph jumps
    traversal_cost: float = Field(default=5.0, ge=0.0, description="Cross-graph traversal is expensive by default")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
