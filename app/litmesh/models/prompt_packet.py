"""
PromptPacket: the final structured context package delivered to the LLM.

This is LitMesh's output. It's what makes LitMesh different from RAG:
instead of a flat list of chunks, the model receives typed, scoped, confidence-annotated
context with explicit generation policy.

Key design:
- Every section (claims, evidence, limitations, conflicts, etc.) is explicitly separated
- generation_policy tells the model what it can and cannot do with each section
- citations are embedded per-claim, not appended at the end
- low_confidence_candidates are labeled clearly so the model knows not to rely on them
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TraversalMode(str, Enum):
    EXPLAIN = "explain"
    AUDIT = "audit"
    COMPARE = "compare"
    TRACE = "trace"
    CONFLICT = "conflict"
    SYNTHESIS = "synthesis"
    TRANSFER = "transfer"


class PointerType(str, Enum):
    """Typed pointers for traversal. First edition priority set."""
    BELONGS_TO = "belongs_to"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    CONSTRAINS = "constrains"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    SECTION_PARENT = "section_parent"
    SECTION_NEXT = "section_next"
    # Extended (v0.5+)
    EXTENDS = "extends"
    SUPERSEDES = "supersedes"
    SAME_AS = "same_as"
    ANALOGOUS_TO_BRIDGE = "analogous_to_bridge"
    TRANSFERS_TO_BRIDGE = "transfers_to_bridge"
    CONFLICTS_WITH_BRIDGE = "conflicts_with_bridge"


class TraversalPlan(BaseModel):
    """LLM-generated plan for how to traverse the knowledge mesh.

    The LLM decides WHICH pointer types to walk; the program enforces
    HOW (depth, node count, confidence gates, cycle detection).
    """
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:12]}")

    task_type: str = Field(description="Brief description of the user's task type")
    start_nodes: list[str] = Field(description="Starting concept_keys or claim_ids")
    graph_scope: list[str] = Field(default_factory=list, description="Which graph_ids to include")

    pointer_types: list[PointerType] = Field(description="Which edge types to traverse")
    traversal_mode: TraversalMode = TraversalMode.EXPLAIN

    # Constraints (enforced by TraversalExecutor)
    max_depth: int = Field(default=3, description="Maximum hops from start nodes")
    max_nodes: int = Field(default=50, description="Maximum total nodes to visit")
    max_edges_per_pointer_type: int = Field(default=20)
    allow_cross_graph: bool = Field(default=False)
    max_cross_graph_jumps: int = Field(default=2)
    require_source_span: bool = Field(default=True)
    must_include_limitations: bool = Field(default=True)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum edge confidence to traverse")
    budget: int = Field(default=5000, description="Approximate token budget for final PromptPacket")

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class VisitedNode(BaseModel):
    """A node visited during traversal."""
    node_id: str
    node_type: str  # claim, evidence, limitation, concept, section, paper
    title: str = ""
    text: str = ""
    confidence: float = 1.0
    importance: float = 0.5
    source_span_id: Optional[str] = None
    graph_id: str = ""
    paper_id: str = ""
    concept_keys: list[str] = Field(default_factory=list)


class VisitedEdge(BaseModel):
    """An edge traversed during traversal."""
    source_id: str
    target_id: str
    relation_type: str
    is_cross_graph: bool = False
    confidence: float = 1.0


class TraversalResult(BaseModel):
    """Output of TraversalExecutor."""
    plan_id: str
    visited_nodes: list[VisitedNode] = Field(default_factory=list)
    visited_edges: list[VisitedEdge] = Field(default_factory=list)

    grouped_claims: list[str] = Field(default_factory=list)
    grouped_evidence: list[str] = Field(default_factory=list)
    grouped_limitations: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    bridge_relations: list[str] = Field(default_factory=list)

    stopped_reason: str = ""
    total_traversal_cost: float = 0.0


class TraversalTrace(BaseModel):
    """Audit log of a traversal execution."""
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid4().hex[:12]}")
    query: str = Field(description="The user's original query")
    plan: Optional[TraversalPlan] = None
    result: Optional[TraversalResult] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# --- PromptPacket ---

class GenerationPolicy(BaseModel):
    """Instructions to the model about how to use the provided context."""
    may_assert_claims: bool = Field(default=True, description="Can the model assert claims as facts?")
    may_assert_limitations: bool = Field(default=True)
    must_cite_claims: bool = Field(default=True, description="Must cite source when using a claim")
    must_mention_limitations: bool = Field(default=True, description="Must mention relevant limitations")
    must_label_analogy: bool = Field(default=True, description="Must explicitly label cross-graph analogies")
    must_flag_low_confidence: bool = Field(default=True, description="Must flag low-confidence items")
    allow_synthesis: bool = Field(default=True, description="Can the model synthesize across claims?")
    disallow_invention: bool = Field(default=True, description="Must not invent claims not in the packet")


class PacketClaim(BaseModel):
    """A claim included in the PromptPacket."""
    claim_id: str
    claim_text: str
    claim_type: str
    confidence: float
    source_citation: str = ""
    is_cross_graph: bool = False


class PromptPacket(BaseModel):
    """The final compiled context package delivered to the reasoning model.

    This is what distinguishes LitMesh from RAG. The model receives a
    structured, typed, confidence-annotated context bundle.
    """
    packet_id: str = Field(default_factory=lambda: f"pkt_{uuid4().hex[:12]}")

    # User query and intent interpretation
    current_user_query: str = ""
    interpreted_intent: str = ""

    # Scope
    graph_scope: list[str] = Field(default_factory=list, description="Which SeriesGraphs were used")
    active_concepts: list[str] = Field(default_factory=list, description="ConceptKeys active in this packet")

    # Cognitive anchors (manually curated high-confidence reference points)
    cognitive_anchors: list[PacketClaim] = Field(default_factory=list)

    # Typed content blocks
    paper_claims: list[PacketClaim] = Field(default_factory=list)
    supporting_evidence: list[dict] = Field(default_factory=list)
    limitations: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)

    # Explicitly uncertain content
    low_confidence_candidates: list[dict] = Field(default_factory=list)

    # Cross-graph bridges used
    bridge_relations: list[dict] = Field(default_factory=list)

    # Instructions to the model
    generation_policy: GenerationPolicy = Field(default_factory=GenerationPolicy)

    # Trace reference
    trace_id: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
