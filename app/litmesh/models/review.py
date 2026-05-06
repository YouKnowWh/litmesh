"""
ReviewInboxItem: the review queue object.

LitMesh principle 6: Review before high-impact use.
New ConceptKeys, cross-graph Bridges, conflict relations, high-importance Claims,
and cognitive anchors must all enter the review queue before becoming active.

Four inbox types:
- ExtractionInbox: claims/evidence/limitations from LLM extraction
- ConceptInbox: new concept keys proposed by ConceptExtractor
- BridgeInbox: cross-graph bridge relation candidates
- ConflictInbox: detected contradictions between claims

Each supports: approve, reject, edit, merge, split, downgrade, mark_as_limitation, mark_as_conflict
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class InboxType(str, Enum):
    EXTRACTION = "extraction"   # ClaimBlock, EvidenceBlock, LimitationBlock candidates
    CONCEPT = "concept"         # ConceptKey candidates
    BRIDGE = "bridge"           # BridgeRelation candidates
    CONFLICT = "conflict"       # Detected conflicts/contradictions


class InboxDecision(str, Enum):
    PENDING = "pending"
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    MERGE = "merge"                     # Merge into existing item
    SPLIT = "split"                     # Split into multiple items
    DOWNGRADE_CONFIDENCE = "downgrade_confidence"   # Keep but lower confidence
    MARK_AS_LIMITATION = "mark_as_limitation"       # Reclassify claim as limitation
    MARK_AS_CONFLICT = "mark_as_conflict"            # Flag as conflicting
    DEFER = "defer"                     # Not now, keep in inbox


class InboxPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewInboxItem(BaseModel):
    """A single item awaiting human review.

    Design: the item_id refers to the actual object (claim_id, concept_key, etc.),
    NOT a copy. The original object stays in candidate status until this is resolved.
    """
    inbox_id: str = Field(default_factory=lambda: f"inbox_{uuid4().hex[:12]}")
    inbox_type: InboxType

    # Reference to the candidate object
    item_id: str = Field(description="FK to the candidate (claim_id, concept_key, bridge_id, etc.)")
    item_type: str = Field(description="Model type: claim, evidence, limitation, concept, bridge_relation")

    # Display info for reviewer
    title: str = Field(description="Short summary of what this item is")
    description: str = Field(default="", description="Detailed context for the reviewer")
    source_text: str = Field(default="", description="Original source text for context")

    # Confidence and priority
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: InboxPriority = InboxPriority.MEDIUM

    # Decision tracking
    decision: InboxDecision = InboxDecision.PENDING
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    decision_notes: str = ""
    # When merging: the target item to merge into
    merge_target_id: Optional[str] = None

    # Context for the reviewer
    extraction_run_id: Optional[str] = None
    graph_id: Optional[str] = None
    paper_id: Optional[str] = None

    # Suggested actions (pre-populated by the system)
    suggested_actions: list[InboxDecision] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
