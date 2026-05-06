"""
ClaimBlock: the core argument unit.

A claim answers: "What does the author assert?"

Key design decisions:
1. extraction_confidence vs claim_confidence are separate dimensions.
   - extraction_confidence: did we correctly capture what the author said?
   - claim_confidence: is the author's claim itself reliable?
   This separation is critical for audit_mode traversal.

2. Status is a workflow state, not a boolean.
   candidate -> reviewed -> active | rejected | orphan

3. concept_keys is a list of ConceptKey references, populated after ConceptRegistry runs.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Semantic type of claim.

    Why not free-text: typed claims enable typed traversal.
    explain_mode walks definitional claims; audit_mode walks empirical ones.
    """
    DEFINITIONAL = "definitional"       # "X is defined as Y"
    EMPIRICAL = "empirical"              # "We observed X in Y conditions"
    THEORETICAL = "theoretical"          # "X should cause Y because of theory Z"
    METHODOLOGICAL = "methodological"    # "Approach X is better than Y for task Z"
    NORMATIVE = "normative"              # "We should do X"
    CRITICAL = "critical"                # "Prior work X is wrong/incomplete because Y"
    SYNTHESIS = "synthesis"              # "Taking X and Y together, we propose Z"
    FRAMEWORK = "framework"              # "We propose framework X with components A, B, C"


class ClaimStatus(str, Enum):
    """Workflow status for a claim.

    candidate: LLM extracted it, not yet reviewed.
    reviewed: human has looked at it.
    active: approved for use in PromptPacket.
    contested: someone flagged it as potentially wrong.
    rejected: explicitly rejected.
    orphan: no valid source_span could be established.
    superseded: replaced by a newer/refined claim.
    """
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    CONTESTED = "contested"
    REJECTED = "rejected"
    ORPHAN = "orphan"
    SUPERSEDED = "superseded"


class ClaimImportance(str, Enum):
    """How central this claim is to understanding the paper's argument."""
    CORE = "core"            # The paper doesn't make sense without this claim
    SUPPORTING = "supporting"  # Important but not central
    PERIPHERAL = "peripheral"  # Interesting but skippable


class ClaimBlock(BaseModel):
    claim_id: str = Field(default_factory=lambda: f"claim_{uuid4().hex[:12]}")
    graph_id: str
    paper_id: str
    section_id: Optional[str] = None

    # The claim as stated by the author
    claim_text: str = Field(description="Verbatim or near-verbatim claim from source")
    # Normalized version (resolved pronouns, expanded acronyms, single sentence)
    normalized_claim: str = Field(
        default="",
        description="Self-contained, pronoun-resolved version of the claim"
    )

    claim_type: ClaimType = ClaimType.THEORETICAL

    # Concept references (populated by ConceptRegistry)
    concept_keys: list[str] = Field(default_factory=list)

    # Related blocks
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="List of evidence_ids that support this claim"
    )
    limitation_refs: list[str] = Field(
        default_factory=list,
        description="List of limitation_ids that constrain this claim"
    )

    # Dual confidence
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence that the LLM correctly extracted the author's intent (0-1)"
    )
    claim_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Assessment of the claim's own reliability/strength (0-1). Filled by reviewer, not LLM."
    )

    importance: ClaimImportance = ClaimImportance.SUPPORTING
    status: ClaimStatus = ClaimStatus.CANDIDATE

    # The mandatory anchor
    source_span_id: Optional[str] = Field(
        default=None,
        description="FK to SourceSpan. NULL = orphan: claim has no verified source location."
    )

    # Provenance
    extraction_run_id: Optional[str] = Field(
        default=None,
        description="FK to ExtractionRun that produced this claim"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
