"""
LimitationBlock: constraints, risks, challenges, unsolved problems.

Answers: "What limits this claim? What could go wrong? What is unaddressed?"

Key design decision:
Limitation is NOT the same as contradiction. A limitation *constrains* a claim's
scope or applicability; a contradiction *denies* a claim. This distinction matters
for traversal: constrains edges have different routing behavior than contradicts edges.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class RiskType(str, Enum):
    """The kind of risk or limitation."""
    SCOPE = "scope"                    # Limited generalizability
    METHOD = "method"                  # Methodological weakness
    DATA = "data"                      # Data quality/quantity issues
    ETHICS = "ethics"                  # Ethical concern
    IMPLEMENTATION = "implementation"  # Practical deployment challenge
    THEORETICAL = "theoretical"        # Theoretical gap or assumption
    UNEXPLORED = "unexplored"          # Area the authors didn't investigate
    CONFLICT = "conflict"              # Conflicts with other work
    OTHER = "other"


class LimitationSeverity(str, Enum):
    """How severely this limitation affects the associated claims."""
    CRITICAL = "critical"      # The claim is significantly weakened
    MODERATE = "moderate"      # The claim needs qualification
    MINOR = "minor"            # The claim mostly holds
    UNASSESSED = "unassessed"


class LimitationStatus(str, Enum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    REJECTED = "rejected"
    ORPHAN = "orphan"


class LimitationBlock(BaseModel):
    limitation_id: str = Field(default_factory=lambda: f"lim_{uuid4().hex[:12]}")
    graph_id: str
    paper_id: str
    section_id: Optional[str] = None

    limitation_text: str = Field(description="The limitation as stated or inferred")

    affected_claim_ids: list[str] = Field(
        default_factory=list,
        description="FKs to ClaimBlocks this limitation constrains"
    )

    risk_type: RiskType = RiskType.OTHER
    severity: LimitationSeverity = LimitationSeverity.UNASSESSED

    concept_keys: list[str] = Field(default_factory=list)

    source_span_id: Optional[str] = None
    extraction_run_id: Optional[str] = None

    status: LimitationStatus = LimitationStatus.CANDIDATE

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
