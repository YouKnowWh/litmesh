"""
EvidenceBlock: what supports a claim.

Answers: "What does the author use to back up this claim?"

Evidence types are designed for traversal routing:
- audit_mode prioritizes experiment + data evidence
- compare_mode looks at evidence strength across papers
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    CASE = "case"                       # Case study, teaching example
    DATA = "data"                       # Quantitative data, statistics
    EXPERIMENT = "experiment"           # Controlled experiment result
    TEACHING_PRACTICE = "teaching_practice"  # Classroom observation, teaching reflection
    SURVEY = "survey"                   # Survey or questionnaire results
    THEORETICAL_REFERENCE = "theoretical_reference"  # Cites another theory
    POLICY_REFERENCE = "policy_reference"  # Cites policy documents
    CHAPTER_ARGUMENT = "chapter_argument"  # Book chapter logical argument
    LITERATURE_CITATION = "literature_citation"  # References prior literature
    OTHER = "other"


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNASSESSED = "unassessed"


class EvidenceStatus(str, Enum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    ACTIVE = "active"
    REJECTED = "rejected"
    ORPHAN = "orphan"


class EvidenceBlock(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"evid_{uuid4().hex[:12]}")
    graph_id: str
    paper_id: str
    section_id: Optional[str] = None

    supports_claim_ids: list[str] = Field(
        default_factory=list,
        description="FKs to ClaimBlocks this evidence supports"
    )

    evidence_text: str = Field(description="The evidence as stated in source")
    evidence_type: EvidenceType = EvidenceType.OTHER
    strength: EvidenceStrength = EvidenceStrength.UNASSESSED

    concept_keys: list[str] = Field(default_factory=list)

    source_span_id: Optional[str] = None
    extraction_run_id: Optional[str] = None

    status: EvidenceStatus = EvidenceStatus.CANDIDATE

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
