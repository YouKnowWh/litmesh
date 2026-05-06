"""
ExtractionRun: audit trail for LLM extraction jobs.

Every ClaimBlock, EvidenceBlock, LimitationBlock, ConceptKey, and Relation
must carry an extraction_run_id. This enables:
- Rollback: revert all candidates from a bad extraction run
- Audit: compare extraction runs, track extraction quality over time
- Debug: trace why a particular claim was extracted
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ExtractionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some items succeeded, some failed
    ROLLED_BACK = "rolled_back"


class ExtractionTarget(str, Enum):
    CLAIMS = "claims"
    EVIDENCE = "evidence"
    LIMITATIONS = "limitations"
    CONCEPTS = "concepts"
    RELATIONS = "relations"
    ALL = "all"


class ExtractionRun(BaseModel):
    """A single LLM extraction job.

    One ExtractionRun targets one paper + one extraction target type.
    For a full extraction pipeline, you chain multiple runs:
    1. Extract claims from sections
    2. Extract evidence for those claims
    3. Extract limitations
    4. Extract concepts
    5. Link relations
    """
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    paper_id: str
    graph_id: str

    target: ExtractionTarget = ExtractionTarget.ALL
    status: ExtractionStatus = ExtractionStatus.RUNNING

    # LLM configuration
    model: str = Field(default="deepseek-v4-pro[1m]", description="Model used for extraction")
    prompt_version: str = Field(default="v0.2", description="Version of extraction prompt used")
    prompt_template: str = Field(default="", description="Name of the prompt template used")

    # Input scope
    section_ids: list[str] = Field(default_factory=list, description="Which sections were processed")
    input_token_count: int = 0

    # Output stats
    items_produced: int = 0
    items_accepted: int = 0
    items_rejected: int = 0
    output_token_count: int = 0

    # Timing and cost
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_cost_usd: float = 0.0

    # Rollback support
    rolled_back_at: Optional[datetime] = None
    rolled_back_by: Optional[str] = None

    error_message: str = ""

    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionRunItem(BaseModel):
    """Individual item produced by an extraction run.

    One row per extracted claim/evidence/limitation/concept/relation.
    Links back to the ExtractionRun for batching and rollback.
    """
    item_id: str = Field(default_factory=lambda: f"item_{uuid4().hex[:12]}")
    run_id: str = Field(description="FK to ExtractionRun")

    # What was produced
    target_type: ExtractionTarget
    target_id: str = Field(description="FK to the actual block (claim_id, evidence_id, etc.)")

    # LLM output metadata
    raw_llm_output: str = Field(default="", description="Raw LLM response for debugging")
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Status
    accepted: bool = False
    reviewer_notes: str = ""

    created_at: datetime = Field(default_factory=datetime.utcnow)
