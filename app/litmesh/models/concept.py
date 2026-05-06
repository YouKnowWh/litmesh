"""
ConceptKey: concept index unit, not a plain tag.

A ConceptKey is a routing entry point into the knowledge graph. It carries:
- namespace (concept:, framework:, method:, problem:, theory:)
- cross-graph merge policy
- parent/child hierarchy for concept tree traversal
- do_not_merge_with for explicit dedup boundaries

Critical rule: ConceptKey is NOT created directly by LLM. It goes through:
  term extraction -> candidate generation -> ConceptRegistry dedup
  -> alias check -> embedding similarity -> merge_policy gate
  -> ConceptInbox review -> active
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ConceptNamespace(str, Enum):
    """Namespace prefix for concept keys. Every ConceptKey must declare one."""
    CONCEPT = "concept"       # General concept (AIGC_education, cognitive_load)
    FRAMEWORK = "framework"   # Named framework or model (PACADI, CPE-3DF)
    METHOD = "method"         # Method or technique (prompt_engineering, virtual_lab)
    PROBLEM = "problem"       # Problem or challenge (AI_hallucination, data_privacy)
    THEORY = "theory"         # Theoretical foundation (Tao_Xingzhi_life_education)
    TOOL = "tool"             # Tool or technology (ChatGPT, GPT-4)
    METRIC = "metric"         # Measurement or metric (learning_outcome, cognitive_load_index)


class ConceptStatus(str, Enum):
    CANDIDATE = "candidate"   # Proposed by extraction, not yet reviewed
    ACTIVE = "active"         # Approved, usable in traversal and PromptPacket
    DEPRECATED = "deprecated"  # Was active, now superseded
    REJECTED = "rejected"     # Explicitly denied
    MERGED = "merged"         # Merged into another ConceptKey


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_EDIT = "needs_edit"
    REJECTED = "rejected"


class MergePolicy(str, Enum):
    """Controls whether and how this concept can merge across graphs."""
    STRICT = "strict"         # Never auto-merge; must be reviewed
    RELAXED = "relaxed"       # Allow merge candidates within same graph
    OPEN = "open"             # Allow merge candidates across graphs (still requires review)


class ConceptKey(BaseModel):
    """A concept index node in the knowledge mesh.

    Design rationale:
    - concept_key is the canonical ID string (e.g., "concept:cognitive_load")
    - label_zh/label_en are human-readable labels
    - parent_keys/child_keys enable hierarchical traversal (broader/narrower relations)
    - do_not_merge_with explicitly prevents bad auto-merges
    - Every concept belongs to a graph (graph_id); cross-graph equivalence requires BridgeRelation
    """

    concept_key: str = Field(
        default_factory=lambda: f"concept_{uuid4().hex[:12]}",
        description="Canonical key, e.g. 'framework:CPE_3DF', 'concept:AIGC_education'"
    )
    graph_id: str = Field(description="FK to SeriesGraph this concept lives in")
    namespace: ConceptNamespace = ConceptNamespace.CONCEPT

    # Human-readable labels
    label_zh: str = ""
    label_en: str = ""
    definition: str = Field(default="", description="One-paragraph definition")

    # Aliases for matching (e.g., "AI 幻觉", "AI hallucination", "人工智能幻觉")
    aliases: list[str] = Field(default_factory=list)

    # Hierarchical relations within the same graph
    parent_keys: list[str] = Field(default_factory=list)
    child_keys: list[str] = Field(default_factory=list)
    related_keys: list[str] = Field(default_factory=list)

    # Explicit dedup boundary: never merge with these concept_keys
    do_not_merge_with: list[str] = Field(default_factory=list)

    # Lifecycle
    status: ConceptStatus = ConceptStatus.CANDIDATE
    review_status: ReviewStatus = ReviewStatus.PENDING
    merge_policy: MergePolicy = MergePolicy.STRICT

    # Provenance
    extraction_run_id: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def full_key(self) -> str:
        """Return namespace:label format, e.g. 'framework:CPE_3DF'"""
        return f"{self.namespace.value}:{self.concept_key.split(':', 1)[-1]}" if ':' not in self.concept_key else self.concept_key

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
