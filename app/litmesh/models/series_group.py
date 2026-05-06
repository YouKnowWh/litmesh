"""SeriesGroup: index-layer grouping of isolated SeriesGraphs.

Data layer: each PDF = its own SeriesGraph (never merged).
Index layer: SeriesGroup maps which graphs belong to the same series.

This separation means:
- Series membership is cheap to update (just change the array).
- No graph merging needed.
- Traversal uses group.graph_ids as graph_scope.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class SeriesGroup(BaseModel):
    group_id: str = Field(default_factory=lambda: f"sgroup_{uuid4().hex[:12]}")
    name: str = Field(description="Human-readable series name")
    graph_ids: list[str] = Field(description="Member graph IDs (order = reading order)")
    domain: str = Field(default="")
    description: str = Field(default="")

    # Detection confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
