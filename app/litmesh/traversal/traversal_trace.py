"""
TraversalTrace: audit log for traversal executions.

Every traversal is recorded so we can:
- Debug why certain nodes were included/excluded
- Compare traversal quality across modes
- Replay traversals for reproducible PromptPackets
"""

import json

from ..models.prompt_packet import TraversalTrace, TraversalPlan, TraversalResult


class TraceStore:
    """Persists traversal traces to the SQLite traversal_traces table."""

    def __init__(self, db):
        self.db = db

    def save(self, query: str, plan: TraversalPlan, result: TraversalResult) -> str:
        """Save a traversal trace and return the trace_id."""
        trace = TraversalTrace(
            query=query,
            plan=plan,
            result=result,
        )
        self.db.insert_trace(
            trace_id=trace.trace_id,
            query=query,
            plan_json=plan.model_dump_json() if plan else "{}",
            result_json=result.model_dump_json() if result else "{}",
        )
        return trace.trace_id

    def load(self, trace_id: str) -> dict | None:
        """Load a trace by ID."""
        row = self.db.conn.execute(
            "SELECT * FROM traversal_traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "trace_id": row["trace_id"],
            "query": row["query"],
            "plan": json.loads(row["plan_json"]),
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
        }

    def list_recent(self, limit: int = 20) -> list[dict]:
        """List recent traces."""
        rows = self.db.conn.execute(
            "SELECT trace_id, query, created_at FROM traversal_traces ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
