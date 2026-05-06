"""
LitMesh API routes (v0.1-v0.2 minimal API).

Endpoints:
  POST /corpora              - Create a corpus
  POST /graphs               - Create a SeriesGraph
  POST /papers/import        - Import PDF (v0.1)
  POST /papers/{id}/extract  - Run extraction (v0.2)
  GET  /papers               - List papers
  GET  /papers/{id}          - Get paper detail
  GET  /papers/{id}/sections - Get sections for a paper
  GET  /papers/{id}/claims   - Get claims for a paper
  GET  /inbox                - List pending review items
  POST /inbox/{id}/approve   - Approve inbox item
  POST /inbox/{id}/reject    - Reject inbox item
  GET  /stats                - Database statistics
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..models.corpus import CorpusCard, CorpusType, IntegrationPolicy
from ..models.graph import SeriesGraph, GraphType, CrossGraphPolicy
from ..ingestion.pipeline import IngestionPipeline
from ..review.inbox import ReviewManager
from ..extraction.llm_client import LLMClient


# ---- Request/Response schemas ----

class CorpusCreate(BaseModel):
    name: str
    corpus_type: str = "paper_collection"
    domain: str = ""
    description: str = ""
    integration_policy: str = "bridge_review"

class GraphCreate(BaseModel):
    corpus_id: str
    name: str
    graph_type: str = "paper_collection"
    domain: str = ""
    description: str = ""
    concept_namespace: str = ""
    cross_graph_policy: str = "strict"

class ImportRequest(BaseModel):
    pdf_path: str

class InboxDecision(BaseModel):
    reason: str = ""
    new_confidence: Optional[float] = None


def create_app(db, llm_client: Optional[LLMClient] = None) -> FastAPI:
    """Factory function to create the FastAPI app with LitMesh dependencies."""

    if llm_client is None:
        llm_client = LLMClient()

    app = FastAPI(title="LitMesh API", version="0.1.0")
    review_mgr = ReviewManager(db)
    # Lazy init: pipeline needs graph_id per-request

    @app.post("/corpora")
    async def create_corpus(req: CorpusCreate):
        corpus = CorpusCard(
            name=req.name,
            corpus_type=CorpusType(req.corpus_type),
            domain=req.domain,
            description=req.description,
            integration_policy=IntegrationPolicy(req.integration_policy),
        )
        db.insert_corpus(corpus)
        return {"ok": True, "corpus_id": corpus.corpus_id}

    @app.post("/graphs")
    async def create_graph(req: GraphCreate):
        graph = SeriesGraph(
            corpus_id=req.corpus_id,
            name=req.name,
            graph_type=GraphType(req.graph_type),
            domain=req.domain,
            description=req.description,
            concept_namespace=req.concept_namespace,
            cross_graph_policy=CrossGraphPolicy(req.cross_graph_policy),
        )
        db.insert_graph(graph)
        return {"ok": True, "graph_id": graph.graph_id}

    @app.post("/papers/import")
    async def import_paper(req: ImportRequest, graph_id: str = Query(...)):
        """Import a PDF (v0.1). Returns paper_id and section count."""
        pipeline = IngestionPipeline(db, llm_client, graph_id)
        result = pipeline.run_v0_1(req.pdf_path)
        return {"ok": True, **result}

    @app.post("/papers/{paper_id}/extract")
    async def extract_paper(paper_id: str, graph_id: str = Query(...)):
        """Run v0.2 extraction on a paper."""
        pipeline = IngestionPipeline(db, llm_client, graph_id)
        stats = pipeline.run_v0_2(paper_id)
        return {"ok": True, "stats": stats}

    @app.post("/papers/{paper_id}/full")
    async def full_pipeline(paper_id: str, req: ImportRequest, graph_id: str = Query(...)):
        """Run v0.1 + v0.2 together."""
        pipeline = IngestionPipeline(db, llm_client, graph_id)
        result = pipeline.run_full(req.pdf_path)
        return {"ok": True, **result}

    @app.get("/papers")
    async def list_papers(graph_id: Optional[str] = None):
        papers = db.list_papers(graph_id)
        return {"papers": papers, "count": len(papers)}

    @app.get("/papers/{paper_id}")
    async def get_paper(paper_id: str):
        paper = db.get_paper(paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found")
        return paper

    @app.get("/papers/{paper_id}/sections")
    async def get_sections(paper_id: str):
        sections = db.get_sections_by_paper(paper_id)
        return {"sections": sections, "count": len(sections)}

    @app.get("/papers/{paper_id}/claims")
    async def get_claims(paper_id: str, status: Optional[str] = None):
        claims = db.get_claims_by_paper(paper_id, status)
        return {"claims": claims, "count": len(claims)}

    @app.get("/inbox")
    async def list_inbox(inbox_type: Optional[str] = None):
        items = db.get_pending_inbox(inbox_type)
        return {"items": items, "count": len(items)}

    @app.post("/inbox/{inbox_id}/approve")
    async def approve_inbox_item(inbox_id: str):
        result = review_mgr.approve(inbox_id)
        if not result["ok"]:
            raise HTTPException(400, result.get("error", "Unknown error"))
        return result

    @app.post("/inbox/{inbox_id}/reject")
    async def reject_inbox_item(inbox_id: str, body: InboxDecision = InboxDecision()):
        result = review_mgr.reject(inbox_id, reason=body.reason)
        if not result["ok"]:
            raise HTTPException(400, result.get("error", "Unknown error"))
        return result

    @app.post("/inbox/{inbox_id}/downgrade")
    async def downgrade_inbox_item(inbox_id: str, body: InboxDecision):
        confidence = body.new_confidence or 0.3
        result = review_mgr.downgrade_confidence(inbox_id, new_confidence=confidence)
        if not result["ok"]:
            raise HTTPException(400, result.get("error", "Unknown error"))
        return result

    @app.get("/concepts")
    async def list_concepts(graph_id: Optional[str] = None, status: Optional[str] = "active"):
        if graph_id:
            rows = db.conn.execute(
                "SELECT * FROM concept_keys WHERE graph_id = ? AND status = ? ORDER BY namespace, label_zh",
                (graph_id, status or "active")
            ).fetchall()
        else:
            rows = db.conn.execute(
                "SELECT * FROM concept_keys WHERE status = ? ORDER BY graph_id, namespace, label_zh",
                (status or "active",)
            ).fetchall()
        return {"concepts": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/traces")
    async def list_traces(limit: int = 20):
        rows = db.conn.execute(
            "SELECT trace_id, query, created_at FROM traversal_traces ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return {"traces": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        row = db.conn.execute("SELECT * FROM traversal_traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Trace not found")
        d = dict(row)
        d["plan"] = json.loads(d["plan_json"]) if d.get("plan_json") else {}
        d["result"] = json.loads(d["result_json"]) if d.get("result_json") else {}
        return d

    @app.get("/series-groups")
    async def list_series_groups(domain: Optional[str] = None):
        groups = db.list_series_groups(domain or "")
        return {"groups": groups, "count": len(groups)}

    @app.get("/bridges")
    async def list_bridges(graph_id: Optional[str] = None, status: Optional[str] = None):
        if graph_id:
            rows = db.conn.execute(
                "SELECT * FROM bridge_relations WHERE (source_graph_id = ? OR target_graph_id = ?) "
                + ("AND review_status = ?" if status else ""),
                ((graph_id, graph_id, status) if status else (graph_id, graph_id))
            ).fetchall()
        else:
            rows = db.conn.execute(
                "SELECT * FROM bridge_relations" + (" WHERE review_status = ?" if status else ""),
                ((status,) if status else ())
            ).fetchall()
        return {"bridges": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/claims")
    async def list_all_claims(graph_id: Optional[str] = None, status: Optional[str] = "active",
                              limit: int = 50):
        if graph_id:
            rows = db.conn.execute(
                "SELECT * FROM claim_blocks WHERE graph_id = ? AND status = ? ORDER BY extraction_confidence DESC LIMIT ?",
                (graph_id, status or "active", limit)
            ).fetchall()
        else:
            rows = db.conn.execute(
                "SELECT * FROM claim_blocks WHERE status = ? ORDER BY extraction_confidence DESC LIMIT ?",
                (status or "active", limit)
            ).fetchall()
        return {"claims": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/evidence")
    async def list_all_evidence(graph_id: Optional[str] = None, limit: int = 30):
        if graph_id:
            rows = db.conn.execute(
                "SELECT * FROM evidence_blocks WHERE graph_id = ? AND status = 'active' LIMIT ?",
                (graph_id, limit)
            ).fetchall()
        else:
            rows = db.conn.execute(
                "SELECT * FROM evidence_blocks WHERE status = 'active' LIMIT ?", (limit,)
            ).fetchall()
        return {"evidence": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/limitations")
    async def list_all_limitations(graph_id: Optional[str] = None, limit: int = 30):
        if graph_id:
            rows = db.conn.execute(
                "SELECT * FROM limitation_blocks WHERE graph_id = ? AND status = 'active' LIMIT ?",
                (graph_id, limit)
            ).fetchall()
        else:
            rows = db.conn.execute(
                "SELECT * FROM limitation_blocks WHERE status = 'active' LIMIT ?", (limit,)
            ).fetchall()
        return {"limitations": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/stats")
    async def get_stats():
        return db.get_stats()

    @app.get("/ui")
    async def serve_ui():
        from fastapi.responses import HTMLResponse
        from pathlib import Path as FSPath
        ui_file = FSPath(__file__).parent.parent.parent.parent / "app" / "litmesh" / "api" / "ui.html"
        if not ui_file.exists():
            raise HTTPException(404, "UI not found")
        return HTMLResponse(ui_file.read_text(encoding="utf-8"))

    return app
