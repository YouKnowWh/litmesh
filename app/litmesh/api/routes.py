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

import shutil
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
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

DATA_DIR = Path("data")

def _run_extraction_safe(db, graph_id, paper_id):
    """Run v0.2 extraction in a background thread, logging any errors."""
    try:
        from ..extraction.llm_client import LLMClient
        llm = LLMClient()
        pipeline = IngestionPipeline(db, llm, graph_id)
        pipeline.run_v0_2(paper_id)
    except Exception as e:
        import logging
        logging.getLogger("litmesh").error(f"Background extraction failed for {paper_id}: {e}")

class InboxDecision(BaseModel):
    reason: str = ""
    new_confidence: Optional[float] = None


def create_app(db, llm_client: Optional[LLMClient] = None, llm_clients=None,
               embed_provider=None) -> FastAPI:
    """Factory function to create the FastAPI app with LitMesh dependencies.

    Args:
        db: LitMeshDB instance.
        llm_client: Single LLMClient (backward-compatible). Ignored if llm_clients given.
        llm_clients: MultiLLMClient with per-role clients (extraction/review/compilation/default).
        embed_provider: EmbeddingProvider for vector search.
    """

    if llm_clients is not None:
        llm_extraction = llm_clients.extraction
        llm_review = llm_clients.review
    elif llm_client is not None:
        llm_extraction = llm_client
        llm_review = llm_client
    else:
        llm_extraction = LLMClient()
        llm_review = llm_extraction

    # Store embed_provider for use by endpoints that need it
    app_state = {"embed_provider": embed_provider}

    app = FastAPI(title="LitMesh API", version="0.1.0")
    review_mgr = ReviewManager(db)

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
        pipeline = IngestionPipeline(db, llm_extraction, graph_id)
        result = pipeline.run_v0_1(req.pdf_path)
        return {"ok": True, **result}

    @app.post("/papers/upload")
    async def upload_paper(file: UploadFile = File(...), graph_id: str = Query(...)):
        """Upload a PDF file and run v0.1 import. Returns paper_id and section count."""
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are accepted")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = file.filename.replace('/', '_').replace('\\', '_')
        dest = DATA_DIR / safe_name
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        pipeline = IngestionPipeline(db, llm_extraction, graph_id)
        result = pipeline.run_v0_1(str(dest))

        # Trigger v0.2 extraction in background (offloaded to thread to avoid blocking)
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_extraction_safe, db, graph_id, result["paper_id"])

        return {"ok": True, "file": safe_name, **result, "extraction": "started"}

    @app.get("/papers/{paper_id}/extraction-status")
    async def get_extraction_status(paper_id: str):
        """Check extraction progress for a paper."""
        runs = db.conn.execute(
            "SELECT target, status, items_produced, items_accepted, started_at, completed_at "
            "FROM extraction_runs WHERE paper_id = ? ORDER BY created_at DESC",
            (paper_id,)
        ).fetchall()
        return {
            "paper_id": paper_id,
            "runs": [dict(r) for r in runs],
            "total_claims": db.conn.execute("SELECT COUNT(*) FROM claim_blocks WHERE paper_id=?", (paper_id,)).fetchone()[0],
            "total_evidence": db.conn.execute("SELECT COUNT(*) FROM evidence_blocks WHERE paper_id=?", (paper_id,)).fetchone()[0],
            "total_limitations": db.conn.execute("SELECT COUNT(*) FROM limitation_blocks WHERE paper_id=?", (paper_id,)).fetchone()[0],
            "total_concepts": db.conn.execute("SELECT COUNT(*) FROM concept_keys WHERE graph_id IN (SELECT graph_id FROM paper_cards WHERE paper_id=?)", (paper_id,)).fetchone()[0],
        }

    @app.get("/corpora")
    async def list_corpora():
        rows = db.conn.execute("SELECT * FROM corpora ORDER BY created_at DESC").fetchall()
        return {"corpora": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/graphs")
    async def list_graphs(corpus_id: Optional[str] = None):
        if corpus_id:
            rows = db.conn.execute("SELECT * FROM series_graphs WHERE corpus_id = ? ORDER BY created_at DESC", (corpus_id,)).fetchall()
        else:
            rows = db.conn.execute("SELECT * FROM series_graphs ORDER BY created_at DESC").fetchall()
        return {"graphs": [dict(r) for r in rows], "count": len(rows)}

    @app.get("/graph-relations")
    async def list_graph_relations(graph_id: Optional[str] = None, paper_id: Optional[str] = None):
        """Return typed graph relations + node metadata for visualization."""
        # Get relations
        if graph_id:
            rels = db.conn.execute(
                "SELECT * FROM graph_relations WHERE graph_id = ? ORDER BY importance DESC",
                (graph_id,)
            ).fetchall()
        else:
            rels = db.conn.execute("SELECT * FROM graph_relations ORDER BY importance DESC LIMIT 500").fetchall()

        # Collect all node IDs referenced in relations
        node_ids = set()
        for r in rels:
            node_ids.add(r["source_id"])
            node_ids.add(r["target_id"])

        # Fetch claim nodes
        claims = {}
        if node_ids:
            placeholders = ",".join("?" * len(node_ids))
            claim_rows = db.conn.execute(
                f"SELECT claim_id, claim_text, claim_type, extraction_confidence, status, concept_keys FROM claim_blocks WHERE claim_id IN ({placeholders})",
                list(node_ids)
            ).fetchall()
            for cr in claim_rows:
                claims[cr["claim_id"]] = dict(cr)

        # Fetch concept nodes
        concepts = {}
        if graph_id:
            concept_rows = db.conn.execute(
                "SELECT concept_key, namespace, label_zh, definition, status FROM concept_keys WHERE graph_id = ? AND status = 'active'",
                (graph_id,)
            ).fetchall()
            for cc in concept_rows:
                concepts[cc["concept_key"]] = dict(cc)

        return {
            "relations": [dict(r) for r in rels],
            "claims": claims,
            "concepts": concepts,
        }

    @app.get("/graph-full")
    async def get_full_graph(graph_id: str = Query(...), paper_id: Optional[str] = None):
        """Return complete graph data for visualization: claims, evidence, limitations, concepts, relations."""
        # Claims
        claims_sql = "SELECT claim_id, claim_text, claim_type, extraction_confidence, status, concept_keys, evidence_refs, limitation_refs, importance FROM claim_blocks WHERE graph_id = ?"
        params = [graph_id]
        if paper_id:
            claims_sql += " AND paper_id = ?"
            params.append(paper_id)
        claims = [dict(r) for r in db.conn.execute(claims_sql, params).fetchall()]

        # Evidence
        ev_sql = "SELECT evidence_id, evidence_text, evidence_type, strength, supports_claim_ids, status FROM evidence_blocks WHERE graph_id = ?"
        ev_params = [graph_id]
        if paper_id:
            ev_sql += " AND paper_id = ?"
            ev_params.append(paper_id)
        evidence = [dict(r) for r in db.conn.execute(ev_sql, ev_params).fetchall()]

        # Limitations
        lim_sql = "SELECT limitation_id, limitation_text, risk_type, severity, affected_claim_ids, status FROM limitation_blocks WHERE graph_id = ?"
        lim_params = [graph_id]
        if paper_id:
            lim_sql += " AND paper_id = ?"
            lim_params.append(paper_id)
        limitations = [dict(r) for r in db.conn.execute(lim_sql, lim_params).fetchall()]

        # Concepts
        concepts = [dict(r) for r in db.conn.execute(
            "SELECT concept_key, namespace, label_zh, definition, status, parent_keys, child_keys, related_keys FROM concept_keys WHERE graph_id = ? AND status = 'active'",
            (graph_id,)
        ).fetchall()]

        # Relations
        relations = [dict(r) for r in db.conn.execute(
            "SELECT * FROM graph_relations WHERE graph_id = ? ORDER BY importance DESC",
            (graph_id,)
        ).fetchall()]

        # Sections for context ordering
        sections = [dict(r) for r in db.conn.execute(
            "SELECT section_id, heading, heading_path, page_start, page_end FROM section_blocks WHERE graph_id = ? ORDER BY page_start, section_id",
            (graph_id,)
        ).fetchall()]

        return {
            "graph_id": graph_id,
            "paper_id": paper_id,
            "claims": claims,
            "evidence": evidence,
            "limitations": limitations,
            "concepts": concepts,
            "relations": relations,
            "sections": sections,
        }

    @app.post("/papers/{paper_id}/extract")
    async def extract_paper(paper_id: str, graph_id: str = Query(...)):
        """Run v0.2 extraction on a paper."""
        pipeline = IngestionPipeline(db, llm_extraction, graph_id)
        stats = pipeline.run_v0_2(paper_id)
        return {"ok": True, "stats": stats}

    @app.post("/papers/{paper_id}/full")
    async def full_pipeline(paper_id: str, req: ImportRequest, graph_id: str = Query(...)):
        """Run v0.1 + v0.2 together."""
        pipeline = IngestionPipeline(db, llm_extraction, graph_id)
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
