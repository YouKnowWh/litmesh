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
import asyncio
from pathlib import Path
from typing import Optional
from uuid import uuid4

import shutil
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..models.corpus import CorpusCard, CorpusType, IntegrationPolicy
from ..models.graph import SeriesGraph, GraphType, CrossGraphPolicy
from ..ingestion.pipeline import IngestionPipeline
from ..ingestion.progress import ProgressTracker
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
    parser: str = "auto"
    segmenter: str = "auto"

DATA_DIR = Path("data")

class InboxDecision(BaseModel):
    reason: str = ""
    new_confidence: Optional[float] = None


def _run_pipeline_safe(db, llm_extraction, llm_segment, pdf_path, parser, segmenter,
                        graph_id, tracker, task_id, paper_id=""):
    """Run v0.1 + v0.2 with progress reporting in a background thread."""
    import logging
    logger = logging.getLogger("litmesh")
    try:
        tracker.emit(task_id, "parsing", "Parsing PDF...", 5.0,
                     paper_id=paper_id)

        pipeline = IngestionPipeline(
            db, llm_extraction, graph_id or "",
            parser_name=parser,
            segmenter_name=segmenter,
            segment_llm_client=llm_segment,
            progress_tracker=tracker,
            task_id=task_id,
        )

        v01 = pipeline.run_v0_1(pdf_path, existing_paper_id=paper_id)
        tracker.emit(task_id, "v01_done",
                     f"v0.1 complete: {v01['section_count']} sections",
                     50.0, paper_id=v01["paper_id"],
                     section_count=v01["section_count"])

        v02 = pipeline.run_v0_2(v01["paper_id"])
        tracker.emit(task_id, "done", "Full pipeline complete", 100.0,
                     paper_id=v01["paper_id"], graph_id=v01["graph_id"],
                     title=v01.get("title", ""), **v02)

        logger.info("Pipeline complete: paper=%s claims=%s evidence=%s",
                    v01["paper_id"], v02.get("claims", 0), v02.get("evidence", 0))

    except Exception as e:
        logger.exception("Pipeline failed for %s", pdf_path)
        tracker.emit(task_id, "failed", str(e), -1.0)

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
        llm_segment = llm_clients.segment
        llm_review = llm_clients.review
    elif llm_client is not None:
        llm_extraction = llm_client
        llm_segment = llm_client
        llm_review = llm_client
    else:
        llm_extraction = LLMClient()
        llm_segment = llm_extraction
        llm_review = llm_extraction

    # Store embed_provider for use by endpoints that need it
    app_state = {"embed_provider": embed_provider}

    app = FastAPI(title="LitMesh API", version="0.1.0")
    app.state.progress_tracker = ProgressTracker()
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
        pipeline = IngestionPipeline(
            db,
            llm_extraction,
            graph_id,
            parser_name=req.parser,
            segmenter_name=req.segmenter,
            segment_llm_client=llm_segment,
        )
        result = pipeline.run_v0_1(req.pdf_path)
        return {"ok": True, **result}

    @app.post("/papers/upload")
    async def upload_paper(
        file: UploadFile = File(...),
        graph_id: str = Query(default=""),
        parser: str = Query("auto"),
        segmenter: str = Query("auto"),
    ):
        """Upload a PDF and start full pipeline in background. Returns task_id + paper_id for SSE."""
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are accepted")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = file.filename.replace('/', '_').replace('\\', '_')
        dest = DATA_DIR / safe_name
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Pre-create placeholder paper so it survives page refresh
        paper_id = f"paper_{uuid4().hex[:12]}"
        title = safe_name.rsplit('.', 1)[0][:80]
        from ..models.paper import PaperCard, ResearchType
        placeholder = PaperCard(
            paper_id=paper_id, graph_id="", title=title,
            authors=[], year=0, source_file=safe_name,
            research_type=ResearchType.OTHER, main_framework="processing",
        )
        # Auto-create graph (and corpus) if needed
        if not graph_id:
            from ..models.graph import SeriesGraph, GraphType, CrossGraphPolicy
            from ..models.corpus import CorpusCard, CorpusType
            corpus = CorpusCard(
                name="Default Corpus", corpus_type=CorpusType.PAPER_COLLECTION,
                domain="import",
            )
            db.insert_corpus(corpus)
            g = SeriesGraph(
                corpus_id=corpus.corpus_id, name=title, graph_type=GraphType.PAPER_COLLECTION,
                domain="", cross_graph_policy=CrossGraphPolicy.STRICT,
            )
            db.insert_graph(g)
            graph_id = g.graph_id
        placeholder.graph_id = graph_id
        db.insert_paper(placeholder)

        task_id = f"task_{uuid4().hex[:12]}"
        tracker: ProgressTracker = app.state.progress_tracker
        tracker.create_task(task_id, f"Uploaded {file.filename}")

        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            None, _run_pipeline_safe,
            db, llm_extraction, llm_segment,
            str(dest), parser, segmenter, graph_id,
            tracker, task_id, paper_id,
        )

        return {"ok": True, "task_id": task_id, "paper_id": paper_id, "graph_id": graph_id, "file": safe_name}

    @app.get("/papers/{paper_id}/parse-quality")
    async def get_parse_quality(paper_id: str):
        """Return the latest parser quality report for a paper."""
        paper = db.get_paper(paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found")
        return {"paper_id": paper_id, "report": db.get_parse_quality_report(paper_id)}

    @app.get("/papers/{paper_id}/parse-audit")
    async def get_parse_audit(paper_id: str, event: str = "", limit: int = 100):
        """Return audit log events for a paper. Filter by event type."""
        audit_path = Path("logs/parse_audit") / f"{paper_id}.jsonl"
        if not audit_path.exists():
            return {"paper_id": paper_id, "events": [], "count": 0}
        events = []
        with open(audit_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if event and ev.get("event") != event:
                        continue
                    events.append(ev)
                    if len(events) >= limit:
                        break
                except json.JSONDecodeError:
                    pass
        return {"paper_id": paper_id, "events": events, "count": len(events)}

    @app.get("/papers/{paper_id}/segments")
    async def get_paper_segments(
        paper_id: str,
        role: str = "",
        page_start: int = 0,
        page_end: int = 0,
        limit: int = 100,
    ):
        """Query sections by role and/or page range."""
        sections = db.get_sections_by_paper(paper_id)
        result = []
        for s in sections:
            if page_start and (s.get("page_start") or 0) < page_start:
                continue
            if page_end and (s.get("page_start") or 0) > page_end:
                continue
            # Role filter: check if section role matches
            # Note: role is not directly stored on SectionBlock.
            # For now, we return all sections; role filtering happens at ParsedElement level.
            result.append(s)
            if len(result) >= limit:
                break
        return {"paper_id": paper_id, "sections": result, "count": len(result)}

    @app.get("/papers/{paper_id}/outline")
    async def get_paper_outline(paper_id: str):
        """Return the TOC-derived outline, falling back to section paths."""
        paper = db.get_paper(paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found")
        report = db.get_parse_quality_report(paper_id)
        quality = (report or {}).get("quality") or {}
        toc_outline = quality.get("outline") or []
        if toc_outline:
            return {
                "paper_id": paper_id,
                "source": quality.get("toc_source", "toc"),
                "outline": toc_outline,
                "count": len(toc_outline),
            }
        outline = []
        seen = set()
        for section in db.get_sections_by_paper(paper_id):
            try:
                path = json.loads(section.get("heading_path") or "[]")
            except json.JSONDecodeError:
                path = []
            for level, title in enumerate(path[:-1], start=1):
                key = (level, title)
                if title and key not in seen:
                    seen.add(key)
                    outline.append({
                        "level": level,
                        "title": title,
                        "page": section.get("page_start"),
                    })
        return {"paper_id": paper_id, "source": "section_heading_path", "outline": outline, "count": len(outline)}

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

    @app.get("/tasks/{task_id}/events")
    async def stream_task_events(task_id: str):
        """SSE endpoint for real-time pipeline progress."""
        tracker: ProgressTracker = app.state.progress_tracker
        tracker.cleanup_stale()

        async def event_generator():
            last_index = 0
            not_found_attempts = 0
            try:
                while True:
                    latest = tracker.get_latest(task_id)
                    if latest is None:
                        not_found_attempts += 1
                        if not_found_attempts > 20:
                            payload = json.dumps({"stage": "not_found", "message": "Task not found", "percentage": -1.0})
                            yield f"event: error\ndata: {payload}\n\n"
                            return
                        await asyncio.sleep(0.5)
                        continue

                    events = tracker.get_events_since(task_id, last_index)
                    for ev in events:
                        payload = json.dumps({
                            "stage": ev.stage,
                            "message": ev.message,
                            "percentage": ev.percentage,
                            "metadata": ev.metadata,
                            "index": ev.index,
                        })
                        yield f"data: {payload}\n\n"
                        last_index = ev.index + 1

                    if latest.stage == "done":
                        payload = json.dumps({
                            "stage": "done", "message": latest.message,
                            "percentage": 100.0, "metadata": latest.metadata, "index": latest.index,
                        })
                        yield f"event: complete\ndata: {payload}\n\n"
                        return
                    if latest.stage == "failed":
                        payload = json.dumps({
                            "stage": "failed", "message": latest.message,
                            "percentage": -1.0, "metadata": latest.metadata, "index": latest.index,
                        })
                        yield f"event: error\ndata: {payload}\n\n"
                        return

                    not_found_attempts = 0
                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                pass
            finally:
                if tracker.is_done(task_id):
                    tracker.cleanup(task_id)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

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
        claims_sql = "SELECT claim_id, claim_text, claim_type, extraction_confidence, status, concept_keys, evidence_refs, limitation_refs, importance, section_id, paper_id FROM claim_blocks WHERE graph_id = ?"
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
            "SELECT section_id, heading, heading_path, page_start, page_end FROM section_blocks WHERE graph_id = ? ORDER BY global_order_index, page_start, section_id",
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

    # ---- Graph view (lightweight, mode-filtered) ----

    @app.get("/graph-view")
    async def get_graph_view(
        graph_id: str = Query(...),
        paper_id: Optional[str] = None,
        mode: str = "paragraph",
        limit: int = 300,
        center_id: Optional[str] = None,
    ):
        """Return a lightweight subgraph for UI rendering.

        Modes:
          - paragraph: paragraph chain with claims/evidence/limitations/concepts attached.
          - argument: only claim/evidence/limitation/concept nodes.
          - mixed: paragraph window around center_id + argument nodes.
        """
        nodes = []
        edges = []
        stats = {"total_nodes": 0, "total_edges": 0, "returned_nodes": 0, "returned_edges": 0, "has_more": False}

        # Base query filter
        paper_filter = "AND paper_id = ?" if paper_id else ""
        paper_params = [graph_id]
        if paper_id:
            paper_params.append(paper_id)

        if mode == "paragraph":
            # Paragraph nodes sorted by page/order
            p_rows = db.conn.execute(
                f"SELECT section_id, heading, display_title, raw_text, page_start, page_end, paper_id, "
                f"chapter_index, section_index, block_index, global_order_index, parser_name, parser_element_id "
                f"FROM section_blocks WHERE graph_id = ? {paper_filter} "
                f"ORDER BY paper_id, global_order_index, page_start, section_id LIMIT ?",
                (*paper_params, limit)
            ).fetchall()
            p_ids = {r["section_id"] for r in p_rows}
            stats["total_nodes"] = db.conn.execute(
                f"SELECT COUNT(*) FROM section_blocks WHERE graph_id = ? {paper_filter}", paper_params
            ).fetchone()[0]

            for i, r in enumerate(p_rows):
                nodes.append({"id": r["section_id"], "type": "paragraph",
                              "label": (r["display_title"] or r["heading"] or f"P{r['page_start']}")[:80],
                              "text": (r["raw_text"] or "")[:200],
                              "page": r["page_start"] or 0, "order": i,
                              "paper_id": r["paper_id"],
                              "chapter_index": r["chapter_index"],
                              "section_index": r["section_index"],
                              "block_index": r["block_index"],
                              "global_order_index": r["global_order_index"],
                              "parser_name": r["parser_name"],
                              "parser_element_id": r["parser_element_id"]})

            # section_next edges
            for i in range(len(p_rows) - 1):
                edges.append({"source": p_rows[i]["section_id"], "target": p_rows[i+1]["section_id"],
                              "type": "section_next", "confidence": 1.0})

            # Attached claims/evidence/limitations
            if p_ids:
                placeholders = ",".join("?" * len(p_ids))
                c_rows = db.conn.execute(
                    f"SELECT claim_id, section_id, claim_text, extraction_confidence "
                    f"FROM claim_blocks WHERE section_id IN ({placeholders}) LIMIT {limit}",
                    list(p_ids)
                ).fetchall()
                for c in c_rows:
                    nodes.append({"id": c["claim_id"], "type": "claim", "label": c["claim_text"][:60],
                                  "text": c["claim_text"], "page": 0, "order": 0})
                    edges.append({"source": c["section_id"], "target": c["claim_id"],
                                  "type": "belongs_to", "confidence": c["extraction_confidence"]})

                ev_rows = db.conn.execute(
                    f"SELECT evidence_id, section_id, evidence_text FROM evidence_blocks "
                    f"WHERE section_id IN ({placeholders}) LIMIT {limit}", list(p_ids)
                ).fetchall()
                for ev in ev_rows:
                    nodes.append({"id": ev["evidence_id"], "type": "evidence", "label": ev["evidence_text"][:60],
                                  "text": ev["evidence_text"], "page": 0, "order": 0})
                    edges.append({"source": ev["section_id"], "target": ev["evidence_id"],
                                  "type": "belongs_to", "confidence": 0.7})

                lim_rows = db.conn.execute(
                    f"SELECT limitation_id, section_id, limitation_text FROM limitation_blocks "
                    f"WHERE section_id IN ({placeholders}) LIMIT {limit}", list(p_ids)
                ).fetchall()
                for lim in lim_rows:
                    nodes.append({"id": lim["limitation_id"], "type": "limitation",
                                  "label": lim["limitation_text"][:60], "text": lim["limitation_text"],
                                  "page": 0, "order": 0})
                    edges.append({"source": lim["section_id"], "target": lim["limitation_id"],
                                  "type": "belongs_to", "confidence": 0.7})

            # mentions / supports / constrains edges (filtered to visible nodes)
            node_ids = {n["id"] for n in nodes}
            rel_rows = db.conn.execute(
                f"SELECT source_id, target_id, relation_type, confidence FROM graph_relations "
                f"WHERE graph_id = ? LIMIT {limit * 2}", (graph_id,)
            ).fetchall()
            for rel in rel_rows:
                if rel["source_id"] in node_ids and rel["target_id"] in node_ids:
                    edges.append({"source": rel["source_id"], "target": rel["target_id"],
                                  "type": rel["relation_type"], "confidence": rel["confidence"]})

        elif mode == "argument":
            c_rows = db.conn.execute(
                f"SELECT claim_id, claim_text, extraction_confidence, section_id "
                f"FROM claim_blocks WHERE graph_id = ? {paper_filter} ORDER BY extraction_confidence DESC LIMIT ?",
                (*paper_params, limit)
            ).fetchall()
            for c in c_rows:
                nodes.append({"id": c["claim_id"], "type": "claim", "label": c["claim_text"][:60],
                              "text": c["claim_text"], "page": 0, "order": 0})
            # Add evidence/limitations/concepts similarly, plus relations...
            node_ids = {n["id"] for n in nodes}
            rel_rows = db.conn.execute(
                f"SELECT source_id, target_id, relation_type, confidence FROM graph_relations "
                f"WHERE graph_id = ? LIMIT {limit * 2}", (graph_id,)
            ).fetchall()
            for rel in rel_rows:
                if rel["source_id"] in node_ids and rel["target_id"] in node_ids:
                    edges.append({"source": rel["source_id"], "target": rel["target_id"],
                                  "type": rel["relation_type"], "confidence": rel["confidence"]})

        elif mode == "mixed":
            # Paragraph window around center_id + nearby claims
            if center_id:
                center_row = db.conn.execute(
                    "SELECT section_id, page_start FROM section_blocks WHERE section_id = ?", (center_id,)
                ).fetchone()
                if not center_row:
                    center_row = db.conn.execute(
                        "SELECT section_id, page_start FROM claim_blocks WHERE claim_id = ?", (center_id,)
                    ).fetchone()
                window_start = max(0, (center_row["page_start"] or 1) - 3) if center_row else 0
                p_rows = db.conn.execute(
                    f"SELECT section_id, heading, display_title, raw_text, page_start, page_end, paper_id "
                    f"FROM section_blocks WHERE graph_id = ? {paper_filter} "
                    f"AND page_start >= ? AND page_start <= ? "
                    f"ORDER BY global_order_index, page_start, section_id LIMIT {limit}",
                    (*paper_params, window_start, window_start + 6)
                ).fetchall()
            else:
                p_rows = db.conn.execute(
                    f"SELECT section_id, heading, display_title, raw_text, page_start, page_end, paper_id "
                    f"FROM section_blocks WHERE graph_id = ? {paper_filter} "
                    f"ORDER BY global_order_index, page_start, section_id LIMIT 30", paper_params
                ).fetchall()
            p_ids = {r["section_id"] for r in p_rows}
            for i, r in enumerate(p_rows):
                nodes.append({"id": r["section_id"], "type": "paragraph",
                              "label": (r["display_title"] or r["heading"] or f"P{r['page_start']}")[:80],
                              "text": (r["raw_text"] or "")[:200],
                              "page": r["page_start"] or 0, "order": i,
                              "paper_id": r["paper_id"]})
            for i in range(len(p_rows) - 1):
                edges.append({"source": p_rows[i]["section_id"], "target": p_rows[i+1]["section_id"],
                              "type": "section_next", "confidence": 1.0})
            # Add claims within window
            if p_ids:
                for c in db.conn.execute(
                    f"SELECT claim_id, section_id, claim_text, extraction_confidence FROM claim_blocks "
                    f"WHERE section_id IN ({','.join('?'*len(p_ids))}) LIMIT {limit}",
                    list(p_ids)
                ).fetchall():
                    nodes.append({"id": c["claim_id"], "type": "claim", "label": c["claim_text"][:60],
                                  "text": c["claim_text"], "page": 0, "order": 0})
                    edges.append({"source": c["section_id"], "target": c["claim_id"],
                                  "type": "belongs_to", "confidence": c["extraction_confidence"]})

        stats["returned_nodes"] = len(nodes)
        stats["returned_edges"] = len(edges)
        stats["total_edges"] = db.conn.execute(
            "SELECT COUNT(*) FROM graph_relations WHERE graph_id = ?", (graph_id,)
        ).fetchone()[0]
        stats["has_more"] = stats["returned_nodes"] < stats["total_nodes"]

        return {"nodes": nodes, "edges": edges, "stats": stats}

    # ---- Connectivity report ----

    @app.get("/graphs/{graph_id}/connectivity-report")
    async def get_connectivity_report(graph_id: str):
        """Return graph health diagnostics."""
        paragraphs = db.conn.execute("SELECT COUNT(*) FROM section_blocks WHERE graph_id=?", (graph_id,)).fetchone()[0]
        claims = db.conn.execute("SELECT COUNT(*) FROM claim_blocks WHERE graph_id=?", (graph_id,)).fetchone()[0]
        evidence = db.conn.execute("SELECT COUNT(*) FROM evidence_blocks WHERE graph_id=?", (graph_id,)).fetchone()[0]
        limitations = db.conn.execute("SELECT COUNT(*) FROM limitation_blocks WHERE graph_id=?", (graph_id,)).fetchone()[0]
        concepts = db.conn.execute("SELECT COUNT(*) FROM concept_keys WHERE graph_id=?", (graph_id,)).fetchone()[0]

        # Count mentions edges where target concept doesn't exist
        mentions = db.conn.execute(
            "SELECT target_id FROM graph_relations WHERE graph_id=? AND relation_type='mentions'", (graph_id,)
        ).fetchall()
        missing_concepts = 0
        for m in mentions:
            row = db.conn.execute("SELECT 1 FROM concept_keys WHERE concept_key=?", (m["target_id"],)).fetchone()
            if not row:
                missing_concepts += 1

        # Count blocks without belongs_to
        blocks_without = 0
        for tbl, id_col in [("claim_blocks", "claim_id"), ("evidence_blocks", "evidence_id"), ("limitation_blocks", "limitation_id")]:
            rows = db.conn.execute(f"SELECT {id_col} FROM {tbl} WHERE graph_id=? AND paper_id!=''", (graph_id,)).fetchall()
            for r in rows:
                rel = db.conn.execute(
                    "SELECT 1 FROM graph_relations WHERE source_id=? AND relation_type='belongs_to'", (r[id_col],)
                ).fetchone()
                if not rel:
                    blocks_without += 1

        return {
            "graph_id": graph_id,
            "paragraphs": paragraphs, "claims": claims, "evidence": evidence,
            "limitations": limitations, "concepts": concepts,
            "missing_concept_nodes": missing_concepts,
            "blocks_without_belongs_to": blocks_without,
        }

    # ---- Repair connectivity ----

    @app.post("/graphs/{graph_id}/repair-connectivity")
    async def repair_connectivity(graph_id: str):
        """Idempotent repair: section_next, belongs_to, mentions, evidence→claim, limitation→claim."""
        import json as _json
        repaired = {"section_next": 0, "belongs_to": 0, "mentions": 0,
                     "evidence_supports": 0, "limitation_constrains": 0,
                     "concepts_created": 0}

        # 1. section_next for all sections ordered by page_start
        secs = db.conn.execute(
            "SELECT section_id FROM section_blocks WHERE graph_id=? ORDER BY paper_id, global_order_index, page_start, section_id",
            (graph_id,)
        ).fetchall()
        for i in range(len(secs) - 1):
            existing = db.conn.execute(
                "SELECT 1 FROM graph_relations WHERE source_id=? AND target_id=? AND relation_type='section_next'",
                (secs[i]["section_id"], secs[i+1]["section_id"])
            ).fetchone()
            if not existing:
                db.conn.execute(
                    "INSERT INTO graph_relations (relation_id, graph_id, source_id, target_id, source_type, target_type, relation_type, confidence, importance, traversal_cost) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (f"rel_{uuid4().hex[:12]}", graph_id, secs[i]["section_id"], secs[i+1]["section_id"],
                     "section", "section", "section_next", 1.0, 0.3, 0.5)
                )
                repaired["section_next"] += 1
        db.conn.commit()

        # 2. belongs_to for claim/evidence/limitation -> section
        for tbl, id_col, ntype in [("claim_blocks", "claim_id", "claim"),
                                     ("evidence_blocks", "evidence_id", "evidence"),
                                     ("limitation_blocks", "limitation_id", "limitation")]:
            rows = db.conn.execute(
                f"SELECT {id_col}, section_id, graph_id FROM {tbl} WHERE graph_id=? AND section_id IS NOT NULL AND section_id != ''",
                (graph_id,)
            ).fetchall()
            for r in rows:
                existing = db.conn.execute(
                    "SELECT 1 FROM graph_relations WHERE source_id=? AND target_id=? AND relation_type='belongs_to'",
                    (r[id_col], r["section_id"])
                ).fetchone()
                if not existing:
                    db.conn.execute(
                        "INSERT INTO graph_relations (relation_id, graph_id, source_id, target_id, source_type, target_type, relation_type, confidence, importance, traversal_cost) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (f"rel_{uuid4().hex[:12]}", graph_id, r[id_col], r["section_id"],
                         ntype, "section", "belongs_to", 1.0, 0.5, 0.5)
                    )
                    repaired["belongs_to"] += 1
        db.conn.commit()

        # 3. Extract concept_keys from claim concept_keys JSON, create missing concepts, add mentions edges
        claim_rows = db.conn.execute(
            "SELECT claim_id, concept_keys, graph_id FROM claim_blocks WHERE graph_id=? AND concept_keys != '[]'",
            (graph_id,)
        ).fetchall()
        for cr in claim_rows:
            keys = []
            try:
                keys = _json.loads(cr["concept_keys"])
            except Exception:
                continue
            for key in keys:
                # Create concept if missing
                existing = db.conn.execute(
                    "SELECT 1 FROM concept_keys WHERE concept_key=? AND graph_id=?", (key, cr["graph_id"])
                ).fetchone()
                if not existing:
                    try:
                        db.conn.execute(
                            "INSERT INTO concept_keys (concept_key, graph_id, namespace, label_zh, status, review_status) VALUES (?,?,?,?,?,?)",
                            (key, cr["graph_id"], "concept", key.replace("concept:", "").replace("_", " "), "candidate", "pending")
                        )
                        repaired["concepts_created"] += 1
                    except Exception:
                        pass  # Duplicate key, fine

                # Add mentions edge
                existing_edge = db.conn.execute(
                    "SELECT 1 FROM graph_relations WHERE source_id=? AND target_id=? AND relation_type='mentions'",
                    (cr["claim_id"], key)
                ).fetchone()
                if not existing_edge:
                    db.conn.execute(
                        "INSERT INTO graph_relations (relation_id, graph_id, source_id, target_id, source_type, target_type, relation_type, confidence, importance, traversal_cost) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (f"rel_{uuid4().hex[:12]}", graph_id, cr["claim_id"], key,
                         "claim", "concept", "mentions", 0.7, 0.4, 0.5)
                    )
                    repaired["mentions"] += 1
        db.conn.commit()

        # 4. evidence -> claim supports from supports_claim_ids JSON
        ev_rows = db.conn.execute(
            "SELECT evidence_id, supports_claim_ids, graph_id FROM evidence_blocks WHERE graph_id=? AND supports_claim_ids != '[]'",
            (graph_id,)
        ).fetchall()
        for er in ev_rows:
            claim_ids = []
            try:
                claim_ids = _json.loads(er["supports_claim_ids"])
            except Exception:
                continue
            for cid in claim_ids:
                existing = db.conn.execute(
                    "SELECT 1 FROM graph_relations WHERE source_id=? AND target_id=? AND relation_type='supports'",
                    (er["evidence_id"], cid)
                ).fetchone()
                if not existing:
                    db.conn.execute(
                        "INSERT INTO graph_relations (relation_id, graph_id, source_id, target_id, source_type, target_type, relation_type, confidence, importance, traversal_cost) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (f"rel_{uuid4().hex[:12]}", graph_id, er["evidence_id"], cid,
                         "evidence", "claim", "supports", 0.7, 0.5, 0.5)
                    )
                    repaired["evidence_supports"] += 1
        db.conn.commit()

        # 5. limitation -> claim constrains from affected_claim_ids JSON
        lim_rows = db.conn.execute(
            "SELECT limitation_id, affected_claim_ids, graph_id FROM limitation_blocks WHERE graph_id=? AND affected_claim_ids != '[]'",
            (graph_id,)
        ).fetchall()
        for lr in lim_rows:
            claim_ids = []
            try:
                claim_ids = _json.loads(lr["affected_claim_ids"])
            except Exception:
                continue
            for cid in claim_ids:
                existing = db.conn.execute(
                    "SELECT 1 FROM graph_relations WHERE source_id=? AND target_id=? AND relation_type='constrains'",
                    (lr["limitation_id"], cid)
                ).fetchone()
                if not existing:
                    db.conn.execute(
                        "INSERT INTO graph_relations (relation_id, graph_id, source_id, target_id, source_type, target_type, relation_type, confidence, importance, traversal_cost) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (f"rel_{uuid4().hex[:12]}", graph_id, lr["limitation_id"], cid,
                         "limitation", "claim", "constrains", 0.7, 0.5, 0.5)
                    )
                    repaired["limitation_constrains"] += 1
        db.conn.commit()

        return {"ok": True, "graph_id": graph_id, "repaired": repaired}

    @app.post("/papers/{paper_id}/extract")
    async def extract_paper(paper_id: str, graph_id: str = Query(...)):
        """Run v0.2 extraction on a paper."""
        pipeline = IngestionPipeline(db, llm_extraction, graph_id)
        stats = pipeline.run_v0_2(paper_id)
        return {"ok": True, "stats": stats}

    @app.post("/papers/{paper_id}/full")
    async def full_pipeline(paper_id: str, req: ImportRequest, graph_id: str = Query(...)):
        """Run v0.1 + v0.2 together."""
        pipeline = IngestionPipeline(
            db,
            llm_extraction,
            graph_id,
            parser_name=req.parser,
            segmenter_name=req.segmenter,
            segment_llm_client=llm_segment,
        )
        result = pipeline.run_full(req.pdf_path)
        return {"ok": True, **result}

    @app.get("/papers")
    async def list_papers(graph_id: Optional[str] = None):
        papers = db.list_papers(graph_id)
        return {"papers": papers, "count": len(papers)}

    @app.delete("/papers/{paper_id}")
    async def delete_paper(paper_id: str):
        """Delete a paper and all related data."""
        paper = db.get_paper(paper_id)
        if not paper:
            raise HTTPException(404, "Paper not found")
        count = db.delete_paper(paper_id)
        return {"ok": True, "paper_id": paper_id, "deleted_rows": count}

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
        from pathlib import Path as FSPath
        ui_file = FSPath(__file__).parent.parent.parent.parent / "app" / "litmesh" / "api" / "ui.html"
        if not ui_file.exists():
            raise HTTPException(404, "UI not found")
        return HTMLResponse(ui_file.read_text(encoding="utf-8"))

    return app
