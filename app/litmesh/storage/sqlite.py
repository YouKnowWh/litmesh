"""
SQLite storage layer for LitMesh.

Usage:
    db = LitMeshDB("litmesh.db")
    db.init_schema()

    # v0.1
    corpus = db.insert_corpus(corpus_card)
    paper = db.insert_paper(paper_card)
    section = db.insert_section(section_block)

    # v0.2
    claim = db.insert_claim(claim_block)
    run = db.create_extraction_run(paper_id, graph_id, target="claims")
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _now() -> str:
    return datetime.utcnow().isoformat()


def _json_list(obj) -> str:
    """Serialize a Python list to a JSON string for SQLite storage."""
    return json.dumps(obj, ensure_ascii=False)


def _parse_json_list(raw: str) -> list:
    """Parse a JSON string from SQLite back to a Python list."""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


class LitMeshDB:
    """SQLite database wrapper for LitMesh."""

    def __init__(self, db_path: str = "litmesh.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self):
        if self.conn:
            self.conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def init_schema(self):
        """Execute schema.sql to create all tables."""
        if not self.conn:
            raise RuntimeError("Not connected. Use db.connect() or context manager.")
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(schema)
        self.conn.commit()

    # ---- v0.1: Corpus ----

    def insert_corpus(self, corpus) -> str:
        self.conn.execute(
            """INSERT INTO corpora (corpus_id, name, corpus_type, domain, description,
               source_items, default_graph_id, integration_policy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (corpus.corpus_id, corpus.name, corpus.corpus_type.value, corpus.domain,
             corpus.description, _json_list(corpus.source_items),
             corpus.default_graph_id, corpus.integration_policy.value)
        )
        self.conn.commit()
        return corpus.corpus_id

    def get_corpus(self, corpus_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM corpora WHERE corpus_id = ?", (corpus_id,)).fetchone()
        if not row:
            return None
        return dict(row)

    # ---- SeriesGraph ----

    def insert_graph(self, graph) -> str:
        self.conn.execute(
            """INSERT INTO series_graphs (graph_id, corpus_id, name, graph_type, domain,
               description, concept_namespace, merge_policy, cross_graph_policy, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (graph.graph_id, graph.corpus_id, graph.name, graph.graph_type.value,
             graph.domain, graph.description, graph.concept_namespace,
             graph.merge_policy, graph.cross_graph_policy.value, graph.status.value)
        )
        self.conn.commit()
        return graph.graph_id

    # ---- PaperCard ----

    def insert_paper(self, paper) -> str:
        self.conn.execute(
            """INSERT INTO paper_cards (paper_id, graph_id, title, authors, year,
               source_file, abstract, abstract_summary, keywords, research_type,
               main_framework, domain_keys, raw_text_hash, page_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (paper.paper_id, paper.graph_id, paper.title, _json_list(paper.authors),
             paper.year, paper.source_file, paper.abstract, paper.abstract_summary,
             _json_list(paper.keywords), paper.research_type.value,
             paper.main_framework, _json_list(paper.domain_keys),
             paper.raw_text_hash, paper.page_count)
        )
        self.conn.commit()
        return paper.paper_id

    def get_paper(self, paper_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM paper_cards WHERE paper_id = ?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def list_papers(self, graph_id: Optional[str] = None) -> list[dict]:
        if graph_id:
            rows = self.conn.execute(
                "SELECT * FROM paper_cards WHERE graph_id = ? ORDER BY year DESC", (graph_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM paper_cards ORDER BY year DESC").fetchall()
        return [dict(r) for r in rows]

    # ---- SectionBlock ----

    def insert_section(self, section) -> str:
        self.conn.execute(
            """INSERT INTO section_blocks (section_id, graph_id, paper_id, heading,
               heading_path, heading_level, raw_text, summary, page_start, page_end,
               concept_keys, parent_section_id, prev_section_id, next_section_id, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (section.section_id, section.graph_id, section.paper_id, section.heading,
             _json_list(section.heading_path), section.heading_level.value,
             section.raw_text, section.summary, section.page_start, section.page_end,
             _json_list(section.concept_keys), section.parent_section_id,
             section.prev_section_id, section.next_section_id, section.content_hash)
        )
        self.conn.commit()
        return section.section_id

    def get_sections_by_paper(self, paper_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM section_blocks WHERE paper_id = ? ORDER BY page_start, section_id",
            (paper_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def update_section_summary(self, section_id: str, summary: str):
        self.conn.execute(
            "UPDATE section_blocks SET summary = ?, updated_at = ? WHERE section_id = ?",
            (summary, _now(), section_id)
        )
        self.conn.commit()

    # ---- SourceSpan ----

    def insert_span(self, span) -> str:
        self.conn.execute(
            """INSERT INTO source_spans (span_id, paper_id, section_id, span_type,
               source_text, char_start, char_end, page_start, page_end, line_start,
               line_end, pdf_bbox, content_hash, normalized_text, verified, verified_by, verified_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (span.span_id, span.paper_id, span.section_id, span.span_type.value,
             span.source_text, span.position.char_start, span.position.char_end,
             span.position.page_start, span.position.page_end,
             span.position.line_start, span.position.line_end,
             span.position.pdf_bbox, span.content_hash, span.normalized_text,
             int(span.verified), span.verified_by,
             span.verified_at.isoformat() if span.verified_at else None)
        )
        self.conn.commit()
        return span.span_id

    def get_spans_by_paper(self, paper_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM source_spans WHERE paper_id = ? ORDER BY char_start",
            (paper_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- v0.2: ClaimBlock ----

    def insert_claim(self, claim) -> str:
        self.conn.execute(
            """INSERT INTO claim_blocks (claim_id, graph_id, paper_id, section_id,
               claim_text, normalized_claim, claim_type, concept_keys, evidence_refs,
               limitation_refs, extraction_confidence, claim_confidence, importance,
               status, source_span_id, extraction_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (claim.claim_id, claim.graph_id, claim.paper_id, claim.section_id,
             claim.claim_text, claim.normalized_claim, claim.claim_type.value,
             _json_list(claim.concept_keys), _json_list(claim.evidence_refs),
             _json_list(claim.limitation_refs), claim.extraction_confidence,
             claim.claim_confidence, claim.importance.value,
             claim.status.value, claim.source_span_id, claim.extraction_run_id)
        )
        self.conn.commit()
        return claim.claim_id

    def get_claims_by_paper(self, paper_id: str, status: Optional[str] = None) -> list[dict]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM claim_blocks WHERE paper_id = ? AND status = ?",
                (paper_id, status)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM claim_blocks WHERE paper_id = ?", (paper_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_claim_status(self, claim_id: str, status: str):
        self.conn.execute(
            "UPDATE claim_blocks SET status = ?, updated_at = ? WHERE claim_id = ?",
            (status, _now(), claim_id)
        )
        self.conn.commit()

    # ---- EvidenceBlock ----

    def insert_evidence(self, evidence) -> str:
        self.conn.execute(
            """INSERT INTO evidence_blocks (evidence_id, graph_id, paper_id, section_id,
               supports_claim_ids, evidence_text, evidence_type, strength, concept_keys,
               source_span_id, extraction_run_id, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (evidence.evidence_id, evidence.graph_id, evidence.paper_id,
             evidence.section_id, _json_list(evidence.supports_claim_ids),
             evidence.evidence_text, evidence.evidence_type.value,
             evidence.strength.value, _json_list(evidence.concept_keys),
             evidence.source_span_id, evidence.extraction_run_id, evidence.status.value)
        )
        self.conn.commit()
        return evidence.evidence_id

    # ---- LimitationBlock ----

    def insert_limitation(self, limitation) -> str:
        self.conn.execute(
            """INSERT INTO limitation_blocks (limitation_id, graph_id, paper_id, section_id,
               limitation_text, affected_claim_ids, risk_type, severity, concept_keys,
               source_span_id, extraction_run_id, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (limitation.limitation_id, limitation.graph_id, limitation.paper_id,
             limitation.section_id, limitation.limitation_text,
             _json_list(limitation.affected_claim_ids), limitation.risk_type.value,
             limitation.severity.value, _json_list(limitation.concept_keys),
             limitation.source_span_id, limitation.extraction_run_id,
             limitation.status.value)
        )
        self.conn.commit()
        return limitation.limitation_id

    # ---- ConceptKey ----

    def insert_concept(self, concept) -> str:
        self.conn.execute(
            """INSERT INTO concept_keys (concept_key, graph_id, namespace, label_zh, label_en,
               definition, aliases, parent_keys, child_keys, related_keys, do_not_merge_with,
               status, review_status, merge_policy, extraction_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (concept.concept_key, concept.graph_id, concept.namespace.value,
             concept.label_zh, concept.label_en, concept.definition,
             _json_list(concept.aliases), _json_list(concept.parent_keys),
             _json_list(concept.child_keys), _json_list(concept.related_keys),
             _json_list(concept.do_not_merge_with), concept.status.value,
             concept.review_status.value, concept.merge_policy.value,
             concept.extraction_run_id)
        )
        self.conn.commit()
        return concept.concept_key

    def find_concept_by_alias(self, alias: str, graph_id: Optional[str] = None) -> list[dict]:
        """Search for concepts by alias (exact match)."""
        if graph_id:
            rows = self.conn.execute(
                "SELECT * FROM concept_keys WHERE graph_id = ? AND aliases LIKE ?",
                (graph_id, f'%"{alias}"%')
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM concept_keys WHERE aliases LIKE ?",
                (f'%"{alias}"%',)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- GraphRelation ----

    def insert_relation(self, relation) -> str:
        self.conn.execute(
            """INSERT INTO graph_relations (relation_id, graph_id, source_id, target_id,
               source_type, target_type, relation_type, confidence, importance,
               traversal_cost, extraction_run_id, evidence_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (relation.relation_id, relation.graph_id, relation.source_id,
             relation.target_id, relation.source_type, relation.target_type,
             relation.relation_type.value, relation.confidence, relation.importance,
             relation.traversal_cost, relation.extraction_run_id, relation.evidence_json)
        )
        self.conn.commit()
        return relation.relation_id

    def get_relations_from(self, source_id: str, relation_type: Optional[str] = None) -> list[dict]:
        if relation_type:
            rows = self.conn.execute(
                "SELECT * FROM graph_relations WHERE source_id = ? AND relation_type = ?",
                (source_id, relation_type)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM graph_relations WHERE source_id = ?", (source_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- BridgeRelation ----

    def insert_bridge(self, bridge) -> str:
        self.conn.execute(
            """INSERT INTO bridge_relations (bridge_id, source_graph_id, target_graph_id,
               source_key, target_key, bridge_type, bridge_confidence, evidence_json,
               warning, review_status, extraction_run_id, traversal_cost)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bridge.bridge_id, bridge.source_graph_id, bridge.target_graph_id,
             bridge.source_key, bridge.target_key, bridge.bridge_type.value,
             bridge.bridge_confidence, bridge.evidence_json, bridge.warning,
             bridge.review_status.value, bridge.extraction_run_id, bridge.traversal_cost)
        )
        self.conn.commit()
        return bridge.bridge_id

    # ---- ExtractionRun ----

    def create_extraction_run(self, run) -> str:
        self.conn.execute(
            """INSERT INTO extraction_runs (run_id, paper_id, graph_id, target, status, model,
               prompt_version, prompt_template, section_ids, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run.run_id, run.paper_id, run.graph_id, run.target.value,
             run.status.value, run.model, run.prompt_version, run.prompt_template,
             _json_list(run.section_ids), _now())
        )
        self.conn.commit()
        return run.run_id

    def complete_extraction_run(self, run_id: str, items_produced: int,
                                 items_accepted: int, items_rejected: int,
                                 input_tokens: int = 0, output_tokens: int = 0,
                                 cost: float = 0.0):
        self.conn.execute(
            """UPDATE extraction_runs SET status = 'completed', items_produced = ?,
               items_accepted = ?, items_rejected = ?, input_token_count = ?,
               output_token_count = ?, total_cost_usd = ?, completed_at = ?
               WHERE run_id = ?""",
            (items_produced, items_accepted, items_rejected, input_tokens,
             output_tokens, cost, _now(), run_id)
        )
        self.conn.commit()

    def fail_extraction_run(self, run_id: str, error: str):
        self.conn.execute(
            "UPDATE extraction_runs SET status = 'failed', error_message = ?, completed_at = ? WHERE run_id = ?",
            (error, _now(), run_id)
        )
        self.conn.commit()

    def rollback_extraction_run(self, run_id: str, rolled_back_by: str = "system"):
        """Mark a run as rolled back; items remain but can be filtered by status."""
        self.conn.execute(
            "UPDATE extraction_runs SET status = 'rolled_back', rolled_back_at = ?, rolled_back_by = ? WHERE run_id = ?",
            (_now(), rolled_back_by, run_id)
        )
        # Also mark all items from this run as rejected in inbox
        self.conn.execute(
            "UPDATE review_inbox SET decision = 'reject', decision_notes = 'Run rolled back' WHERE extraction_run_id = ? AND decision = 'pending'",
            (run_id,)
        )
        self.conn.commit()

    def insert_extraction_item(self, run_id: str, target_type: str, target_id: str,
                                raw_output: str = "", confidence: float = 0.0, accepted: bool = False):
        import uuid
        item_id = f"item_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            """INSERT INTO extraction_run_items (item_id, run_id, target_type, target_id,
               raw_llm_output, extraction_confidence, accepted)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (item_id, run_id, target_type, target_id, raw_output, confidence, int(accepted))
        )
        self.conn.commit()
        return item_id

    # ---- ReviewInbox ----

    def insert_inbox_item(self, inbox_item) -> str:
        self.conn.execute(
            """INSERT INTO review_inbox (inbox_id, inbox_type, item_id, item_type, title,
               description, source_text, extraction_confidence, priority, decision,
               extraction_run_id, graph_id, paper_id, suggested_actions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (inbox_item.inbox_id, inbox_item.inbox_type.value, inbox_item.item_id,
             inbox_item.item_type, inbox_item.title, inbox_item.description,
             inbox_item.source_text, inbox_item.extraction_confidence,
             inbox_item.priority.value, inbox_item.decision.value,
             inbox_item.extraction_run_id, inbox_item.graph_id, inbox_item.paper_id,
             _json_list([a.value for a in inbox_item.suggested_actions]))
        )
        self.conn.commit()
        return inbox_item.inbox_id

    def get_pending_inbox(self, inbox_type: Optional[str] = None) -> list[dict]:
        if inbox_type:
            rows = self.conn.execute(
                "SELECT * FROM review_inbox WHERE inbox_type = ? AND decision = 'pending' ORDER BY priority DESC, created_at",
                (inbox_type,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM review_inbox WHERE decision = 'pending' ORDER BY priority DESC, created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_inbox_item(self, inbox_id: str, decision: str, notes: str = "",
                            decided_by: str = "human", merge_target_id: Optional[str] = None):
        self.conn.execute(
            """UPDATE review_inbox SET decision = ?, decision_notes = ?, decided_by = ?,
               decided_at = ?, merge_target_id = ?
               WHERE inbox_id = ?""",
            (decision, notes, decided_by, _now(), merge_target_id, inbox_id)
        )
        self.conn.commit()

    # ---- TraversalTrace ----

    def insert_trace(self, trace_id: str, query: str, plan_json: str = "{}",
                      result_json: str = "{}"):
        self.conn.execute(
            "INSERT INTO traversal_traces (trace_id, query, plan_json, result_json) VALUES (?, ?, ?, ?)",
            (trace_id, query, plan_json, result_json)
        )
        self.conn.commit()

    # ---- SeriesGroup ----

    def insert_series_group(self, group) -> str:
        self.conn.execute(
            """INSERT INTO series_groups (group_id, name, graph_ids, domain, description, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (group.group_id, group.name, json.dumps(group.graph_ids, ensure_ascii=False),
             group.domain, group.description, group.confidence)
        )
        self.conn.commit()
        return group.group_id

    def get_series_group(self, group_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM series_groups WHERE group_id = ?", (group_id,)).fetchone()
        return dict(row) if row else None

    def list_series_groups(self, domain: str = "") -> list[dict]:
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM series_groups WHERE domain = ? ORDER BY created_at DESC", (domain,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM series_groups ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def find_series_group_for_graph(self, graph_id: str) -> dict | None:
        rows = self.conn.execute("SELECT * FROM series_groups").fetchall()
        for r in rows:
            graph_ids = _parse_json_list(r["graph_ids"]) if isinstance(r["graph_ids"], str) else r["graph_ids"]
            if graph_id in graph_ids:
                return dict(r)
        return None

    def add_graph_to_series_group(self, group_id: str, graph_id: str):
        row = self.conn.execute("SELECT graph_ids FROM series_groups WHERE group_id = ?", (group_id,)).fetchone()
        if row:
            ids = _parse_json_list(row["graph_ids"]) if isinstance(row["graph_ids"], str) else row["graph_ids"]
            if graph_id not in ids:
                ids.append(graph_id)
                self.conn.execute(
                    "UPDATE series_groups SET graph_ids = ?, updated_at = datetime('now') WHERE group_id = ?",
                    (json.dumps(ids, ensure_ascii=False), group_id)
                )
                self.conn.commit()

    # ---- FTS search helpers ----

    def search_sections(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 search on sections. Falls back to LIKE if FTS has no content."""
        try:
            rows = self.conn.execute(
                "SELECT section_id, heading, raw_text, summary FROM section_fts WHERE section_fts MATCH ? LIMIT ?",
                (query, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return self._like_search("section_blocks", "heading, raw_text", query, limit, "section_id")

    def search_claims(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 search on claims. Falls back to LIKE if FTS has no content."""
        try:
            rows = self.conn.execute(
                "SELECT claim_id, claim_text FROM claim_fts WHERE claim_fts MATCH ? LIMIT ?",
                (query, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return self._like_search("claim_blocks", "claim_text", query, limit, "claim_id")

    def search_papers(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 search on papers. Falls back to LIKE if FTS has no content."""
        try:
            rows = self.conn.execute(
                "SELECT paper_id, title, abstract FROM paper_fts WHERE paper_fts MATCH ? LIMIT ?",
                (query, limit)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return self._like_search("paper_cards", "title, abstract", query, limit, "paper_id")

    def _like_search(self, table: str, cols: str, query: str, limit: int, id_col: str) -> list[dict]:
        """Fallback LIKE search when FTS is unavailable."""
        words = [w for w in query.split() if len(w) >= 2]
        if not words:
            return []
        conditions = " OR ".join([f"{c} LIKE '%{w}%'" for c in cols.split(", ") for w in words[:3]])
        rows = self.conn.execute(
            f"SELECT {id_col}, {cols} FROM {table} WHERE {conditions} LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Concept graph traversal ----

    def get_concept_hierarchy(self, concept_key: str) -> dict:
        """Get parent chain and immediate children for a concept."""
        row = self.conn.execute(
            "SELECT concept_key, parent_keys, child_keys, label_zh, label_en, definition, namespace FROM concept_keys WHERE concept_key = ?",
            (concept_key,)
        ).fetchone()
        if not row:
            return {"concept_key": concept_key, "parents": [], "children": [], "siblings": []}

        parents = [dict(r) for r in self.conn.execute(
            "SELECT concept_key, label_zh, label_en, namespace FROM concept_keys WHERE concept_key IN (SELECT value FROM json_each(?))",
            (row["parent_keys"],)
        ).fetchall()]

        children = [dict(r) for r in self.conn.execute(
            "SELECT concept_key, label_zh, label_en, namespace FROM concept_keys WHERE concept_key IN (SELECT value FROM json_each(?))",
            (row["child_keys"],)
        ).fetchall()]

        # Find siblings (share at least one parent)
        siblings = []
        if row["parent_keys"]:
            for parent_key in json.loads(row["parent_keys"]):
                p_row = self.conn.execute(
                    "SELECT child_keys FROM concept_keys WHERE concept_key = ?", (parent_key,)
                ).fetchone()
                if p_row:
                    sibling_keys = json.loads(p_row["child_keys"])
                    for sk in sibling_keys:
                        if sk != concept_key:
                            s = self.conn.execute(
                                "SELECT concept_key, label_zh, namespace FROM concept_keys WHERE concept_key = ?",
                                (sk,)
                            ).fetchone()
                            if s and dict(s) not in siblings:
                                siblings.append(dict(s))

        return {
            "concept_key": concept_key,
            "label_zh": row["label_zh"],
            "label_en": row["label_en"],
            "definition": row["definition"],
            "namespace": row["namespace"],
            "parents": parents,
            "children": children,
            "siblings": siblings,
        }

    def get_blocks_by_concept(self, concept_key: str, graph_id: str) -> dict:
        """Get all claims, evidence, and limitations linked to a concept."""
        claims = [dict(r) for r in self.conn.execute(
            "SELECT claim_id, claim_text, claim_type, status, extraction_confidence FROM claim_blocks WHERE graph_id = ? AND concept_keys LIKE ?",
            (graph_id, f'%"{concept_key}"%')
        ).fetchall()]

        evidence = [dict(r) for r in self.conn.execute(
            "SELECT evidence_id, evidence_text, evidence_type, status FROM evidence_blocks WHERE graph_id = ? AND concept_keys LIKE ?",
            (graph_id, f'%"{concept_key}"%')
        ).fetchall()]

        limitations = [dict(r) for r in self.conn.execute(
            "SELECT limitation_id, limitation_text, risk_type, severity, status FROM limitation_blocks WHERE graph_id = ? AND concept_keys LIKE ?",
            (graph_id, f'%"{concept_key}"%')
        ).fetchall()]

        return {"claims": claims, "evidence": evidence, "limitations": limitations}

    def expand_concept_neighborhood(
        self, concept_key: str, graph_id: str, max_depth: int = 2
    ) -> list[dict]:
        """BFS from a concept through related concepts (parent/child/related).
        Stays within the same graph.
        """
        visited = set()
        queue = [(concept_key, 0)]
        result = []

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            row = self.conn.execute(
                "SELECT concept_key, label_zh, namespace, parent_keys, child_keys, related_keys FROM concept_keys WHERE concept_key = ? AND graph_id = ?",
                (current, graph_id)
            ).fetchone()
            if not row:
                continue

            node = dict(row)
            node["depth"] = depth
            result.append(node)

            if depth < max_depth:
                for key_list in ["parent_keys", "child_keys", "related_keys"]:
                    try:
                        keys = json.loads(node.get(key_list, "[]"))
                    except (json.JSONDecodeError, TypeError):
                        keys = []
                    for k in keys:
                        if k not in visited:
                            queue.append((k, depth + 1))

        return result

    def find_concepts_by_namespace(self, graph_id: str, namespace: str) -> list[dict]:
        """List all concepts in a namespace within a graph."""
        rows = self.conn.execute(
            "SELECT * FROM concept_keys WHERE graph_id = ? AND namespace = ? ORDER BY label_zh",
            (graph_id, namespace)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_relations_between(
        self, source_id: str, target_id: str
    ) -> list[dict]:
        """Get all relations between two nodes."""
        rows = self.conn.execute(
            "SELECT * FROM graph_relations WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
            (source_id, target_id, target_id, source_id)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Stats ----

    def get_stats(self) -> dict:
        tables = [
            "corpora", "series_graphs", "paper_cards", "section_blocks", "source_spans",
            "claim_blocks", "evidence_blocks", "limitation_blocks", "concept_keys",
            "graph_relations", "bridge_relations", "extraction_runs", "review_inbox"
        ]
        stats = {}
        for t in tables:
            row = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {t}").fetchone()
            stats[t] = row["cnt"]
        return stats
