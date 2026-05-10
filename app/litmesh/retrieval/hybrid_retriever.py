"""
HybridRetriever: combines vector search + FTS5 + graph expansion.

The retrieval pipeline:
  1. ConceptRouter determines relevant concepts
  2. Vector search finds semantically similar claims
  3. FTS5 finds keyword-matching claims
  4. Graph expansion walks from matched claims through typed pointers
  5. Results are merged, deduplicated, and ranked
"""

import json

from .retrieval_gate import decide_retrieval, GateInput
from .concept_router import ConceptRouter


class HybridRetriever:
    """Combined retrieval: vector + full-text + graph expansion."""

    def __init__(self, db, vector_store=None, embed_fn=None, graph_expander=None):
        self.db = db
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.graph_expander = graph_expander
        self.concept_router = ConceptRouter(db)

    def retrieve(
        self,
        query: str,
        graph_scope: list[str],
        top_k: int = 20,
        include_limitations: bool = True,
        include_context_blocks: bool = True,
        context_window: int = 1,
        max_context_blocks: int = 10,
        gate_kwargs: dict | None = None,
    ) -> dict:
        """Run hybrid retrieval.

        Returns:
            {
                "claims": [...],
                "evidence": [...],
                "limitations": [...],
                "concepts": [...],
                "context_blocks": [...],
                "fallback_reason": str,
                "decision": GateDecision,
                "method": "vector" | "fts" | "graph" | "combined",
            }
        """
        # 1. Retrieval gate decision
        gate_inp = GateInput(
            query=query,
            existing_anchor_count=gate_kwargs.get("anchor_count", 0) if gate_kwargs else 0,
            existing_confidence_sum=gate_kwargs.get("confidence_sum", 0) if gate_kwargs else 0,
        )
        decision = decide_retrieval(gate_inp)

        all_claims = {}
        all_evidence = {}
        all_limitations = {}
        context_blocks = {}
        concepts = []

        # 2. Concept routing (always)
        concepts = self.concept_router.route(query, graph_scope, top_k=5)

        method = decision.mode

        # 3. Vector search
        if decision.mode in ("vector", "full") and self.vector_store and self.embed_fn:
            query_vec = self.embed_fn(query) if callable(self.embed_fn) else self.embed_fn.embed(query)
            vec_results = self.vector_store.search(query_vec, top_k=top_k, graph_ids=graph_scope)
            for r in vec_results:
                all_claims[r["id"]] = {**r, "source": "vector"}

        # 4. FTS5 full-text search
        if decision.mode in ("fts_only", "full"):
            fts_results = self.db.search_claims(query, limit=top_k)
            for r in fts_results:
                rid = r["claim_id"]
                if rid not in all_claims:
                    all_claims[rid] = {**r, "source": "fts"}

            if include_context_blocks:
                for block in self.retrieve_context_blocks(
                    query=query,
                    graph_scope=graph_scope,
                    top_k=max(3, top_k // 2),
                    window=context_window,
                ):
                    context_blocks[block["section_id"]] = block

        # 5. Graph expansion: walk from matched claims
        if decision.mode in ("graph_only", "full") and self.graph_expander:
            for claim_id in list(all_claims.keys())[:5]:
                neighbors = self._expand_from_claim(claim_id, graph_scope)
                for n in neighbors:
                    if n["type"] == "evidence" and n["id"] not in all_evidence:
                        all_evidence[n["id"]] = n
                    elif n["type"] == "limitation" and n["id"] not in all_limitations:
                        all_limitations[n["id"]] = n

        # 6. Always inject limitations if requested
        if include_limitations:
            claim_ids = list(all_claims.keys())
            lim_ids = set()
            for cid in claim_ids[:10]:
                rows = self.db.conn.execute(
                    "SELECT source_id, target_id FROM graph_relations WHERE "
                    "(source_id = ? OR target_id = ?) AND relation_type IN ('constrains', 'supports')",
                    (cid, cid)
                ).fetchall()
                for r in rows:
                    other = r["target_id"] if r["source_id"] == cid else r["source_id"]
                    lim_ids.add(other)

            for lim_id in lim_ids:
                row = self.db.conn.execute(
                    "SELECT limitation_id, limitation_text, risk_type, severity FROM limitation_blocks WHERE limitation_id = ?",
                    (lim_id,)
                ).fetchone()
                if row and row["limitation_id"] not in all_limitations:
                    all_limitations[row["limitation_id"]] = dict(row)
                    all_limitations[row["limitation_id"]]["source"] = "limitation_injection"

        # 7. If structured traversal found too little, fall back to paragraph
        # context walking. This is traditional RAG-like chunk traversal, but it
        # is still bounded by graph scope and paragraph window.
        fallback_reason = ""
        if include_context_blocks and len(all_claims) + len(all_evidence) + len(all_limitations) < 2:
            fallback_reason = (
                f"Structured traversal returned only "
                f"{len(all_claims)} claims, {len(all_evidence)} evidence, "
                f"{len(all_limitations)} limitations. Falling back to paragraph context."
            )
            for block in self.retrieve_context_blocks(
                query=query,
                graph_scope=graph_scope,
                top_k=max_context_blocks,
                window=context_window,
            ):
                if len(context_blocks) >= max_context_blocks:
                    break
                context_blocks[block["section_id"]] = block
            if context_blocks and method not in ("full", "chunk_walk"):
                method = f"{method}+chunk_walk" if method != "skip" else "chunk_walk"

        return {
            "claims": list(all_claims.values()),
            "evidence": list(all_evidence.values()),
            "limitations": list(all_limitations.values()),
            "concepts": concepts,
            "context_blocks": list(context_blocks.values()),
            "fallback_reason": fallback_reason,
            "decision": decision,
            "method": method,
        }

    def retrieve_context_blocks(
        self,
        query: str,
        graph_scope: list[str],
        top_k: int = 10,
        window: int = 1,
    ) -> list[dict]:
        """Traditional RAG fallback over paragraph-level SectionBlocks.

        This does not let the LLM roam freely. The program finds matching
        paragraph blocks, expands a small prev/next window, then returns blocks
        in paper order so the model can read local context sequentially.
        """
        seeds = self._search_section_blocks(query, graph_scope, max(1, top_k))
        selected: dict[str, dict] = {}

        for seed in seeds:
            for block in self._context_window(seed["section_id"], window):
                if graph_scope and block["graph_id"] not in graph_scope:
                    continue
                selected[block["section_id"]] = block

        ordered = sorted(
            selected.values(),
            key=lambda b: (
                b.get("paper_id") or "",
                b.get("page_start") or 0,
                b.get("created_at") or "",
                b.get("section_id") or "",
            ),
        )
        for block in ordered:
            block["source"] = "chunk_walk"
        return ordered[:top_k]

    def _search_section_blocks(self, query: str, graph_scope: list[str], limit: int) -> list[dict]:
        """Search paragraph blocks with LIKE terms.

        FTS5 content tables are not always rebuilt in tests/imports, so this
        uses deterministic SQLite LIKE matching for the fallback path.
        """
        terms = [t.strip() for t in query.replace("，", " ").replace("。", " ").split() if len(t.strip()) >= 2]
        if not terms:
            terms = [query.strip()] if len(query.strip()) >= 2 else []
        if not terms:
            return []

        where = []
        params = []
        for term in terms[:4]:
            where.append("(heading LIKE ? OR raw_text LIKE ? OR summary LIKE ?)")
            like = f"%{term}%"
            params.extend([like, like, like])

        graph_clause = ""
        if graph_scope:
            graph_clause = f" AND graph_id IN ({','.join('?' for _ in graph_scope)})"
            params.extend(graph_scope)

        rows = self.db.conn.execute(
            "SELECT section_id, graph_id, paper_id, heading, heading_path, raw_text, "
            "page_start, page_end, prev_section_id, next_section_id, created_at "
            f"FROM section_blocks WHERE ({' OR '.join(where)}){graph_clause} "
            "ORDER BY page_start, created_at LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def _context_window(self, section_id: str, window: int) -> list[dict]:
        """Return a bounded prev/current/next paragraph window."""
        center = self._get_section(section_id)
        if not center:
            return []

        blocks = [center]
        current = center
        for _ in range(window):
            prev_id = current.get("prev_section_id")
            if not prev_id:
                break
            prev = self._get_section(prev_id)
            if not prev:
                break
            blocks.insert(0, prev)
            current = prev

        current = center
        for _ in range(window):
            next_id = current.get("next_section_id")
            if not next_id:
                next_id = self._next_section_from_relation(current["section_id"])
            if not next_id:
                break
            nxt = self._get_section(next_id)
            if not nxt:
                break
            blocks.append(nxt)
            current = nxt

        return blocks

    def _get_section(self, section_id: str) -> dict | None:
        row = self.db.conn.execute(
            "SELECT section_id, graph_id, paper_id, heading, heading_path, raw_text, "
            "page_start, page_end, prev_section_id, next_section_id, created_at "
            "FROM section_blocks WHERE section_id = ?",
            (section_id,),
        ).fetchone()
        return dict(row) if row else None

    def _next_section_from_relation(self, section_id: str) -> str | None:
        row = self.db.conn.execute(
            "SELECT target_id FROM graph_relations WHERE source_id = ? "
            "AND relation_type = 'section_next' ORDER BY importance DESC LIMIT 1",
            (section_id,),
        ).fetchone()
        return row["target_id"] if row else None

    def _expand_from_claim(self, claim_id: str, graph_scope: list[str]) -> list[dict]:
        """Walk one hop from a claim to find related evidence and limitations."""
        neighbors = []

        for edge_type, neighbor_type in [("supports", "evidence"), ("constrains", "limitation")]:
            # Forward: claim -> other
            forward = self.db.conn.execute(
                "SELECT source_id, target_id FROM graph_relations WHERE source_id = ? AND relation_type = ?",
                (claim_id, edge_type)
            ).fetchall()
            for r in forward:
                neighbors.append({"id": r["target_id"], "type": neighbor_type})

            # Backward: other -> claim
            backward = self.db.conn.execute(
                "SELECT source_id, target_id FROM graph_relations WHERE target_id = ? AND relation_type = ?",
                (claim_id, edge_type)
            ).fetchall()
            for r in backward:
                neighbors.append({"id": r["source_id"], "type": neighbor_type})

        return neighbors
