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
        gate_kwargs: dict | None = None,
    ) -> dict:
        """Run hybrid retrieval.

        Returns:
            {
                "claims": [...],
                "evidence": [...],
                "limitations": [...],
                "concepts": [...],
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

        return {
            "claims": list(all_claims.values()),
            "evidence": list(all_evidence.values()),
            "limitations": list(all_limitations.values()),
            "concepts": concepts,
            "decision": decision,
            "method": method,
        }

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
