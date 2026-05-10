"""
TraversalExecutor: program-controlled typed pointer traversal (v0.9).

Optimizations over v0.4:
- node_index for O(1) node type resolution (avoids 5-table probing)
- Priority queue BFS: low traversal_cost + high importance edges first
- Batch node resolve for queue head expansion
- Fixed confidence vs require_source_span comparison bug
- Enforced graph_scope filtering

Principle: LLM decides pointer types; program controls HOW.
"""

import json
from collections import deque
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Optional

from ..models.prompt_packet import (
    TraversalMode, PointerType, TraversalPlan,
    TraversalResult, VisitedNode, VisitedEdge,
)
from ..models.relation import GraphRelationType, BridgeRelationType


# Map PointerType -> GraphRelationType for SQL queries
_POINTER_TO_RELATION: dict[PointerType, str] = {
    PointerType.BELONGS_TO: GraphRelationType.BELONGS_TO.value,
    PointerType.DERIVED_FROM: GraphRelationType.DERIVED_FROM.value,
    PointerType.SUPPORTS: GraphRelationType.SUPPORTS.value,
    PointerType.CONSTRAINS: GraphRelationType.CONSTRAINS.value,
    PointerType.CONTRADICTS: GraphRelationType.CONTRADICTS.value,
    PointerType.REFINES: GraphRelationType.REFINES.value,
    PointerType.SECTION_PARENT: GraphRelationType.PARENT.value,
    PointerType.SECTION_NEXT: GraphRelationType.NEXT.value,
    PointerType.EXTENDS: GraphRelationType.EXTENDS.value,
    PointerType.SUPERSEDES: GraphRelationType.SUPERSEDES.value,
    PointerType.SAME_AS: GraphRelationType.SAME_AS.value,
}

_BRIDGE_POINTER_TO_TYPE: dict[PointerType, str] = {
    PointerType.ANALOGOUS_TO_BRIDGE: BridgeRelationType.ANALOGOUS_TO.value,
    PointerType.TRANSFERS_TO_BRIDGE: BridgeRelationType.TRANSFERS_TO.value,
    PointerType.CONFLICTS_WITH_BRIDGE: BridgeRelationType.CONFLICTS_WITH.value,
}

# Bidirectional pointer types: walk both outgoing and incoming edges
_BIDIRECTIONAL = {
    PointerType.SUPPORTS, PointerType.CONSTRAINS,
    PointerType.CONTRADICTS, PointerType.REFINES,
    PointerType.EXTENDS, PointerType.SUPERSEDES,
    PointerType.DERIVED_FROM,
}


class TraversalExecutor:
    """Executes a TraversalPlan against the SQLite knowledge graph."""

    def __init__(self, db):
        self.db = db

    def execute(self, plan: TraversalPlan) -> TraversalResult:
        visited_nodes: dict[str, VisitedNode] = {}
        visited_edges: list[VisitedEdge] = []
        cross_graph_jumps = 0
        budget_used = 0

        # Priority queue: (priority_score, node_id, depth)
        # Lower score = higher priority
        # Score = depth * 100 + traversal_cost * 10 - importance * 3 - confidence * 2
        queue = []
        for start in plan.start_nodes:
            heappush(queue, (0, start, 0))

        # Batch resolve the initial nodes
        start_types = self.db.batch_resolve_nodes([s for _, s, _ in queue])
        for i, (score, nid, depth) in enumerate(list(queue)):
            if nid not in start_types:
                queue.pop(0)  # Node not in index, skip
                continue

        while queue and len(visited_nodes) < plan.max_nodes and budget_used < plan.budget:
            _, current_id, depth = heappop(queue)

            if current_id in visited_nodes:
                continue
            if depth > plan.max_depth:
                continue

            # Fast resolve via node_index
            node = self._resolve_node_fast(current_id, plan)
            if not node:
                continue

            # Enforce graph_scope
            if plan.graph_scope and node.graph_id not in plan.graph_scope:
                continue

            # Source span gate: only enforce for claim/evidence/limitation
            if (plan.require_source_span
                    and node.node_type in ("claim", "evidence", "limitation")
                    and not node.source_span_id):
                continue

            visited_nodes[current_id] = node
            budget_used += len(node.text) // 4

            if depth >= plan.max_depth:
                continue

            # Walk each enabled pointer type
            for pointer_type in plan.pointer_types:
                edge_count = sum(1 for e in visited_edges if _pointer_key(e) == pointer_type.value)
                if edge_count >= plan.max_edges_per_pointer_type:
                    continue

                if pointer_type in _POINTER_TO_RELATION:
                    neighbors = self._walk_relation(
                        current_id, _POINTER_TO_RELATION[pointer_type],
                        pointer_type, plan
                    )
                elif pointer_type in _BRIDGE_POINTER_TO_TYPE:
                    if not plan.allow_cross_graph or cross_graph_jumps >= plan.max_cross_graph_jumps:
                        continue
                    neighbors = self._walk_bridge(
                        current_id, node.graph_id,
                        _BRIDGE_POINTER_TO_TYPE[pointer_type], pointer_type, plan
                    )
                    cross_graph_jumps += len(neighbors)
                else:
                    continue

                for neighbor_id, edge_info in neighbors:
                    if edge_info["confidence"] < plan.min_confidence:
                        continue

                    visited_edges.append(VisitedEdge(
                        source_id=current_id,
                        target_id=neighbor_id,
                        relation_type=pointer_type.value,
                        is_cross_graph=pointer_type in _BRIDGE_POINTER_TO_TYPE,
                        confidence=edge_info["confidence"],
                    ))

                    if neighbor_id not in visited_nodes:
                        # Priority: low cost + high importance + high confidence
                        priority = (
                            (depth + 1) * 100
                            + edge_info.get("traversal_cost", 1.0) * 10
                            - edge_info.get("importance", 0.5) * 3
                            - edge_info["confidence"] * 2
                        )
                        heappush(queue, (priority, neighbor_id, depth + 1))

        stopped_reason = ""
        if len(visited_nodes) >= plan.max_nodes:
            stopped_reason = f"Reached max_nodes limit ({plan.max_nodes})"
        elif budget_used >= plan.budget:
            stopped_reason = f"Reached budget limit ({plan.budget})"
        elif not queue:
            stopped_reason = "No more nodes to traverse"

        grouped_claims = [nid for nid, n in visited_nodes.items() if n.node_type == "claim"]
        grouped_evidence = [nid for nid, n in visited_nodes.items() if n.node_type == "evidence"]
        grouped_limitations = [nid for nid, n in visited_nodes.items() if n.node_type == "limitation"]

        conflicts = [
            nid for nid, n in visited_nodes.items()
            if n.node_type == "claim" and any(
                e.relation_type == "contradicts" and (e.source_id == nid or e.target_id == nid)
                for e in visited_edges
            )
        ]

        bridge_ids = [e.target_id for e in visited_edges if e.is_cross_graph]
        total_cost = sum(e.confidence for e in visited_edges)

        return TraversalResult(
            plan_id=plan.plan_id,
            visited_nodes=list(visited_nodes.values()),
            visited_edges=visited_edges,
            grouped_claims=grouped_claims,
            grouped_evidence=grouped_evidence,
            grouped_limitations=grouped_limitations,
            conflicts=conflicts,
            bridge_relations=bridge_ids,
            stopped_reason=stopped_reason,
            total_traversal_cost=round(total_cost, 2),
        )

    def _resolve_node_fast(self, node_id: str, plan: TraversalPlan) -> Optional[VisitedNode]:
        """Resolve a node using node_index for type lookup (O(1) instead of O(5) table probes)."""
        node_type = self.db.get_node_type(node_id)
        if not node_type:
            return None

        if node_type == "claim":
            row = self.db.conn.execute(
                "SELECT claim_id, claim_text, claim_type, extraction_confidence, importance, "
                "source_span_id, graph_id, paper_id, concept_keys "
                "FROM claim_blocks WHERE claim_id = ?", (node_id,)
            ).fetchone()
            if row:
                return VisitedNode(
                    node_id=row["claim_id"], node_type="claim",
                    title=row["claim_text"][:80], text=row["claim_text"],
                    confidence=row["extraction_confidence"],
                    importance=_importance_value(row["importance"]),
                    source_span_id=row["source_span_id"],
                    graph_id=row["graph_id"], paper_id=row["paper_id"],
                    concept_keys=json.loads(row["concept_keys"]) if row["concept_keys"] else [],
                )

        elif node_type == "evidence":
            row = self.db.conn.execute(
                "SELECT evidence_id, evidence_text, evidence_type, graph_id, paper_id, source_span_id "
                "FROM evidence_blocks WHERE evidence_id = ?", (node_id,)
            ).fetchone()
            if row:
                return VisitedNode(
                    node_id=row["evidence_id"], node_type="evidence",
                    title=row["evidence_text"][:80], text=row["evidence_text"],
                    confidence=0.7, source_span_id=row["source_span_id"],
                    graph_id=row["graph_id"], paper_id=row["paper_id"],
                )

        elif node_type == "limitation":
            row = self.db.conn.execute(
                "SELECT limitation_id, limitation_text, risk_type, severity, graph_id, paper_id, source_span_id "
                "FROM limitation_blocks WHERE limitation_id = ?", (node_id,)
            ).fetchone()
            if row:
                return VisitedNode(
                    node_id=row["limitation_id"], node_type="limitation",
                    title=row["limitation_text"][:80], text=row["limitation_text"],
                    confidence=0.7, source_span_id=row["source_span_id"],
                    graph_id=row["graph_id"], paper_id=row["paper_id"],
                )

        elif node_type == "concept":
            row = self.db.conn.execute(
                "SELECT concept_key, label_zh, definition, graph_id "
                "FROM concept_keys WHERE concept_key = ?", (node_id,)
            ).fetchone()
            if row:
                return VisitedNode(
                    node_id=row["concept_key"], node_type="concept",
                    title=row["label_zh"] or row["concept_key"], text=row["definition"],
                    confidence=0.8, graph_id=row["graph_id"], paper_id="",
                )

        elif node_type == "section":
            row = self.db.conn.execute(
                "SELECT section_id, heading, raw_text, graph_id, paper_id "
                "FROM section_blocks WHERE section_id = ?", (node_id,)
            ).fetchone()
            if row:
                return VisitedNode(
                    node_id=row["section_id"], node_type="section",
                    title=row["heading"], text=row["raw_text"][:500],
                    confidence=1.0, graph_id=row["graph_id"], paper_id=row["paper_id"],
                )

        return None

    def _walk_relation(
        self, node_id: str, rel_type: str, pointer_type: PointerType, plan: TraversalPlan
    ) -> list[tuple[str, dict]]:
        """Walk graph_relations. Uses composite index for fast lookup."""
        rows = self.db.conn.execute(
            "SELECT target_id, confidence, importance, traversal_cost "
            "FROM graph_relations WHERE source_id = ? AND relation_type = ? AND confidence >= ? "
            "ORDER BY importance DESC, traversal_cost ASC",
            (node_id, rel_type, plan.min_confidence)
        ).fetchall()

        if pointer_type in _BIDIRECTIONAL:
            in_rows = self.db.conn.execute(
                "SELECT source_id as target_id, confidence, importance, traversal_cost "
                "FROM graph_relations WHERE target_id = ? AND relation_type = ? AND confidence >= ?",
                (node_id, rel_type, plan.min_confidence)
            ).fetchall()
            rows = list(rows) + list(in_rows)

        results = []
        seen = set()
        for r in rows:
            tid = r["target_id"]
            if tid != node_id and tid not in seen:
                seen.add(tid)
                # graph_scope filter
                results.append((tid, dict(r)))
        return results

    def _walk_bridge(
        self, node_id: str, graph_id: str, bridge_type: str,
        pointer_type: PointerType, plan: TraversalPlan
    ) -> list[tuple[str, dict]]:
        """Walk bridge_relations for cross-graph traversal."""
        rows = self.db.conn.execute(
            "SELECT target_key as target_id, bridge_confidence as confidence, "
            "traversal_cost, warning "
            "FROM bridge_relations WHERE source_key = ? AND source_graph_id = ? "
            "AND bridge_type = ? AND bridge_confidence >= ? "
            "AND review_status = 'active'",
            (node_id, graph_id, bridge_type, plan.min_confidence)
        ).fetchall()
        return [(r["target_id"], {**dict(r), "confidence": r["confidence"]}) for r in rows]


def _pointer_key(edge: VisitedEdge) -> str:
    return edge.relation_type


def _importance_value(imp: str) -> float:
    mapping = {"core": 1.0, "supporting": 0.5, "peripheral": 0.2}
    return mapping.get(imp, 0.5)
