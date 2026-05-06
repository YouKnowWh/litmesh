"""
TraversalExecutor: program-controlled typed pointer traversal.

Takes a TraversalPlan and executes it against the SQLite knowledge graph.
The executor enforces:
- max_depth (BFS level limit)
- max_nodes (total visited node limit)
- confidence gates (skip low-confidence edges)
- cycle detection (don't revisit nodes)
- cross-graph jump limits
- source_span requirement
- budget enforcement
- traversal_cost awareness (cheaper edges first)

Principle: LLM decides pointer types; program controls HOW.
"""

import json
from collections import deque
from dataclasses import dataclass, field

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

# Bridge pointers map to bridge_relation types
_BRIDGE_POINTER_TO_TYPE: dict[PointerType, str] = {
    PointerType.ANALOGOUS_TO_BRIDGE: BridgeRelationType.ANALOGOUS_TO.value,
    PointerType.TRANSFERS_TO_BRIDGE: BridgeRelationType.TRANSFERS_TO.value,
    PointerType.CONFLICTS_WITH_BRIDGE: BridgeRelationType.CONFLICTS_WITH.value,
}


class TraversalExecutor:
    """Executes a TraversalPlan against the SQLite knowledge graph."""

    def __init__(self, db):
        self.db = db

    def execute(self, plan: TraversalPlan) -> TraversalResult:
        """Execute a traversal plan and return the result.

        Algorithm: BFS with typed edges. Each level processes all
        enabled pointer types before advancing depth.
        """
        visited_nodes: dict[str, VisitedNode] = {}
        visited_edges: list[VisitedEdge] = []
        cross_graph_jumps = 0
        budget_used = 0

        # BFS queue: (node_id, depth)
        queue = deque()
        for start in plan.start_nodes:
            queue.append((start, 0))

        while queue and len(visited_nodes) < plan.max_nodes and budget_used < plan.budget:
            current_id, depth = queue.popleft()

            if current_id in visited_nodes:
                continue
            if depth > plan.max_depth:
                continue

            # Resolve the node
            node = self._resolve_node(current_id, plan)
            if not node:
                continue

            # Source span gate
            if plan.require_source_span and not node.source_span_id:
                # Only enforce for claim/evidence/limitation nodes
                if node.node_type in ("claim", "evidence", "limitation"):
                    continue

            visited_nodes[current_id] = node
            budget_used += len(node.text) // 4  # Approximate token count

            if depth >= plan.max_depth:
                continue

            # Traverse each enabled pointer type from this node
            for pointer_type in plan.pointer_types:
                edge_count = sum(1 for e in visited_edges if _pointer_key(e) == pointer_type.value)
                if edge_count >= plan.max_edges_per_pointer_type:
                    continue

                if pointer_type in _POINTER_TO_RELATION:
                    neighbors = self._walk_relation(current_id, _POINTER_TO_RELATION[pointer_type], pointer_type, plan)
                elif pointer_type in _BRIDGE_POINTER_TO_TYPE:
                    if not plan.allow_cross_graph or cross_graph_jumps >= plan.max_cross_graph_jumps:
                        continue
                    neighbors = self._walk_bridge(current_id, node.graph_id, _BRIDGE_POINTER_TO_TYPE[pointer_type], pointer_type, plan)
                    cross_graph_jumps += len(neighbors)
                else:
                    continue

                for neighbor_id, edge_info in neighbors:
                    if edge_info["confidence"] < plan.require_source_span and plan.min_confidence > 0.5:
                        continue
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
                        queue.append((neighbor_id, depth + 1))

        stopped_reason = ""
        if len(visited_nodes) >= plan.max_nodes:
            stopped_reason = f"Reached max_nodes limit ({plan.max_nodes})"
        elif budget_used >= plan.budget:
            stopped_reason = f"Reached budget limit ({plan.budget})"
        elif not queue:
            stopped_reason = "No more nodes to traverse"

        # Group by type
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

        total_cost = sum(e.confidence for e in visited_edges)  # Simplified cost

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

    def _resolve_node(self, node_id: str, plan: TraversalPlan) -> VisitedNode | None:
        """Resolve a node ID to a VisitedNode by checking each table."""
        # Try claim
        row = self.db.conn.execute(
            "SELECT claim_id, claim_text, claim_type, extraction_confidence, importance, "
            "source_span_id, graph_id, paper_id, concept_keys "
            "FROM claim_blocks WHERE claim_id = ?", (node_id,)
        ).fetchone()
        if row:
            return VisitedNode(
                node_id=row["claim_id"],
                node_type="claim",
                title=row["claim_text"][:80],
                text=row["claim_text"],
                confidence=row["extraction_confidence"],
                importance=_importance_value(row["importance"]),
                source_span_id=row["source_span_id"],
                graph_id=row["graph_id"],
                paper_id=row["paper_id"],
                concept_keys=json.loads(row["concept_keys"]) if row["concept_keys"] else [],
            )

        # Try evidence
        row = self.db.conn.execute(
            "SELECT evidence_id, evidence_text, evidence_type, graph_id, paper_id, source_span_id "
            "FROM evidence_blocks WHERE evidence_id = ?", (node_id,)
        ).fetchone()
        if row:
            return VisitedNode(
                node_id=row["evidence_id"],
                node_type="evidence",
                title=row["evidence_text"][:80],
                text=row["evidence_text"],
                confidence=0.7,
                source_span_id=row["source_span_id"],
                graph_id=row["graph_id"],
                paper_id=row["paper_id"],
            )

        # Try limitation
        row = self.db.conn.execute(
            "SELECT limitation_id, limitation_text, risk_type, severity, graph_id, paper_id, source_span_id "
            "FROM limitation_blocks WHERE limitation_id = ?", (node_id,)
        ).fetchone()
        if row:
            return VisitedNode(
                node_id=row["limitation_id"],
                node_type="limitation",
                title=row["limitation_text"][:80],
                text=row["limitation_text"],
                confidence=0.7,
                source_span_id=row["source_span_id"],
                graph_id=row["graph_id"],
                paper_id=row["paper_id"],
            )

        # Try concept
        row = self.db.conn.execute(
            "SELECT concept_key, label_zh, definition, graph_id "
            "FROM concept_keys WHERE concept_key = ?", (node_id,)
        ).fetchone()
        if row:
            return VisitedNode(
                node_id=row["concept_key"],
                node_type="concept",
                title=row["label_zh"] or row["concept_key"],
                text=row["definition"],
                confidence=0.8,
                graph_id=row["graph_id"],
                paper_id="",
            )

        # Try section
        row = self.db.conn.execute(
            "SELECT section_id, heading, raw_text, graph_id, paper_id "
            "FROM section_blocks WHERE section_id = ?", (node_id,)
        ).fetchone()
        if row:
            return VisitedNode(
                node_id=row["section_id"],
                node_type="section",
                title=row["heading"],
                text=row["raw_text"][:500],
                confidence=1.0,
                graph_id=row["graph_id"],
                paper_id=row["paper_id"],
            )

        return None

    def _walk_relation(
        self, node_id: str, rel_type: str, pointer_type: PointerType, plan: TraversalPlan
    ) -> list[tuple[str, dict]]:
        """Walk graph_relations from a node by relation type."""
        # Query outgoing edges
        rows = self.db.conn.execute(
            "SELECT target_id, confidence, importance, traversal_cost "
            "FROM graph_relations WHERE source_id = ? AND relation_type = ? AND confidence >= ? "
            "ORDER BY importance DESC, traversal_cost ASC",
            (node_id, rel_type, plan.min_confidence)
        ).fetchall()

        # Also query incoming edges (for bidirectional/symmetric pointer types)
        # SUPPORTS: evidence -> claim, but we may start from claim and need to find evidence
        # CONSTRAINS: limitation -> claim, same pattern
        # CONTRADICTS/REFINES/EXTENDS/SUPERSEDES: claim <-> claim
        # DERIVED_FROM: claim A -> claim B, but B may reference A
        if pointer_type in (PointerType.SUPPORTS, PointerType.CONSTRAINS,
                             PointerType.CONTRADICTS, PointerType.REFINES,
                             PointerType.EXTENDS, PointerType.SUPERSEDES,
                             PointerType.DERIVED_FROM):
            in_rows = self.db.conn.execute(
                "SELECT source_id as target_id, confidence, importance, traversal_cost "
                "FROM graph_relations WHERE target_id = ? AND relation_type = ? AND confidence >= ?",
                (node_id, rel_type, plan.min_confidence)
            ).fetchall()
            rows = list(rows) + list(in_rows)

        return [(r["target_id"], dict(r)) for r in rows if r["target_id"] != node_id]

    def _walk_bridge(
        self, node_id: str, graph_id: str, bridge_type: str, pointer_type: PointerType, plan: TraversalPlan
    ) -> list[tuple[str, dict]]:
        """Walk bridge_relations from a node."""
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
