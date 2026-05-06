"""
PromptPacket compiler (v0.4): TraversalResult -> PromptPacket.

Compiles traversal output into a structured, typed, confidence-annotated
context package for the reasoning LLM.

Key design:
- Separates claims/evidence/limitations/conflicts into labeled sections
- Labels each item with confidence and source citation
- Low-confidence items go in a separate section
- GenerationPolicy tells the model what it can and cannot do
"""

from datetime import datetime

from ..models.prompt_packet import (
    PromptPacket, GenerationPolicy, PacketClaim,
    TraversalResult, TraversalTrace, TraversalPlan,
)


class PromptPacketCompiler:
    """Compiles TraversalResult into a PromptPacket."""

    def compile(
        self,
        user_query: str,
        intent: str,
        result: TraversalResult,
        plan: TraversalPlan,
        trace_id: str = "",
    ) -> PromptPacket:
        """Compile traversal output into a structured prompt packet."""

        packet = PromptPacket(
            current_user_query=user_query,
            interpreted_intent=intent,
            graph_scope=plan.graph_scope,
            active_concepts=[n.node_id for n in result.visited_nodes if n.node_type == "concept"],
            trace_id=trace_id,
        )

        # Separate visited nodes by type and confidence
        claims = []
        evidence = []
        limitations = []
        low_conf = []
        anchors = []

        for node in result.visited_nodes:
            if node.node_type == "claim":
                claim = PacketClaim(
                    claim_id=node.node_id,
                    claim_text=node.text,
                    claim_type="claim",
                    confidence=node.confidence,
                    source_citation=f"paper:{node.paper_id}",
                    is_cross_graph=node.graph_id not in plan.graph_scope,
                )
                if node.confidence >= 0.7 and node.importance >= 0.8:
                    anchors.append(claim)
                elif node.confidence < 0.5:
                    low_conf.append({"type": "claim", "text": node.text,
                                     "confidence": node.confidence, "id": node.node_id})
                else:
                    claims.append(claim)

            elif node.node_type == "evidence":
                if node.confidence < 0.5:
                    low_conf.append({"type": "evidence", "text": node.text,
                                     "confidence": node.confidence, "id": node.node_id})
                else:
                    evidence.append({
                        "evidence_id": node.node_id,
                        "text": node.text,
                        "confidence": node.confidence,
                        "source_paper": node.paper_id,
                    })

            elif node.node_type == "limitation":
                limitations.append({
                    "limitation_id": node.node_id,
                    "text": node.text,
                    "confidence": node.confidence,
                    "source_paper": node.paper_id,
                })

        # Conflicts: edges with contradicts type
        conflict_pairs = []
        for edge in result.visited_edges:
            if edge.relation_type == "contradicts":
                conflict_pairs.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "confidence": edge.confidence,
                })

        # Bridge relations
        bridges = [
            {"target_id": e.target_id, "type": e.relation_type, "confidence": e.confidence}
            for e in result.visited_edges if e.is_cross_graph
        ]

        packet.cognitive_anchors = anchors
        packet.paper_claims = claims
        packet.supporting_evidence = evidence
        packet.limitations = limitations
        packet.conflicts = conflict_pairs
        packet.low_confidence_candidates = low_conf
        packet.bridge_relations = bridges

        # Generation policy
        packet.generation_policy = GenerationPolicy(
            may_assert_claims=True,
            may_assert_limitations=True,
            must_cite_claims=True,
            must_mention_limitations=len(limitations) > 0,
            must_label_analogy=len(bridges) > 0,
            must_flag_low_confidence=len(low_conf) > 0,
            allow_synthesis=True,
            disallow_invention=True,
        )

        return packet

    def compile_to_text(self, packet: PromptPacket) -> str:
        """Render PromptPacket as a structured text block for the LLM context window."""
        parts = []

        parts.append("=== LitMesh Structured Context ===")
        parts.append(f"Intent: {packet.interpreted_intent}")
        parts.append("")

        if packet.cognitive_anchors:
            parts.append("## Cognitive Anchors (high-confidence core claims)")
            for a in packet.cognitive_anchors:
                parts.append(f"- [{a.confidence:.0%}] {a.claim_text}")
            parts.append("")

        if packet.paper_claims:
            parts.append("## Author Claims")
            for c in packet.paper_claims:
                parts.append(f"- [{c.confidence:.0%}] {c.claim_text}")
            parts.append("")

        if packet.supporting_evidence:
            parts.append("## Supporting Evidence")
            for e in packet.supporting_evidence:
                parts.append(f"- [{e['confidence']:.0%}] {e['text']}")
            parts.append("")

        if packet.limitations:
            parts.append("## Limitations & Risks")
            for l in packet.limitations:
                parts.append(f"- {l['text']}")
            parts.append("")

        if packet.conflicts:
            parts.append("## Known Conflicts")
            for c in packet.conflicts:
                parts.append(f"- Conflict between {c['source_id']} and {c['target_id']}")
            parts.append("")

        if packet.low_confidence_candidates:
            parts.append("## Low Confidence Candidates (WEAK REFERENCE ONLY)")
            for lc in packet.low_confidence_candidates:
                parts.append(f"- [WEAK] {lc['text']}")
            parts.append("")

        if packet.bridge_relations:
            parts.append("## Cross-Graph Analogies (label as analogy)")
            for b in packet.bridge_relations:
                parts.append(f"- Bridge: {b['target_id']} (type: {b['type']})")
            parts.append("")

        # Generation policy
        gp = packet.generation_policy
        parts.append("## Generation Constraints")
        if gp.must_cite_claims:
            parts.append("- You MUST cite sources when using claims.")
        if gp.must_mention_limitations:
            parts.append("- You MUST mention relevant limitations.")
        if gp.must_label_analogy:
            parts.append("- You MUST explicitly label cross-graph analogies.")
        if gp.must_flag_low_confidence:
            parts.append("- You MUST flag low-confidence items as uncertain.")
        if gp.disallow_invention:
            parts.append("- Do NOT invent claims not present in this context.")

        return "\n".join(parts)
