"""
KnowledgeQueryEngine: end-to-end query pipeline (v0.6).

Flow:
  User Query
    → ConceptRouter (find relevant concepts)
    → RetrievalGate (decide retrieval strategy)
    → HybridRetriever (vector + FTS + graph expansion)
    → TraversalPlanner (preset-based mode selection)
    → TraversalExecutor (typed pointer walk)
    → PromptPacketCompiler (compile structured context)
    → render_to_text (LLM-ready context block)
"""

from ..retrieval.concept_router import ConceptRouter
from ..retrieval.retrieval_gate import decide_retrieval, GateInput
from ..retrieval.hybrid_retriever import HybridRetriever
from ..traversal.traversal_presets import build_preset_plan
from ..traversal.traversal_executor import TraversalExecutor
from ..traversal.traversal_trace import TraceStore
from ..compiler.prompt_packet_compiler import PromptPacketCompiler
from ..models.prompt_packet import TraversalMode


_MODE_KEYWORDS = {
    "explain": ["什么是", "解释", "定义", "概念", "是什么", "含义", "理解"],
    "audit": ["验证", "证据", "支撑", "可信", "可靠", "检查", "审查"],
    "compare": ["对比", "区别", "比较", "异同", "差异", "vs", "不同"],
    "trace": ["原文", "出处", "引用", "页码", "来源", "在哪里提到的"],
    "conflict": ["矛盾", "冲突", "不一致", "争议", "反对", "质疑"],
    "synthesis": ["综述", "总结", "综合", "归纳", "概述", "梳理", "整理"],
    "transfer": ["迁移", "跨领域", "类比", "应用于", "借鉴"],
}


class KnowledgeQueryEngine:
    """End-to-end query engine: from question to LLM-ready context."""

    def __init__(self, db, llm_client=None, vector_store=None, embed_fn=None):
        self.db = db
        self.llm = llm_client
        self.concept_router = ConceptRouter(db)
        self.retriever = HybridRetriever(db, vector_store, embed_fn)
        self.executor = TraversalExecutor(db)
        self.compiler = PromptPacketCompiler()
        self.trace_store = TraceStore(db)

    def query(self, user_query: str, graph_scope: list[str] = None,
              mode: TraversalMode = None) -> dict:
        """Run the full pipeline.

        Args:
            user_query: The user's question.
            graph_scope: Optional list of graph_ids to search within.
                         If None, search all graphs.
            mode: Optional traversal mode override. If None, auto-detect.

        Returns:
            {
                "packet": PromptPacket,
                "text": str (LLM-ready context block),
                "trace_id": str,
                "mode": str,
                "stats": {...},
            }
        """
        if graph_scope is None:
            graph_scope = self._all_graph_ids()

        # 1. Auto-detect traversal mode from query
        if mode is None:
            mode = self._detect_mode(user_query)
        intent = self._infer_intent(user_query, mode)

        # 2. Route to relevant concepts
        concepts = self.concept_router.route(user_query, graph_scope, top_k=5)
        concept_keys = [c["concept_key"] for c in concepts]

        # 3. Retrieval gate + retrieval
        gate_kwargs = {"anchor_count": len(concepts), "confidence_sum": sum(c.get("relevance", 0) for c in concepts)}
        retrieval_result = self.retriever.retrieve(
            user_query, graph_scope, top_k=20,
            gate_kwargs=gate_kwargs,
        )

        # 4. Build traversal start nodes (concepts + top retrieved claim IDs)
        start_nodes = list(concept_keys)
        for claim in retrieval_result["claims"][:5]:
            cid = claim.get("claim_id") or claim.get("id", "")
            if cid and cid not in start_nodes:
                start_nodes.append(cid)

        # If we have nothing, just use concepts
        if not start_nodes:
            start_nodes = concept_keys

        # 5. Build and execute traversal plan
        plan = build_preset_plan(mode, start_nodes, graph_scope, task_type=intent)
        traversal_result = self.executor.execute(plan)

        # 6. Merge retrieval results into traversal (claims not found by traversal)
        self._merge_retrieval_into_traversal(retrieval_result, traversal_result)

        # 7. Compile PromptPacket
        packet = self.compiler.compile(
            user_query=user_query,
            intent=intent,
            result=traversal_result,
            plan=plan,
            trace_id="",
        )

        # 8. Save trace
        trace_id = self.trace_store.save(user_query, plan, traversal_result)
        packet.trace_id = trace_id

        # 9. Render to text
        text = self.compiler.compile_to_text(packet)

        return {
            "packet": packet,
            "text": text,
            "trace_id": trace_id,
            "mode": mode.value,
            "stats": {
                "nodes_visited": len(traversal_result.visited_nodes),
                "edges_traversed": len(traversal_result.visited_edges),
                "claims": len(packet.paper_claims),
                "evidence": len(packet.supporting_evidence),
                "limitations": len(packet.limitations),
                "conflicts": len(packet.conflicts),
                "low_confidence": len(packet.low_confidence_candidates),
                "concepts_routed": len(concepts),
                "stopped_reason": traversal_result.stopped_reason,
            },
        }

    def _detect_mode(self, query: str) -> TraversalMode:
        """Auto-detect traversal mode from query keywords."""
        scores = {}
        for mode, keywords in _MODE_KEYWORDS.items():
            scores[mode] = sum(1 for kw in keywords if kw in query)
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return TraversalMode(best)
        return TraversalMode.EXPLAIN  # Default

    def _infer_intent(self, query: str, mode: TraversalMode) -> str:
        intents = {
            TraversalMode.EXPLAIN: f"explain:{query[:50]}",
            TraversalMode.AUDIT: f"audit:{query[:50]}",
            TraversalMode.COMPARE: f"compare:{query[:50]}",
            TraversalMode.TRACE: f"trace:{query[:50]}",
            TraversalMode.CONFLICT: f"conflict:{query[:50]}",
            TraversalMode.SYNTHESIS: f"synthesis:{query[:50]}",
            TraversalMode.TRANSFER: f"transfer:{query[:50]}",
        }
        return intents.get(mode, query[:60])

    def _merge_retrieval_into_traversal(self, retrieval_result, traversal_result):
        """Add claims found by retrieval but not by traversal."""
        from ..models.prompt_packet import VisitedNode
        existing_ids = {n.node_id for n in traversal_result.visited_nodes}
        for claim in retrieval_result.get("claims", []):
            cid = claim.get("claim_id") or claim.get("id", "")
            if cid and cid not in existing_ids:
                traversal_result.visited_nodes.append(VisitedNode(
                    node_id=cid,
                    node_type="claim",
                    title=claim.get("claim_text", claim.get("text", ""))[:80],
                    text=claim.get("claim_text", claim.get("text", "")),
                    confidence=claim.get("extraction_confidence", 0.5),
                    source_span_id=claim.get("source_span_id"),
                    graph_id=claim.get("graph_id", ""),
                    paper_id=claim.get("paper_id", ""),
                ))
                traversal_result.grouped_claims.append(cid)

    def _all_graph_ids(self) -> list[str]:
        rows = self.db.conn.execute("SELECT graph_id FROM series_graphs").fetchall()
        return [r["graph_id"] for r in rows]
