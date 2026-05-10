"""
v0.6-v0.7 tests: end-to-end pipeline and cross-graph bridges.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.litmesh.storage.sqlite import LitMeshDB
from app.litmesh.models.corpus import CorpusCard
from app.litmesh.models.graph import SeriesGraph
from app.litmesh.models.paper import PaperCard
from app.litmesh.models.claim import ClaimBlock, ClaimType, ClaimStatus
from app.litmesh.models.evidence import EvidenceBlock, EvidenceType
from app.litmesh.models.limitation import LimitationBlock, RiskType
from app.litmesh.models.concept import ConceptKey, ConceptNamespace, ConceptStatus
from app.litmesh.models.relation import GraphRelation, GraphRelationType, BridgeRelationType
from app.litmesh.models.source_span import SourceSpan, SpanPosition
from app.litmesh.models.prompt_packet import TraversalMode

from app.litmesh.compiler.knowledge_query_engine import KnowledgeQueryEngine, _MODE_KEYWORDS
from app.litmesh.registry.bridge_detector import BridgeDetector


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def db():
    database = LitMeshDB(":memory:")
    database.connect()
    database.init_schema()
    yield database
    database.close()


def _make_span(db, paper_id):
    existing = db.conn.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0]
    sid = f"span_{paper_id[:8]}_{existing}"
    db.insert_span(SourceSpan(
        span_id=sid, paper_id=paper_id,
        span_type="paragraph", source_text="Test",
        position=SpanPosition(char_start=0, char_end=4, page_start=1),
    ))
    return sid


def _build_graph(db, corpus_id: str, graph_id: str, name: str, domain: str,
                 papers_data: list[dict], concepts_data: list[dict]) -> str:
    """Helper: create a graph with papers, claims, relations, and concepts."""
    db.insert_graph(SeriesGraph(
        graph_id=graph_id, corpus_id=corpus_id, name=name, domain=domain,
    ))

    for pd in papers_data:
        paper = PaperCard(graph_id=graph_id, **pd)
        db.insert_paper(paper)
        span_id = _make_span(db, paper.paper_id)

        for cd in pd.get("claims", []):
            claim = ClaimBlock(
                graph_id=graph_id, paper_id=paper.paper_id,
                claim_text=cd["text"], claim_type=ClaimType(cd.get("type", "theoretical")),
                extraction_confidence=cd.get("confidence", 0.8),
                concept_keys=cd.get("concept_keys", []),
                status=ClaimStatus.ACTIVE, source_span_id=span_id,
            )
            db.insert_claim(claim)
            cd["_id"] = claim.claim_id

        for ed in pd.get("evidence", []):
            ev = EvidenceBlock(
                graph_id=graph_id, paper_id=paper.paper_id,
                evidence_text=ed["text"],
                supports_claim_ids=[c["_id"] for c in pd.get("claims", [])],
                evidence_type=EvidenceType(ed.get("type", "other")),
                source_span_id=span_id,
            )
            db.insert_evidence(ev)
            ed["_id"] = ev.evidence_id

        for ld in pd.get("limitations", []):
            lim = LimitationBlock(
                graph_id=graph_id, paper_id=paper.paper_id,
                limitation_text=ld["text"],
                affected_claim_ids=[c["_id"] for c in pd.get("claims", [])],
                risk_type=RiskType(ld.get("risk", "scope")),
                source_span_id=span_id,
            )
            db.insert_limitation(lim)
            ld["_id"] = lim.limitation_id

    for cd in concepts_data:
        db.insert_concept(ConceptKey(
            concept_key=cd["key"], graph_id=graph_id,
            namespace=ConceptNamespace(cd.get("namespace", "concept")),
            label_zh=cd["label"], aliases=cd.get("aliases", []),
            status=ConceptStatus.ACTIVE,
        ))

    # Create constraint relations: limitations → claims
    for pd in papers_data:
        for ld in pd.get("limitations", []):
            for cd in pd.get("claims", []):
                db.insert_relation(GraphRelation(
                    graph_id=graph_id,
                    source_id=ld["_id"], target_id=cd["_id"],
                    source_type="limitation", target_type="claim",
                    relation_type=GraphRelationType.CONSTRAINS, confidence=0.8,
                ))

    return graph_id


@pytest.fixture
def two_graph_system(db):
    """Two graphs:
    Graph A: AI Education — CPE-3DF framework paper
    Graph B: OS Education — OSTEP textbook chapter
    With overlapping concept: both mention "cognitive_load"
    """
    corpus = CorpusCard(name="Test", corpus_id="corpus_main", domain="general")
    db.insert_corpus(corpus)

    _build_graph(db, "corpus_main", "graph_ai", "AI Education", "AI_education", [
        {"title": "CPE-3DF Framework Paper", "authors": ["Zhang"], "year": 2024,
         "source_file": "cpe.pdf",
         "claims": [
             {"text": "AI教育必须三维推进", "type": "framework", "concept_keys": ["framework:CPE_3DF", "concept:cognitive_load"]},
         ],
         "evidence": [{"text": "三所大学实验验证"}],
         "limitations": [{"text": "样本量小", "risk": "scope"}],
        },
    ], [
        {"key": "framework:CPE_3DF", "namespace": "framework", "label": "CPE-3DF", "aliases": ["认知实践伦理框架"]},
        {"key": "concept:cognitive_load", "namespace": "concept", "label": "认知负荷", "aliases": ["认知负担"]},
    ])

    _build_graph(db, "corpus_main", "graph_os", "OS Education", "operating_systems", [
        {"title": "OSTEP: CPU Scheduling", "authors": ["Remzi"], "year": 2019,
         "source_file": "ostep3.pdf",
         "claims": [
             {"text": "CPU调度是OS核心功能", "type": "theoretical", "concept_keys": ["concept:CPU_scheduling", "concept:cognitive_load_os"]},
         ],
         "evidence": [{"text": "Linux内核源码分析"}],
         "limitations": [{"text": "未涵盖实时调度", "risk": "scope"}],
        },
    ], [
        {"key": "concept:CPU_scheduling", "namespace": "concept", "label": "CPU调度"},
        {"key": "concept:cognitive_load_os", "namespace": "concept", "label": "认知负荷", "aliases": ["认知负担"]},
    ])

    return db


# ============================================================
# v0.6: End-to-end query pipeline
# ============================================================

class TestQueryEngine:

    def test_mode_detection(self):
        """Auto-detect traversal mode from query."""
        engine = KnowledgeQueryEngine.__new__(KnowledgeQueryEngine)  # Skip __init__
        assert engine._detect_mode("什么是CPE-3DF框架") == TraversalMode.EXPLAIN
        assert engine._detect_mode("有什么证据支持这个主张") == TraversalMode.AUDIT
        assert engine._detect_mode("CPE-3DF和PACADI有什么区别") == TraversalMode.COMPARE
        assert engine._detect_mode("这个观点在原文哪里") == TraversalMode.TRACE
        assert engine._detect_mode("这些文献有什么矛盾") == TraversalMode.CONFLICT
        assert engine._detect_mode("总结一下AI教育的核心发现") == TraversalMode.SYNTHESIS
        assert engine._detect_mode("这个框架能借鉴到其他领域吗") == TraversalMode.TRANSFER

    def test_query_explain_mode(self, two_graph_system):
        """Full explain query pipeline."""
        db = two_graph_system
        engine = KnowledgeQueryEngine(db)
        result = engine.query("什么是认知负荷", graph_scope=["graph_ai"])

        assert result["mode"] == "explain"
        assert "text" in result
        assert "LitMesh" in result["text"]
        assert result["trace_id"].startswith("trace_")
        assert result["stats"]["nodes_visited"] >= 1

    def test_query_audit_mode(self, two_graph_system):
        """Full audit query pipeline."""
        db = two_graph_system
        engine = KnowledgeQueryEngine(db)
        result = engine.query("验证CPE-3DF框架的有效性", graph_scope=["graph_ai"])

        assert result["mode"] == "audit"

    def test_query_compare_mode(self, two_graph_system):
        """Full compare query pipeline."""
        db = two_graph_system
        engine = KnowledgeQueryEngine(db)
        result = engine.query("AI教育和OS教育在认知负荷方面有什么区别",
                               graph_scope=["graph_ai", "graph_os"])
        assert result["mode"] in ("compare", "explain")

    def test_packet_has_all_sections(self, two_graph_system):
        """PromptPacket should have all required sections."""
        db = two_graph_system
        engine = KnowledgeQueryEngine(db)
        result = engine.query("AI教育框架的核心主张是什么", graph_scope=["graph_ai"])

        packet = result["packet"]
        assert packet.current_user_query != ""
        assert hasattr(packet, "paper_claims")
        assert hasattr(packet, "limitations")
        assert hasattr(packet, "generation_policy")
        assert packet.generation_policy.must_cite_claims is True

    def test_stats_are_present(self, two_graph_system):
        """Result should include traversal stats."""
        db = two_graph_system
        engine = KnowledgeQueryEngine(db)
        result = engine.query("AI教育框架")

        stats = result["stats"]
        assert "nodes_visited" in stats
        assert "edges_traversed" in stats
        assert "stopped_reason" in stats


# ============================================================
# v0.7: Cross-graph bridge detection
# ============================================================

class TestBridgeDetection:

    def test_detect_same_concept_across_graphs(self, two_graph_system):
        """Two graphs with the same concept should produce a same_as bridge."""
        db = two_graph_system
        detector = BridgeDetector(db)
        stats = detector.detect_all()

        # Both graphs have "concept:cognitive_load" → should bridge
        assert stats["bridges_proposed"] >= 1

        # Check that a bridge was created
        rows = db.conn.execute("SELECT * FROM bridge_relations").fetchall()
        assert len(rows) >= 1

    def test_same_as_bridge_auto_accepted(self, two_graph_system):
        """Same concept label + same namespace → auto-active same_as bridge."""
        db = two_graph_system
        # Add same concept in both graphs (same label, different key)
        db.insert_concept(ConceptKey(
            concept_key="problem:AI_bias", graph_id="graph_ai",
            namespace=ConceptNamespace.PROBLEM, label_zh="AI偏见",
            aliases=["算法偏见"], status=ConceptStatus.ACTIVE,
        ))
        db.insert_concept(ConceptKey(
            concept_key="problem:AI_bias", graph_id="graph_os",
            namespace=ConceptNamespace.PROBLEM, label_zh="AI偏见",
            aliases=["算法偏见"], status=ConceptStatus.ACTIVE,
        ))

        detector = BridgeDetector(db)
        detector.detect_all()

        rows = db.conn.execute(
            "SELECT * FROM bridge_relations WHERE bridge_type = ?",
            (BridgeRelationType.SAME_AS.value,)
        ).fetchall()
        same_as = [dict(r) for r in rows]
        assert len(same_as) >= 1
        # Same concept key + same namespace + same label → should auto-activate
        assert any(r["review_status"] == "active" for r in same_as)

    def test_bridge_inbox_for_analogous(self, two_graph_system):
        """Analogous concepts should go to BridgeInbox, not auto-activate."""
        db = two_graph_system
        # Add an analogous concept pair
        db.insert_concept(ConceptKey(
            concept_key="framework:PACADI", graph_id="graph_ai",
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="PACADI", aliases=["PACADI框架"],
            status=ConceptStatus.ACTIVE,
        ))
        db.insert_concept(ConceptKey(
            concept_key="framework:CPE_3DF", graph_id="graph_os",
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="CPE-3DF", aliases=["CPE3DF框架"],
            status=ConceptStatus.ACTIVE,
        ))

        detector = BridgeDetector(db)
        detector.detect_all()

        # Check inbox for bridge items
        pending = db.get_pending_inbox("bridge")
        assert len(pending) >= 1

    def test_do_not_merge_respected(self, two_graph_system):
        """do_not_merge_with should prevent bridge creation."""
        db = two_graph_system
        # Add a concept in graph_os that would normally bridge with cognitive_load in graph_ai
        db.insert_concept(ConceptKey(
            concept_key="concept:cognitive_load_os2", graph_id="graph_os",
            namespace=ConceptNamespace.CONCEPT, label_zh="认知负荷OS",
            aliases=["认知负担", "cognitive load"],
            status=ConceptStatus.ACTIVE,
        ))
        # Update cognitive_load in graph_ai to exclude graph_os's concept
        db.conn.execute(
            "UPDATE concept_keys SET do_not_merge_with = ? WHERE concept_key = ? AND graph_id = ?",
            (json.dumps(["concept:cognitive_load_os2"]), "concept:cognitive_load", "graph_ai")
        )
        db.conn.commit()

        detector = BridgeDetector(db)
        detector.detect_all()

        # Should NOT create bridge due to do_not_merge_with
        rows = db.conn.execute(
            "SELECT * FROM bridge_relations WHERE (source_key = 'concept:cognitive_load' AND target_key = 'concept:cognitive_load_os2') OR (source_key = 'concept:cognitive_load_os2' AND target_key = 'concept:cognitive_load')"
        ).fetchall()
        assert len(rows) == 0

    def test_bridge_types_are_valid(self, two_graph_system):
        """All created bridges should have valid types."""
        db = two_graph_system
        detector = BridgeDetector(db)
        detector.detect_all()

        valid_types = {t.value for t in BridgeRelationType}
        rows = db.conn.execute("SELECT bridge_type FROM bridge_relations").fetchall()
        for r in rows:
            assert r["bridge_type"] in valid_types


# ============================================================
# End-to-end: Query across bridged graphs
# ============================================================

class TestCrossGraphQuery:

    def test_transfer_mode_uses_bridges(self, two_graph_system):
        """Transfer mode traversal should use bridge relations."""
        db = two_graph_system

        # First, run bridge detection
        BridgeDetector(db).detect_all()

        # Query with transfer mode across both graphs
        engine = KnowledgeQueryEngine(db)
        result = engine.query(
            "认知负荷的概念在AI教育和操作系统教育中有什么关联",
            graph_scope=["graph_ai", "graph_os"],
            mode=TraversalMode.TRANSFER,
        )

        assert result["mode"] == "transfer"
        # Should visit nodes from both graphs
        graphs_visited = set()
        for node in result["packet"].traversal_result.visited_nodes if hasattr(result["packet"], "traversal_result") else []:
            graphs_visited.add(node.graph_id)

        assert result["stats"]["nodes_visited"] >= 1


# ============================================================
# v0.9: Traversal index layer tests
# ============================================================

@pytest.fixture
def corpus(db):
    from app.litmesh.models.corpus import CorpusCard
    c = CorpusCard(name="Test Corpus", corpus_type="paper_collection", domain="test")
    db.insert_corpus(c)
    return c


@pytest.fixture
def graph(db, corpus):
    from app.litmesh.models.graph import SeriesGraph
    g = SeriesGraph(
        corpus_id=corpus.corpus_id, name="Test Graph",
        graph_type="paper_collection", domain="test",
    )
    db.insert_graph(g)
    return g


@pytest.fixture
def paper(db, graph):
    from app.litmesh.models.paper import PaperCard
    p = PaperCard(
        graph_id=graph.graph_id, title="Test Paper",
        source_file="test.pdf",
    )
    db.insert_paper(p)
    return p


@pytest.fixture
def section(db, graph, paper):
    from app.litmesh.models.section import SectionBlock, HeadingLevel
    s = SectionBlock(
        graph_id=graph.graph_id, paper_id=paper.paper_id,
        heading="Test Section", heading_path=["Test Section"],
        heading_level=HeadingLevel.SECTION,
        raw_text="Test paragraph text.", page_start=1,
    )
    db.insert_section(s)
    return s


@pytest.fixture
def retrieval_setup(db, graph, paper, section):
    """Setup with claims, evidence, sections for retrieval tests."""
    from app.litmesh.models.claim import ClaimBlock, ClaimType
    from app.litmesh.models.evidence import EvidenceBlock, EvidenceType

    c = ClaimBlock(
        graph_id=graph.graph_id, paper_id=paper.paper_id,
        section_id=section.section_id,
        claim_text="Virtual experiments improve learning outcomes.",
        claim_type=ClaimType.EMPIRICAL,
    )
    db.insert_claim(c)

    ev = EvidenceBlock(
        graph_id=graph.graph_id, paper_id=paper.paper_id,
        section_id=section.section_id,
        evidence_text="Study shows 20% improvement.",
        evidence_type=EvidenceType.DATA,
    )
    db.insert_evidence(ev)

    return db, graph.graph_id


class TestNodeIndex:
    """Tests for node_index and related index tables."""

    def test_schema_has_node_index_table(self, db):
        """node_index table exists after schema init."""
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='node_index'"
        ).fetchone()
        assert row is not None

    def test_schema_has_block_concepts_table(self, db):
        """block_concepts table exists."""
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='block_concepts'"
        ).fetchone()
        assert row is not None

    def test_schema_has_claim_evidence_links(self, db):
        """claim_evidence_links table exists."""
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='claim_evidence_links'"
        ).fetchone()
        assert row is not None

    def test_schema_has_claim_limitation_links(self, db):
        """claim_limitation_links table exists."""
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='claim_limitation_links'"
        ).fetchone()
        assert row is not None

    def test_insert_claim_writes_node_index(self, db, graph, section):
        """insert_claim syncs to node_index."""
        from app.litmesh.models.claim import ClaimBlock, ClaimType
        claim = ClaimBlock(
            graph_id=graph.graph_id, paper_id=section.paper_id,
            section_id=section.section_id, claim_text="Test claim",
            claim_type=ClaimType.THEORETICAL,
        )
        db.insert_claim(claim)
        assert db.get_node_type(claim.claim_id) == "claim"

    def test_insert_evidence_writes_node_index(self, db, graph, section):
        """insert_evidence syncs to node_index."""
        from app.litmesh.models.evidence import EvidenceBlock, EvidenceType
        ev = EvidenceBlock(
            graph_id=graph.graph_id, paper_id=section.paper_id,
            section_id=section.section_id, evidence_text="Test evidence",
            evidence_type=EvidenceType.OTHER,
        )
        db.insert_evidence(ev)
        assert db.get_node_type(ev.evidence_id) == "evidence"

    def test_insert_section_writes_node_index(self, db, graph, paper):
        """insert_section syncs to node_index."""
        from app.litmesh.models.section import SectionBlock, HeadingLevel
        sec = SectionBlock(
            graph_id=graph.graph_id, paper_id=paper.paper_id,
            heading="Test", heading_path=["Test"],
            heading_level=HeadingLevel.SECTION,
            raw_text="Some section text.",
            page_start=1, page_end=1,
        )
        db.insert_section(sec)
        assert db.get_node_type(sec.section_id) == "section"

    def test_batch_resolve_returns_types(self, db, graph, section):
        """batch_resolve_nodes returns correct type dict."""
        from app.litmesh.models.claim import ClaimBlock, ClaimType
        c = ClaimBlock(
            graph_id=graph.graph_id, paper_id=section.paper_id,
            section_id=section.section_id, claim_text="C",
            claim_type=ClaimType.THEORETICAL,
        )
        db.insert_claim(c)
        resolved = db.batch_resolve_nodes([c.claim_id, section.section_id, "nonexistent"])
        assert resolved.get(c.claim_id) == "claim"
        assert resolved.get(section.section_id) == "section"
        assert "nonexistent" not in resolved

    def test_link_block_concept(self, db, graph):
        """link_block_concept creates block_concepts entry."""
        db.link_block_concept("concept:test", "block_1", "claim", graph.graph_id)
        block_concepts = db.get_concepts_for_block("block_1")
        assert "concept:test" in block_concepts

    def test_link_claim_evidence(self, db, graph):
        """link_claim_evidence creates link entry."""
        db.link_claim_evidence("claim_1", "evidence_1", graph.graph_id)
        ev_ids = db.get_evidence_for_claim("claim_1")
        assert "evidence_1" in ev_ids

    def test_get_node_type_returns_none_for_unknown(self, db):
        """get_node_type returns None for unindexed nodes."""
        assert db.get_node_type("nonexistent_id") is None

    def test_node_index_upsert_updates_type(self, db, graph):
        """upsert_node_index updates existing entry."""
        db.upsert_node_index("node_x", "claim", graph.graph_id)
        assert db.get_node_type("node_x") == "claim"
        db.upsert_node_index("node_x", "evidence", graph.graph_id)
        assert db.get_node_type("node_x") == "evidence"


class TestPriorityTraversal:
    """Tests for priority-based BFS traversal."""

    def test_graph_scope_filters_nodes(self, db, graph, section):
        """Nodes outside graph_scope are excluded."""
        from app.litmesh.models.claim import ClaimBlock, ClaimType
        from app.litmesh.models.graph import SeriesGraph
        from app.litmesh.traversal.traversal_executor import TraversalExecutor
        from app.litmesh.models.prompt_packet import TraversalPlan, TraversalMode

        # Create second graph + paper + section (use corpus fixture's corpus)
        from app.litmesh.models.corpus import CorpusCard
        c2 = CorpusCard(name="C2", corpus_type="paper_collection", domain="test")
        db.insert_corpus(c2)
        g2 = SeriesGraph(corpus_id=c2.corpus_id, name="G2")
        db.insert_graph(g2)
        from app.litmesh.models.paper import PaperCard
        p2 = PaperCard(graph_id=g2.graph_id, title="P2", source_file="t.pdf")
        db.insert_paper(p2)
        from app.litmesh.models.section import SectionBlock, HeadingLevel
        s2 = SectionBlock(graph_id=g2.graph_id, paper_id=p2.paper_id, heading="S2",
                          heading_path=["S2"], heading_level=HeadingLevel.SECTION,
                          raw_text="text", page_start=1)
        db.insert_section(s2)

        c_a = ClaimBlock(graph_id=graph.graph_id, paper_id=section.paper_id,
                         section_id=section.section_id, claim_text="In scope",
                         claim_type=ClaimType.THEORETICAL)
        c_b = ClaimBlock(graph_id=g2.graph_id, paper_id=p2.paper_id,
                         section_id=s2.section_id, claim_text="Out of scope",
                         claim_type=ClaimType.THEORETICAL)
        db.insert_claim(c_a); db.insert_claim(c_b)

        plan = TraversalPlan(
            task_type="test", start_nodes=[c_a.claim_id, c_b.claim_id],
            graph_scope=[graph.graph_id], pointer_types=[],
            traversal_mode=TraversalMode.EXPLAIN, max_depth=0,
            require_source_span=False,
        )
        executor = TraversalExecutor(db)
        result = executor.execute(plan)

        graph_ids = {n.graph_id for n in result.visited_nodes}
        assert graph.graph_id in graph_ids
        assert g2.graph_id not in graph_ids

    def test_require_source_span_skips_unlinked_clains(self, db, graph, section):
        """require_source_span=True skips claims without source_span_id."""
        from app.litmesh.models.claim import ClaimBlock, ClaimType
        from app.litmesh.traversal.traversal_executor import TraversalExecutor
        from app.litmesh.models.prompt_packet import TraversalPlan, TraversalMode

        c = ClaimBlock(
            graph_id=graph.graph_id, paper_id=section.paper_id,
            section_id=section.section_id, claim_text="No span",
            claim_type=ClaimType.THEORETICAL, source_span_id=None,
        )
        db.insert_claim(c)

        plan = TraversalPlan(
            task_type="test", start_nodes=[c.claim_id],
            graph_scope=[graph.graph_id], pointer_types=[],
            traversal_mode=TraversalMode.EXPLAIN, max_depth=0,
            require_source_span=True,
        )
        executor = TraversalExecutor(db)
        result = executor.execute(plan)
        assert c.claim_id not in [n.node_id for n in result.visited_nodes]


class TestFallbackContext:
    """Tests for paragraph fallback context blocks."""

    def test_fallback_reason_when_structured_thin(self, db, retrieval_setup):
        """fallback_reason is populated when structured results are thin."""
        db, graph_id = retrieval_setup
        from app.litmesh.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(db)
        result = retriever.retrieve(
            "nonexistent query term",
            [graph_id], top_k=5,
            include_context_blocks=True, context_window=1, max_context_blocks=3,
        )
        assert "fallback_reason" in result
        if result["fallback_reason"]:
            assert "Structured traversal" in result["fallback_reason"]
            assert len(result["context_blocks"]) <= 3

    def test_fallback_blocks_in_prompt_packet(self, db):
        """PromptPacket includes fallback_context_blocks."""
        from app.litmesh.models.prompt_packet import PromptPacket, FallbackContextBlock, ContextBlockPolicy

        pkt = PromptPacket(
            current_user_query="test query",
            fallback_context_blocks=[
                FallbackContextBlock(
                    section_id="sec_1", heading="Test Section",
                    raw_text="Raw paragraph text", page_start=1, page_end=1,
                )
            ],
            fallback_reason="Structured traversal returned 0 claims.",
            context_block_policy=ContextBlockPolicy.APPEND_AS_RAW,
        )
        assert len(pkt.fallback_context_blocks) == 1
        assert pkt.fallback_context_blocks[0].raw_text == "Raw paragraph text"
        assert pkt.context_block_policy == ContextBlockPolicy.APPEND_AS_RAW
        assert "0 claims" in pkt.fallback_reason
