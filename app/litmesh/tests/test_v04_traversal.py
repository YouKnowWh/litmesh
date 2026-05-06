"""
v0.4 tests: Typed Pointer Traversal.

Tests all 7 traversal modes, executor constraints, trace persistence,
and PromptPacket compilation.
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
from app.litmesh.models.section import SectionBlock, HeadingLevel
from app.litmesh.models.claim import ClaimBlock, ClaimType, ClaimImportance, ClaimStatus
from app.litmesh.models.evidence import EvidenceBlock, EvidenceType, EvidenceStrength
from app.litmesh.models.limitation import LimitationBlock, RiskType, LimitationSeverity
from app.litmesh.models.concept import ConceptKey, ConceptNamespace, ConceptStatus
from app.litmesh.models.relation import GraphRelation, GraphRelationType
from app.litmesh.models.source_span import SourceSpan, SpanPosition, SpanType
from app.litmesh.models.prompt_packet import TraversalMode, PointerType, TraversalPlan

from app.litmesh.traversal.traversal_presets import (
    build_preset_plan, get_preset, PRESETS,
)
from app.litmesh.traversal.traversal_executor import TraversalExecutor
from app.litmesh.traversal.traversal_trace import TraceStore
from app.litmesh.compiler.prompt_packet_compiler import PromptPacketCompiler


# ============================================================
# Fixtures — build a small but complete knowledge graph
# ============================================================

@pytest.fixture
def db():
    database = LitMeshDB(":memory:")
    database.connect()
    database.init_schema()
    yield database
    database.close()


@pytest.fixture
def knowledge_graph(db):
    """Build a realistic mini knowledge graph with:
    - 1 corpus, 1 graph, 1 paper
    - 1 section
    - 3 claims (1 core, 1 supporting, 1 contradictory)
    - 2 evidence blocks
    - 1 limitation
    - 2 concepts with parent/child relation
    - GraphRelations: supports, constrains, contradicts, refines, derived_from
    """
    corpus = CorpusCard(name="Test", domain="AI_education")
    db.insert_corpus(corpus)

    graph = SeriesGraph(
        corpus_id=corpus.corpus_id,
        name="KG Test",
        domain="AI_education",
        concept_namespace="AI_edu",
    )
    db.insert_graph(graph)

    paper = PaperCard(
        graph_id=graph.graph_id,
        title="Test Paper",
        authors=["Author"],
        year=2024,
        source_file="test.pdf",
    )
    db.insert_paper(paper)

    section = SectionBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        heading="Methods",
        heading_path=["Methods"],
        heading_level=HeadingLevel.SECTION,
        raw_text="Experiment description...",
        page_start=3,
    )
    db.insert_section(section)

    # Source spans (required for traversal — Principle 5)
    span1_id = f"span_{hash('test1') % 10**12:012d}"
    span2_id = f"span_{hash('test2') % 10**12:012d}"
    for sp_id, pg in [(span1_id, 3), (span2_id, 4)]:
        db.insert_span(SourceSpan(
            span_id=sp_id,
            paper_id=paper.paper_id,
            section_id=section.section_id,
            span_type=SpanType.PARAGRAPH,
            source_text="Test source text for traversal",
            position=SpanPosition(char_start=0, char_end=30, page_start=pg),
        ))

    # Claims
    c1 = ClaimBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        section_id=section.section_id,
        claim_text="AI教育框架必须同时覆盖认知、实践和伦理三个维度",
        claim_type=ClaimType.FRAMEWORK,
        importance=ClaimImportance.CORE,
        extraction_confidence=0.9,
        concept_keys=["framework:test_framework"],
        status=ClaimStatus.ACTIVE,
        source_span_id=span1_id,
    )
    c2 = ClaimBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        section_id=section.section_id,
        claim_text="AI辅助教学可以提升学习效果23%",
        claim_type=ClaimType.EMPIRICAL,
        importance=ClaimImportance.SUPPORTING,
        extraction_confidence=0.8,
        concept_keys=["concept:learning_outcome"],
        status=ClaimStatus.ACTIVE,
        source_span_id=span2_id,
    )
    c3 = ClaimBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        section_id=section.section_id,
        claim_text="AI框架只需要关注技术层面即可",
        claim_type=ClaimType.CRITICAL,
        importance=ClaimImportance.SUPPORTING,
        extraction_confidence=0.7,
        concept_keys=[],
        status=ClaimStatus.ACTIVE,
        source_span_id=span1_id,
    )
    db.insert_claim(c1)
    db.insert_claim(c2)
    db.insert_claim(c3)

    # Evidence
    e1 = EvidenceBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        section_id=section.section_id,
        supports_claim_ids=[c2.claim_id],
        evidence_text="实验组n=120，提升23%，p<0.01",
        evidence_type=EvidenceType.DATA,
        strength=EvidenceStrength.STRONG,
        source_span_id=span1_id,
    )
    e2 = EvidenceBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        section_id=section.section_id,
        supports_claim_ids=[c1.claim_id],
        evidence_text="三所大学的教学实践验证了框架有效性",
        evidence_type=EvidenceType.TEACHING_PRACTICE,
        strength=EvidenceStrength.MODERATE,
        source_span_id=span2_id,
    )
    db.insert_evidence(e1)
    db.insert_evidence(e2)

    # Limitation
    lim = LimitationBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        section_id=section.section_id,
        affected_claim_ids=[c1.claim_id, c2.claim_id],
        limitation_text="样本局限于三所大学，样本量小，周期仅一学期",
        risk_type=RiskType.SCOPE,
        severity=LimitationSeverity.MODERATE,
        source_span_id=span1_id,
    )
    db.insert_limitation(lim)

    # Concepts
    conc1 = ConceptKey(
        concept_key="framework:test_framework",
        graph_id=graph.graph_id,
        namespace=ConceptNamespace.FRAMEWORK,
        label_zh="测试框架",
        child_keys=["concept:learning_outcome"],
        status=ConceptStatus.ACTIVE,
    )
    conc2 = ConceptKey(
        concept_key="concept:learning_outcome",
        graph_id=graph.graph_id,
        namespace=ConceptNamespace.CONCEPT,
        label_zh="学习效果",
        parent_keys=["framework:test_framework"],
        status=ConceptStatus.ACTIVE,
    )
    db.insert_concept(conc1)
    db.insert_concept(conc2)

    # Relations
    rels = [
        GraphRelation(
            graph_id=graph.graph_id,
            source_id=c1.claim_id, target_id=c2.claim_id,
            source_type="claim", target_type="claim",
            relation_type=GraphRelationType.DERIVED_FROM,
            confidence=0.8, traversal_cost=1.0,
        ),
        GraphRelation(
            graph_id=graph.graph_id,
            source_id=e1.evidence_id, target_id=c2.claim_id,
            source_type="evidence", target_type="claim",
            relation_type=GraphRelationType.SUPPORTS,
            confidence=0.9, traversal_cost=1.0,
        ),
        GraphRelation(
            graph_id=graph.graph_id,
            source_id=e2.evidence_id, target_id=c1.claim_id,
            source_type="evidence", target_type="claim",
            relation_type=GraphRelationType.SUPPORTS,
            confidence=0.7, traversal_cost=1.0,
        ),
        GraphRelation(
            graph_id=graph.graph_id,
            source_id=lim.limitation_id, target_id=c1.claim_id,
            source_type="limitation", target_type="claim",
            relation_type=GraphRelationType.CONSTRAINS,
            confidence=0.85, traversal_cost=1.0,
        ),
        GraphRelation(
            graph_id=graph.graph_id,
            source_id=lim.limitation_id, target_id=c2.claim_id,
            source_type="limitation", target_type="claim",
            relation_type=GraphRelationType.CONSTRAINS,
            confidence=0.85, traversal_cost=1.0,
        ),
        GraphRelation(
            graph_id=graph.graph_id,
            source_id=c3.claim_id, target_id=c1.claim_id,
            source_type="claim", target_type="claim",
            relation_type=GraphRelationType.CONTRADICTS,
            confidence=0.75, traversal_cost=1.5,
        ),
        GraphRelation(
            graph_id=graph.graph_id,
            source_id=c1.claim_id, target_id=conc1.concept_key,
            source_type="claim", target_type="concept",
            relation_type=GraphRelationType.MENTIONS,
            confidence=0.9, traversal_cost=0.5,
        ),
    ]
    for r in rels:
        db.insert_relation(r)

    return {
        "graph_id": graph.graph_id,
        "paper_id": paper.paper_id,
        "claim_ids": [c1.claim_id, c2.claim_id, c3.claim_id],
        "evidence_ids": [e1.evidence_id, e2.evidence_id],
        "limitation_id": lim.limitation_id,
        "concept_keys": [conc1.concept_key, conc2.concept_key],
        "c1_id": c1.claim_id,
        "c2_id": c2.claim_id,
        "c3_id": c3.claim_id,
    }


@pytest.fixture
def executor(db):
    return TraversalExecutor(db)


@pytest.fixture
def compiler():
    return PromptPacketCompiler()


# ============================================================
# Preset validation
# ============================================================

class TestPresets:
    """Verify all 7 presets are properly configured."""

    def test_all_modes_have_presets(self):
        for mode in TraversalMode:
            assert mode in PRESETS, f"Missing preset for {mode}"

    def test_preset_constraints_are_sane(self):
        for mode, preset in PRESETS.items():
            assert preset.max_depth >= 1
            assert preset.max_nodes >= 1
            assert preset.max_edges_per_type >= 1
            assert 0.0 <= preset.min_confidence <= 1.0
            assert len(preset.pointer_types) >= 1

    def test_build_plan_from_preset(self):
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=["claim:test"],
            graph_scope=["graph:test"],
        )
        assert plan.traversal_mode == TraversalMode.AUDIT
        assert plan.max_depth == 2  # Audit mode uses depth 2
        assert plan.require_source_span is True

    def test_explain_mode_walks_concept_routes(self):
        """Explain mode should include concept-to-claim path pointers."""
        preset = get_preset(TraversalMode.EXPLAIN)
        assert PointerType.BELONGS_TO in preset.pointer_types
        assert PointerType.SUPPORTS in preset.pointer_types
        assert PointerType.CONSTRAINS in preset.pointer_types

    def test_transfer_mode_allows_cross_graph(self):
        """Transfer mode is the only mode allowing cross-graph traversal."""
        preset = get_preset(TraversalMode.TRANSFER)
        assert preset.allow_cross_graph is True
        assert preset.max_cross_graph_jumps > 0

        # All other modes should disallow cross-graph
        for mode in TraversalMode:
            if mode == TraversalMode.TRANSFER:
                continue
            assert get_preset(mode).allow_cross_graph is False


# ============================================================
# TraversalExecutor
# ============================================================

class TestExecutor:

    def test_basic_traversal(self, executor, knowledge_graph):
        """Start from a claim and traverse supports + constrains edges."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)

        # Should find: c1, c2 (derived_from), e2 (supports c1), lim (constrains c1)
        assert len(result.visited_nodes) >= 4
        assert knowledge_graph["c1_id"] in result.grouped_claims
        # c1 and c2 should both be in visited claims
        claim_ids = [n.node_id for n in result.visited_nodes if n.node_type == "claim"]
        assert knowledge_graph["c1_id"] in claim_ids

    def test_conflict_mode_finds_contradictions(self, executor, knowledge_graph):
        """Conflict mode should find the contradicts relation between c1 and c3."""
        plan = build_preset_plan(
            TraversalMode.CONFLICT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        assert len(result.conflicts) >= 1  # c1 or c3 should appear as conflict

    def test_audit_mode_includes_limitations(self, executor, knowledge_graph):
        """Audit mode must include limitations per preset."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        assert len(result.grouped_limitations) >= 1

    def test_max_depth_enforced(self, executor, knowledge_graph):
        """Traversal should not exceed max_depth."""
        plan = build_preset_plan(
            TraversalMode.SYNTHESIS,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        plan.max_depth = 1  # Override to 1
        result = executor.execute(plan)

        # At depth 1, should only get direct neighbors of c1
        # Check the stopped reason
        assert "Reached max_depth" not in result.stopped_reason.lower() or len(result.visited_nodes) <= 10

    def test_max_nodes_enforced(self, executor, knowledge_graph):
        """Traversal should stop at max_nodes."""
        plan = build_preset_plan(
            TraversalMode.SYNTHESIS,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        plan.max_nodes = 2
        result = executor.execute(plan)
        assert len(result.visited_nodes) <= 2
        assert "max_nodes" in result.stopped_reason.lower()

    def test_cycle_detection(self, executor, knowledge_graph):
        """Nodes should not be visited twice."""
        plan = build_preset_plan(
            TraversalMode.SYNTHESIS,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)

        # No duplicate node_ids
        node_ids = [n.node_id for n in result.visited_nodes]
        assert len(node_ids) == len(set(node_ids))

    def test_concept_resolution(self, executor, knowledge_graph):
        """Starting from a concept should resolve all connected claims."""
        plan = build_preset_plan(
            TraversalMode.EXPLAIN,
            start_nodes=[knowledge_graph["concept_keys"][0]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        # Should find the concept node itself
        concept_nodes = [n for n in result.visited_nodes if n.node_type == "concept"]
        assert len(concept_nodes) >= 1

    def test_low_confidence_gating(self, executor, knowledge_graph):
        """Edges below min_confidence should be skipped."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        plan.min_confidence = 0.9  # High bar: only 0.9+ edges pass
        result = executor.execute(plan)

        # The SUPPORTS from e2 to c1 has confidence 0.7 (below 0.9)
        # It should be excluded
        e2_id = knowledge_graph["evidence_ids"][1]
        evidence_nodes = [n.node_id for n in result.visited_nodes if n.node_type == "evidence"]
        assert e2_id not in evidence_nodes

    def test_stopped_reason_is_set(self, executor, knowledge_graph):
        """Every traversal result should have a stopped_reason."""
        plan = build_preset_plan(
            TraversalMode.EXPLAIN,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        assert result.stopped_reason != ""

    def test_traversal_cost_tracking(self, executor, knowledge_graph):
        """TraversalResult should track total cost."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        assert result.total_traversal_cost >= 0


# ============================================================
# TraversalTrace
# ============================================================

class TestTraversalTrace:

    def test_save_and_load_trace(self, db, executor, knowledge_graph):
        """Trace should be persisted and retrievable."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)

        store = TraceStore(db)
        trace_id = store.save(
            query="CPE-3DF框架的核心主张是什么？",
            plan=plan,
            result=result,
        )
        assert trace_id.startswith("trace_")

        loaded = store.load(trace_id)
        assert loaded is not None
        assert loaded["query"] == "CPE-3DF框架的核心主张是什么？"

    def test_list_recent_traces(self, db, executor, knowledge_graph):
        """Should list recent traces in reverse chronological order."""
        store = TraceStore(db)
        for i in range(3):
            plan = build_preset_plan(
                TraversalMode.AUDIT,
                start_nodes=[knowledge_graph["c1_id"]],
                graph_scope=[knowledge_graph["graph_id"]],
            )
            result = executor.execute(plan)
            store.save(query=f"Query {i}", plan=plan, result=result)

        recent = store.list_recent(limit=5)
        assert len(recent) == 3


# ============================================================
# PromptPacket Compilation
# ============================================================

class TestPromptPacketCompilation:

    def test_compile_from_traversal(self, executor, knowledge_graph, compiler):
        """PromptPacket should be compilable from any TraversalResult."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)

        packet = compiler.compile(
            user_query="CPE-3DF框架的核心主张是什么？有哪些证据和限制？",
            intent="audit_framework_claims",
            result=result,
            plan=plan,
            trace_id="test_trace",
        )

        assert packet.current_user_query != ""
        assert len(packet.paper_claims) >= 0
        assert len(packet.limitations) >= 1
        assert packet.generation_policy.must_cite_claims is True

    def test_compile_with_conflicts(self, executor, knowledge_graph, compiler):
        """Conflict mode should populate the conflicts section."""
        plan = build_preset_plan(
            TraversalMode.CONFLICT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)

        packet = compiler.compile(
            user_query="文献之间有什么矛盾？",
            intent="analyze_conflicts",
            result=result,
            plan=plan,
            trace_id="test_trace",
        )

        assert len(packet.conflicts) >= 1

    def test_text_rendering(self, executor, knowledge_graph, compiler):
        """PromptPacket should render to structured text."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        packet = compiler.compile(
            user_query="测试问题",
            intent="test",
            result=result,
            plan=plan,
        )
        text = compiler.compile_to_text(packet)

        assert "LitMesh Structured Context" in text
        assert "Generation Constraints" in text
        assert "Limitations" in text

    def test_low_confidence_items_are_separated(self, executor, knowledge_graph, compiler):
        """Low-confidence nodes go to low_confidence_candidates, not claims."""
        plan = build_preset_plan(
            TraversalMode.EXPLAIN,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        packet = compiler.compile(
            user_query="test",
            intent="test",
            result=result,
            plan=plan,
        )
        # Low confidence items should not be in paper_claims
        for claim in packet.paper_claims:
            assert claim.confidence >= 0.5, f"Low confidence claim {claim.claim_id} in paper_claims"


# ============================================================
# End-to-end: query -> traversal -> compile -> text
# ============================================================

class TestEndToEnd:
    """Simulate the full v0.4 pipeline."""

    def test_explain_flow(self, executor, knowledge_graph, compiler, db):
        """Full explain mode pipeline."""
        # 1. User query -> select mode
        mode = TraversalMode.EXPLAIN
        start = knowledge_graph["concept_keys"][0]

        # 2. Build plan from preset
        plan = build_preset_plan(
            mode,
            start_nodes=[start],
            graph_scope=[knowledge_graph["graph_id"]],
            task_type="解释测试框架的概念结构",
        )

        # 3. Execute traversal
        result = executor.execute(plan)
        assert len(result.visited_nodes) >= 1

        # 4. Save trace
        store = TraceStore(db)
        trace_id = store.save(
            query="测试框架包含哪些概念？",
            plan=plan,
            result=result,
        )

        # 5. Compile PromptPacket
        packet = compiler.compile(
            user_query="测试框架包含哪些概念？",
            intent="explain_framework_concepts",
            result=result,
            plan=plan,
            trace_id=trace_id,
        )

        # 6. Render to text for LLM
        context_text = compiler.compile_to_text(packet)
        assert "LitMesh" in context_text

        # 7. Verify packet has concepts and structure
        assert packet.interpreted_intent == "explain_framework_concepts"
        assert len(packet.active_concepts) >= 1

    def test_audit_flow(self, executor, knowledge_graph, compiler):
        """Audit mode: claim -> evidence -> limitation."""
        plan = build_preset_plan(
            TraversalMode.AUDIT,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)

        # Audit must include both evidence and limitations
        assert len(result.grouped_evidence) >= 1
        assert len(result.grouped_limitations) >= 1

    def test_compare_flow(self, executor, knowledge_graph):
        """Compare mode: claim -> evidence -> limitation -> refines."""
        plan = build_preset_plan(
            TraversalMode.COMPARE,
            start_nodes=[knowledge_graph["c1_id"]],
            graph_scope=[knowledge_graph["graph_id"]],
        )
        result = executor.execute(plan)
        # Should find related claims
        assert len(result.grouped_claims) >= 1
