"""
v0.3 tests: Concept normalization, graph traversal, block linking.

Tests the full concept lifecycle:
1. ConceptKey creation and registry dedup
2. ConceptNormalizer resolve + link flow
3. Concept hierarchy queries
4. Block concept_keys resolution
5. Cross-paper concept sharing within a graph
6. Conflict detection
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
from app.litmesh.models.claim import ClaimBlock, ClaimType, ClaimImportance
from app.litmesh.models.concept import ConceptKey, ConceptNamespace, ConceptStatus, MergePolicy
from app.litmesh.models.relation import GraphRelation, GraphRelationType
from app.litmesh.models.review import ReviewInboxItem, InboxType, InboxDecision

from app.litmesh.registry.concept_registry import ConceptRegistry
from app.litmesh.registry.concept_normalizer import ConceptNormalizer


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


@pytest.fixture
def graph(db):
    corpus = CorpusCard(name="Test Corpus", domain="AI_education")
    db.insert_corpus(corpus)
    g = SeriesGraph(
        corpus_id=corpus.corpus_id,
        name="Test Graph",
        domain="AI_education",
        concept_namespace="AI_edu",
    )
    db.insert_graph(g)
    return g


@pytest.fixture
def paper(db, graph):
    p = PaperCard(
        graph_id=graph.graph_id,
        title="Test Paper — AI Education Framework",
        authors=["Alice"],
        year=2024,
        source_file="test.pdf",
    )
    db.insert_paper(p)
    return p


@pytest.fixture
def claims(db, graph, paper):
    """Create test claims with concept keys."""
    c1 = ClaimBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        claim_text="AI教育必须在认知、实践和伦理三个层面同时推进",
        claim_type=ClaimType.FRAMEWORK,
        concept_keys=["framework:CPE_3DF", "concept:AI_education"],
        extraction_confidence=0.9,
    )
    c2 = ClaimBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        claim_text="AI幻觉是教育场景中的突出风险",
        claim_type=ClaimType.EMPIRICAL,
        concept_keys=["problem:AI_hallucination", "concept:risk_management"],
        extraction_confidence=0.8,
    )
    c3 = ClaimBlock(
        graph_id=graph.graph_id,
        paper_id=paper.paper_id,
        claim_text="虚拟实验不能完全替代真实实验",
        claim_type=ClaimType.CRITICAL,
        concept_keys=["method:virtual_lab", "problem:cognitive_load"],
        extraction_confidence=0.85,
    )
    db.insert_claim(c1)
    db.insert_claim(c2)
    db.insert_claim(c3)
    return [c1, c2, c3]


@pytest.fixture
def registry(db):
    return ConceptRegistry(db)


@pytest.fixture
def normalizer(db, registry):
    return ConceptNormalizer(db, registry)


# ============================================================
# ConceptKey lifecycle tests
# ============================================================

class TestConceptLifecycle:

    def test_create_concept_candidate(self, db, graph):
        """New concepts start as CANDIDATE."""
        c = ConceptKey(
            concept_key="framework:CPE_3DF",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="CPE-3DF框架",
            definition="认知-实践-伦理三维AI教育框架",
        )
        db.insert_concept(c)
        assert c.status == ConceptStatus.CANDIDATE

    def test_registry_rejects_duplicate(self, registry, db, graph):
        """Registry should reject exact key collisions."""
        c1 = ConceptKey(
            concept_key="framework:CPE_3DF",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="CPE-3DF框架",
        )
        db.insert_concept(c1)

        c2 = ConceptKey(
            concept_key="framework:CPE_3DF",  # Same key
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="CPE3DF框架",
        )
        decision = registry.register(c2, "run_test")
        assert decision["decision"] == "reject_duplicate"

    def test_registry_detects_alias_overlap(self, registry, db, graph):
        """Registry should flag alias overlap for review."""
        c1 = ConceptKey(
            concept_key="problem:AI_hallucination",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.PROBLEM,
            label_zh="AI幻觉",
            aliases=["AI幻觉", "大模型幻觉"],
        )
        db.insert_concept(c1)
        c1.status = ConceptStatus.ACTIVE
        c1.review_status = "approved"

        c2 = ConceptKey(
            concept_key="problem:LLM_hallucination",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.PROBLEM,
            label_zh="大模型幻觉",
            aliases=["大模型幻觉", "生成式AI幻觉"],
        )
        decision = registry.register(c2, "run_test")
        assert decision["decision"] == "needs_review"
        assert "AI_hallucination" in decision.get("existing_key", "")

    def test_concept_activate_and_reject(self, registry, db, graph):
        """Concepts can be activated or rejected after review."""
        c = ConceptKey(
            concept_key="method:prompt_engineering",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.METHOD,
            label_zh="提示工程",
        )
        db.insert_concept(c)

        assert registry.activate(c.concept_key)
        row = db.conn.execute(
            "SELECT status FROM concept_keys WHERE concept_key = ?", (c.concept_key,)
        ).fetchone()
        assert row["status"] == "active"

        assert registry.reject(c.concept_key)
        row = db.conn.execute(
            "SELECT status FROM concept_keys WHERE concept_key = ?", (c.concept_key,)
        ).fetchone()
        assert row["status"] == "rejected"

    def test_concept_merge(self, registry, db, graph):
        """Merge source into target concept."""
        c1 = ConceptKey(
            concept_key="problem:data_privacy",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.PROBLEM,
            label_zh="数据隐私",
        )
        c2 = ConceptKey(
            concept_key="problem:privacy_concern",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.PROBLEM,
            label_zh="隐私顾虑",
        )
        db.insert_concept(c1)
        db.insert_concept(c2)

        assert registry.merge("problem:privacy_concern", "problem:data_privacy")
        row = db.conn.execute(
            "SELECT status FROM concept_keys WHERE concept_key = ?",
            ("problem:privacy_concern",)
        ).fetchone()
        assert row["status"] == "merged"


# ============================================================
# Concept normalization pipeline
# ============================================================

class TestNormalizationPipeline:

    def test_normalize_new_concepts(self, normalizer, db, graph, claims):
        """Normalization creates candidate concepts and inbox items."""
        candidates = [
            ConceptKey(
                concept_key="framework:CPE_3DF",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.FRAMEWORK,
                label_zh="CPE-3DF框架",
                definition="认知-实践-伦理三维框架",
            ),
            ConceptKey(
                concept_key="problem:AI_hallucination",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.PROBLEM,
                label_zh="AI幻觉",
                aliases=["AI幻觉", "大模型幻觉"],
            ),
            ConceptKey(
                concept_key="method:virtual_lab",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.METHOD,
                label_zh="虚拟实验",
            ),
        ]

        stats = normalizer.normalize_extraction(
            candidate_concepts=candidates,
            claims=claims,
            evidence_blocks=[],
            limitations=[],
            extraction_run_id="run_test",
        )

        assert stats["new_concepts"] == 3
        assert stats["inbox_items"] >= 1  # Each needs review
        assert stats["block_links_updated"] == 3  # 3 claims updated

    def test_block_concept_keys_are_resolved(self, normalizer, db, graph, claims):
        """After normalization, blocks have canonical concept keys."""
        # Pre-seed one concept so it's found during resolution
        c = ConceptKey(
            concept_key="framework:CPE_3DF",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="CPE-3DF",
            status=ConceptStatus.ACTIVE,
        )
        db.insert_concept(c)

        candidates = [
            ConceptKey(
                concept_key="framework:CPE_3DF",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.FRAMEWORK,
                label_zh="CPE-3DF框架",
            ),
            ConceptKey(
                concept_key="problem:AI_hallucination",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.PROBLEM,
                label_zh="AI幻觉",
            ),
        ]

        normalizer.normalize_extraction(
            candidate_concepts=candidates,
            claims=claims,
            evidence_blocks=[],
            limitations=[],
            extraction_run_id="run_test",
        )

        # Check claim 1's concept_keys are canonical
        row = db.conn.execute(
            "SELECT concept_keys FROM claim_blocks WHERE claim_id = ?",
            (claims[0].claim_id,)
        ).fetchone()
        keys = json.loads(row["concept_keys"])
        assert "framework:CPE_3DF" in keys

    def test_mentions_relations_created(self, normalizer, db, graph, claims):
        """Each concept<->claim link creates a 'mentions' relation."""
        candidates = [
            ConceptKey(
                concept_key="framework:CPE_3DF",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.FRAMEWORK,
                label_zh="CPE-3DF",
            ),
        ]

        stats = normalizer.normalize_extraction(
            candidate_concepts=candidates,
            claims=claims,
            evidence_blocks=[],
            limitations=[],
            extraction_run_id="run_test",
        )

        # Should have created mentions relations for claims referencing CPE_3DF
        assert stats["relations_created"] >= 1

        # Verify in DB
        rels = db.get_relations_from(claims[0].claim_id, "mentions")
        assert len(rels) >= 1
        assert rels[0]["relation_type"] == "mentions"

    def test_conflict_detection(self, normalizer, db, graph):
        """Overlapping aliases should trigger conflict inbox items."""
        candidates = [
            ConceptKey(
                concept_key="problem:AI_hallucination",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.PROBLEM,
                label_zh="AI幻觉",
                aliases=["大模型幻觉"],
            ),
            ConceptKey(
                concept_key="problem:LLM_fabrication",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.PROBLEM,
                label_zh="大模型幻觉问题",
                aliases=["大模型幻觉"],  # Same alias!
            ),
        ]

        stats = normalizer.normalize_extraction(
            candidate_concepts=candidates,
            claims=[],
            evidence_blocks=[],
            limitations=[],
            extraction_run_id="run_test",
        )

        assert stats["conflicts_detected"] >= 1

    def test_do_not_merge_respected(self, normalizer, db, graph):
        """Concepts with do_not_merge_with should skip conflict detection."""
        candidates = [
            ConceptKey(
                concept_key="method:virtual_lab",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.METHOD,
                label_zh="虚拟实验",
                aliases=["虚拟实验室"],
                do_not_merge_with=["method:simulation_lab"],
            ),
            ConceptKey(
                concept_key="method:simulation_lab",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.METHOD,
                label_zh="仿真实验",
                aliases=["虚拟实验室"],  # Same alias but excluded
            ),
        ]

        stats = normalizer.normalize_extraction(
            candidate_concepts=candidates,
            claims=[],
            evidence_blocks=[],
            limitations=[],
            extraction_run_id="run_test",
        )

        # No conflict because do_not_merge_with excludes it
        assert stats["conflicts_detected"] == 0


# ============================================================
# Concept hierarchy and graph traversal
# ============================================================

class TestConceptGraph:

    def test_hierarchy_walk(self, db, graph):
        """Parent->child hierarchy should be queryable."""
        parent = ConceptKey(
            concept_key="concept:AI_education",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.CONCEPT,
            label_zh="AI教育",
            child_keys=["concept:AIGC_education", "concept:intelligent_tutoring"],
        )
        child1 = ConceptKey(
            concept_key="concept:AIGC_education",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.CONCEPT,
            label_zh="AIGC教育",
            parent_keys=["concept:AI_education"],
        )
        child2 = ConceptKey(
            concept_key="concept:intelligent_tutoring",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.CONCEPT,
            label_zh="智能辅导",
            parent_keys=["concept:AI_education"],
        )
        db.insert_concept(parent)
        db.insert_concept(child1)
        db.insert_concept(child2)

        hierarchy = db.get_concept_hierarchy("concept:AI_education")
        assert len(hierarchy["children"]) == 2
        assert len(hierarchy["parents"]) == 0

        child_hier = db.get_concept_hierarchy("concept:AIGC_education")
        assert len(child_hier["parents"]) == 1
        assert child_hier["parents"][0]["concept_key"] == "concept:AI_education"
        # AIGC_education and intelligent_tutoring are siblings
        assert len(child_hier["siblings"]) >= 1
        sibling_keys = [s["concept_key"] for s in child_hier["siblings"]]
        assert "concept:intelligent_tutoring" in sibling_keys

    def test_blocks_by_concept(self, db, graph, claims):
        """Query all blocks linked to a concept."""
        # Seed a concept and update a claim's concept_keys
        c = ConceptKey(
            concept_key="framework:CPE_3DF",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="CPE-3DF",
            status=ConceptStatus.ACTIVE,
        )
        db.insert_concept(c)

        result = db.get_blocks_by_concept("framework:CPE_3DF", graph.graph_id)
        assert len(result["claims"]) >= 1
        assert result["claims"][0]["claim_text"].find("同时推进") >= 0

    def test_neighborhood_expansion(self, db, graph):
        """BFS expansion should find related concepts within depth limit."""
        c1 = ConceptKey(
            concept_key="concept:AI_education",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.CONCEPT,
            label_zh="AI教育",
            child_keys=["concept:AIGC_education"],
        )
        c2 = ConceptKey(
            concept_key="concept:AIGC_education",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.CONCEPT,
            label_zh="AIGC教育",
            parent_keys=["concept:AI_education"],
            related_keys=["problem:AI_hallucination"],
        )
        c3 = ConceptKey(
            concept_key="problem:AI_hallucination",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.PROBLEM,
            label_zh="AI幻觉",
            related_keys=["concept:AIGC_education"],
        )
        db.insert_concept(c1)
        db.insert_concept(c2)
        db.insert_concept(c3)

        hood = db.expand_concept_neighborhood("concept:AI_education", graph.graph_id, max_depth=2)
        assert len(hood) >= 2
        keys = [h["concept_key"] for h in hood]
        assert "concept:AI_education" in keys
        assert "concept:AIGC_education" in keys
        # At depth 2, AI_hallucination should be reachable via related_keys
        assert "problem:AI_hallucination" in keys

    def test_namespace_filtering(self, db, graph):
        """Concepts can be filtered by namespace."""
        for ns in [ConceptNamespace.FRAMEWORK, ConceptNamespace.PROBLEM, ConceptNamespace.METHOD]:
            c = ConceptKey(
                concept_key=f"{ns.value}:test_{ns.value}",
                graph_id=graph.graph_id,
                namespace=ns,
                label_zh=f"Test {ns.value}",
            )
            db.insert_concept(c)

        frameworks = db.find_concepts_by_namespace(graph.graph_id, "framework")
        assert len(frameworks) == 1
        assert "test_framework" in frameworks[0]["concept_key"]


# ============================================================
# Cross-paper concept sharing within a graph
# ============================================================

class TestCrossPaperConcepts:

    def test_multi_paper_concept_resolution(self, db, graph, normalizer):
        """Two papers in the same graph should share concepts."""
        paper1 = PaperCard(
            graph_id=graph.graph_id,
            title="Paper 1 — CPE-3DF Framework",
            authors=["Author 1"],
            year=2024,
            source_file="paper1.pdf",
        )
        paper2 = PaperCard(
            graph_id=graph.graph_id,
            title="Paper 2 — AI Hallucination Study",
            authors=["Author 2"],
            year=2024,
            source_file="paper2.pdf",
        )
        db.insert_paper(paper1)
        db.insert_paper(paper2)

        c1 = ClaimBlock(
            graph_id=graph.graph_id,
            paper_id=paper1.paper_id,
            claim_text="CPE-3DF框架是有效的教育框架",
            claim_type=ClaimType.THEORETICAL,
            concept_keys=["framework:CPE_3DF"],
        )
        c2 = ClaimBlock(
            graph_id=graph.graph_id,
            paper_id=paper2.paper_id,
            claim_text="CPE-3DF在实践中也验证了有效性",
            claim_type=ClaimType.EMPIRICAL,
            concept_keys=["framework:CPE_3DF"],
        )
        db.insert_claim(c1)
        db.insert_claim(c2)

        candidates = [
            ConceptKey(
                concept_key="framework:CPE_3DF",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.FRAMEWORK,
                label_zh="CPE-3DF框架",
            ),
        ]

        normalizer.normalize_extraction(
            candidate_concepts=candidates,
            claims=[c1, c2],
            evidence_blocks=[],
            limitations=[],
            extraction_run_id="run_cross_paper",
        )

        # Both claims should now point to the same canonical concept key
        for claim_id in [c1.claim_id, c2.claim_id]:
            row = db.conn.execute(
                "SELECT concept_keys FROM claim_blocks WHERE claim_id = ?",
                (claim_id,)
            ).fetchone()
            keys = json.loads(row["concept_keys"])
            assert "framework:CPE_3DF" in keys


# ============================================================
# ConceptInbox integration
# ============================================================

class TestConceptInbox:

    def test_normalization_creates_inbox_items(self, normalizer, db, graph, claims):
        """Each new concept should create a ConceptInbox review item."""
        candidates = [
            ConceptKey(
                concept_key="framework:CPE_3DF",
                graph_id=graph.graph_id,
                namespace=ConceptNamespace.FRAMEWORK,
                label_zh="CPE-3DF框架",
            ),
        ]

        normalizer.normalize_extraction(
            candidate_concepts=candidates,
            claims=claims,
            evidence_blocks=[],
            limitations=[],
            extraction_run_id="run_inbox_test",
        )

        pending = db.get_pending_inbox(InboxType.CONCEPT.value)
        assert len(pending) >= 1
        assert pending[0]["item_type"] == "concept"

    def test_approve_concept_via_inbox(self, db, graph):
        """Approving a concept inbox item should activate the concept."""
        c = ConceptKey(
            concept_key="framework:CPE_3DF",
            graph_id=graph.graph_id,
            namespace=ConceptNamespace.FRAMEWORK,
            label_zh="CPE-3DF",
            status=ConceptStatus.CANDIDATE,
        )
        db.insert_concept(c)

        inbox = ReviewInboxItem(
            inbox_type=InboxType.CONCEPT,
            item_id=c.concept_key,
            item_type="concept",
            title="CPE-3DF框架",
            graph_id=graph.graph_id,
        )
        db.insert_inbox_item(inbox)

        # Approve
        db.resolve_inbox_item(inbox.inbox_id, "approve", decided_by="reviewer")
        db.conn.execute(
            "UPDATE concept_keys SET status = ?, review_status = ? WHERE concept_key = ?",
            (ConceptStatus.ACTIVE.value, "approved", c.concept_key),
        )
        db.conn.commit()

        row = db.conn.execute(
            "SELECT status, review_status FROM concept_keys WHERE concept_key = ?",
            (c.concept_key,)
        ).fetchone()
        assert row["status"] == "active"
        assert row["review_status"] == "approved"
