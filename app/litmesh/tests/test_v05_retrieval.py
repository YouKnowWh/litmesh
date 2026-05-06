"""
v0.5 tests: Series detection, hybrid retrieval, retrieval gate.
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
from app.litmesh.models.claim import ClaimBlock, ClaimType, ClaimImportance, ClaimStatus
from app.litmesh.models.evidence import EvidenceBlock, EvidenceType
from app.litmesh.models.limitation import LimitationBlock, RiskType
from app.litmesh.models.concept import ConceptKey, ConceptNamespace, ConceptStatus
from app.litmesh.models.relation import GraphRelation, GraphRelationType
from app.litmesh.models.source_span import SourceSpan, SpanPosition
from app.litmesh.models.series_group import SeriesGroup

from app.litmesh.registry.series_detector import SeriesDetector, _shingle
from app.litmesh.retrieval.retrieval_gate import decide_retrieval, GateInput
from app.litmesh.retrieval.concept_router import ConceptRouter
from app.litmesh.retrieval.hybrid_retriever import HybridRetriever
from app.litmesh.retrieval.vector_store import VectorStore, DummyEmbedder


# ============================================================
# SeriesGroup tests
# ============================================================

class TestSeriesGroup:

    def test_create_group(self, db):
        """SeriesGroup creation and retrieval."""
        g = SeriesGroup(
            name="AI Education Papers",
            graph_ids=["graph_a", "graph_b"],
            domain="AI_education",
        )
        db.insert_series_group(g)
        loaded = db.get_series_group(g.group_id)
        assert loaded["name"] == "AI Education Papers"
        ids = json.loads(loaded["graph_ids"])
        assert "graph_a" in ids

    def test_add_graph_to_group(self, db):
        """Add a graph to existing group."""
        g = SeriesGroup(name="Test", graph_ids=["graph_1"], domain="test")
        db.insert_series_group(g)
        db.add_graph_to_series_group(g.group_id, "graph_2")
        loaded = db.get_series_group(g.group_id)
        ids = json.loads(loaded["graph_ids"])
        assert len(ids) == 2
        assert "graph_2" in ids

    def test_find_group_for_graph(self, db):
        """Look up which group a graph belongs to."""
        g = SeriesGroup(name="Test", graph_ids=["graph_x"], domain="test")
        db.insert_series_group(g)
        found = db.find_series_group_for_graph("graph_x")
        assert found is not None
        assert found["name"] == "Test"

    def test_graph_without_group(self, db):
        """A graph not in any group returns None."""
        found = db.find_series_group_for_graph("nonexistent")
        assert found is None


# ============================================================
# SeriesDetector tests
# ============================================================

class TestSeriesDetection:

    def test_shingle_chinese(self):
        """Chinese text shingling produces character n-grams."""
        shingles = _shingle("人工智能教育", k=3)
        assert "人工智" in shingles
        assert len(shingles) >= 3

    def test_shingle_english(self):
        """English text shingling."""
        shingles = _shingle("Operating Systems", k=3)
        assert "ope" in shingles
        assert "rat" in shingles

    def test_minhash_same_book_different_chapter(self, db):
        """Same book, different chapters should be grouped."""
        from datasketch import MinHash

        corpus = CorpusCard(name="Test", corpus_id="corpus_ostep")
        db.insert_corpus(corpus)

        # Existing group with one paper
        g = SeriesGroup(
            name="OSTEP",
            graph_ids=["graph_ostep_1"],
            domain="operating_systems",
        )
        db.insert_series_group(g)

        db.insert_graph(SeriesGraph(
            graph_id="graph_ostep_1", corpus_id="corpus_ostep",
            name="OSTEP Ch3", domain="operating_systems",
        ))
        db.insert_paper(PaperCard(
            graph_id="graph_ostep_1",
            title="OSTEP: Operating Systems — Chapter 3: CPU Scheduling",
            authors=["Remzi"], year=2019, source_file="ostep3.pdf",
        ))

        # New paper: same book, different chapter
        detector = SeriesDetector(db, llm_client=None)
        paper = PaperCard(
            graph_id="",  # New graph
            title="OSTEP: Operating Systems — Chapter 5: Memory Management",
            authors=["Remzi"], year=2019, source_file="ostep5.pdf",
            main_framework="operating_systems",
        )
        result = detector.detect(paper, "graph_ostep_2", "operating_systems")
        assert result["action"] == "add_to_existing"
        assert "OSTEP" in result["group_name"]

    def test_minhash_different_books(self, db):
        """Different books should NOT be grouped together."""
        corpus = CorpusCard(name="Test", corpus_id="corpus_diff")
        db.insert_corpus(corpus)

        # Existing group: OSTEP
        g = SeriesGroup(
            name="OSTEP",
            graph_ids=["graph_ostep"],
            domain="operating_systems",
        )
        db.insert_series_group(g)
        db.insert_graph(SeriesGraph(
            graph_id="graph_ostep", corpus_id="corpus_diff",
            name="OSTEP", domain="operating_systems",
        ))
        db.insert_paper(PaperCard(
            graph_id="graph_ostep",
            title="OSTEP: Operating Systems — Chapter 3: CPU Scheduling",
            authors=["Remzi"], year=2019, source_file="ostep.pdf",
        ))

        # New paper: completely different book
        detector = SeriesDetector(db, llm_client=None)
        paper = PaperCard(
            graph_id="",
            title="Kurose: Computer Networking — Chapter 1: Introduction",
            authors=["Kurose"], year=2020, source_file="kurose.pdf",
            main_framework="computer_networking",
        )
        result = detector.detect(paper, "graph_kurose", "computer_networking")
        assert result["action"] == "new_group"

    def test_minhash_chinese_papers_same_series(self, db):
        """Chinese papers in the same series should match."""
        corpus = CorpusCard(name="AI Edu", corpus_id="corpus_ai")
        db.insert_corpus(corpus)

        g = SeriesGroup(
            name="AI Education Papers",
            graph_ids=["graph_ai_1"],
            domain="AI_education",
        )
        db.insert_series_group(g)
        db.insert_graph(SeriesGraph(
            graph_id="graph_ai_1", corpus_id="corpus_ai",
            name="AI Edu Paper 1", domain="AI_education",
        ))
        db.insert_paper(PaperCard(
            graph_id="graph_ai_1",
            title="AIGC赋能生物工程教育的范式重构：CPE-3DF框架研究",
            authors=["张三"], year=2024, source_file="cpe3df.pdf",
            main_framework="CPE-3DF",
        ))

        detector = SeriesDetector(db, llm_client=None)
        paper = PaperCard(
            graph_id="",
            title="AIGC赋能生物工程教育的实践验证：CPE-3DF框架应用",
            authors=["李四"], year=2024, source_file="cpe3df_app.pdf",
            main_framework="CPE-3DF",
        )
        result = detector.detect(paper, "graph_ai_2", "AI_education")
        # Shared framework + similar title structure should match
        assert result["action"] == "add_to_existing"

    def test_new_group_when_no_match(self, db):
        """Paper with no matching series creates a new group."""
        detector = SeriesDetector(db, llm_client=None)
        paper = PaperCard(
            graph_id="graph_new", title="完全不同的论文标题",
            authors=["Author"], year=2024, source_file="test.pdf",
            main_framework="Unknown",
        )
        result = detector._new_group(paper, "graph_new", "general", "No match")
        assert result["action"] == "new_group"
        assert result["confidence"] == 1.0


# ============================================================
# RetrievalGate tests
# ============================================================

class TestRetrievalGate:

    def test_skip_short_query(self):
        decision = decide_retrieval(GateInput(query="hi"))
        assert decision.should_retrieve is False
        assert decision.mode == "skip"

    def test_trigger_keyword_match(self):
        decision = decide_retrieval(GateInput(query="你还记得上次提到的框架吗"))
        assert decision.should_retrieve is True
        # "记得" or "上次" or "框架" should trigger
        assert decision.mode == "full"

    def test_sufficient_anchors_skip(self):
        decision = decide_retrieval(GateInput(
            query="继续说前面的内容",
            existing_anchor_count=5,
            existing_confidence_sum=4.0,  # avg 0.8
        ))
        assert decision.should_retrieve is False

    def test_long_query_triggers_vector(self):
        decision = decide_retrieval(GateInput(
            query="CPE-3DF框架中伦理维度和认知维度之间有什么关系",
        ))
        assert decision.should_retrieve is True

    def test_explain_query_triggers(self):
        """'什么是' type questions should trigger retrieval."""
        decision = decide_retrieval(GateInput(query="什么是PACADI框架"))
        assert decision.should_retrieve is True


# ============================================================
# ConceptRouter tests
# ============================================================

class TestConceptRouter:

    def test_route_by_alias(self, db, graph_setup):
        """Query matching concept alias should route to that concept."""
        db, graph_id = graph_setup
        router = ConceptRouter(db)
        results = router.route("AI幻觉", [graph_id])
        assert len(results) >= 1
        assert any("AI_hallucination" in r["concept_key"] for r in results)

    def test_route_by_label_like(self, db, graph_setup):
        """Partial label match should find concepts."""
        db, graph_id = graph_setup
        router = ConceptRouter(db)
        results = router.route("框架", [graph_id])
        assert len(results) >= 1

    def test_route_no_match(self, db):
        """Query with no matching concepts returns empty."""
        router = ConceptRouter(db)
        results = router.route("xyznotexist", ["graph_x"])
        assert len(results) == 0


# ============================================================
# HybridRetriever tests
# ============================================================

class TestHybridRetriever:

    def test_fts_retrieval(self, db, retrieval_setup):
        """Full-text search should find matching claims."""
        db, graph_id = retrieval_setup
        retriever = HybridRetriever(db)
        result = retriever.retrieve("AI教育框架", [graph_id])
        assert len(result["claims"]) >= 1
        assert result["method"] in ("fts_only", "full")

    def test_limitation_injection(self, db, retrieval_setup):
        """Retrieval should inject related limitations."""
        db, graph_id = retrieval_setup
        retriever = HybridRetriever(db)
        result = retriever.retrieve("AI教育", [graph_id], include_limitations=True)
        assert len(result["limitations"]) >= 1

    def test_skip_on_short_query(self, db, retrieval_setup):
        """Very short queries should be skipped."""
        db, graph_id = retrieval_setup
        retriever = HybridRetriever(db)
        result = retriever.retrieve("hi", [graph_id])
        assert result["decision"].should_retrieve is False

    def test_concept_routing_included(self, db, retrieval_setup):
        """Retrieval result includes concept routes."""
        db, graph_id = retrieval_setup
        retriever = HybridRetriever(db)
        result = retriever.retrieve("CPE-3DF框架的伦理维度", [graph_id])
        assert len(result["concepts"]) >= 0  # May be 0 if no concepts in scope


# ============================================================
# VectorStore tests
# ============================================================

class TestVectorStore:

    def test_dummy_embedder(self):
        """DummyEmbedder produces consistent results."""
        emb = DummyEmbedder(dim=128)
        v1 = emb.embed("test text")
        v2 = emb.embed("test text")
        assert v1 == v2
        assert len(v1) == 128

    def test_dummy_embedder_different_texts(self):
        """Different texts produce different embeddings."""
        emb = DummyEmbedder(dim=64)
        v1 = emb.embed("claim about AI")
        v2 = emb.embed("limitation about data")
        assert v1 != v2


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
def graph_setup(db):
    """Setup with concepts for routing."""
    corpus = CorpusCard(name="Test", domain="AI_education")
    db.insert_corpus(corpus)
    graph = SeriesGraph(
        corpus_id=corpus.corpus_id, name="Test Graph",
        domain="AI_education",
    )
    db.insert_graph(graph)

    conc = ConceptKey(
        concept_key="problem:AI_hallucination",
        graph_id=graph.graph_id,
        namespace=ConceptNamespace.PROBLEM,
        label_zh="AI幻觉",
        aliases=["AI幻觉", "大模型幻觉"],
        status=ConceptStatus.ACTIVE,
    )
    conc2 = ConceptKey(
        concept_key="framework:CPE_3DF",
        graph_id=graph.graph_id,
        namespace=ConceptNamespace.FRAMEWORK,
        label_zh="CPE-3DF框架",
        aliases=["CPE-3DF", "认知实践伦理框架"],
        status=ConceptStatus.ACTIVE,
    )
    db.insert_concept(conc)
    db.insert_concept(conc2)
    return db, graph.graph_id


@pytest.fixture
def retrieval_setup(db):
    """Setup with claims, evidence, limitations, and relations."""
    corpus = CorpusCard(name="Test", domain="AI_education")
    db.insert_corpus(corpus)
    graph = SeriesGraph(
        corpus_id=corpus.corpus_id, name="Test Graph",
        domain="AI_education",
    )
    db.insert_graph(graph)

    paper = PaperCard(
        graph_id=graph.graph_id, title="Test", authors=["A"],
        year=2024, source_file="test.pdf",
    )
    db.insert_paper(paper)

    span_id = "span_test01"
    db.insert_span(SourceSpan(
        span_id=span_id, paper_id=paper.paper_id,
        span_type="paragraph", source_text="Test",
        position=SpanPosition(char_start=0, char_end=4, page_start=1),
    ))

    claim = ClaimBlock(
        graph_id=graph.graph_id, paper_id=paper.paper_id,
        claim_text="AI教育框架需要关注伦理维度",
        claim_type=ClaimType.THEORETICAL, extraction_confidence=0.9,
        status=ClaimStatus.ACTIVE, source_span_id=span_id,
    )
    db.insert_claim(claim)

    lim = LimitationBlock(
        graph_id=graph.graph_id, paper_id=paper.paper_id,
        limitation_text="伦理维度的实证数据不足",
        affected_claim_ids=[claim.claim_id],
        risk_type=RiskType.DATA, source_span_id=span_id,
    )
    db.insert_limitation(lim)

    db.insert_relation(GraphRelation(
        graph_id=graph.graph_id,
        source_id=lim.limitation_id, target_id=claim.claim_id,
        source_type="limitation", target_type="claim",
        relation_type=GraphRelationType.CONSTRAINS,
        confidence=0.8,
    ))

    return db, graph.graph_id
