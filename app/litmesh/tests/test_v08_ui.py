"""
v0.8 tests: Admin UI endpoints and HTML serving.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.litmesh.storage.sqlite import LitMeshDB
from app.litmesh.models.corpus import CorpusCard
from app.litmesh.models.graph import SeriesGraph
from app.litmesh.models.paper import PaperCard
from app.litmesh.models.claim import ClaimBlock, ClaimStatus
from app.litmesh.models.concept import ConceptKey, ConceptNamespace, ConceptStatus
from app.litmesh.models.relation import BridgeRelation, BridgeRelationType, BridgeStatus
from app.litmesh.models.review import ReviewInboxItem, InboxType
from app.litmesh.models.source_span import SourceSpan, SpanPosition
from app.litmesh.models.series_group import SeriesGroup
from app.litmesh.api.routes import create_app


@pytest.fixture
def client(db):
    """Create a FastAPI test client with a fresh in-memory DB."""
    app = create_app(db)
    return TestClient(app)


@pytest.fixture
def db():
    database = LitMeshDB(":memory:")
    database.connect()
    database.init_schema()
    _seed_data(database)
    yield database
    database.close()


def _seed_data(db):
    """Seed minimal data for UI endpoints."""
    c = CorpusCard(name="Test Corpus", domain="test")
    db.insert_corpus(c)
    g = SeriesGraph(corpus_id=c.corpus_id, name="Test Graph", domain="test")
    db.insert_graph(g)

    p = PaperCard(graph_id=g.graph_id, title="Test Paper", authors=["A"],
                  year=2024, source_file="test.pdf", main_framework="TestFW")
    db.insert_paper(p)

    span = SourceSpan(span_id="span_ui01", paper_id=p.paper_id,
                      span_type="paragraph", source_text="Test",
                      position=SpanPosition(char_start=0, char_end=4, page_start=1))
    db.insert_span(span)

    claim = ClaimBlock(graph_id=g.graph_id, paper_id=p.paper_id,
                       claim_text="A test claim", concept_keys=["concept:test"],
                       status=ClaimStatus.ACTIVE, source_span_id="span_ui01")
    db.insert_claim(claim)

    concept = ConceptKey(concept_key="concept:test", graph_id=g.graph_id,
                         namespace=ConceptNamespace.CONCEPT, label_zh="测试概念",
                         status=ConceptStatus.ACTIVE)
    db.insert_concept(concept)

    inbox = ReviewInboxItem(inbox_type=InboxType.EXTRACTION, item_id=claim.claim_id,
                            item_type="claim", title="Review test claim",
                            graph_id=g.graph_id, paper_id=p.paper_id)
    db.insert_inbox_item(inbox)

    bridge = BridgeRelation(source_graph_id=g.graph_id, target_graph_id=g.graph_id,
                            source_key="concept:test", target_key="concept:other",
                            bridge_type=BridgeRelationType.SAME_AS,
                            bridge_confidence=0.8)
    db.insert_bridge(bridge)

    sgroup = SeriesGroup(name="Test Series", graph_ids=[g.graph_id], domain="test")
    db.insert_series_group(sgroup)

    db.insert_trace("trace_ui01", "Test query", '{"plan":{}}', '{"result":{}}')


class TestUIEndpoints:

    def test_papers_list(self, client):
        r = client.get("/papers")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_papers_detail(self, client):
        r = client.get("/papers")
        pid = r.json()["papers"][0]["paper_id"]
        r2 = client.get(f"/papers/{pid}")
        assert r2.status_code == 200
        assert "Test Paper" in r2.json()["title"]

    def test_claims_list(self, client):
        r = client.get("/claims")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_concepts_list(self, client):
        r = client.get("/concepts")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_inbox_list(self, client):
        r = client.get("/inbox")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_inbox_approve(self, client):
        r = client.get("/inbox")
        inbox_id = r.json()["items"][0]["inbox_id"]
        r2 = client.post(f"/inbox/{inbox_id}/approve")
        assert r2.status_code == 200
        assert r2.json()["ok"] is True

    def test_inbox_reject(self, client):
        r = client.get("/inbox")
        items = r.json()["items"]
        if not items:
            pytest.skip("No pending items")
        inbox_id = items[0]["inbox_id"]
        r2 = client.post(f"/inbox/{inbox_id}/reject", json={"reason": "test"})
        assert r2.status_code == 200

    def test_bridges_list(self, client):
        r = client.get("/bridges")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_traces_list(self, client):
        r = client.get("/traces")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_traces_detail(self, client):
        r = client.get("/traces")
        tid = r.json()["traces"][0]["trace_id"]
        r2 = client.get(f"/traces/{tid}")
        assert r2.status_code == 200
        assert r2.json()["query"] == "Test query"

    def test_series_groups(self, client):
        r = client.get("/series-groups")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_stats(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        assert r.json()["paper_cards"] >= 1

    def test_ui_served(self, client):
        r = client.get("/ui")
        assert r.status_code == 200
        assert "LitMesh 管理后台" in r.text

    def test_evidence_list(self, client):
        r = client.get("/evidence")
        assert r.status_code == 200

    def test_limitations_list(self, client):
        r = client.get("/limitations")
        assert r.status_code == 200
