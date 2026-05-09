"""
Full ingestion + extraction pipeline for v0.1-v0.2.

Orchestrates:
  v0.1: PDF -> auto graph -> PaperCard -> SectionBlocks -> SourceSpans
        -> SeriesDetector -> SeriesGroup (index layer)
  v0.2: SectionBlocks -> Claims -> Evidence -> Limitations -> Concepts -> Relations

Each PDF gets its own isolated SeriesGraph. SeriesDetector assigns to a SeriesGroup.
"""

from ..models.graph import SeriesGraph, GraphType, CrossGraphPolicy
from ..models.corpus import CorpusCard, CorpusType, IntegrationPolicy
from ..models.extraction_run import ExtractionRun, ExtractionTarget, ExtractionStatus
from ..models.review import ReviewInboxItem, InboxType, InboxPriority, InboxDecision
from ..models.relation import GraphRelation, GraphRelationType

from .pdf_parser import PDFExtractor
from .metadata_extractor import MetadataExtractor
from .section_splitter import split_sections

from ..extraction.claim_extractor import ClaimExtractor
from ..extraction.evidence_extractor import EvidenceExtractor
from ..extraction.limitation_extractor import LimitationExtractor
from ..extraction.concept_extractor import ConceptExtractor
from ..extraction.relation_linker import RelationLinker
from ..registry.concept_registry import ConceptRegistry
from ..registry.concept_normalizer import ConceptNormalizer
from ..registry.series_detector import SeriesDetector


class IngestionPipeline:
    """Runs the full v0.1 -> v0.2 pipeline for a single paper.

    Two modes:
    1. Explicit graph: pass graph_id (existing usage)
    2. Auto graph: pass corpus_id, each PDF gets its own SeriesGraph,
       then SeriesDetector decides series grouping.
    """

    def __init__(self, db, llm_client, graph_id: str = "", corpus_id: str = ""):
        self.db = db
        self.llm = llm_client
        self.graph_id = graph_id
        self.corpus_id = corpus_id

    def _ensure_graph(self, paper_card) -> str:
        """Auto-create a SeriesGraph if none provided."""
        if self.graph_id:
            return self.graph_id
        if not self.corpus_id:
            # Auto-create corpus too
            corpus = CorpusCard(
                name="Default Corpus",
                corpus_type=CorpusType.PAPER_COLLECTION,
                domain=paper_card.main_framework or "general",
            )
            self.db.insert_corpus(corpus)
            self.corpus_id = corpus.corpus_id

        graph = SeriesGraph(
            corpus_id=self.corpus_id,
            name=paper_card.title[:80],
            graph_type=GraphType.PAPER_COLLECTION,
            domain=paper_card.main_framework or "general",
            concept_namespace="",
            cross_graph_policy=CrossGraphPolicy.STRICT,
        )
        self.db.insert_graph(graph)
        self.graph_id = graph.graph_id
        return self.graph_id

    def _detect_series(self, paper_card) -> dict:
        """Run series detection after metadata extraction."""
        detector = SeriesDetector(self.db, self.llm)
        return detector.detect(paper_card, self.graph_id,
                                paper_card.main_framework or "general")

    def run_v0_1(self, pdf_path: str) -> dict:
        """v0.1: PDF -> PaperCard -> SectionBlocks -> SourceSpans."""
        # 1. Extract PDF text
        extractor = PDFExtractor(pdf_path)
        result = extractor.extract()

        # 2. Extract metadata -> PaperCard
        meta_extractor = MetadataExtractor(self.llm)
        paper_card = meta_extractor.extract(
            full_text=result["full_text"],
            source_file=pdf_path,
            graph_id="",  # Will be set after graph creation
        )

        # 3. Auto-create graph or use provided one
        self._ensure_graph(paper_card)
        paper_card.graph_id = self.graph_id

        paper_card.raw_text_hash = result["raw_text_hash"]
        paper_card.page_count = result["page_count"]
        self.db.insert_paper(paper_card)

        # 4. Split into sections
        sections = split_sections(
            full_text=result["full_text"],
            paper_id=paper_card.paper_id,
            graph_id=self.graph_id,
            pages=result["pages"],
        )
        for i, section in enumerate(sections):
            self.db.insert_section(section)

        # Link next_section_id after all sections exist
        for i in range(len(sections) - 1):
            self.db.conn.execute(
                "UPDATE section_blocks SET next_section_id = ? WHERE section_id = ?",
                (sections[i + 1].section_id, sections[i].section_id)
            )
            self.db.insert_relation(GraphRelation(
                graph_id=self.graph_id,
                source_id=sections[i].section_id,
                target_id=sections[i + 1].section_id,
                source_type="section",
                target_type="section",
                relation_type=GraphRelationType.NEXT,
                confidence=1.0,
                importance=0.4,
                traversal_cost=0.2,
                evidence_json='{"source":"section_splitter"}',
            ))

        for section in sections:
            if section.parent_section_id:
                self.db.insert_relation(GraphRelation(
                    graph_id=self.graph_id,
                    source_id=section.section_id,
                    target_id=section.parent_section_id,
                    source_type="section",
                    target_type="section",
                    relation_type=GraphRelationType.PARENT,
                    confidence=1.0,
                    importance=0.5,
                    traversal_cost=0.2,
                    evidence_json='{"source":"section_splitter"}',
                ))
        self.db.conn.commit()

        # 5. Detect series membership
        series_info = self._detect_series(paper_card)

        return {
            "paper_id": paper_card.paper_id,
            "graph_id": self.graph_id,
            "section_count": len(sections),
            "page_count": result["page_count"],
            "title": paper_card.title,
            "series": series_info,
        }

    def run_v0_2(self, paper_id: str) -> dict:
        """v0.2: SectionBlocks -> Claims/Evidence/Limitations/Concepts/Relations."""
        sections = self.db.get_sections_by_paper(paper_id)
        stats = {"claims": 0, "evidence": 0, "limitations": 0,
                  "concepts": 0, "relations": 0}

        claim_extractor = ClaimExtractor(self.llm)
        evidence_extractor = EvidenceExtractor(self.llm)
        limitation_extractor = LimitationExtractor(self.llm)
        concept_extractor = ConceptExtractor(self.llm)
        relation_linker = RelationLinker(self.llm)

        all_claims = []
        all_evidence = []
        all_limitations = []

        # Phase 1: Extract claims
        claim_run = ExtractionRun(
            paper_id=paper_id, graph_id=self.graph_id,
            target=ExtractionTarget.CLAIMS, model=self.llm.model,
        )
        self.db.create_extraction_run(claim_run)

        for s in sections:
            section = _section_from_dict(s)
            claims = claim_extractor.extract_from_section(section, extraction_run_id=claim_run.run_id)
            for claim in claims:
                self.db.insert_claim(claim)
                self._link_block_to_section(
                    block_id=claim.claim_id,
                    block_type="claim",
                    section_id=claim.section_id,
                    extraction_run_id=claim_run.run_id,
                )
                self._add_to_inbox(claim, paper_id, InboxType.EXTRACTION)
                all_claims.append(claim)
                stats["claims"] += 1

        self.db.complete_extraction_run(claim_run.run_id, items_produced=stats["claims"],
                                         items_accepted=0, items_rejected=0)

        # Phase 2: Evidence
        if all_claims:
            ev_run = ExtractionRun(paper_id=paper_id, graph_id=self.graph_id,
                                    target=ExtractionTarget.EVIDENCE, model=self.llm.model)
            self.db.create_extraction_run(ev_run)
            for s in sections:
                section = _section_from_dict(s)
                section_claims = [c for c in all_claims if c.section_id == s["section_id"]]
                for ev in evidence_extractor.extract_for_claims(section, section_claims, ev_run.run_id):
                    self.db.insert_evidence(ev)
                    self._link_block_to_section(
                        block_id=ev.evidence_id,
                        block_type="evidence",
                        section_id=ev.section_id,
                        extraction_run_id=ev_run.run_id,
                    )
                    self._add_to_inbox(ev, paper_id, InboxType.EXTRACTION)
                    all_evidence.append(ev)
                    stats["evidence"] += 1
            self.db.complete_extraction_run(ev_run.run_id, items_produced=stats["evidence"],
                                             items_accepted=0, items_rejected=0)
            for rel in relation_linker.link_evidence_to_claims(all_evidence, all_claims, self.graph_id, ev_run.run_id):
                self.db.insert_relation(rel)
                stats["relations"] += 1

        # Phase 3: Limitations
        if all_claims:
            lim_run = ExtractionRun(paper_id=paper_id, graph_id=self.graph_id,
                                     target=ExtractionTarget.LIMITATIONS, model=self.llm.model)
            self.db.create_extraction_run(lim_run)
            for s in sections:
                section = _section_from_dict(s)
                section_claims = [c for c in all_claims if c.section_id == s["section_id"]]
                for lim in limitation_extractor.extract_for_claims(section, section_claims, lim_run.run_id):
                    self.db.insert_limitation(lim)
                    self._link_block_to_section(
                        block_id=lim.limitation_id,
                        block_type="limitation",
                        section_id=lim.section_id,
                        extraction_run_id=lim_run.run_id,
                    )
                    self._add_to_inbox(lim, paper_id, InboxType.EXTRACTION)
                    all_limitations.append(lim)
                    stats["limitations"] += 1
            self.db.complete_extraction_run(lim_run.run_id, items_produced=stats["limitations"],
                                             items_accepted=0, items_rejected=0)
            for rel in relation_linker.link_limitations_to_claims(all_limitations, all_claims, self.graph_id, lim_run.run_id):
                self.db.insert_relation(rel)
                stats["relations"] += 1

        # Phase 4: Concepts + normalization
        if all_claims:
            conc_run = ExtractionRun(paper_id=paper_id, graph_id=self.graph_id,
                                      target=ExtractionTarget.CONCEPTS, model=self.llm.model)
            self.db.create_extraction_run(conc_run)
            concepts = concept_extractor.extract_from_claims(all_claims, self.graph_id, conc_run.run_id)
            for concept in concepts:
                self.db.insert_concept(concept)
                self._add_to_inbox(concept, paper_id, InboxType.CONCEPT)
                stats["concepts"] += 1

            normalizer = ConceptNormalizer(self.db, ConceptRegistry(self.db))
            norm_stats = normalizer.normalize_extraction(
                candidate_concepts=concepts, claims=all_claims,
                evidence_blocks=all_evidence, limitations=all_limitations,
                extraction_run_id=conc_run.run_id,
            )
            stats["concept_relations"] = norm_stats["relations_created"]
            self.db.complete_extraction_run(conc_run.run_id, items_produced=stats["concepts"],
                                             items_accepted=norm_stats.get("activated_concepts", 0),
                                             items_rejected=0)

        # Phase 5: Claim-to-claim relations
        if len(all_claims) >= 2:
            for rel in relation_linker.link_claims(all_claims, self.graph_id, claim_run.run_id):
                self.db.insert_relation(rel)
                stats["relations"] += 1

        return stats

    def _link_block_to_section(
        self,
        block_id: str,
        block_type: str,
        section_id: str | None,
        extraction_run_id: str | None = None,
    ):
        """Create the structural pointer back to the source section."""
        if not section_id:
            return
        self.db.insert_relation(GraphRelation(
            graph_id=self.graph_id,
            source_id=block_id,
            target_id=section_id,
            source_type=block_type,
            target_type="section",
            relation_type=GraphRelationType.BELONGS_TO,
            confidence=1.0,
            importance=0.6,
            traversal_cost=0.1,
            extraction_run_id=extraction_run_id,
            evidence_json='{"source":"ingestion_pipeline"}',
        ))

    def run_full(self, pdf_path: str) -> dict:
        """Run v0.1 + v0.2 for a single PDF."""
        v01 = self.run_v0_1(pdf_path)
        v02 = self.run_v0_2(v01["paper_id"])
        return {**v01, **v02}

    def _add_to_inbox(self, item, paper_id: str, inbox_type: InboxType):
        item_type = item.__class__.__name__.replace("Block", "").replace("Key", "").lower()
        title = (getattr(item, 'claim_text', '') or getattr(item, 'evidence_text', '') or
                 getattr(item, 'limitation_text', '') or getattr(item, 'label_zh', '') or "")[:100]
        item_id = (getattr(item, 'claim_id', '') or getattr(item, 'evidence_id', '') or
                   getattr(item, 'limitation_id', '') or getattr(item, 'concept_key', ''))

        inbox = ReviewInboxItem(
            inbox_type=inbox_type, item_id=item_id, item_type=item_type,
            title=title, extraction_confidence=getattr(item, 'extraction_confidence', 0.5),
            priority=InboxPriority.MEDIUM,
            extraction_run_id=getattr(item, 'extraction_run_id', None),
            graph_id=self.graph_id, paper_id=paper_id,
            suggested_actions=[InboxDecision.APPROVE, InboxDecision.EDIT,
                                InboxDecision.REJECT, InboxDecision.DOWNGRADE_CONFIDENCE],
        )
        self.db.insert_inbox_item(inbox)


def _section_from_dict(d: dict):
    from ..models.section import SectionBlock, HeadingLevel
    import json
    try:
        heading_path = json.loads(d.get("heading_path", "[]"))
    except (json.JSONDecodeError, TypeError):
        heading_path = []
    return SectionBlock(
        section_id=d["section_id"], graph_id=d["graph_id"], paper_id=d["paper_id"],
        heading=d.get("heading", ""), heading_path=heading_path,
        heading_level=HeadingLevel(d.get("heading_level", "section")),
        raw_text=d.get("raw_text", ""), summary=d.get("summary", ""),
        page_start=d.get("page_start"), page_end=d.get("page_end"),
    )
