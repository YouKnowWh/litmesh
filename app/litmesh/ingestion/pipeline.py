"""
Full ingestion + extraction pipeline for v0.1-v0.2.

Orchestrates:
  v0.1: PDF -> auto graph -> PaperCard -> SectionBlocks -> SourceSpans
        -> SeriesDetector -> SeriesGroup (index layer)
  v0.2: SectionBlocks -> Claims -> Evidence -> Limitations -> Concepts -> Relations

Each PDF gets its own isolated SeriesGraph. SeriesDetector assigns to a SeriesGroup.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.graph import SeriesGraph, GraphType, CrossGraphPolicy

logger = logging.getLogger("litmesh.pipeline")
from ..models.corpus import CorpusCard, CorpusType, IntegrationPolicy
from ..models.extraction_run import ExtractionRun, ExtractionTarget, ExtractionStatus
from ..models.review import ReviewInboxItem, InboxType, InboxPriority, InboxDecision
from ..models.relation import GraphRelation, GraphRelationType

import os

from .metadata_extractor import MetadataExtractor
from .parsers import parse_document
from .section_splitter import split_parsed_document, split_sections

from ..repair.candidate_detector import CandidateDetector
from ..repair.reranker_client import RerankerClient
from ..repair.repair_policy import RepairPolicy
from ..repair.repair_log import RepairLog
from ..repair.fallback_llm import FallbackLLM
from ..repair.repair_executor import RepairExecutor
from ..repair.page_number_stripper import PageNumberStripper
from ..structure.group_builder import GroupBuilder

from ..extraction.combined_extractor import CombinedExtractor
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

    def __init__(
        self,
        db,
        llm_client,
        graph_id: str = "",
        corpus_id: str = "",
        parser_name: str = "auto",
        segmenter_name: str = "auto",
        segment_llm_client=None,
        progress_tracker=None,
        task_id: str = "",
        repair_mode: str = "",
    ):
        self.db = db
        self.llm = llm_client
        self.graph_id = graph_id
        self.corpus_id = corpus_id
        self.parser_name = parser_name
        self.segmenter_name = segmenter_name
        self.segment_llm_client = segment_llm_client or llm_client
        self._tracker = progress_tracker
        self._task_id = task_id
        self._repair_mode = repair_mode or os.environ.get("LITMESH_REPAIR_MODE", "dry_run")
        self._repair_executor = None
        self._page_stripper_enabled = os.environ.get("LITMESH_PAGE_NUMBER_STRIPPER", "llm") != "off"
        self._page_stripper = None

    def _emit(self, stage: str, message: str, percentage: float, **metadata):
        if self._tracker and self._task_id:
            self._tracker.emit(self._task_id, stage, message, percentage, **metadata)

    def _get_repair_executor(self) -> RepairExecutor:
        """Lazy-init the repair executor with optional LLM fallback."""
        if self._repair_executor is not None:
            return self._repair_executor
        log_dir = os.environ.get("LITMESH_REPAIR_LOG_DIR", "logs/repair_audit/")
        log = RepairLog(log_dir=log_dir)
        policy = RepairPolicy.from_env()
        fallback_llm = None
        if self._repair_mode == "full":
            try:
                from ..extraction.llm_config import load_endpoint
                endpoint = load_endpoint("REPAIR")
                repair_llm = endpoint.create_client()
                fallback_llm = FallbackLLM(repair_llm)
            except Exception as e:
                logger.warning("Failed to create repair LLM client: %s", e)
        self._repair_executor = RepairExecutor(
            policy=policy,
            fallback_llm=fallback_llm,
            log=log,
        )
        return self._repair_executor

    def _get_page_number_stripper(self) -> PageNumberStripper:
        """Lazy-init the LLM-based page number stripper."""
        if self._page_stripper is not None:
            return self._page_stripper
        mode = os.environ.get("LITMESH_PAGE_NUMBER_STRIPPER", "llm")
        llm = None
        if mode == "llm":
            try:
                from ..extraction.llm_config import load_endpoint
                endpoint = load_endpoint("REPAIR")
                llm = endpoint.create_client()
            except Exception as e:
                logger.warning("Failed to create page number LLM client: %s — "
                               "falling back to regex", e)
        self._page_stripper = PageNumberStripper(llm_client=llm)
        return self._page_stripper

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

    def run_v0_1(self, pdf_path: str, existing_paper_id: str = "") -> dict:
        """v0.1: PDF -> PaperCard -> SectionBlocks -> SourceSpans."""
        is_existing = bool(existing_paper_id)
        t_start = time.monotonic()

        # 1. Parse PDF into a unified document structure.
        logger.info("v0.1 start: pdf=%s existing_paper=%s", pdf_path, is_existing)
        self._emit("parsing", "Parsing PDF...", 5.0)
        last_reported = [0]  # mutable closure
        def _parse_progress(current, total):
            if current < 0:
                # Heartbeat: remote API waiting, total = elapsed seconds
                self._emit("parsing", f"PDF sent to remote server, waiting... ({total}s)", 8.0,
                           elapsed_s=total)
                return
            if total <= 100:
                # Window-level progress from LLM segmenter
                pct = 8.0 + (current / max(total, 1)) * 5.0
                if current - last_reported[0] >= max(1, total // 10) or current == total:
                    self._emit("parsing", f"Segmenting text: window {current}/{total}", pct,
                               window=current, total_windows=total)
                    last_reported[0] = current
                return
            # Page-level progress from pdfplumber/PyMuPDF
            pct = 5.0 + (current / max(total, 1)) * 10.0
            if current - last_reported[0] >= max(1, total // 20) or current == total:
                self._emit("parsing", f"Parsing PDF: page {current}/{total}", pct,
                           current=current, total=total)
                last_reported[0] = current
        parsed = parse_document(
            pdf_path,
            self.parser_name,
            segmenter_name=self.segmenter_name,
            segment_llm_client=self.segment_llm_client,
            progress_callback=_parse_progress,
            paper_id=existing_paper_id if is_existing else "",
            audit_dir="logs/parse_audit",
        )
        t_parsed = time.monotonic()
        logger.info("v0.1 parsing done: pages=%d parser=%s time=%.1fs",
                    len(parsed.pages), parsed.parser_name, t_parsed - t_start)
        self._emit("parsing", f"PDF parsed: {len(parsed.pages)} pages, using {parsed.parser_name}", 15.0,
                   parser=parsed.parser_name, pages=len(parsed.pages))

        # 2. Extract metadata -> PaperCard
        self._emit("metadata", "Extracting metadata via LLM...", 15.0)
        meta_extractor = MetadataExtractor(self.llm)
        paper_card = meta_extractor.extract(
            full_text=parsed.full_text,
            source_file=pdf_path,
            graph_id=self.graph_id if is_existing else "",
        )
        t_meta = time.monotonic()
        logger.info("v0.1 metadata done: title=%s time=%.1fs",
                    paper_card.title[:60], t_meta - t_parsed)
        self._emit("metadata", f"Metadata extracted: {paper_card.title[:60]}", 22.0,
                   title=paper_card.title)

        # Override paper_id with the pre-created one so it survives page refresh
        if is_existing:
            paper_card.paper_id = existing_paper_id

        # 3. Auto-create graph or use provided one
        if not self.graph_id:
            self._emit("graph", "Creating graph...", 25.0)
            self._ensure_graph(paper_card)
        paper_card.graph_id = self.graph_id

        import hashlib

        paper_card.raw_text_hash = hashlib.sha256(parsed.full_text.encode("utf-8")).hexdigest()
        paper_card.page_count = len(parsed.pages)

        if is_existing:
            self.db.update_paper(paper_card)
        else:
            self.db.insert_paper(paper_card)

        self._emit("graph", f"Paper ready: {paper_card.paper_id}", 30.0,
                   paper_id=paper_card.paper_id)

        # 4. Split into sections
        self._emit("sections", "Splitting into sections...", 35.0)
        sections = split_parsed_document(
            parsed=parsed,
            paper_id=paper_card.paper_id,
            graph_id=self.graph_id,
        )
        if not sections:
            sections = split_sections(
                full_text=parsed.full_text,
                paper_id=paper_card.paper_id,
                graph_id=self.graph_id,
                pages=parsed.pages,
                min_section_chars=40,
            )

        # Persist outline nodes (TOC tree) from parsed document
        if parsed.outline:
            try:
                from ..ingestion.section_splitter import _outline_items_to_nodes
                outline_nodes = _outline_items_to_nodes(
                    parsed.outline, paper_card.paper_id, self.graph_id,
                )
                if outline_nodes:
                    self.db.delete_outline_nodes(paper_card.paper_id)
                    count = self.db.insert_outline_nodes(outline_nodes)
                    logger.info("Outline nodes persisted: %d entries", count)
            except Exception as e:
                logger.warning("Failed to persist outline nodes: %s", e)

        # Page number stripping (LLM-based, before structural repair)
        if self._page_stripper_enabled:
            stripper = self._get_page_number_stripper()
            sections, pn_report = stripper.process(
                sections, paper_id=paper_card.paper_id,
            )
            logger.info("Page number stripping: %s", pn_report)

        # Repair layer: structural diagnosis and repair (v0.3)
        if self._repair_mode != "off":
            repair_executor = self._get_repair_executor()
            sections, repair_report = repair_executor.repair(
                sections,
                paper_id=paper_card.paper_id,
                graph_id=self.graph_id,
                mode=self._repair_mode,
            )
            logger.info("Repair report: %s", {
                k: v for k, v in repair_report.items() if k != "elapsed_ms"
            })

        # Structure layer: build StructureGroup hierarchy (v1)
        try:
            builder = GroupBuilder()
            outline = parsed.outline if hasattr(parsed, 'outline') else None
            groups = builder.build(
                sections, paper_id=paper_card.paper_id,
                graph_id=self.graph_id, outline_nodes=outline,
            )
            if groups:
                self.db.delete_structure_groups(paper_card.paper_id)
                count = self.db.insert_structure_groups(groups)
                logger.info("Structure groups persisted: %d groups", count)
        except Exception as e:
            logger.warning("Failed to build structure groups: %s", e)

        if parsed.quality_report:
            self.db.insert_parse_quality_report(
                paper_card.paper_id, self.graph_id, parsed.quality_report
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
        t_sections = time.monotonic()
        logger.info("v0.1 sections done: count=%d time=%.1fs",
                    len(sections), t_sections - t_meta)
        self._emit("sections", f"{len(sections)} sections created", 42.0,
                   section_count=len(sections))

        # 5. Detect series membership
        self._emit("series", "Detecting series membership...", 45.0)
        series_info = self._detect_series(paper_card)
        t_series = time.monotonic()
        logger.info("v0.1 series done: action=%s time=%.1fs",
                    series_info.get("action", "?"), t_series - t_sections)
        self._emit("series", f"Series detection: {series_info.get('action', '?')}", 48.0)

        logger.info("v0.1 complete: total_time=%.1fs paper=%s sections=%d pages=%d",
                    t_series - t_start, paper_card.paper_id, len(sections), len(parsed.pages))
        return {
            "paper_id": paper_card.paper_id,
            "graph_id": self.graph_id,
            "section_count": len(sections),
            "page_count": len(parsed.pages),
            "title": paper_card.title,
            "series": series_info,
            "parser_used": parsed.parser_name,
            "quality_report": (
                parsed.quality_report.__dict__ if parsed.quality_report else {}
            ),
            "needs_structure_review": (
                bool(parsed.quality_report.needs_structure_review)
                if parsed.quality_report else False
            ),
        }

    def run_v0_2(self, paper_id: str) -> dict:
        """v0.2: SectionBlocks -> Claims/Evidence/Limitations/Concepts/Relations."""
        t0 = time.monotonic()
        sections = self.db.get_sections_by_paper(paper_id)
        stats = {"claims": 0, "evidence": 0, "limitations": 0,
                  "concepts": 0, "relations": 0}

        combined_extractor = CombinedExtractor(self.llm)
        concept_extractor = ConceptExtractor(self.llm)
        relation_linker = RelationLinker(self.llm)

        all_claims = []
        all_evidence = []
        all_limitations = []

        # Count qualifying sections
        qual_section_list = []
        for s in sections:
            sec = _section_from_dict(s)
            if len(sec.raw_text) >= 150 and not _is_reserved_structure_section(sec):
                qual_section_list.append(sec)
        n_qual = len(qual_section_list) or 1

        # Extraction run for combined claims+evidence+limitations
        ext_run = ExtractionRun(
            paper_id=paper_id, graph_id=self.graph_id,
            target=ExtractionTarget.CLAIMS, model=self.llm.model,
        )
        self.db.create_extraction_run(ext_run)

        # Phase 1: Combined claims+evidence+limitations (parallel, 3 workers)
        completed = 0
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(combined_extractor.extract_from_section, sec, ext_run.run_id): (i, sec)
                for i, sec in enumerate(qual_section_list)
            }
            self._emit("claims",
                       f"Extracting {n_qual} sections (parallel, combined claims+evidence+limitations)...",
                       52.0, total=n_qual)

            for future in as_completed(futures):
                i, section = futures[future]
                result = future.result()
                for claim in result["claims"]:
                    self.db.insert_claim(claim)
                    self._link_block_to_section(block_id=claim.claim_id, block_type="claim",
                                                section_id=claim.section_id, extraction_run_id=ext_run.run_id)
                    self._add_to_inbox(claim, paper_id, InboxType.EXTRACTION)
                    all_claims.append(claim)
                    stats["claims"] += 1
                for ev in result["evidence"]:
                    self.db.insert_evidence(ev)
                    self._link_block_to_section(block_id=ev.evidence_id, block_type="evidence",
                                                section_id=ev.section_id, extraction_run_id=ext_run.run_id)
                    self._add_to_inbox(ev, paper_id, InboxType.EXTRACTION)
                    all_evidence.append(ev)
                    stats["evidence"] += 1
                for lim in result["limitations"]:
                    self.db.insert_limitation(lim)
                    self._link_block_to_section(block_id=lim.limitation_id, block_type="limitation",
                                                section_id=lim.section_id, extraction_run_id=ext_run.run_id)
                    self._add_to_inbox(lim, paper_id, InboxType.EXTRACTION)
                    all_limitations.append(lim)
                    stats["limitations"] += 1
                completed += 1
                self._emit("claims",
                           f"Section {completed}/{n_qual}: {stats['claims']}C/{stats['evidence']}E/{stats['limitations']}L",
                           52.0 + completed/n_qual * 36.0,
                           current=completed, total=n_qual, claims=stats["claims"],
                           evidence=stats["evidence"], limitations=stats["limitations"])

        t_extract = time.monotonic()
        logger.info("v0.2 extraction done: claims=%d evidence=%d limitations=%d "
                    "sections=%d time=%.1fs",
                    stats["claims"], stats["evidence"], stats["limitations"],
                    completed, t_extract - t0)

        self.db.complete_extraction_run(ext_run.run_id,
                                         items_produced=stats["claims"] + stats["evidence"] + stats["limitations"],
                                         items_accepted=0, items_rejected=0)

        # Link evidence/limitations to claims
        if all_evidence and all_claims:
            for rel in relation_linker.link_evidence_to_claims(all_evidence, all_claims, self.graph_id, ext_run.run_id):
                self.db.insert_relation(rel)
                stats["relations"] += 1
        if all_limitations and all_claims:
            for rel in relation_linker.link_limitations_to_claims(all_limitations, all_claims, self.graph_id, ext_run.run_id):
                self.db.insert_relation(rel)
                stats["relations"] += 1

        self._emit("concepts", f"Extraction done ({stats['claims']}C/{stats['evidence']}E/{stats['limitations']}L), extracting concepts...", 88.0)

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

            self._emit("concepts", f"Normalizing {len(concepts)} concepts...", 88.0)
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
        self._emit("relations", f"Linking {len(all_claims)} claims...", 92.0)
        if len(all_claims) >= 2:
            for rel in relation_linker.link_claims(all_claims, self.graph_id, ext_run.run_id):
                self.db.insert_relation(rel)
                stats["relations"] += 1

        self._emit("v02_done",
                   f"Extraction complete: {stats['claims']}C/{stats['evidence']}E/{stats['limitations']}L/{stats['concepts']}K",
                   96.0, **stats)
        logger.info("v0.2 complete: total_time=%.1fs claims=%d evidence=%d limitations=%d concepts=%d relations=%d",
                    time.monotonic() - t0, stats["claims"], stats["evidence"],
                    stats["limitations"], stats["concepts"], stats["relations"])
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
        """Auto-approve high-confidence items, only flag low-confidence for review."""
        conf = getattr(item, 'extraction_confidence', 0.5)
        # Auto-approve: confidence >= 0.7
        if conf >= 0.7:
            return  # Skip inbox entirely — item is auto-approved
        # Auto-approve with downgrade: 0.5 <= conf < 0.7
        if conf >= 0.5:
            return  # Also skip — medium confidence still auto-approved

        # Only low-confidence items (<0.5) go to inbox for review
        item_type = item.__class__.__name__.replace("Block", "").replace("Key", "").lower()
        title = (getattr(item, 'claim_text', '') or getattr(item, 'evidence_text', '') or
                 getattr(item, 'limitation_text', '') or getattr(item, 'label_zh', '') or "")[:100]
        item_id = (getattr(item, 'claim_id', '') or getattr(item, 'evidence_id', '') or
                   getattr(item, 'limitation_id', '') or getattr(item, 'concept_key', ''))

        inbox = ReviewInboxItem(
            inbox_type=inbox_type, item_id=item_id, item_type=item_type,
            title=title, extraction_confidence=conf,
            priority=InboxPriority.LOW,
            extraction_run_id=getattr(item, 'extraction_run_id', None),
            graph_id=self.graph_id, paper_id=paper_id,
            suggested_actions=[InboxDecision.APPROVE, InboxDecision.REJECT],
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


def _is_reserved_structure_section(section) -> bool:
    import re
    if not section.heading_path:
        return False
    label = re.sub(r"\s+", "", section.heading_path[0] or "")
    return label in {"目录", "前言", "编写说明"}
