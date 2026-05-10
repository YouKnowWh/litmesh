"""
pdfplumber parser adapter with secondary text segmentation.

Extracts page-level text with pdfplumber, then runs PageTextSegmenter
to produce cleaned, paragraph-level ParsedElements.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from ..parsed_document import ParsedDocument
from ..page_segmenter import PageTextSegmenter, SegmenterConfig
from ..llm_page_segmenter import LLMPageSegmenter
from ..toc_extractor import TOCExtractor

logger = logging.getLogger("litmesh.parser")


class PdfPlumberAdapter:
    """Parser using pdfplumber for text extraction + segmenter for paragraph rebuild."""

    def __init__(self, segmenter_name: str = "auto", segment_llm_client=None,
                 paper_id: str = "", audit_dir: str = ""):
        self.name = "pdfplumber"
        self.progress_callback = None
        import pdfplumber
        self._version = pdfplumber.__version__
        self.segmenter_name = segmenter_name
        self.segment_llm_client = segment_llm_client
        self.paper_id = paper_id
        self.audit_dir = audit_dir

    def parse(self, pdf_path: str) -> ParsedDocument:
        import pdfplumber
        from ..pdf_parser import _clean_cjk_spaces

        t0 = time.monotonic()
        pdf_path = Path(pdf_path)
        pages = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                if self.progress_callback and i % 5 == 0:
                    self.progress_callback(i, total)
                text = page.extract_text(
                    x_tolerance=2, y_tolerance=2, keep_blank_chars=False,
                ) or ""
                text = _clean_cjk_spaces(text)
                pages.append({"page_num": i, "text": text})
        t_extract = time.monotonic()
        toc_extractor = TOCExtractor(paper_id=self.paper_id, audit_dir=self.audit_dir)
        outline, toc_meta = toc_extractor.extract(pages)

        # Secondary segmentation
        segmenter = self._make_segmenter()
        elements, quality, body_start = segmenter.segment(pages)
        t_total = time.monotonic() - t0
        logger.info("pdfplumber parse done: file=%s pages=%d elements=%d "
                    "segmenter=%s extract=%.1fs total=%.1fs",
                    pdf_path.name, total, len(elements),
                    type(segmenter).__name__, t_extract - t0, t_total)

        # Update quality with body info
        quality.body_start_page = body_start
        quality.parser_name = self.name
        if not quality.segmenter_name:
            quality.segmenter_name = "rule_page_segmenter"
        quality.toc_detected = bool(outline) or quality.toc_detected
        quality.toc_entry_count = int(toc_meta.get("toc_entry_count", 0))
        quality.toc_page_count = int(toc_meta.get("toc_page_count", 0))
        quality.toc_source = str(toc_meta.get("toc_source", ""))
        quality.toc_alignment_confidence = float(toc_meta.get("toc_alignment_confidence", 0.0))
        quality.toc_printed_page_offset = int(toc_meta.get("toc_printed_page_offset", 0))
        quality.toc_unaligned_entries = int(toc_meta.get("toc_unaligned_entries", 0))
        quality.outline = [asdict(item) for item in outline]
        if outline:
            quality.warnings = [w for w in quality.warnings if w != "body_start_unknown"]
            if quality.toc_alignment_confidence < 0.5:
                quality.quality_gate_reasons.append("toc_alignment_low")
                quality.warnings.append("toc_alignment_low")
        else:
            quality.quality_gate_reasons.append("toc_missing")
        quality.needs_structure_review = quality.needs_structure_review or (
            bool(outline) and quality.toc_alignment_confidence < 0.5
        )
        logger.info(
            "quality_gate_done needs_structure_review=%s reasons=%s",
            quality.needs_structure_review,
            quality.quality_gate_reasons or quality.warnings,
        )

        full_text = "\n\n".join(e.text for e in elements)

        return ParsedDocument(
            pages=pages, elements=elements, outline=outline,
            metadata={"filename": pdf_path.name},
            parser_name=self.name, parser_version=self._version,
            quality_report=quality, full_text=full_text,
        )

    def _make_segmenter(self):
        if self.segmenter_name == "rule":
            return PageTextSegmenter(SegmenterConfig())
        if self.segmenter_name in ("llm", "hybrid"):
            return LLMPageSegmenter(self.segment_llm_client, paper_id=self.paper_id,
                                    audit_dir=self.audit_dir)
        if self.segmenter_name == "auto":
            has_key = bool(getattr(self.segment_llm_client, "api_key", ""))
            if has_key:
                return LLMPageSegmenter(self.segment_llm_client, paper_id=self.paper_id,
                                        audit_dir=self.audit_dir)
            return PageTextSegmenter(SegmenterConfig())
        raise ValueError("Unknown segmenter. Supported: auto, rule, llm")
