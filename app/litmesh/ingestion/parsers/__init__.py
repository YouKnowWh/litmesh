"""
Document parser adapters. Each adapter converts a specific parser's output
into the unified ParsedDocument format.

Supported:
- mineru_api: remote PDF-to-Markdown API (preferred outsourced parser)
- external_markdown: configured command/sidecar Markdown parser
- markdown: existing Markdown file or sidecar
- pdfplumber: pdfplumber text extraction (default source of truth)
- pymupdf_blocks: PyMuPDF block-level extraction (explicit diagnostics only)
- docling: IBM Docling (install: pip install docling)
- mineru: MinerU (requires separate installation)
"""

import os

from ..parsed_document import ParsedDocument, ParsedElement, ElementType, OutlineItem, QualityReport
from .pymupdf_adapter import PyMuPDFBlockAdapter
from .pdfplumber_adapter import PdfPlumberAdapter
from .markdown_adapter import (
    ExternalMarkdownAdapter,
    MarkdownAdapter,
    MarkerMarkdownAdapter,
    MinerUMarkdownAdapter,
    RemoteMarkdownAdapter,
)

# Auto parser selection outsources PDF structure first. Remote/sidecar Markdown
# parsers are skipped when unavailable; pdfplumber remains the local fallback.
PARSER_PRIORITY = ["mineru_api", "external_markdown", "pdfplumber"]


def create_parser(name: str = "auto", segmenter_name: str = "auto", segment_llm_client=None,
                  paper_id: str = "", audit_dir: str = ""):
    """Factory: return the appropriate parser adapter."""
    if name == "auto":
        for p in PARSER_PRIORITY:
            try:
                return _instantiate(p, segmenter_name, segment_llm_client, paper_id, audit_dir)
            except ImportError:
                continue
        raise RuntimeError("No document parser available")

    if name == "docling":
        from .docling_adapter import DoclingAdapter
        return DoclingAdapter()
    if name in ("mineru_api", "remote_markdown"):
        return RemoteMarkdownAdapter("mineru_api")
    if name in ("external_markdown", "outsourced_markdown"):
        return ExternalMarkdownAdapter("external_markdown")
    if name in ("markdown", "md"):
        return MarkdownAdapter("markdown")
    if name == "mineru_markdown":
        return MinerUMarkdownAdapter()
    if name == "marker_markdown":
        return MarkerMarkdownAdapter()
    if name in ("pymupdf_blocks", "pymupdf"):
        return PyMuPDFBlockAdapter()
    if name in ("pdfplumber", "fallback"):
        return PdfPlumberAdapter(segmenter_name=segmenter_name, segment_llm_client=segment_llm_client,
                                 paper_id=paper_id, audit_dir=audit_dir)

    raise ValueError(
        "Unknown parser: "
        f"{name}. Supported: auto, mineru_api, external_markdown, markdown, "
        "mineru_markdown, marker_markdown, pymupdf_blocks, pdfplumber"
    )


def parse_document(
    pdf_path: str,
    name: str = "auto",
    segmenter_name: str = "auto",
    segment_llm_client=None,
    progress_callback=None,
    paper_id: str = "",
    audit_dir: str = "",
) -> ParsedDocument:
    """Parse with quality-aware fallback.

    In auto mode, LitMesh first tries outsourced Markdown structure recovery:
    remote MinerU API, configured external command/sidecar Markdown, then local
    pdfplumber fallback. PyMuPDF is never attempted automatically; use
    parser=pymupdf_blocks explicitly for diagnostics.

    progress_callback(page_num, total_pages) is called during page extraction.
    """
    if name != "auto":
        parser = create_parser(name, segmenter_name, segment_llm_client,
                               paper_id=paper_id, audit_dir=audit_dir)
        parser.progress_callback = progress_callback
        return parser.parse(pdf_path)

    last_doc = None
    fallback_notes = []
    for parser_name in PARSER_PRIORITY:
        try:
            import logging
            logging.getLogger("litmesh.parser").info("parser_attempt parser=%s mode=auto", parser_name)
            parser = create_parser(parser_name, segmenter_name, segment_llm_client,
                                   paper_id=paper_id, audit_dir=audit_dir)
            parser.progress_callback = progress_callback
            doc = parser.parse(pdf_path)
        except ImportError as exc:
            note = f"{parser_name}: unavailable ({exc})"
            fallback_notes.append(note)
            logging.getLogger("litmesh.parser").info("parser_fallback from=%s reason=%s", parser_name, note)
            continue
        except Exception as exc:
            note = f"{parser_name}: failed ({type(exc).__name__}: {exc})"
            fallback_notes.append(note)
            logging.getLogger("litmesh.parser").info("parser_fallback from=%s reason=%s", parser_name, note)
            continue

        if doc.quality_report:
            doc.quality_report.warnings.extend(list(fallback_notes))
        if not (doc.quality_report and doc.quality_report.needs_structure_review):
            return doc

        if parser_name == "pymupdf_blocks" and _pymupdf_should_be_disabled(doc):
            note = f"{parser_name}: disabled by strict quality gate"
            fallback_notes.append(note)
            logging.getLogger("litmesh.parser").info(
                "parser_disabled parser=%s reason=%s warnings=%s",
                parser_name, note, doc.quality_report.warnings if doc.quality_report else [],
            )
            if last_doc is not None:
                continue

        # Keep the first completed source-of-truth result. PyMuPDF is excluded
        # from auto priority, but this guard remains defensive for future edits.
        if last_doc is None or parser_name != "pymupdf_blocks":
            last_doc = doc
        note = f"{parser_name}: low quality"
        fallback_notes.append(note)
        logging.getLogger("litmesh.parser").info("parser_fallback from=%s reason=%s", parser_name, note)

    if last_doc is not None:
        if last_doc.quality_report:
            existing = set(last_doc.quality_report.warnings)
            for note in fallback_notes:
                if note not in existing:
                    last_doc.quality_report.warnings.append(note)
        return last_doc
    raise RuntimeError("No document parser available")


def _pymupdf_should_be_disabled(doc: ParsedDocument) -> bool:
    """Auto-mode guardrail for sparse PyMuPDF block extraction."""
    quality = doc.quality_report
    if not quality or quality.parser_name != "pymupdf_blocks":
        return False
    warnings = set(quality.warnings or [])
    hard_reasons = {
        "pymupdf_too_sparse_vs_toc",
        "pymupdf_too_sparse_vs_pages",
        "pymupdf_too_little_text",
    }
    return bool(warnings & hard_reasons)


def _instantiate(name: str, segmenter_name: str = "auto", segment_llm_client=None,
                  paper_id: str = "", audit_dir: str = ""):
    if name == "docling":
        from .docling_adapter import DoclingAdapter
        return DoclingAdapter()
    if name == "mineru_api":
        if not os.getenv("LITMESH_MINERU_API_URL", "").strip():
            raise ImportError("LITMESH_MINERU_API_URL is not configured")
        return RemoteMarkdownAdapter("mineru_api")
    if name == "external_markdown":
        return ExternalMarkdownAdapter("external_markdown")
    if name == "markdown":
        return MarkdownAdapter("markdown")
    if name == "mineru_markdown":
        return MinerUMarkdownAdapter()
    if name == "marker_markdown":
        return MarkerMarkdownAdapter()
    if name == "pymupdf_blocks":
        return PyMuPDFBlockAdapter()
    if name == "pdfplumber":
        return PdfPlumberAdapter(segmenter_name=segmenter_name, segment_llm_client=segment_llm_client,
                                 paper_id=paper_id, audit_dir=audit_dir)
    raise ImportError(name)
