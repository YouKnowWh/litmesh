"""
PyMuPDF block-level parser. Extracts text blocks from PDF layout,
merges adjacent continuations, and returns ParsedElements.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from ..parsed_document import ParsedDocument, ParsedElement, ElementType, QualityReport
from ..toc_extractor import TOCExtractor

logger = logging.getLogger("litmesh.parser")


class PyMuPDFBlockAdapter:
    def __init__(self):
        self.name = "pymupdf_blocks"
        self.progress_callback = None
        import fitz
        self._version = fitz.version[0]

    def parse(self, pdf_path: str) -> ParsedDocument:
        import fitz
        from ..pdf_parser import _clean_cjk_spaces

        t0 = time.monotonic()
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pages = []
        raw_blocks = []
        global_order = 0

        for i, page in enumerate(doc, start=1):
            if self.progress_callback and i % 3 == 0:
                self.progress_callback(i, total_pages)
            text = page.get_text("text") or ""
            text = _clean_cjk_spaces(text)
            pages.append({"page_num": i, "text": text})

            blocks = page.get_text("blocks")
            for b in blocks:
                x0, y0, x1, y1, text, block_type, _ = b
                if block_type != 0:
                    continue
                text = _clean_cjk_spaces(text.strip())
                if not text or len(text) < 15:
                    continue
                global_order += 1
                raw_blocks.append({
                    "text": text, "page": i, "order": global_order,
                    "y": y0,
                })

        doc.close()
        t_extract = time.monotonic()

        # Merge adjacent blocks that are continuations
        merged = []
        for b in raw_blocks:
            if merged and _should_merge(merged[-1]["text"], b["text"]):
                merged[-1]["text"] = merged[-1]["text"] + " " + b["text"]
                merged[-1]["order"] = b["order"]
            else:
                merged.append(b)

        # Build elements, filtering noise
        elements = []
        order = 0
        for b in merged:
            text = b["text"]
            if _is_noise(text):
                continue
            order += 1
            elements.append(ParsedElement(
                element_id=f"elem_{order}",
                type=ElementType.PARAGRAPH,
                text=text, page_start=b["page"], page_end=b["page"],
                order_index=order,
            ))

        full_text = "\n\n".join(e.text for e in elements)
        t_total = time.monotonic() - t0
        logger.info("pymupdf parse done: file=%s pages=%d blocks=%d elements=%d "
                    "extract=%.1fs total=%.1fs",
                    Path(pdf_path).name, total_pages, len(raw_blocks),
                    len(elements), t_extract - t0, t_total)

        outline, toc_meta = TOCExtractor().extract(pages)
        page_count = len(pages)
        paragraph_count = len(elements)
        toc_entry_count = int(toc_meta.get("toc_entry_count", 0))
        warnings = []
        if paragraph_count < 10:
            warnings.append("PyMuPDF produced fewer than 10 text elements")
        if page_count and paragraph_count < page_count:
            warnings.append("PyMuPDF element count is smaller than page count")
        if page_count and paragraph_count < max(20, page_count * 1.5):
            warnings.append("pymupdf_too_sparse_vs_pages")
        if len(full_text) < max(2000, page_count * 200):
            warnings.append("pymupdf_too_little_text")
        if toc_entry_count and paragraph_count < max(toc_entry_count * 2, toc_entry_count + 10):
            warnings.append("pymupdf_too_sparse_vs_toc")
        if toc_entry_count:
            logger.info(
                "pymupdf_quality_gate toc_entries=%d elements=%d ratio=%.2f warnings=%s",
                toc_entry_count,
                paragraph_count,
                paragraph_count / max(toc_entry_count, 1),
                warnings,
            )

        quality = QualityReport(
            parser_name=self.name, parser_version=self._version,
            segmenter_name="pymupdf_blocks",
            total_elements=paragraph_count,
            paragraph_count=paragraph_count,
            toc_detected=bool(outline),
            toc_entry_count=toc_entry_count,
            toc_page_count=int(toc_meta.get("toc_page_count", 0)),
            toc_source=str(toc_meta.get("toc_source", "")),
            toc_alignment_confidence=float(toc_meta.get("toc_alignment_confidence", 0.0)),
            toc_printed_page_offset=int(toc_meta.get("toc_printed_page_offset", 0)),
            toc_unaligned_entries=int(toc_meta.get("toc_unaligned_entries", 0)),
            outline=[asdict(item) for item in outline],
            average_paragraph_length=round(sum(len(e.text) for e in elements) / max(len(elements), 1), 1),
            body_page_count=page_count,
            needs_structure_review=bool(warnings),
            warnings=warnings,
            quality_gate_reasons=list(warnings),
        )

        return ParsedDocument(
            pages=pages, elements=elements, outline=outline,
            parser_name=self.name, parser_version=self._version,
            quality_report=quality, full_text=full_text,
        )


_SENTENCE_END = set('。！？!?」】）)')
_NOISE_RE = __import__('re').compile(r'^\.{3,}$|^\d{1,3}$|^[・·.]{5,}$')


def _should_merge(prev: str, curr: str) -> bool:
    """Check if curr is a continuation of prev."""
    if not prev or not curr:
        return False
    prev_last = prev.strip()[-1] if prev.strip() else ''
    # Don't end properly → continuation
    if prev_last in _SENTENCE_END:
        return False
    # Prev is short fragment → merge
    if len(prev) < 40:
        return True
    # Prev doesn't end and both are on the shorter side
    return len(prev) < 200 and len(curr) < 200 and prev_last not in '.!?;'


def _is_noise(text: str) -> bool:
    """Filter obvious noise blocks."""
    stripped = text.strip()
    if len(stripped) < 10:
        return True
    if _NOISE_RE.match(stripped):
        return True
    # Block that's just numbers and dots (TOC artifact)
    if __import__('re').sub(r'[\d\s.]', '', stripped) == '':
        return True
    return False
