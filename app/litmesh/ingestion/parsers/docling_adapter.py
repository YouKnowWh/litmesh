"""
Docling parser adapter.

Docling owns PDF layout recovery. This adapter uses its Markdown export as a
stable protocol and converts headings/paragraphs into LitMesh's ParsedDocument.
"""

from pathlib import Path

from ..parsed_document import ParsedDocument, ParsedElement, ElementType, OutlineItem, QualityReport


class DoclingAdapter:
    """Parse documents with IBM Docling when it is installed."""

    def __init__(self):
        self.name = "docling"
        import docling

        self._version = getattr(docling, "__version__", "")

    def parse(self, pdf_path: str) -> ParsedDocument:
        from docling.document_converter import DocumentConverter

        pdf_path = Path(pdf_path)
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        markdown = doc.export_to_markdown()
        plain_text = doc.export_to_text()

        elements = self._elements_from_markdown(markdown)
        outline = [
            OutlineItem(title=e.text, level=max(1, e.level), page=e.page_start, element_id=e.element_id)
            for e in elements
            if e.type == ElementType.HEADING
        ]
        pages = [{"page_num": 1, "text": plain_text or markdown}]

        paragraphs = sum(1 for e in elements if e.type == ElementType.PARAGRAPH)
        headings = sum(1 for e in elements if e.type == ElementType.HEADING)
        avg_len = sum(len(e.text) for e in elements) / max(len(elements), 1)
        suspicious_long_heading_count = sum(
            1 for e in elements if e.type == ElementType.HEADING and len(e.text) > 80
        )
        empty_fragment_count = sum(1 for e in elements if len(e.text.strip()) < 20)

        quality = QualityReport(
            parser_name=self.name,
            parser_version=self._version,
            total_elements=len(elements),
            paragraph_count=paragraphs,
            heading_count=headings,
            toc_detected=bool(outline),
            suspicious_long_heading_count=suspicious_long_heading_count,
            empty_fragment_count=empty_fragment_count,
            average_paragraph_length=round(avg_len, 1),
            needs_structure_review=paragraphs < 10 or suspicious_long_heading_count > 5,
            warnings=[],
        )
        if paragraphs < 10:
            quality.warnings.append("docling returned too few paragraph elements")
        if suspicious_long_heading_count > 5:
            quality.warnings.append("many headings look too long; layout may be noisy")

        return ParsedDocument(
            pages=pages,
            elements=elements,
            outline=outline,
            metadata={"filename": pdf_path.name},
            parser_name=self.name,
            parser_version=self._version,
            quality_report=quality,
            full_text=plain_text or markdown,
        )

    @staticmethod
    def _elements_from_markdown(markdown: str) -> list[ParsedElement]:
        elements: list[ParsedElement] = []
        order = 0
        paragraph_lines: list[str] = []

        def flush_paragraph():
            nonlocal order, paragraph_lines
            text = "\n".join(paragraph_lines).strip()
            paragraph_lines = []
            if not text:
                return
            order += 1
            elements.append(ParsedElement(
                element_id=f"docling_elem_{order}",
                type=ElementType.PARAGRAPH,
                text=text,
                page_start=1,
                page_end=1,
                order_index=order,
                confidence=0.85,
            ))

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                continue
            if line.startswith("#"):
                flush_paragraph()
                hashes = len(line) - len(line.lstrip("#"))
                text = line[hashes:].strip()
                if not text:
                    continue
                order += 1
                elements.append(ParsedElement(
                    element_id=f"docling_elem_{order}",
                    type=ElementType.HEADING,
                    text=text,
                    page_start=1,
                    page_end=1,
                    level=max(1, min(hashes, 6)),
                    order_index=order,
                    confidence=0.9,
                ))
                continue
            paragraph_lines.append(line)

        flush_paragraph()
        return elements
