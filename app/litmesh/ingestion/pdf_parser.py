"""
PDF text extraction.

Uses pdfplumber for high-quality text extraction with page-level granularity.
Falls back to PyPDF2 if pdfplumber is unavailable.
"""

import hashlib
from pathlib import Path
from typing import Optional


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PDFExtractor:
    """Extract raw text from PDF with page-level metadata."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def extract(self) -> dict:
        """Extract full text and per-page text.

        Returns:
            {
                "full_text": str,
                "pages": [{"page_num": 1, "text": "..."}, ...],
                "page_count": int,
                "raw_text_hash": str (SHA256 of full_text)
            }
        """
        try:
            return self._extract_with_pdfplumber()
        except ImportError:
            return self._extract_with_pypdf2()

    def _extract_with_pdfplumber(self) -> dict:
        import pdfplumber

        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                pages.append({"page_num": i, "text": text})

        full_text = "\n\n".join(p["text"] for p in pages)
        return {
            "full_text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "raw_text_hash": sha256(full_text),
        }

    def _extract_with_pypdf2(self) -> dict:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(self.pdf_path))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append({"page_num": i, "text": text})

        full_text = "\n\n".join(p["text"] for p in pages)
        return {
            "full_text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "raw_text_hash": sha256(full_text),
        }
