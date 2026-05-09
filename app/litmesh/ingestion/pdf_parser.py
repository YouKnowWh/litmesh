"""
PDF text extraction with CJK support.

Uses pdfplumber with tolerance tuning for Chinese text layout.
Falls back to PyPDF2 if pdfplumber is unavailable.
"""

import hashlib
import re
from pathlib import Path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# CJK character range for cleaning
_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')


def _clean_cjk_text(text: str) -> str:
    """Fix common CJK PDF extraction artifacts from complex layouts.

    Chinese PDFs with multi-column, embedded fonts often produce:
      - Characters split across lines with spaces
      - Duplicate characters from overlapping text boxes
      - Random line breaks mid-sentence
    """
    # 1. Remove space between two CJK chars: "生 物" -> "生物"
    text = re.sub(r'(?<=[一-鿿㐀-䶿豈-﫿])\s+(?=[一-鿿㐀-䶿豈-﫿])', '', text)
    # 2. Remove space between CJK and punctuation
    text = re.sub(r'(?<=[一-鿿㐀-䶿豈-﫿])\s+(?=[　-〿＀-￯])', '', text)
    # 3. Merge lines that are single CJK chars with spaces (column-split artifacts)
    #    "普\n通\n高\n中" -> "普通高中"
    lines = text.split('\n')
    merged = []
    buf = ''
    for line in lines:
        stripped = line.strip()
        # If line is just 1-3 CJK chars (possibly with spaces), buffer it
        if stripped and len(stripped) <= 6 and all(
            '一' <= c <= '鿿' or '　' <= c <= '〿' or c == ' '
            for c in stripped if c != ' '
        ):
            buf += stripped.replace(' ', '')
        else:
            if buf:
                merged.append(buf)
                buf = ''
            merged.append(stripped)
    if buf:
        merged.append(buf)
    text = '\n'.join(merged)
    # 4. Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 5. Remove lines that are just repeated chars (noise from PDF structure)
    text = re.sub(r'\n(.)\\1{15,}\n', '\n', text)
    return text


class PDFExtractor:
    """Extract raw text from PDF with page-level granularity and CJK cleanup."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def extract(self) -> dict:
        try:
            return self._extract_with_pdfplumber()
        except ImportError:
            return self._extract_with_pypdf2()
        except Exception as e:
            # If pdfplumber completely fails, try PyPDF2
            print(f"pdfplumber failed: {e}, falling back to PyPDF2")
            try:
                return self._extract_with_pypdf2()
            except ImportError:
                raise

    def _extract_with_pdfplumber(self) -> dict:
        import pdfplumber

        pages = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                # Higher tolerance helps with Chinese PDF column layouts
                text = page.extract_text(
                    x_tolerance=2,
                    y_tolerance=2,
                    keep_blank_chars=False,
                ) or ""
                text = _clean_cjk_text(text)
                pages.append({"page_num": i, "text": text})

        full_text = "\n\n".join(p["text"] for p in pages)
        full_text = _clean_cjk_text(full_text)
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
            text = _clean_cjk_text(text)
            pages.append({"page_num": i, "text": text})

        full_text = "\n\n".join(p["text"] for p in pages)
        full_text = _clean_cjk_text(full_text)
        return {
            "full_text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "raw_text_hash": sha256(full_text),
        }
