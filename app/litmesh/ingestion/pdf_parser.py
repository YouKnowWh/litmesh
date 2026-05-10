"""
PDF text extraction with multi-engine support.

Priority:
1. PyMuPDF (fitz) — best CJK text extraction
2. pdfplumber — fallback with CJK cleanup
3. Tesseract OCR — for pages where text extraction produces garbage

The extractor automatically detects poor-quality text (high ratio of
fragmented/single-char lines) and falls back to OCR for those pages.
"""

import hashlib
import re
from pathlib import Path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- CJK text cleaning (line-break-preserving) ----

def _clean_cjk_spaces(text: str) -> str:
    """Merge spaces/tabs between CJK characters WITHOUT merging newlines."""
    text = re.sub(r'(?<=[一-鿿㐀-䶿豈-﫿])[ \t]+(?=[一-鿿㐀-䶿豈-﫿])', '', text)
    text = re.sub(r'(?<=[一-鿿㐀-䶿豈-﫿])[ \t]+(?=[　-〿＀-￯])', '', text)
    return text


# ---- Noise filters ----

_PAGE_NUMBER_RE = re.compile(r'^\s*\d{1,4}\s*$')
_RATIO_FRAGMENT_RE = re.compile(r'^[\d\s.:：∶]+$')
_UNIT_FRAGMENT_RE = re.compile(r'^\d+\s*(nm|μm|mm|cm|m|km|g|kg|ml|L|mol|℃|%)\s*$')
_CHAR_REPEAT_RE = re.compile(r'^(.)\1{15,}$')


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    if _RATIO_FRAGMENT_RE.match(stripped) and len(stripped) <= 15:
        return True
    if _UNIT_FRAGMENT_RE.match(stripped):
        return True
    if _CHAR_REPEAT_RE.match(stripped):
        return True
    return False


def _filter_lines(text: str) -> tuple[str, int]:
    lines = text.split('\n')
    kept, noise = [], 0
    for line in lines:
        if _is_noise_line(line):
            noise += 1
        else:
            kept.append(line)
    return '\n'.join(kept), noise


# ---- Text quality check ----

def _is_poor_text(text: str) -> bool:
    """Detect pages where text extraction produced garbage.

    Heuristic: if >60% of lines are single CJK characters (fragmented text
    from multi-column layouts), the page needs OCR.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 3:
        return False
    single_char = sum(1 for l in lines if len(l) == 1 and '一' <= l <= '鿿')
    return single_char > len(lines) * 0.6


# ---- PDF Extractor ----

class PDFExtractor:
    """Extract text from PDF with multi-engine support and OCR fallback."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    def extract(self) -> dict:
        # Try PyMuPDF first (best CJK support)
        try:
            return self._extract_with_pymupdf()
        except ImportError:
            pass

        # Fall back to pdfplumber
        try:
            return self._extract_with_pdfplumber()
        except ImportError:
            return self._extract_with_pypdf2()

    def _extract_with_pymupdf(self) -> dict:
        import fitz  # PyMuPDF

        doc = fitz.open(str(self.pdf_path))
        blocks = []       # All text blocks across pages
        pages = []        # Page-level text for TOC parsing
        ocr_pages = 0
        total_noise = 0
        global_block_id = 0

        for i, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            page_text = _clean_cjk_spaces(page_text)

            if _is_poor_text(page_text):
                page_text = self._ocr_page(page, i)
                ocr_pages += 1

            page_text, noise = _filter_lines(page_text)
            total_noise += noise
            pages.append({"page_num": i, "text": page_text})

            # Extract blocks (natural paragraph units from PDF layout)
            page_blocks = page.get_text("blocks")
            for b in page_blocks:
                x0, y0, x1, y1, text, block_type, block_no = b
                if block_type != 0:  # Skip images
                    continue
                text = _clean_cjk_spaces(text.strip())
                if len(text) < 20:  # Skip tiny fragments
                    continue
                global_block_id += 1
                blocks.append({
                    "block_id": global_block_id,
                    "page_num": i,
                    "text": text,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                })

        doc.close()

        # Build full_text from blocks (preserves paragraph structure)
        full_text = "\n\n".join(b["text"] for b in blocks)
        full_text = _clean_cjk_spaces(full_text)

        return {
            "full_text": full_text,
            "pages": pages,
            "blocks": blocks,
            "page_count": len(pages),
            "block_count": len(blocks),
            "raw_text_hash": sha256(full_text),
            "structure_report": {
                "total_pages": len(pages),
                "total_blocks": len(blocks),
                "ocr_pages": ocr_pages,
                "noise_lines_filtered": total_noise,
                "engine": "pymupdf+blocks",
            },
            "layout_lines": [],
        }

    def _ocr_page(self, page, page_num: int) -> str:
        """Run Tesseract OCR on a page rendered as image."""
        try:
            import pytesseract
            from PIL import Image
            import io

            # Render page at 300 DPI for good OCR quality
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="chi_sim+chi_tra")
            return text
        except Exception:
            # OCR failed, return the garbled text as-is
            return page.get_text("text") or ""

    def _extract_with_pdfplumber(self) -> dict:
        import pdfplumber

        pages = []
        total_noise = 0

        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text(
                    x_tolerance=2, y_tolerance=2, keep_blank_chars=False,
                ) or ""
                text = _clean_cjk_spaces(text)
                text, noise = _filter_lines(text)
                total_noise += noise
                pages.append({"page_num": i, "text": text})

        full_text = "\n".join(p["text"] for p in pages)
        full_text = _clean_cjk_spaces(full_text)

        return {
            "full_text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "raw_text_hash": sha256(full_text),
            "structure_report": {
                "total_pages": len(pages),
                "ocr_pages": 0,
                "noise_lines_filtered": total_noise,
                "engine": "pdfplumber",
            },
            "layout_lines": [],
        }

    def _extract_with_pypdf2(self) -> dict:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(self.pdf_path))
        pages = []
        total_noise = 0

        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            text = _clean_cjk_spaces(text)
            text, noise = _filter_lines(text)
            total_noise += noise
            pages.append({"page_num": i, "text": text})

        full_text = "\n".join(p["text"] for p in pages)
        full_text = _clean_cjk_spaces(full_text)

        return {
            "full_text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "raw_text_hash": sha256(full_text),
            "structure_report": {
                "total_pages": len(pages),
                "ocr_pages": 0,
                "noise_lines_filtered": total_noise,
                "engine": "pypdf2",
            },
            "layout_lines": [],
        }
